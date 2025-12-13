"""
Handcrafted feature extraction pipeline.

Extracts handcrafted features from videos including:
- Noise residual features
- DCT statistics
- Blur/sharpness metrics
- Boundary inconsistency
- Codec cues
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path, get_video_metadata_cache_path, load_metadata_flexible, write_metadata_atomic
from lib.utils.memory import aggressive_gc, log_memory_stats, safe_execute
from lib.utils.schemas import validate_stage_output, PANDERA_AVAILABLE
from lib.features.handcrafted import extract_all_features, HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


def extract_features_from_video(
    video_path: str,
    num_frames: Optional[int] = None,
    extractor: Optional[HandcraftedFeatureExtractor] = None,
    frame_percentage: Optional[float] = None,
    min_frames: int = 5,
    max_frames: int = 50,
    project_root: Optional[str] = None
) -> dict:
    """
    Extract features from a video by sampling frames.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (if provided, overrides percentage-based calculation)
        extractor: Feature extractor instance
        frame_percentage: Percentage of frames to sample (default: 0.10 = 10% if num_frames not provided)
        min_frames: Minimum frames to sample (for percentage-based sampling, default: 5)
        max_frames: Maximum frames to sample (for percentage-based sampling, default: 50)
        project_root: Project root directory (optional, used for cache file path)
    
    Returns:
        Dictionary of aggregated features
    """
    # Input validation
    if not video_path or not isinstance(video_path, str):
        raise ValueError(f"video_path must be a non-empty string, got: {type(video_path)}")
    if num_frames is not None and (not isinstance(num_frames, int) or num_frames <= 0):
        raise ValueError(f"num_frames must be a positive integer, got: {num_frames}")
    if frame_percentage is not None and (not isinstance(frame_percentage, (int, float)) or not (0 < frame_percentage <= 1)):
        raise ValueError(f"frame_percentage must be between 0 and 1, got: {frame_percentage}")
    if not isinstance(min_frames, int) or min_frames <= 0:
        raise ValueError(f"min_frames must be a positive integer, got: {min_frames}")
    if not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError(f"max_frames must be a positive integer, got: {max_frames}")
    if min_frames > max_frames:
        raise ValueError(f"min_frames ({min_frames}) must be <= max_frames ({max_frames})")
    
    if extractor is None:
        extractor = HandcraftedFeatureExtractor()
    
    # Use cached metadata to avoid duplicate frame counting
    from lib.utils.video_cache import get_video_metadata
    from lib.utils.paths import calculate_adaptive_num_frames
    
    # Use persistent cache file for cross-stage caching (if project_root provided)
    try:
        if project_root:
            cache_file = get_video_metadata_cache_path(project_root)
        else:
            cache_file = None
        metadata = get_video_metadata(video_path, use_cache=True, cache_file=cache_file)
        if metadata is None:
            logger.warning(f"Could not get video metadata for {video_path}")
            return {}
        total_frames = metadata.get('total_frames', 0)
    except Exception as e:
        logger.error(f"Failed to get video metadata for {video_path}: {e}")
        return {}
    
    if total_frames == 0:
        logger.warning(f"Video has no frames: {video_path}")
        return {}
    
    # Calculate number of frames to sample
    if num_frames is not None:
        # Use fixed number of frames (backward compatible)
        frames_to_sample = num_frames
    else:
        # Use percentage-based adaptive sampling
        if frame_percentage is None:
            frame_percentage = 0.10  # Default 10%
        frames_to_sample = calculate_adaptive_num_frames(
            total_frames, frame_percentage, min_frames, max_frames
        )
        logger.debug(f"Adaptive sampling: {total_frames} total frames -> {frames_to_sample} frames ({frame_percentage*100:.1f}%, bounded [{min_frames}, {max_frames}])")
    
    container = None
    try:
        # Validate video file exists
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return {}
        if not video_path_obj.is_file():
            logger.error(f"Video path is not a file: {video_path}")
            return {}
        
        container = av.open(str(video_path))
        if len(container.streams.video) == 0:
            logger.error(f"Video has no video streams: {video_path}")
            container.close()
            return {}
        
        stream = container.streams.video[0]
        
        # Sample frames uniformly
        if total_frames <= 0:
            logger.warning(f"Video has {total_frames} frames: {video_path}")
            container.close()
            return {}
        
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
        
        all_features = []
        frame_count = 0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_count in frame_indices:
                    try:
                        frame_array = frame.to_ndarray(format='rgb24')
                        features = extractor.extract(frame_array, video_path)
                        if features:
                            all_features.append(features)
                    except Exception as e:
                        logger.warning(f"Failed to extract features from frame {frame_count} of {video_path}: {e}")
                    finally:
                        # Aggressive GC after each frame extraction
                        if 'frame_array' in locals():
                            del frame_array
                        aggressive_gc(clear_cuda=False)
                
                frame_count += 1
                if frame_count >= total_frames or len(all_features) >= frames_to_sample:
                    break
            
            if frame_count >= total_frames or len(all_features) >= frames_to_sample:
                break
        
        # Aggregate features across frames (mean)
        if not all_features:
            logger.warning(f"No features extracted from {video_path}")
            return {}
        
        aggregated = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features if key in f]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        
        return aggregated
        
    except av.AVError as e:
        logger.error(f"PyAV error extracting features from {video_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to extract features from {video_path}: {e}", exc_info=True)
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception as e:
                logger.debug(f"Error closing video container: {e}")
        aggressive_gc(clear_cuda=False)


def stage2_extract_features(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/features_stage2",
    num_frames: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    delete_existing: bool = False,
    resume: bool = True,
    frame_percentage: Optional[float] = None,
    min_frames: int = 5,
    max_frames: int = 50,
    execution_order: str = "forward"
) -> pl.DataFrame:
    """
    Stage 2: Extract handcrafted features from all augmented videos.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to augmented metadata CSV/Arrow/Parquet
        output_dir: Directory to save features
        num_frames: Number of frames to sample per video (if provided, overrides percentage-based calculation)
        start_idx: Start index for video range (0-based, inclusive). If None, starts from 0.
        end_idx: End index for video range (0-based, exclusive). If None, processes all videos.
        delete_existing: If True, delete existing feature files before regenerating (clean mode)
        resume: If True, skip videos where feature files already exist (resume mode)
        frame_percentage: Percentage of frames to sample (default: 0.10 = 10% if num_frames not provided)
        min_frames: Minimum frames to sample (for percentage-based sampling, default: 5)
        max_frames: Maximum frames to sample (for percentage-based sampling, default: 50)
    
    Returns:
        DataFrame with feature metadata
    """
    # Input validation
    if not project_root or not isinstance(project_root, str):
        raise ValueError(f"project_root must be a non-empty string, got: {type(project_root)}")
    if not augmented_metadata_path or not isinstance(augmented_metadata_path, str):
        raise ValueError(f"augmented_metadata_path must be a non-empty string, got: {type(augmented_metadata_path)}")
    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be a string, got: {type(output_dir)}")
    if start_idx is not None and (not isinstance(start_idx, int) or start_idx < 0):
        raise ValueError(f"start_idx must be a non-negative integer, got: {start_idx}")
    if end_idx is not None and (not isinstance(end_idx, int) or end_idx < 0):
        raise ValueError(f"end_idx must be a non-negative integer, got: {end_idx}")
    if execution_order not in ["forward", "reverse"]:
        raise ValueError(f"execution_order must be 'forward' or 'reverse', got: {execution_order}")
    
    try:
        project_root_path = Path(project_root).resolve()
        if not project_root_path.exists():
            raise FileNotFoundError(f"Project root directory does not exist: {project_root_path}")
        if not project_root_path.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {project_root_path}")
    except (OSError, ValueError) as e:
        logger.error(f"Invalid project_root path: {project_root} - {e}")
        raise ValueError(f"Invalid project_root path: {project_root}") from e
    
    project_root_str = str(project_root_path)  # Keep as string for function calls
    
    try:
        output_dir_path = project_root_path / output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise ValueError(f"Cannot create output directory: {output_dir}") from e
    
    output_dir = output_dir_path  # Use Path object for path operations
    
    # Load augmented metadata (support CSV, Arrow, and Parquet)
    logger.info("Stage 2: Loading augmented metadata...")
    from lib.utils.paths import load_metadata_flexible, validate_metadata_columns, write_metadata_atomic
    
    try:
        df = load_metadata_flexible(augmented_metadata_path)
    except Exception as e:
        logger.error(f"Failed to load augmented metadata from {augmented_metadata_path}: {e}")
        raise
    if df is None:
        logger.error(f"Augmented metadata not found: {augmented_metadata_path} (checked .arrow, .parquet, .csv)")
        return pl.DataFrame()
    
    # Validate required columns
    try:
        validate_metadata_columns(df, ["video_path", "label"], "Stage 2")
    except ValueError as e:
        logger.error(f"{e}")
        return pl.DataFrame()
    
    # Validate Stage 1 output schema using Pandera
    if PANDERA_AVAILABLE:
        logger.info("Stage 2: Validating Stage 1 output schema...")
        if not validate_stage_output(df, stage=1):
            logger.warning("Stage 1 output validation failed, but continuing...")
    else:
        logger.warning("Pandera not available, skipping schema validation. Install with: pip install pandera")
    
    # Apply range filtering if specified
    total_videos = df.height
    if total_videos == 0:
        logger.warning("Stage 2: No videos found in metadata")
        return pl.DataFrame()
    
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else total_videos
        if start < 0:
            start = 0
        if end > total_videos:
            end = total_videos
        if start >= end:
            logger.warning(f"Invalid range: start_idx={start}, end_idx={end}, total_videos={total_videos}. Skipping.")
            return pl.DataFrame()
        df = df.slice(start, end - start)
        logger.info(f"Stage 2: Processing video range [{start}, {end}) of {total_videos} total videos")
        
        # Check if range resulted in empty DataFrame
        if df.height == 0:
            logger.warning(f"Stage 2: Range [{start}, {end}) resulted in empty DataFrame")
            return pl.DataFrame()
    else:
        logger.info(f"Stage 2: Processing all {total_videos} videos")
    
    # Log detailed video count information
    logger.info(f"Stage 2: Processing {df.height} videos in this range")
    
    # Check for expected video count (298 original * 11 augmentations = 3278)
    if 'is_original' in df.columns:
        # CODE QUALITY: Use Polars boolean filtering instead of == True/False
        original_count = df.filter(pl.col('is_original')).height
        augmented_count = df.filter(~pl.col('is_original')).height
        logger.info(f"  - Original videos: {original_count}")
        logger.info(f"  - Augmented videos: {augmented_count}")
        logger.info(f"  - Total videos: {df.height}")
        
        # Warn if count doesn't match expected
        if original_count > 0:
            expected_total = original_count * 11  # Assuming 10 augmentations + 1 original
            if df.height < expected_total:
                missing = expected_total - df.height
                logger.warning(f"  - Expected ~{expected_total} videos (298 * 11), but found {df.height} videos")
                logger.warning(f"  - Missing {missing} videos - some augmentations may have failed")
                
                # Try to identify which videos are missing
                aug_per_original = augmented_count / original_count
                logger.info(f"  - Average augmentations per original video: {aug_per_original:.2f}")
    else:
        logger.info(f"  - Total videos in metadata: {df.height}")
        logger.info("  - Note: 'is_original' column not found, cannot determine original vs augmented count")
    
    # Load existing metadata if it exists (for resume mode)
    existing_metadata = None
    existing_feature_paths = set()
    
    if resume and not delete_existing:
        # Try to load existing metadata (check all formats)
        # Use retry logic to handle race conditions and corrupted files
        existing_metadata_path = output_dir / "features_metadata"
        try:
            existing_metadata = load_metadata_flexible(str(existing_metadata_path), max_retries=5, retry_delay=1.0)
            if existing_metadata is not None and existing_metadata.height > 0:
                existing_feature_paths = set(existing_metadata["feature_path"].to_list())
                logger.info(f"Stage 2: Found {len(existing_feature_paths)} existing feature files (resume mode)")
            else:
                logger.info("Stage 2: No existing metadata found or metadata is empty, starting fresh")
                existing_metadata = None
                existing_feature_paths = set()
        except Exception as e:
            logger.warning(f"Stage 2: Could not load existing metadata (will start fresh): {e}")
            existing_metadata = None
            existing_feature_paths = set()
    
    # Delete existing feature files if clean mode
    if delete_existing:
        logger.info("Stage 2: Deleting existing feature files (clean mode)...")
        deleted_count = 0
        for feature_file in output_dir.glob("*_features.*"):
            if feature_file.name not in ["features_metadata.arrow", "features_metadata.parquet"]:
                try:
                    feature_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {feature_file}: {e}")
        logger.info(f"Stage 2: Deleted {deleted_count} existing feature files")
        existing_feature_paths = set()  # Clear after deletion
    
    extractor = HandcraftedFeatureExtractor()
    feature_rows = []
    skipped_count = 0
    processed_count = 0
    
    # Determine iteration order
    if execution_order == "reverse":
        indices = range(df.height - 1, -1, -1)  # Reverse: from end to start
        logger.info("Stage 2: Processing videos in REVERSE order (from end to start)")
    else:
        indices = range(df.height)  # Forward: from start to end (default)
        logger.info("Stage 2: Processing videos in FORWARD order (from start to end)")
    
    iteration_count = 0
    for idx in indices:
        iteration_count += 1
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root_str)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            if iteration_count % 10 == 0:
                log_memory_stats(f"Stage 2: processing video {iteration_count}/{df.height} (index {idx})")
            
            # Check if feature file already exists (resume mode)
            video_id = Path(video_path).stem
            feature_path_parquet = output_dir / f"{video_id}_features.parquet"
            feature_path_npy = output_dir / f"{video_id}_features.npy"
            feature_path_rel = str(feature_path_parquet.relative_to(project_root_path))
            
            if resume and not delete_existing:
                # Check if feature file exists
                if feature_path_parquet.exists() or feature_path_npy.exists():
                    if feature_path_rel in existing_feature_paths or feature_path_parquet.exists() or feature_path_npy.exists():
                        logger.debug(f"Skipping {video_path} - feature file already exists")
                        skipped_count += 1
                        # Still add to metadata if not already present
                        if existing_metadata is None or feature_path_rel not in existing_feature_paths:
                            # Try to load existing features to add to metadata
                            try:
                                if feature_path_parquet.exists():
                                    import pyarrow.parquet as pq
                                    table = pq.read_table(str(feature_path_parquet))
                                    features_dict = {col: table[col][0].as_py() for col in table.column_names}
                                else:
                                    features_dict = np.load(str(feature_path_npy), allow_pickle=True).item()
                                
                                feature_row = {
                                    "video_path": video_rel,
                                    "label": label,
                                    "feature_path": feature_path_rel,
                                }
                                feature_row.update(features_dict)
                                feature_rows.append(feature_row)
                            except Exception as e:
                                logger.warning(f"Could not load existing features from {feature_path_parquet}: {e}")
                        continue
            
            # Extract features with OOM handling
            features = safe_execute(
                extract_features_from_video,
                video_path, 
                num_frames=num_frames,
                extractor=extractor,
                frame_percentage=frame_percentage,
                min_frames=min_frames,
                max_frames=max_frames,
                project_root=project_root_str,
                oom_retry=True,
                max_retries=1,
                context=f"Stage 2: extracting features from {Path(video_path).name}"
            )
            
            if not features:
                logger.warning(f"No features extracted from {video_path}")
                continue
            
            # Save features as Arrow/Parquet (better than .npy)
            feature_path = feature_path_parquet
            
            # Convert features dict to Arrow table and save as Parquet
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                # Convert dict to columnar format (each key becomes a column with single value)
                feature_dict = {k: [v] for k, v in features.items()}
                table = pa.Table.from_pydict(feature_dict)
                pq.write_table(table, str(feature_path), compression='snappy')
            except ImportError:
                # Fallback to numpy if pyarrow not available
                logger.warning("PyArrow not available, falling back to .npy format")
                feature_path = output_dir / f"{video_id}_features.npy"
                np.save(str(feature_path), features)
            
            # Create metadata row
            feature_row = {
                "video_path": video_rel,
                "label": label,
                "feature_path": str(feature_path.relative_to(project_root_path)),
            }
            feature_row.update(features)  # Add all feature values
            feature_rows.append(feature_row)
            processed_count += 1
            
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            continue
    
    logger.info(f"Stage 2: Processed {processed_count} videos, skipped {skipped_count} videos")
    
    if not feature_rows:
        logger.warning("Stage 2: No new features extracted! (may have all been skipped)")
        # Return existing metadata if available
        if existing_metadata is not None:
            logger.info("Stage 2: Returning existing metadata")
            return existing_metadata
        return pl.DataFrame()
    
    # Merge new metadata_rows with existing metadata and write final file
    logger.info("=" * 80)
    logger.info("Stage 2: Merging new features with existing metadata...")
    logger.info("=" * 80)
    
    # Create DataFrame from new feature_rows
    new_features_df = pl.DataFrame(feature_rows) if feature_rows else pl.DataFrame()
    
    if new_features_df.height > 0:
        logger.info(f"New features to add: {new_features_df.height} entries")
        
        # Merge with existing metadata (avoid duplicates)
        if existing_metadata is not None and existing_metadata.height > 0 and not delete_existing:
            # Create a set of existing entries to avoid duplicates
            existing_keys = set()
            for row in existing_metadata.iter_rows(named=True):
                key = row.get("video_path", "")
                existing_keys.add(key)
            
            # Filter out duplicates from new entries
            new_entries_filtered = []
            for row in feature_rows:
                key = row.get("video_path", "")
                if key not in existing_keys:
                    new_entries_filtered.append(row)
            
            if new_entries_filtered:
                new_features_df = pl.DataFrame(new_entries_filtered)
                logger.info(f"After deduplication: {new_features_df.height} new entries to add")
                combined_features_df = pl.concat([existing_metadata, new_features_df])
            else:
                logger.info("All new entries already exist in metadata, no merge needed")
                combined_features_df = existing_metadata
        else:
            combined_features_df = new_features_df
        
        # Write final metadata file (prefer Arrow, fallback to Parquet, then CSV)
        logger.info(f"Writing final metadata file with {combined_features_df.height} total entries...")
        
        # Try Arrow first
        final_metadata_path = output_dir / "features_metadata.arrow"
        success = write_metadata_atomic(combined_features_df, final_metadata_path, append=False)
        
        if not success:
            # Fallback to Parquet
            final_metadata_path = output_dir / "features_metadata.parquet"
            success = write_metadata_atomic(combined_features_df, final_metadata_path, append=False)
            if success:
                logger.info(f"✓ Saved metadata as Parquet: {final_metadata_path}")
        else:
            logger.info(f"✓ Saved metadata as Arrow IPC: {final_metadata_path}")
        
        if not success:
            # Final fallback to CSV
            final_metadata_path = output_dir / "features_metadata.csv"
            try:
                combined_features_df.write_csv(final_metadata_path)
                logger.info(f"✓ Saved metadata as CSV: {final_metadata_path}")
                success = True
            except Exception as e:
                logger.error(f"Failed to save metadata as CSV: {e}")
                success = False
        
        # Remove old metadata files if format changed
        metadata_paths_to_check = [
            output_dir / "features_metadata.arrow",
            output_dir / "features_metadata.parquet",
            output_dir / "features_metadata.csv"
        ]
        for old_path in metadata_paths_to_check:
            if old_path != final_metadata_path and old_path.exists():
                try:
                    old_path.unlink()
                    logger.debug(f"Removed old metadata file: {old_path}")
                except Exception:
                    pass
        
        if success:
            logger.info(f"✓ Final metadata file written: {final_metadata_path}")
            logger.info(f"  Total entries: {combined_features_df.height}")
            original_count = existing_metadata.height if existing_metadata is not None else 0
            new_count = len(new_entries_filtered) if 'new_entries_filtered' in locals() and new_entries_filtered else new_features_df.height
            logger.info(f"  Original entries: {original_count}")
            logger.info(f"  New entries added: {new_count}")
            return combined_features_df
        else:
            logger.error("Failed to write final metadata file!")
            # Try to return existing metadata
            if existing_metadata is not None:
                return existing_metadata
            return pl.DataFrame()
    else:
        logger.info("✓ Stage 2 complete: No new features to save (all may have been skipped)")
        # Reload final metadata file to return complete dataset
        metadata_paths_to_check = [
            output_dir / "features_metadata.arrow",
            output_dir / "features_metadata.parquet",
            output_dir / "features_metadata.csv"
        ]
        for metadata_path in metadata_paths_to_check:
            if metadata_path.exists():
                try:
                    final_metadata = load_metadata_flexible(str(metadata_path), max_retries=3, retry_delay=0.5)
                    if final_metadata is not None:
                        logger.info(f"Stage 2: Returning complete metadata from {metadata_path} ({final_metadata.height} entries)")
                        return final_metadata
                except Exception as e:
                    logger.debug(f"Could not load metadata from {metadata_path}: {e}")
        
        # Fallback to existing_metadata if available
        if existing_metadata is not None:
            logger.info("Stage 2: Returning existing metadata")
            return existing_metadata
        return pl.DataFrame()

