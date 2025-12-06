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
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from lib.utils.schemas import validate_stage_output, PANDERA_AVAILABLE
from lib.features.handcrafted import extract_all_features, HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


def extract_features_from_video(
    video_path: str,
    num_frames: Optional[int] = None,
    extractor: Optional[HandcraftedFeatureExtractor] = None,
    frame_percentage: Optional[float] = None,
    min_frames: int = 5,
    max_frames: int = 50
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
    
    Returns:
        Dictionary of aggregated features
    """
    if extractor is None:
        extractor = HandcraftedFeatureExtractor()
    
    # Use cached metadata to avoid duplicate frame counting
    from lib.utils.video_cache import get_video_metadata
    from lib.utils.paths import calculate_adaptive_num_frames
    
    metadata = get_video_metadata(video_path, use_cache=True)
    total_frames = metadata['total_frames']
    
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
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
        
        all_features = []
        frame_count = 0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_count in frame_indices:
                    frame_array = frame.to_ndarray(format='rgb24')
                    features = extractor.extract(frame_array, video_path)
                    all_features.append(features)
                    
                    # Aggressive GC after each frame extraction
                    del frame_array
                    aggressive_gc(clear_cuda=False)
                
                frame_count += 1
                if frame_count >= total_frames or len(all_features) >= frames_to_sample:
                    break
            
            if frame_count >= total_frames or len(all_features) >= frames_to_sample:
                break
        
        # Aggregate features across frames (mean)
        if not all_features:
            return {}
        
        aggregated = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features if key in f]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Failed to extract features from {video_path}: {e}")
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
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
    max_frames: int = 50
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
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented metadata (support CSV, Arrow, and Parquet)
    logger.info("Stage 2: Loading augmented metadata...")
    from lib.utils.paths import load_metadata_flexible, validate_metadata_columns
    
    df = load_metadata_flexible(augmented_metadata_path)
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
        original_count = df.filter(pl.col('is_original') == True).height
        augmented_count = df.filter(pl.col('is_original') == False).height
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
        existing_metadata_path = output_dir / "features_metadata"
        existing_metadata = load_metadata_flexible(str(existing_metadata_path))
        if existing_metadata is not None:
            existing_feature_paths = set(existing_metadata["feature_path"].to_list())
            logger.info(f"Stage 2: Found {len(existing_feature_paths)} existing feature files (resume mode)")
    
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
    
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            if idx % 10 == 0:
                log_memory_stats(f"Stage 2: processing video {idx + 1}/{df.height}")
            
            # Check if feature file already exists (resume mode)
            video_id = Path(video_path).stem
            feature_path_parquet = output_dir / f"{video_id}_features.parquet"
            feature_path_npy = output_dir / f"{video_id}_features.npy"
            feature_path_rel = str(feature_path_parquet.relative_to(project_root))
            
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
            
            # Extract features
            features = extract_features_from_video(
                video_path, 
                num_frames=num_frames,
                extractor=extractor,
                frame_percentage=frame_percentage,
                min_frames=min_frames,
                max_frames=max_frames
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
                "feature_path": str(feature_path.relative_to(project_root)),
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
    
    # Create DataFrame from new features
    new_features_df = pl.DataFrame(feature_rows)
    
    # Merge with existing metadata if available
    if existing_metadata is not None and not delete_existing:
        # Combine existing and new features, removing duplicates
        features_df = pl.concat([existing_metadata, new_features_df]).unique(subset=["video_path"], keep="last")
        logger.info(f"Stage 2: Merged with existing metadata. Total: {features_df.height} videos")
    else:
        features_df = new_features_df
    
    # Save metadata as Arrow IPC (faster and type-safe)
    metadata_path = output_dir / "features_metadata.arrow"
    try:
        features_df.write_ipc(str(metadata_path))
        logger.debug(f"Saved metadata as Arrow IPC: {metadata_path}")
    except Exception as e:
        # Fallback to Parquet if IPC fails
        logger.warning(f"Arrow IPC write failed, using Parquet: {e}")
        metadata_path = output_dir / "features_metadata.parquet"
        features_df.write_parquet(str(metadata_path))
    logger.info(f"✓ Stage 2 complete: Saved features to {output_dir}")
    logger.info(f"✓ Stage 2: Extracted features from {len(feature_rows)} videos")
    
    return features_df

