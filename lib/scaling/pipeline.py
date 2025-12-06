"""
Video scaling pipeline.

Scales videos to target resolutions using letterbox resizing or autoencoder
methods while preserving aspect ratios. Can both downscale and upscale videos
to ensure max(width, height) = target_size.
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
from lib.utils.memory import aggressive_gc, log_memory_stats, check_oom_error, handle_oom_error, get_memory_stats
from lib.scaling.methods import (
    scale_video_frames,
    letterbox_resize,
    load_hf_autoencoder
)
from lib.augmentation.io import load_frames, save_frames, concatenate_videos

logger = logging.getLogger(__name__)


def scale_video(
    video_path: str,
    output_path: str,
    target_size: int = 256,
    max_frames: Optional[int] = 100,
    chunk_size: int = 100,
    method: str = "autoencoder",
    autoencoder: Optional[object] = None
) -> bool:
    """
    Scale a single video to target max dimension using chunked processing to avoid OOM.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size)
    to ensure max(width, height) = target_size.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        target_size: Target max dimension (max(width, height) will be target_size)
        max_frames: Maximum frames to process per chunk (default: 100 for 64GB RAM)
        chunk_size: Number of frames to process per chunk (default: 100 for 64GB RAM)
        method: Scaling method ("letterbox" or "autoencoder")
        autoencoder: Optional autoencoder model for autoencoder method
    
    Returns:
        True if successful, False otherwise
    """
    import tempfile
    import shutil
    
    if max_frames is None:
        max_frames = 100
    if chunk_size is None:
        chunk_size = 100
    
    container = None
    temp_dir = None
    try:
        # Use cached metadata to avoid duplicate frame counting
        from lib.utils.video_cache import get_video_metadata
        
        # Get video metadata - handle any format from Stage 1
        try:
            metadata = get_video_metadata(video_path, use_cache=True)
            total_frames = metadata.get('total_frames', 0)
            fps = metadata.get('fps', 30.0)  # Default FPS if not available
            
            # Validate metadata
            if total_frames <= 0:
                logger.warning(f"Video {video_path} has {total_frames} frames, may be empty or corrupted")
                # Continue anyway - let the processing attempt to handle it
            if fps <= 0:
                logger.warning(f"Video {video_path} has invalid FPS {fps}, using default 30.0")
                fps = 30.0
        except Exception as e:
            logger.error(f"Could not get video metadata for {video_path}: {e}")
            return False
        
        container = None
        aggressive_gc(clear_cuda=False)
        
        logger.debug(f"Video has {total_frames} frames, processing in chunks of {chunk_size}")
        
        # Create temporary directory for intermediate chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="scale_chunks_"))
        intermediate_files = []
        
        # Process video in chunks
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, total_frames)
            
            if start_frame >= total_frames:
                break
            
            logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame-1})")
            
            # Proactive memory check before processing chunk
            try:
                mem_stats = get_memory_stats()
                cpu_memory_gb = mem_stats.get("cpu_memory_gb", 0)
                gpu_allocated_gb = mem_stats.get("gpu_allocated_gb", 0)
                
                # Warn if memory usage is high (>50GB CPU or >10GB GPU)
                if cpu_memory_gb > 50:
                    logger.warning(f"High CPU memory usage before chunk {chunk_idx + 1}: {cpu_memory_gb:.2f}GB")
                    aggressive_gc(clear_cuda=True)
                if gpu_allocated_gb > 10:
                    logger.warning(f"High GPU memory usage before chunk {chunk_idx + 1}: {gpu_allocated_gb:.2f}GB")
                    aggressive_gc(clear_cuda=True)
            except Exception as e:
                logger.debug(f"Could not check memory stats: {e}")
            
            # Load chunk - handle any video format/codec from Stage 1
            try:
                chunk_frames, chunk_fps = load_frames(video_path, max_frames=chunk_size, start_frame=start_frame)
            except Exception as e:
                if check_oom_error(e):
                    handle_oom_error(e, f"loading chunk {chunk_idx + 1}")
                    logger.warning(f"OOM while loading chunk {chunk_idx + 1}, skipping chunk")
                    aggressive_gc(clear_cuda=True)
                    continue
                # Handle codec/format errors gracefully
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['codec', 'format', 'decode', 'corrupt', 'invalid']):
                    logger.warning(f"Video codec/format error loading chunk {chunk_idx + 1} from {video_path}: {e}")
                    logger.warning(f"This may indicate an incompatible or corrupted video from Stage 1")
                    continue
                raise
            
            if not chunk_frames:
                logger.warning(f"No frames loaded for chunk {chunk_idx + 1}, skipping")
                continue
            
            # Scale frames in chunk with OOM handling
            try:
                if method == "autoencoder" and autoencoder is not None:
                    # Use autoencoder for scaling (preserves aspect ratio)
                    try:
                        scaled_chunk = scale_video_frames(
                            chunk_frames,
                            method="autoencoder",
                            target_size=target_size,
                            autoencoder=autoencoder,
                            preserve_aspect_ratio=True
                        )
                        # Aggressive GC after autoencoder processing
                        aggressive_gc(clear_cuda=True)
                    except Exception as e:
                        if check_oom_error(e):
                            handle_oom_error(e, f"autoencoder scaling chunk {chunk_idx + 1}")
                            logger.warning(f"OOM during autoencoder scaling, falling back to letterbox for chunk {chunk_idx + 1}")
                            # Fallback to letterbox for this chunk
                            scaled_chunk = []
                            for frame_idx, frame in enumerate(chunk_frames):
                                try:
                                    scaled_frame = letterbox_resize(frame, target_size)
                                    scaled_chunk.append(scaled_frame)
                                    # Aggressive GC every 25 frames (more frequent for safety)
                                    if (frame_idx + 1) % 25 == 0:
                                        aggressive_gc(clear_cuda=False)
                                except Exception as frame_e:
                                    if check_oom_error(frame_e):
                                        handle_oom_error(frame_e, f"letterbox frame {frame_idx}")
                                        logger.warning(f"OOM on frame {frame_idx}, skipping frame")
                                        aggressive_gc(clear_cuda=True)
                                        continue
                                    raise
                            aggressive_gc(clear_cuda=True)
                        else:
                            raise
                else:
                    # Use letterbox resize
                    scaled_chunk = []
                    for frame_idx, frame in enumerate(chunk_frames):
                        try:
                            scaled_frame = letterbox_resize(frame, target_size)
                            scaled_chunk.append(scaled_frame)
                            
                            # Aggressive GC every 25 frames (more frequent for safety)
                            if (frame_idx + 1) % 25 == 0:
                                aggressive_gc(clear_cuda=False)
                        except Exception as frame_e:
                            if check_oom_error(frame_e):
                                handle_oom_error(frame_e, f"letterbox frame {frame_idx}")
                                logger.warning(f"OOM on frame {frame_idx}, skipping frame")
                                aggressive_gc(clear_cuda=True)
                                continue
                            raise
            except Exception as e:
                if check_oom_error(e):
                    handle_oom_error(e, f"scaling chunk {chunk_idx + 1}")
                    logger.error(f"OOM error in chunk {chunk_idx + 1}, skipping chunk")
                    aggressive_gc(clear_cuda=True)
                    continue
                raise
            
            # Save chunk to intermediate file
            intermediate_path = temp_dir / f"chunk_{chunk_idx}.mp4"
            if save_frames(scaled_chunk, str(intermediate_path), fps=chunk_fps):
                intermediate_files.append(str(intermediate_path))
                logger.debug(f"Saved chunk {chunk_idx + 1} with {len(scaled_chunk)} frames")
            else:
                logger.error(f"Failed to save chunk {chunk_idx + 1}")
            
            # Clear chunk memory immediately
            del chunk_frames, scaled_chunk
            aggressive_gc(clear_cuda=False)
        
        # Concatenate all chunks into final video
        if intermediate_files:
            logger.debug(f"Concatenating {len(intermediate_files)} chunks into final video...")
            success = concatenate_videos(intermediate_files, output_path, fps=fps)
            
            # Clean up intermediate files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug(f"Could not delete temp directory: {e}")
            
            aggressive_gc(clear_cuda=False)
            return success
        else:
            logger.warning(f"No chunks processed from {video_path}")
            return False
        
    except Exception as e:
        if check_oom_error(e):
            handle_oom_error(e, f"scaling video {video_path}")
            logger.error(f"OOM error while scaling video {video_path}")
        else:
            logger.error(f"Failed to scale video {video_path}: {e}")
        return False
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        aggressive_gc(clear_cuda=False)


def stage3_scale_videos(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/scaled_videos",
    target_size: int = 256,
    max_frames: Optional[int] = 100,
    chunk_size: int = 100,
    method: str = "autoencoder",
    autoencoder_model: Optional[str] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    delete_existing: bool = False,
    resume: bool = True
) -> pl.DataFrame:
    """
    Stage 3: Scale all videos to target max dimension.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size)
    to ensure max(width, height) = target_size.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to augmented metadata CSV/Arrow/Parquet
        output_dir: Directory to save scaled videos
        target_size: Target max dimension (max(width, height) = target_size, default: 256)
        max_frames: Maximum frames to process per video
        method: Scaling method ("letterbox" or "autoencoder")
        autoencoder_model: Hugging Face model name for autoencoder (e.g., "stabilityai/sd-vae-ft-mse")
                          If None and method="autoencoder", uses default model
        start_idx: Start index for video range (0-based, inclusive). If None, starts from 0.
        end_idx: End index for video range (0-based, exclusive). If None, processes all videos.
        delete_existing: If True, delete existing scaled video files before regenerating (clean mode)
        resume: If True, skip videos where scaled files already exist (resume mode)
    
    Returns:
        DataFrame with scaled video metadata (includes original_width and original_height)
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented metadata (support both CSV and Arrow/Parquet)
    logger.info("Stage 3: Loading augmented metadata...")
    from lib.utils.paths import load_metadata_flexible, validate_metadata_columns
    
    df = load_metadata_flexible(augmented_metadata_path)
    if df is None:
        logger.error(f"Augmented metadata not found: {augmented_metadata_path} (checked .arrow, .parquet, .csv)")
        return pl.DataFrame()
    
    # Validate required columns
    try:
        validate_metadata_columns(df, ["video_path", "label"], "Stage 3")
    except ValueError as e:
        logger.error(f"{e}")
        return pl.DataFrame()
    
    # Apply range filtering if specified
    total_videos = df.height
    if total_videos == 0:
        logger.warning("Stage 3: No videos found in metadata")
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
        logger.info(f"Stage 3: Processing video range [{start}, {end}) of {total_videos} total videos")
        
        # Check if range resulted in empty DataFrame
        if df.height == 0:
            logger.warning(f"Stage 3: Range [{start}, {end}) resulted in empty DataFrame")
            return pl.DataFrame()
    else:
        logger.info(f"Stage 3: Processing all {total_videos} videos")
    
    logger.info(f"Stage 3: Processing {df.height} videos in this range")
    logger.info(f"Stage 3: Target max dimension: {target_size} pixels")
    
    # Map "resolution" to "letterbox" for backward compatibility
    if method == "resolution":
        method = "letterbox"
    
    logger.info(f"Stage 3: Method: {method}")
    logger.info(f"Stage 3: Chunk size: {chunk_size} frames (optimized for 64GB RAM)")
    
    # Load autoencoder if needed
    autoencoder = None
    if method == "autoencoder":
        try:
            model_name = autoencoder_model or "stabilityai/sd-vae-ft-mse"
            logger.info(f"Stage 3: Loading Hugging Face autoencoder: {model_name}")
            autoencoder = load_hf_autoencoder(model_name)
            logger.info("✓ Autoencoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load autoencoder: {e}")
            logger.warning("Falling back to letterbox method")
            method = "letterbox"
            autoencoder = None
    
    # Load existing metadata if it exists (for resume mode)
    existing_metadata = None
    existing_video_paths = set()
    
    if resume and not delete_existing:
        # Try to load existing metadata (check all formats)
        existing_metadata_path = output_dir / "scaled_metadata"
        existing_metadata = load_metadata_flexible(str(existing_metadata_path))
        if existing_metadata is not None:
            existing_video_paths = set(existing_metadata["video_path"].to_list())
            logger.info(f"Stage 3: Found {len(existing_video_paths)} existing scaled videos (resume mode)")
    
    # Delete existing scaled video files if clean mode
    if delete_existing:
        logger.info("Stage 3: Deleting existing scaled video files (clean mode)...")
        deleted_count = 0
        for video_file in output_dir.glob("*_scaled*.mp4"):
            try:
                video_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {video_file}: {e}")
        logger.info(f"Stage 3: Deleted {deleted_count} existing scaled video files")
        existing_video_paths = set()  # Clear after deletion
    
    # Use incremental Arrow/Parquet writing (more efficient than CSV)
    # We'll collect rows and write at the end, or use streaming if needed
    metadata_path = output_dir / "scaled_metadata.arrow"
    metadata_rows = []
    
    total_videos_processed = 0
    skipped_count = 0
    
    # Process each video
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        # Handle optional columns from Stage 1 - use safe defaults
        # Stage 1 may or may not have these columns depending on version
        original_video = row.get("original_video", video_rel)
        aug_idx = row.get("augmentation_idx", -1)
        # Handle both boolean and string representations of is_original
        is_original_raw = row.get("is_original", False)
        if isinstance(is_original_raw, str):
            is_original = is_original_raw.lower() in ('true', '1', 'yes', 't')
        else:
            is_original = bool(is_original_raw)
        
        try:
            # Robustly handle video path resolution
            try:
                video_path = resolve_video_path(video_rel, project_root)
            except Exception as e:
                logger.warning(f"Could not resolve video path '{video_rel}': {e}, skipping")
                continue
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path} (from metadata: {video_rel})")
                continue
            
            # Verify it's actually a file (not a directory)
            if not Path(video_path).is_file():
                logger.warning(f"Video path is not a file: {video_path}, skipping")
                continue
            
            if idx % 10 == 0:
                log_memory_stats(f"Stage 3: processing video {idx + 1}/{df.height}")
            
            # Create output path - handle edge cases in video_id and aug_idx
            video_id = Path(video_path).stem
            # Sanitize video_id to avoid filesystem issues
            video_id = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in video_id)
            
            # Handle augmentation_idx edge cases (could be None, -1, or missing)
            if is_original or aug_idx == -1 or aug_idx is None:
                output_filename = f"{video_id}_scaled_original.mp4"
            else:
                # Ensure aug_idx is a valid integer
                try:
                    aug_idx_int = int(aug_idx) if aug_idx is not None else -1
                    output_filename = f"{video_id}_scaled_aug{aug_idx_int}.mp4"
                except (ValueError, TypeError):
                    # Fallback if aug_idx is invalid
                    logger.warning(f"Invalid augmentation_idx '{aug_idx}' for {video_path}, using 'original' suffix")
                    output_filename = f"{video_id}_scaled_original.mp4"
            
            output_path = output_dir / output_filename
            output_rel = str(output_path.relative_to(project_root))
            
            # Check if scaled video already exists (resume mode)
            if resume and not delete_existing:
                if output_path.exists() or output_rel in existing_video_paths:
                    logger.debug(f"Skipping {video_path} - scaled video already exists")
                    skipped_count += 1
                    # Still add to metadata if not already present
                    if output_rel not in existing_video_paths:
                        # Get original video dimensions
                        original_width = None
                        original_height = None
                        try:
                            container = av.open(video_path)
                            stream = container.streams.video[0]
                            original_width = stream.width
                            original_height = stream.height
                            container.close()
                        except Exception as e:
                            logger.warning(f"Could not get dimensions for {video_path}: {e}")
                        
                        metadata_row = {
                            "video_path": output_rel,
                            "label": label,
                            "original_video": original_video,
                            "augmentation_idx": aug_idx,
                            "is_original": is_original
                        }
                        if original_width is not None and original_height is not None:
                            metadata_row["original_width"] = original_width
                            metadata_row["original_height"] = original_height
                        metadata_rows.append(metadata_row)
                    continue
            
            # Get original video dimensions - handle any video format/codec from Stage 1
            original_width = None
            original_height = None
            try:
                container = av.open(video_path)
                if len(container.streams.video) == 0:
                    logger.warning(f"No video stream found in {video_path}, skipping")
                    container.close()
                    continue
                stream = container.streams.video[0]
                original_width = stream.width
                original_height = stream.height
                container.close()
                
                # Validate dimensions
                if original_width is None or original_height is None or original_width <= 0 or original_height <= 0:
                    logger.warning(f"Invalid dimensions for {video_path}: {original_width}x{original_height}, skipping")
                    continue
            except Exception as e:
                logger.warning(f"Could not get dimensions for {video_path}: {e}, will continue without dimensions")
                # Don't skip the video, just proceed without dimension info
            
            # Scale video (downscale or upscale to target_size) with OOM handling
            logger.info(f"Scaling {Path(video_path).name} to {output_path.name}")
            try:
                success = scale_video(
                    video_path,
                    str(output_path),
                    target_size=target_size,
                    max_frames=max_frames,
                    chunk_size=chunk_size,
                    method=method,
                    autoencoder=autoencoder
                )
            except Exception as e:
                if check_oom_error(e):
                    handle_oom_error(e, f"scaling video {video_path}")
                    logger.error(f"OOM error while scaling video {video_path}")
                    # Try fallback to letterbox if autoencoder was being used
                    if method == "autoencoder" and autoencoder is not None:
                        logger.warning(f"Retrying video {video_path} with letterbox method due to OOM")
                        try:
                            success = scale_video(
                                video_path,
                                str(output_path),
                                target_size=target_size,
                                max_frames=max_frames,
                                chunk_size=chunk_size,
                                method="letterbox",
                                autoencoder=None
                            )
                        except Exception as e2:
                            if check_oom_error(e2):
                                handle_oom_error(e2, f"letterbox fallback for {video_path}")
                                logger.error(f"OOM even with letterbox method for {video_path}, skipping video")
                                success = False
                            else:
                                raise
                    else:
                        success = False
                else:
                    raise
            
            if success:
                metadata_row = {
                    "video_path": output_rel,
                    "label": label,
                    "original_video": original_video,
                    "augmentation_idx": aug_idx,
                    "is_original": is_original
                }
                if original_width is not None and original_height is not None:
                    metadata_row["original_width"] = original_width
                    metadata_row["original_height"] = original_height
                metadata_rows.append(metadata_row)
                total_videos_processed += 1
                logger.info(f"✓ Scaled: {output_path.name}")
            else:
                logger.error(f"✗ Failed to scale: {video_path}")
            
            # Aggressive GC after each video
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            if check_oom_error(e):
                handle_oom_error(e, f"processing video {video_rel}")
                logger.error(f"OOM error processing {video_rel}: {e}")
            else:
                logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            aggressive_gc(clear_cuda=True)
            continue
    
    logger.info(f"Stage 3: Processed {total_videos_processed} videos, skipped {skipped_count} videos")
    
    # Save final metadata as Arrow IPC
    if metadata_rows or total_videos_processed > 0 or skipped_count > 0:
        try:
            # Create DataFrame from new metadata
            new_metadata_df = pl.DataFrame(metadata_rows) if metadata_rows else pl.DataFrame()
            
            # Merge with existing metadata if available
            if existing_metadata is not None and not delete_existing:
                # Combine existing and new metadata, removing duplicates
                metadata_df = pl.concat([existing_metadata, new_metadata_df]).unique(subset=["video_path"], keep="last")
                logger.info(f"Stage 3: Merged with existing metadata. Total: {metadata_df.height} videos")
            else:
                metadata_df = new_metadata_df
            
            if metadata_df.height > 0:
                # Try Arrow IPC first, fallback to Parquet
                try:
                    metadata_df.write_ipc(str(metadata_path))
                    logger.debug(f"Saved metadata as Arrow IPC: {metadata_path}")
                except Exception as e:
                    logger.warning(f"Arrow IPC write failed, using Parquet: {e}")
                    metadata_path = output_dir / "scaled_metadata.parquet"
                    metadata_df.write_parquet(str(metadata_path))
                
                logger.info(f"\n✓ Stage 3 complete: Saved metadata to {metadata_path}")
                logger.info(f"✓ Stage 3: Scaled {total_videos_processed} videos, skipped {skipped_count} videos")
                return metadata_df
            else:
                logger.warning("Stage 3: No metadata to save (all videos may have been skipped)")
                return existing_metadata if existing_metadata is not None else pl.DataFrame()
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return existing_metadata if existing_metadata is not None else pl.DataFrame()
    else:
        logger.warning("Stage 3: No videos processed!")
        return existing_metadata if existing_metadata is not None else pl.DataFrame()

