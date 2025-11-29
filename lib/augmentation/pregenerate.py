"""
Pre-generation pipeline for video augmentations.

Generates and stores augmented video data BEFORE training for:
- Faster training (no augmentation overhead)
- Reproducibility (same augmentations across runs)
- Memory efficiency (can pre-process and cache)
"""

from __future__ import annotations

import os
import logging
import random
import csv
from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

# Import from lib.models - handle import error gracefully
try:
    from lib.models import VideoConfig, _read_video_wrapper, uniform_sample_indices
except ImportError:
    # Fallback: import directly from video module
    try:
        from lib.models.video import VideoConfig, _read_video_wrapper, uniform_sample_indices
    except ImportError:
        # If still fails, this module won't work but won't break Stage 1
        VideoConfig = None
        _read_video_wrapper = None
        uniform_sample_indices = None
from lib.utils.paths import resolve_video_path
from lib.utils.memory import log_memory_stats, aggressive_gc
from lib.augmentation.transforms import build_comprehensive_frame_transforms, apply_temporal_augmentations

logger = logging.getLogger(__name__)


def generate_augmented_clips(
    video_path: str,
    config: VideoConfig,
    num_augmentations: int = 1,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Generate augmented clips from a video.
    
    Args:
        video_path: Path to source video
        config: VideoConfig with augmentation settings
        num_augmentations: Number of augmented versions to generate
        save_dir: Directory to save augmented clips (optional)
        seed: Random seed for deterministic augmentations
    
    Returns:
        List of augmented clip tensors (each is T, C, H, W)
    """
    # Generate deterministic seed from video path if not provided
    if seed is None:
        import hashlib
        video_path_str = str(video_path)
        seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # Get video frame count without loading all frames
    container = None
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames == 0:
            total_frames = int(stream.duration * stream.average_rate / stream.time_base) if stream.duration else 0
    except Exception:
        # Fallback: load video to get frame count
        log_memory_stats(f"before loading video (fallback): {Path(video_path).name}")
        video = _read_video_wrapper(video_path)
        total_frames = video.shape[0]
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            return []
        del video
        aggressive_gc(clear_cuda=False)
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
    
    if total_frames == 0:
        logger.warning(f"Video has no frames: {video_path}")
        return []
    
    augmented_clips = []
    
    # Build transforms
    spatial_transform, post_tensor_transform = build_comprehensive_frame_transforms(
        train=True,
        fixed_size=config.fixed_size,
        max_size=config.max_size,
        augmentation_config=config.augmentation_config,
    )
    
    for aug_idx in range(num_augmentations):
        aug_seed = seed + aug_idx
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        torch.manual_seed(aug_seed)
        
        # Sample frame indices
        indices = uniform_sample_indices(total_frames, config.num_frames)
        
        # Decode only selected frames using PyAV
        frames = []
        container = None
        try:
            import av
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
            
            max_decode_attempts = len(indices) * 10
            decode_attempts = 0
            for frame_idx in sorted(indices):
                if decode_attempts >= max_decode_attempts:
                    logger.warning(f"Max decode attempts reached for {video_path}, stopping")
                    break
                timestamp_sec = frame_idx / fps
                timestamp_pts = int(timestamp_sec / time_base)
                
                try:
                    container.seek(timestamp_pts, stream=stream)
                    frame_count = 0
                    for packet in container.demux(stream):
                        if decode_attempts >= max_decode_attempts:
                            break
                        for frame in packet.decode():
                            decode_attempts += 1
                            if frame_count == frame_idx or abs(frame_count - frame_idx) < 2:
                                frame_array = frame.to_ndarray(format='rgb24')
                                frame_tensor = spatial_transform(frame_array)
                                if post_tensor_transform is not None:
                                    frame_tensor = post_tensor_transform(frame_tensor)
                                frames.append(frame_tensor)
                                break
                            frame_count += 1
                        if len(frames) > len(indices):
                            break
                    if len(frames) >= len(indices):
                        break
                except Exception as seek_error:
                    logger.debug(f"Seek failed for frame {frame_idx}: {seek_error}")
                    container.seek(0)
                    frame_count = 0
                    for packet in container.demux(stream):
                        if decode_attempts >= max_decode_attempts:
                            break
                        for frame in packet.decode():
                            decode_attempts += 1
                            if frame_count in indices:
                                frame_array = frame.to_ndarray(format='rgb24')
                                frame_tensor = spatial_transform(frame_array)
                                if post_tensor_transform is not None:
                                    frame_tensor = post_tensor_transform(frame_tensor)
                                frames.append(frame_tensor)
                            frame_count += 1
                            if len(frames) >= len(indices):
                                break
                        if len(frames) >= len(indices):
                            break
                    break
            
            if len(frames) < len(indices):
                raise ValueError(f"Only decoded {len(frames)}/{len(indices)} frames")
        except Exception as e:
            # Fallback: load entire video
            logger.warning(f"Frame-by-frame decoding failed: {e}. Loading full video.")
            log_memory_stats(f"before loading full video (fallback): {Path(video_path).name}")
            video = _read_video_wrapper(video_path)
            video_size_mb = video.numel() * video.element_size() / 1024 / 1024
            logger.warning("Loaded full video: %.2f MB", video_size_mb)
            
            for i in indices:
                if i < video.shape[0]:
                    frame = video[i].numpy().astype(np.uint8)
                    frame_tensor = spatial_transform(frame)
                    if post_tensor_transform is not None:
                        frame_tensor = post_tensor_transform(frame_tensor)
                    frames.append(frame_tensor)
            
            del video
            aggressive_gc(clear_cuda=False)
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
        
        # Apply temporal augmentations
        if config.temporal_augmentation_config:
            frames = apply_temporal_augmentations(frames, config.temporal_augmentation_config)
        
        # Stack frames into clip tensor (T, C, H, W)
        if len(frames) < config.num_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else torch.zeros(3, config.fixed_size or 224, config.fixed_size or 224)
            while len(frames) < config.num_frames:
                frames.append(last_frame.clone())
        
        clip = torch.stack(frames[:config.num_frames], dim=0)  # (T, C, H, W)
        
        # Save if requested
        if save_dir:
            save_path = Path(save_dir) / f"{Path(video_path).stem}_aug{aug_idx}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(clip, save_path)
            logger.debug("Saved augmented clip: %s", save_path)
        
        augmented_clips.append(clip)
        del clip
        aggressive_gc(clear_cuda=False)
    
    return augmented_clips


def pregenerate_augmented_dataset(
    df: pl.DataFrame,
    project_root: str,
    config: VideoConfig,
    output_dir: str,
    num_augmentations_per_video: int = 3,
    batch_size: int = 1,
) -> pl.DataFrame:
    """
    Pre-generate augmented clips for all videos in DataFrame.
    
    Args:
        df: DataFrame with video_path and label columns
        project_root: Project root directory
        config: VideoConfig with augmentation settings
        output_dir: Directory to save augmented clips
        num_augmentations_per_video: Number of augmentations per video
        batch_size: Process videos in batches (default: 1 for memory efficiency)
    
    Returns:
        DataFrame with metadata for all augmented clips
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Pre-generating augmented clips for %d videos...", df.height)
    logger.info("Output directory: %s", output_dir)
    logger.info("Augmentations per video: %d", num_augmentations_per_video)
    
    # Incremental CSV writing
    metadata_path = output_dir / "augmented_metadata_temp.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "original_video", "augmentation_idx"])
    
    total_rows_written = 0
    
    for batch_start in tqdm(range(0, df.height, batch_size), desc="Processing videos"):
        batch_end = min(batch_start + batch_size, df.height)
        batch_df = df.slice(batch_start, batch_end - batch_start)
        
        if batch_start % 10 == 0:
            log_memory_stats(f"before processing video {batch_start + 1}", detailed=True)
        
        for idx in range(batch_df.height):
            row = batch_df.row(idx, named=True)
            video_rel = row["video_path"]
            label = row["label"]
            
            try:
                video_path = resolve_video_path(video_rel, project_root)
                
                if idx == 0:
                    log_memory_stats(f"before processing video: {Path(video_path).name}")
                
                # Generate augmented clips
                clips = generate_augmented_clips(
                    video_path,
                    config,
                    num_augmentations=num_augmentations_per_video,
                    save_dir=str(output_dir),
                )
                
                # Write metadata for each clip
                for aug_idx, clip in enumerate(clips):
                    clip_filename = f"{Path(video_path).stem}_aug{aug_idx}.pt"
                    clip_path = output_dir / clip_filename
                    
                    with open(metadata_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            str(clip_path.relative_to(project_root)),
                            label,
                            video_rel,
                            aug_idx
                        ])
                    total_rows_written += 1
                
                del clips
                aggressive_gc(clear_cuda=False)
                
            except Exception as e:
                logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
                continue
    
    # Load final metadata
    if metadata_path.exists() and total_rows_written > 0:
        try:
            metadata_df = pl.read_csv(str(metadata_path))
            final_metadata_path = output_dir / "augmented_metadata.csv"
            metadata_df.write_csv(str(final_metadata_path))
            logger.info(f"Saved metadata to {final_metadata_path}")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to read metadata CSV: {e}")
            return pl.DataFrame()
    else:
        logger.error("No augmented clips generated!")
        return pl.DataFrame()


def load_precomputed_clip(clip_path: str) -> torch.Tensor:
    """Load a pre-computed augmented clip."""
    return torch.load(clip_path)

