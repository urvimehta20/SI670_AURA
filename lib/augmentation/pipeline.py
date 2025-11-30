"""
Video augmentation pipeline.

Generates multiple augmented versions of each video using spatial and temporal
transformations. Creates augmented clips for training data augmentation.
"""

from __future__ import annotations

import logging
import hashlib
import random
import csv
from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata, filter_existing_videos
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from .io import load_frames, save_frames
from .transforms import apply_simple_augmentation

logger = logging.getLogger(__name__)


def augment_video(
    video_path: str,
    num_augmentations: int = 10,
    augmentation_types: Optional[List[str]] = None,
    max_frames: Optional[int] = 1000
) -> List[List[np.ndarray]]:
    """
    Generate augmented versions of a video.
    
    Args:
        video_path: Path to video file
        num_augmentations: Number of augmentations to generate
        augmentation_types: List of augmentation types to use
        max_frames: Maximum frames to load (default: 1000)
    
    Returns:
        List of augmented frame sequences
    """
    logger.info(f"Loading video: {video_path}")
    
    if max_frames is None:
        max_frames = 1000
        logger.warning(f"No max_frames specified, limiting to {max_frames} frames")
    
    original_frames, fps = load_frames(video_path, max_frames=max_frames)
    
    if not original_frames:
        logger.warning(f"No frames loaded from {video_path}")
        return []
    
    logger.info(f"Loaded {len(original_frames)} frames from {video_path}")
    
    # Generate deterministic seed from video path
    video_path_str = str(video_path)
    base_seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # Default augmentation types
    if augmentation_types is None:
        augmentation_types = [
            'rotation', 'flip', 'brightness', 'contrast', 'saturation',
            'gaussian_noise', 'gaussian_blur', 'affine', 'elastic', 'cutout'
        ]
    
    # Ensure diversity: use each augmentation type at least once
    # If num_augmentations <= len(augmentation_types), use each type exactly once (shuffled)
    # If num_augmentations > len(augmentation_types), cycle through types
    if num_augmentations <= len(augmentation_types):
        # Use each type exactly once, shuffled deterministically for variety
        selected_types = augmentation_types[:num_augmentations].copy()
        random.seed(base_seed)  # Deterministic shuffle based on video path
        random.shuffle(selected_types)
    else:
        # Cycle through types if we need more augmentations than types available
        selected_types = []
        for i in range(num_augmentations):
            selected_types.append(augmentation_types[i % len(augmentation_types)])
        # Shuffle the first len(augmentation_types) to ensure initial diversity
        random.seed(base_seed)
        random.shuffle(selected_types[:len(augmentation_types)])
    
    augmented_videos = []
    
    for aug_idx in range(num_augmentations):
        aug_seed = base_seed + aug_idx
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        
        # Use pre-selected augmentation type (ensures diversity - no duplicates)
        aug_type = selected_types[aug_idx]
        
        # Apply augmentation to all frames
        augmented_frames = []
        for frame_idx, frame in enumerate(original_frames):
            augmented_frame = apply_simple_augmentation(frame, aug_type, aug_seed + frame_idx)
            augmented_frames.append(augmented_frame)
            
            # Aggressive GC every 100 frames
            if (frame_idx + 1) % 100 == 0:
                aggressive_gc(clear_cuda=False)
        
        augmented_videos.append(augmented_frames)
        logger.info(f"Generated augmentation {aug_idx + 1}/{num_augmentations} with type '{aug_type}'")
        
        del augmented_frames
        aggressive_gc(clear_cuda=False)
    
    del original_frames
    aggressive_gc(clear_cuda=False)
    
    return augmented_videos


def stage1_augment_videos(
    project_root: str,
    num_augmentations: int = 10,
    output_dir: str = "data/augmented_videos",
    delete_existing: bool = False
) -> pl.DataFrame:
    """
    Stage 1: Augment all videos.
    
    Args:
        project_root: Project root directory
        num_augmentations: Number of augmentations per video (default: 10)
        output_dir: Directory to save augmented videos
        delete_existing: If True, delete existing augmentations before regenerating (default: False)
    
    Returns:
        DataFrame with metadata for all videos (original + augmented)
    """
    import numpy as np
    
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata - check for FVC_dup.csv first, then video_index_input.csv
    logger.info("Stage 1: Loading video metadata...")
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            logger.info(f"Using metadata file: {input_metadata_path}")
            break
    
    if input_metadata_path is None:
        logger.error(f"Metadata file not found. Expected: {project_root / 'data' / 'FVC_dup.csv'} or {project_root / 'data' / 'video_index_input.csv'}")
        return pl.DataFrame()
    
    df = load_metadata(str(input_metadata_path))
    df = filter_existing_videos(df, str(project_root))
    
    logger.info(f"Stage 1: Found {df.height} original videos")
    logger.info(f"Stage 1: Generating {num_augmentations} augmentation(s) per video")
    logger.info(f"Stage 1: Output directory: {output_dir}")
    logger.info(f"Stage 1: Delete existing augmentations: {delete_existing}")
    
    # Use incremental CSV writing to avoid memory accumulation
    metadata_path = output_dir / "augmented_metadata.csv"
    
    # Load existing metadata if it exists and we're not deleting
    existing_metadata = None
    existing_video_ids = set()
    if metadata_path.exists() and not delete_existing:
        try:
            existing_metadata = pl.read_csv(str(metadata_path))
            # Extract video IDs that already have all augmentations
            for row in existing_metadata.iter_rows(named=True):
                original_video = row.get("original_video", "")
                aug_idx = row.get("augmentation_idx", -1)
                if aug_idx >= 0:  # This is an augmentation
                    # Extract video_id from original_video path
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        existing_video_ids.add(video_id)
            logger.info(f"Stage 1: Found {len(existing_video_ids)} videos with existing augmentations")
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}, will regenerate")
            existing_metadata = None
    
    # If deleting existing, remove all augmented files
    if delete_existing:
        logger.info("Stage 1: Deleting existing augmentations...")
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted existing metadata: {metadata_path}")
        
        # Delete all augmented video files (keep original videos)
        for aug_file in output_dir.glob("*_aug*.mp4"):
            aug_file.unlink()
            logger.debug(f"Deleted existing augmentation: {aug_file}")
        logger.info("Stage 1: Deleted all existing augmentations")
    
    # Open metadata file for writing (append if resuming, write if new)
    mode = 'a' if metadata_path.exists() and not delete_existing else 'w'
    if mode == 'w':
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "label", "original_video", "augmentation_idx", "is_original"])
    
    total_videos_processed = 0
    
    # Process each video one at a time
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Stage 1: Processing video {idx + 1}/{df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 1: before video {idx + 1}")
            
            # Save original video metadata
            # Extract unique ID from video path (e.g., "IJfOsFABDwY" from "FVC1/youtube/IJfOsFABDwY/video.mp4")
            video_path_obj = Path(video_path)
            video_path_parts = video_path_obj.parts
            
            # Get unique identifier from parent directory or use hash
            if len(video_path_parts) >= 2:
                # Parent directory is usually the unique ID (e.g., "IJfOsFABDwY")
                video_id = video_path_parts[-2]  # Parent of "video.mp4"
            else:
                # Fallback: use hash of full path
                import hashlib
                video_id = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
            
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            # Check if this video already has all augmentations
            if video_id in existing_video_ids and not delete_existing:
                # Check if all augmentations exist
                all_augmentations_exist = True
                for aug_idx in range(num_augmentations):
                    aug_path = output_dir / f"{video_id}_aug{aug_idx}.mp4"
                    if not aug_path.exists():
                        all_augmentations_exist = False
                        break
                
                if all_augmentations_exist:
                    logger.info(f"Video {video_id} already has all {num_augmentations} augmentations, skipping...")
                    # Still write metadata if not already in existing metadata
                    if existing_metadata is None or video_id not in existing_video_ids:
                        original_output = output_dir / f"{video_id}_original.mp4"
                        if original_output.exists():
                            with open(metadata_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    str(original_output.relative_to(project_root)),
                                    label,
                                    video_rel,
                                    -1,  # -1 indicates original
                                    True
                                ])
                    continue
            
            original_output = output_dir / f"{video_id}_original.mp4"
            if not original_output.exists():
                import shutil
                shutil.copy2(video_path, original_output)
            
            # Write original video metadata immediately to CSV (only if not already exists)
            if not (existing_metadata is not None and video_id in existing_video_ids):
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        str(original_output.relative_to(project_root)),
                        label,
                        video_rel,
                        -1,  # -1 indicates original
                        True
                    ])
                total_videos_processed += 1
            
            # Generate augmentations with frame limit
            max_frames_per_video = 1000
            augmented_videos = augment_video(
                video_path, 
                num_augmentations=num_augmentations,
                max_frames=max_frames_per_video
            )
            
            # Get FPS from original video
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                container.close()
            except Exception as e:
                logger.warning(f"Failed to get FPS from {video_path}: {e}, using default 30.0")
                fps = 30.0
            
            if not augmented_videos:
                logger.warning(f"No augmentations generated for {video_path}")
                continue
            
            # Save augmented videos
            logger.info(f"Generated {len(augmented_videos)} augmentations, saving...")
            for aug_idx, aug_frames in enumerate(augmented_videos):
                aug_filename = f"{video_id}_aug{aug_idx}.mp4"
                aug_path = output_dir / aug_filename
                
                # Skip if augmentation already exists and we're not deleting
                if aug_path.exists() and not delete_existing:
                    logger.info(f"Augmentation {aug_idx + 1}/{len(augmented_videos)} already exists: {aug_path}, skipping...")
                    # Still write metadata if needed
                    aug_path_rel = str(aug_path.relative_to(project_root))
                    with open(metadata_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            aug_path_rel,
                            label,
                            video_rel,
                            aug_idx,
                            False
                        ])
                    continue
                
                logger.info(f"Saving augmentation {aug_idx + 1}/{len(augmented_videos)} to {aug_path}")
                
                # Validate frames
                if not aug_frames or len(aug_frames) == 0:
                    logger.error(f"✗ Augmentation {aug_idx + 1} has no frames, skipping")
                    continue
                
                success = save_frames(aug_frames, str(aug_path), fps=fps)
                
                if success:
                    aug_path_rel = str(aug_path.relative_to(project_root))
                    # Write augmented video metadata immediately to CSV
                    with open(metadata_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            aug_path_rel,
                            label,
                            video_rel,
                            aug_idx,
                            False
                        ])
                    total_videos_processed += 1
                    logger.info(f"✓ Saved: {aug_path}")
                else:
                    logger.error(f"✗ Failed to save: {aug_path} - save_frames returned False")
                
                # Clear frames immediately
                del aug_frames
                aggressive_gc(clear_cuda=False)
            
            # Clear all augmented videos
            del augmented_videos
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing video {video_rel}: {e}", exc_info=True)
            continue
    
    # Load final metadata from CSV
    if metadata_path.exists() and total_videos_processed > 0:
        try:
            metadata_df = pl.read_csv(str(metadata_path))
            logger.info(f"\n✓ Stage 1 complete: Saved metadata to {metadata_path}")
            logger.info(f"✓ Stage 1: Generated {total_videos_processed} total videos ({df.height} original + {total_videos_processed - df.height} augmented)")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to read metadata CSV: {e}")
            return pl.DataFrame()
    else:
        logger.error("Stage 1: No videos processed!")
        return pl.DataFrame()

