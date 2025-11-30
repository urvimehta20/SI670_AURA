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


def _reconstruct_metadata_from_files(
    metadata_path: Path,
    output_dir: Path,
    project_root: Path,
    df: pl.DataFrame,
    num_augmentations: int
) -> None:
    """
    Reconstruct metadata CSV from existing augmentation files in the output directory.
    
    Scans for all *_aug*.mp4 and *_original.mp4 files and creates metadata entries.
    This function loads the full original metadata to match video_ids correctly.
    """
    logger.info("Reconstructing metadata from existing augmentation files...")
    
    # Load full original metadata (not just the filtered df) to match all video_ids
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if input_metadata_path is None:
        logger.error("Cannot reconstruct metadata: original metadata file not found")
        return
    
    try:
        full_df = load_metadata(str(input_metadata_path))
    except Exception as e:
        logger.error(f"Cannot reconstruct metadata: failed to load original metadata: {e}")
        return
    
    # Create a mapping from video_id to original video path and label
    # Check all videos in the full dataset, not just the filtered range
    video_id_to_info = {}
    for row in full_df.iter_rows(named=True):
        video_rel = row["video_path"]
        label = row["label"]
        try:
            video_path = resolve_video_path(video_rel, project_root)
            if Path(video_path).exists():
                video_path_obj = Path(video_path)
                video_path_parts = video_path_obj.parts
                if len(video_path_parts) >= 2:
                    video_id = video_path_parts[-2]
                    video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                    video_id_to_info[video_id] = {
                        'original_video': video_rel,
                        'label': label
                    }
        except Exception:
            continue
    
    # Scan for all augmentation and original files in the output directory
    entries = []
    
    # Find all *_aug*.mp4 files
    for aug_file in output_dir.glob("*_aug*.mp4"):
        aug_filename = aug_file.stem  # Remove .mp4 extension
        if "_aug" in aug_filename:
            video_id = aug_filename.split("_aug")[0]
            aug_idx_str = aug_filename.split("_aug")[1]
            try:
                aug_idx = int(aug_idx_str)
            except ValueError:
                continue
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                aug_path_rel = str(aug_file.relative_to(project_root))
                entries.append({
                    'video_path': aug_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': aug_idx,
                    'is_original': False
                })
    
    # Find all *_original.mp4 files
    for orig_file in output_dir.glob("*_original.mp4"):
        orig_filename = orig_file.stem  # Remove .mp4 extension
        if orig_filename.endswith("_original"):
            video_id = orig_filename[:-9]  # Remove "_original" suffix
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                orig_path_rel = str(orig_file.relative_to(project_root))
                entries.append({
                    'video_path': orig_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': -1,  # -1 for original videos
                    'is_original': True
                })
    
    # Write metadata CSV (append if file exists, write if new)
    mode = 'a' if metadata_path.exists() else 'w'
    if mode == 'w':
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "label", "original_video", "augmentation_idx", "is_original"])
    
    # Append entries (avoid duplicates by checking existing entries if appending)
    existing_entries = set()
    if mode == 'a' and metadata_path.exists():
        try:
            existing_df = pl.read_csv(str(metadata_path))
            for row in existing_df.iter_rows(named=True):
                existing_entries.add((
                    row.get('video_path', ''),
                    row.get('original_video', ''),
                    row.get('augmentation_idx', -999)
                ))
        except Exception:
            pass
    
    new_entries = []
    with open(metadata_path, mode, newline='') as f:
        writer = csv.writer(f)
        for entry in entries:
            entry_key = (entry['video_path'], entry['original_video'], entry['augmentation_idx'])
            if entry_key not in existing_entries:
                writer.writerow([
                    entry['video_path'],
                    entry['label'],
                    entry['original_video'],
                    entry['augmentation_idx'],
                    entry['is_original']
                ])
                new_entries.append(entry)
                existing_entries.add(entry_key)
    
    if new_entries:
        logger.info(f"✓ Reconstructed metadata: Added {len(new_entries)} new entries (total {len(entries)} found)")
    else:
        logger.info(f"✓ Metadata reconstruction: All {len(entries)} entries already exist in metadata")


def stage1_augment_videos(
    project_root: str,
    num_augmentations: int = 10,
    output_dir: str = "data/augmented_videos",
    delete_existing: bool = False,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None
) -> pl.DataFrame:
    """
    Stage 1: Augment all videos.
    
    Args:
        project_root: Project root directory
        num_augmentations: Number of augmentations per video (default: 10)
        output_dir: Directory to save augmented videos
        delete_existing: If True, delete existing augmentations before regenerating (default: False)
        start_idx: Start index for video range (0-based, inclusive). If None, starts from 0.
        end_idx: End index for video range (0-based, exclusive). If None, processes all videos.
    
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
    
    total_videos = df.height
    
    # Apply range filtering if specified
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
        logger.info(f"Stage 1: Processing video range [{start}, {end}) of {total_videos} total videos")
    else:
        logger.info(f"Stage 1: Processing all {total_videos} videos")
    
    logger.info(f"Stage 1: Found {df.height} videos to process")
    logger.info(f"Stage 1: Generating {num_augmentations} augmentation(s) per video")
    logger.info(f"Stage 1: Output directory: {output_dir}")
    logger.info(f"Stage 1: Delete existing augmentations: {delete_existing}")
    
    # Use incremental CSV writing to avoid memory accumulation
    metadata_path = output_dir / "augmented_metadata.csv"
    
    # Load existing metadata if it exists and we're not deleting
    existing_metadata = None
    existing_video_ids_with_all_augs = set()  # Videos that have all augmentations
    if metadata_path.exists() and not delete_existing:
        try:
            existing_metadata = pl.read_csv(str(metadata_path))
            # Count augmentations per video to find which have all augmentations
            video_aug_counts = {}
            for row in existing_metadata.iter_rows(named=True):
                original_video = row.get("original_video", "")
                aug_idx = row.get("augmentation_idx", -1)
                if aug_idx >= 0:  # This is an augmentation
                    # Extract video_id from original_video path
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        video_aug_counts[video_id] = video_aug_counts.get(video_id, 0) + 1
            
            # Only mark videos that have all required augmentations
            for video_id, count in video_aug_counts.items():
                if count >= num_augmentations:
                    existing_video_ids_with_all_augs.add(video_id)
            
            logger.info(f"Stage 1: Found {len(existing_video_ids_with_all_augs)} videos with all {num_augmentations} augmentations")
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}, will regenerate")
            existing_metadata = None
    
    # If deleting existing, remove augmented files only in the specified range
    if delete_existing:
        logger.info("Stage 1: Deleting existing augmentations in range...")
        
        # Get the video IDs that will be processed in this range
        video_ids_in_range = set()
        for idx in range(df.height):
            row = df.row(idx, named=True)
            video_rel = row["video_path"]
            try:
                video_path = resolve_video_path(video_rel, project_root)
                if Path(video_path).exists():
                    video_path_obj = Path(video_path)
                    video_path_parts = video_path_obj.parts
                    if len(video_path_parts) >= 2:
                        video_id = video_path_parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        video_ids_in_range.add(video_id)
            except Exception:
                continue
        
        logger.info(f"Stage 1: Will delete augmentations for {len(video_ids_in_range)} videos in range")
        
        # Delete augmented video files only for videos in this range
        aug_files_deleted = 0
        for aug_file in output_dir.glob("*_aug*.mp4"):
            # Extract video_id from filename (format: {video_id}_aug{idx}.mp4)
            aug_filename = aug_file.stem  # Remove .mp4 extension
            if "_aug" in aug_filename:
                file_video_id = aug_filename.split("_aug")[0]
                if file_video_id in video_ids_in_range:
                    aug_file.unlink()
                    aug_files_deleted += 1
                    logger.debug(f"Deleted existing augmentation: {aug_file}")
        
        logger.info(f"Stage 1: Deleted {aug_files_deleted} existing augmentation files in range")
        
        # Delete metadata entries for videos in this range (if metadata exists)
        if metadata_path.exists():
            try:
                existing_metadata = pl.read_csv(str(metadata_path))
                # Filter out entries for videos in this range
                rows_to_keep = []
                for row in existing_metadata.iter_rows(named=True):
                    original_video = row.get("original_video", "")
                    # Extract video_id from original_video path
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        if video_id not in video_ids_in_range:
                            rows_to_keep.append(row)
                
                # Rewrite metadata without deleted entries
                if len(rows_to_keep) < existing_metadata.height:
                    deleted_count = existing_metadata.height - len(rows_to_keep)
                    logger.info(f"Stage 1: Removing {deleted_count} metadata entries for videos in range")
                    # Create new DataFrame from kept rows
                    if rows_to_keep:
                        new_metadata = pl.DataFrame(rows_to_keep)
                        new_metadata.write_csv(str(metadata_path))
                        logger.info(f"Stage 1: Updated metadata file, kept {len(rows_to_keep)} entries")
                    else:
                        # No entries left, delete metadata file
                        metadata_path.unlink()
                        logger.info(f"Stage 1: Deleted metadata file (no entries remaining)")
                else:
                    logger.info(f"Stage 1: No metadata entries to delete for this range")
            except Exception as e:
                logger.warning(f"Could not update metadata file: {e}, will regenerate entries")
        
        logger.info("Stage 1: Range-specific cleanup complete")
    
    # Open metadata file for writing (append if file exists, write if new)
    # Always append if file exists to preserve entries from other ranges, even when delete_existing=True
    mode = 'a' if metadata_path.exists() else 'w'
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
            if video_id in existing_video_ids_with_all_augs and not delete_existing:
                # Double-check that all augmentation files actually exist
                all_augmentations_exist = True
                for aug_idx in range(num_augmentations):
                    aug_path = output_dir / f"{video_id}_aug{aug_idx}.mp4"
                    if not aug_path.exists():
                        all_augmentations_exist = False
                        logger.warning(f"Video {video_id} missing augmentation {aug_idx}, will regenerate")
                        break
                
                if all_augmentations_exist:
                    logger.info(f"Video {video_id} already has all {num_augmentations} augmentations, skipping...")
                    continue
            
            original_output = output_dir / f"{video_id}_original.mp4"
            if not original_output.exists():
                import shutil
                # Ensure output directory exists before copying
                original_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, original_output)
            
            # Write original video metadata immediately to CSV (only if not already exists in metadata)
            # Check if original is already in metadata
            original_already_in_metadata = False
            if existing_metadata is not None:
                for row in existing_metadata.iter_rows(named=True):
                    if row.get("original_video") == video_rel and row.get("is_original") == True:
                        original_already_in_metadata = True
                        break
            
            if not original_already_in_metadata:
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
                    # Check if metadata entry already exists
                    aug_path_rel = str(aug_path.relative_to(project_root))
                    metadata_entry_exists = False
                    if existing_metadata is not None:
                        for row in existing_metadata.iter_rows(named=True):
                            if (row.get("video_path") == aug_path_rel and 
                                row.get("original_video") == video_rel and 
                                row.get("augmentation_idx") == aug_idx):
                                metadata_entry_exists = True
                                break
                    
                    # Only write metadata if it doesn't already exist
                    if not metadata_entry_exists:
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
    
    # Reconstruct metadata from existing files if needed
    # This ensures that if all augmentations already exist and were skipped,
    # or if the metadata file is missing/corrupted, we rebuild it from the files
    needs_reconstruction = False
    if not metadata_path.exists():
        needs_reconstruction = True
        logger.info("Stage 1: Metadata file missing, reconstructing from existing augmentation files...")
    elif metadata_path.exists() and metadata_path.stat().st_size == 0:
        needs_reconstruction = True
        logger.info("Stage 1: Metadata file is empty, reconstructing from existing augmentation files...")
    else:
        # Check if metadata is incomplete (has fewer entries than expected files)
        try:
            existing_metadata_df = pl.read_csv(str(metadata_path))
            # Count expected files: original + augmentations for each video in range
            expected_entries = df.height * (1 + num_augmentations)  # 1 original + num_augmentations per video
            if existing_metadata_df.height < expected_entries * 0.5:  # If less than 50% of expected
                needs_reconstruction = True
                logger.info(f"Stage 1: Metadata appears incomplete ({existing_metadata_df.height} entries, expected ~{expected_entries}), reconstructing...")
        except Exception:
            needs_reconstruction = True
            logger.info("Stage 1: Could not read metadata file, reconstructing from existing augmentation files...")
    
    if needs_reconstruction:
        _reconstruct_metadata_from_files(metadata_path, output_dir, project_root, df, num_augmentations)
    
    # Load final metadata from CSV
    if metadata_path.exists():
        try:
            metadata_df = pl.read_csv(str(metadata_path))
            logger.info(f"\n✓ Stage 1 complete: Metadata available at {metadata_path}")
            logger.info(f"✓ Stage 1: Total entries in metadata: {metadata_df.height}")
            if total_videos_processed > 0:
                logger.info(f"✓ Stage 1: Processed {total_videos_processed} videos in this run")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to read metadata CSV: {e}")
            return pl.DataFrame()
    else:
        logger.error("Stage 1: No metadata file found and could not reconstruct!")
        return pl.DataFrame()

