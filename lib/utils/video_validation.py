"""
Video validation utilities to check for corrupted videos before training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import polars as pl
import numpy as np

from lib.utils.paths import resolve_video_path, validate_video_file as paths_validate_video_file

logger = logging.getLogger(__name__)


def validate_video_file(video_path: str, project_root: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single video file.
    
    Args:
        video_path: Path to video file (relative or absolute)
        project_root: Project root directory
    
    Returns:
        Tuple of (is_valid, error_message)
        is_valid: True if video can be read, False otherwise
        error_message: Error message if invalid, None if valid
    """
    try:
        # Resolve video path
        resolved_path = resolve_video_path(video_path, project_root)
        
        if not Path(resolved_path).exists():
            return False, f"Video file not found: {resolved_path}"
        
        # Use existing validation function from paths.py
        is_valid, error_msg = paths_validate_video_file(resolved_path, check_decode=True)
        
        if not is_valid:
            # Check for common corruption errors in error message
            error_lower = error_msg.lower()
            if 'moov atom' in error_lower:
                return False, "Corrupted video: moov atom not found"
            elif 'corrupt' in error_lower or 'invalid' in error_lower:
                return False, f"Corrupted video: {error_msg}"
            else:
                return False, error_msg
        
        # Valid video
        return True, None
                
    except Exception as e:
        return False, f"Error validating video: {str(e)}"


def validate_videos_batch(
    video_paths: List[str],
    project_root: str,
    max_check: Optional[int] = None,
    sample_rate: float = 1.0
) -> Tuple[List[str], List[str], int]:
    """
    Validate a batch of videos.
    
    Args:
        video_paths: List of video paths
        project_root: Project root directory
        max_check: Maximum number of videos to check (None = check all)
        sample_rate: Fraction of videos to check (1.0 = check all, 0.1 = check 10%)
    
    Returns:
        Tuple of (valid_videos, invalid_videos, corruption_count)
        valid_videos: List of valid video paths
        invalid_videos: List of invalid video paths with reasons
        corruption_count: Number of corrupted videos found
    """
    if max_check is not None:
        video_paths = video_paths[:max_check]
    elif sample_rate < 1.0:
        import random
        random.seed(42)
        n_check = int(len(video_paths) * sample_rate)
        video_paths = random.sample(video_paths, min(n_check, len(video_paths)))
    
    valid_videos = []
    invalid_videos = []
    corruption_count = 0
    
    logger.info(f"Validating {len(video_paths)} videos...")
    
    for i, video_path in enumerate(video_paths):
        if (i + 1) % 100 == 0:
            logger.info(f"Validated {i + 1}/{len(video_paths)} videos...")
        
        is_valid, error_msg = validate_video_file(video_path, project_root)
        
        if is_valid:
            valid_videos.append(video_path)
        else:
            invalid_videos.append(f"{video_path}: {error_msg}")
            if 'corrupt' in error_msg.lower() or 'moov atom' in error_msg.lower():
                corruption_count += 1
    
    return valid_videos, invalid_videos, corruption_count


def filter_valid_videos(
    df: pl.DataFrame,
    project_root: str,
    min_valid_videos: int = 3000,
    check_all: bool = False,
    sample_rate: float = 0.1
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Filter dataframe to only include valid videos.
    
    Args:
        df: DataFrame with video_path column
        project_root: Project root directory
        min_valid_videos: Minimum number of valid videos required
        check_all: If True, check all videos; if False, use sample_rate
        sample_rate: Fraction of videos to check if check_all=False
    
    Returns:
        Tuple of (filtered_df, invalid_video_reasons)
        filtered_df: DataFrame with only valid videos
        invalid_video_reasons: List of invalid video reasons
    
    Raises:
        ValueError: If fewer than min_valid_videos valid videos found
    """
    video_paths = df["video_path"].to_list()
    
    if check_all:
        logger.info(f"Checking all {len(video_paths)} videos for validity...")
        valid_videos, invalid_videos, corruption_count = validate_videos_batch(
            video_paths, project_root, max_check=None, sample_rate=1.0
        )
    else:
        # First, do a quick sample check
        logger.info(f"Sampling {int(len(video_paths) * sample_rate)} videos to check validity...")
        sample_valid, sample_invalid, sample_corruption = validate_videos_batch(
            video_paths, project_root, max_check=None, sample_rate=sample_rate
        )
        
        # Estimate validity rate
        if len(sample_valid) + len(sample_invalid) > 0:
            validity_rate = len(sample_valid) / (len(sample_valid) + len(sample_invalid))
            estimated_valid = int(len(video_paths) * validity_rate)
            
            logger.info(f"Sample validity rate: {validity_rate:.2%} ({len(sample_valid)}/{len(sample_valid) + len(sample_invalid)})")
            logger.info(f"Estimated valid videos: {estimated_valid}/{len(video_paths)}")
            
            if estimated_valid < min_valid_videos:
                logger.warning(f"Estimated valid videos ({estimated_valid}) < minimum required ({min_valid_videos})")
                logger.info("Checking all videos to get exact count...")
                # Check all videos if estimate is too low
                valid_videos, invalid_videos, corruption_count = validate_videos_batch(
                    video_paths, project_root, max_check=None, sample_rate=1.0
                )
            else:
                # Even if estimate looks good, we need to check ALL videos to ensure we have at least min_valid_videos
                # The sample is just for early warning - we still need to validate all videos
                logger.info(f"Sample suggests sufficient valid videos. Checking all videos to confirm...")
                valid_videos, invalid_videos, corruption_count = validate_videos_batch(
                    video_paths, project_root, max_check=None, sample_rate=1.0
                )
        else:
            # No valid samples, check all
            logger.warning("No valid videos in sample. Checking all videos...")
            valid_videos, invalid_videos, corruption_count = validate_videos_batch(
                video_paths, project_root, max_check=None, sample_rate=1.0
            )
    
    # Filter dataframe to only valid videos
    valid_video_set = set(valid_videos)
    filtered_df = df.filter(pl.col("video_path").is_in(valid_video_set))
    
    logger.info(f"Video validation complete:")
    logger.info(f"  Total videos: {len(video_paths)}")
    logger.info(f"  Valid videos: {len(valid_videos)}")
    logger.info(f"  Invalid videos: {len(invalid_videos)}")
    logger.info(f"  Corrupted videos: {corruption_count}")
    
    if len(valid_videos) < min_valid_videos:
        error_msg = (
            f"Insufficient valid videos for training. "
            f"Found {len(valid_videos)} valid videos, but need at least {min_valid_videos}. "
            f"Invalid videos: {len(invalid_videos)} (including {corruption_count} corrupted)."
        )
        logger.error(error_msg)
        if invalid_videos:
            logger.error("Sample of invalid videos:")
            for invalid in invalid_videos[:10]:  # Show first 10
                logger.error(f"  {invalid}")
            if len(invalid_videos) > 10:
                logger.error(f"  ... and {len(invalid_videos) - 10} more")
        raise ValueError(error_msg)
    
    logger.info(f"✓ Sufficient valid videos: {len(valid_videos)} >= {min_valid_videos}")
    
    return filtered_df, invalid_videos

