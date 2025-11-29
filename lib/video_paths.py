"""
Centralized video path resolution utilities.

This module provides a single source of truth for resolving video paths,
ensuring consistency across all components (dataset loading, filtering, verification).
"""
import os
from pathlib import Path
from typing import List


def resolve_video_path(video_rel: str, project_root: str) -> str:
    """
    Resolve a relative video path to an absolute path.
    
    Tries multiple path resolution strategies in order:
    1. Add 'videos/' prefix (most common: CSV has FVC1/... but files are at videos/FVC1/...)
    2. Direct relative path from project_root
    3. Remove 'videos/' prefix if present
    
    Args:
        video_rel: Relative video path from CSV (e.g., "FVC1/youtube/.../video.mp4")
        project_root: Project root directory
        
    Returns:
        Absolute path to the video file (first existing path found, or most likely path)
    """
    if not video_rel:
        raise ValueError("video_rel cannot be empty")
    
    video_rel = str(video_rel).strip()
    
    # Handle absolute paths
    if os.path.isabs(video_rel):
        return video_rel
    
    project_root = Path(project_root)
    candidates: List[Path] = []
    
    # Strategy 1: Add 'videos/' prefix (most common case)
    if not video_rel.startswith("videos/"):
        candidates.append(project_root / "videos" / video_rel)
    
    # Strategy 2: Direct relative path from project_root
    candidates.append(project_root / video_rel)
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        without_prefix = video_rel[7:]  # Remove "videos/"
        candidates.append(project_root / without_prefix)
    
    # Check each candidate and return first existing file
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    
    # Fallback: return the most likely path (with videos/ prefix)
    if not video_rel.startswith("videos/"):
        return str(project_root / "videos" / video_rel)
    return str(project_root / video_rel)


def _check_video_path_exists(video_rel: str, project_root: str) -> bool:
    """
    Check if a video path exists using the same resolution logic.
    
    Args:
        video_rel: Relative video path from CSV
        project_root: Project root directory
        
    Returns:
        True if video file exists, False otherwise
    """
    if not video_rel:
        return False
    
    video_rel = str(video_rel).strip()
    if not video_rel:
        return False
    
    if os.path.isabs(video_rel):
        return os.path.exists(video_rel) and os.path.isfile(video_rel)
    
    project_root = Path(project_root)
    candidates: List[Path] = []
    
    # Strategy 1: Add 'videos/' prefix
    if not video_rel.startswith("videos/"):
        candidates.append(project_root / "videos" / video_rel)
    
    # Strategy 2: Direct relative path
    candidates.append(project_root / video_rel)
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        without_prefix = video_rel[7:]
        candidates.append(project_root / without_prefix)
    
    # Check each candidate
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return True
    
    return False


def _get_video_path_candidates(video_rel: str, project_root: str) -> List[str]:
    """
    Get all candidate paths that would be tried for a video.
    
    Useful for debugging and error messages.
    
    Args:
        video_rel: Relative video path from CSV
        project_root: Project root directory
        
    Returns:
        List of candidate absolute paths (in order they would be tried)
    """
    if not video_rel:
        return []
    
    video_rel = str(video_rel).strip()
    if not video_rel:
        return []
    
    if os.path.isabs(video_rel):
        return [video_rel]
    
    project_root = Path(project_root)
    candidates: List[str] = []
    
    # Strategy 1: Add 'videos/' prefix
    if not video_rel.startswith("videos/"):
        candidates.append(str(project_root / "videos" / video_rel))
    
    # Strategy 2: Direct relative path
    candidates.append(str(project_root / video_rel))
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        without_prefix = video_rel[7:]
        candidates.append(str(project_root / without_prefix))
    
    return candidates


__all__ = ["resolve_video_path", "check_video_path_exists", "get_video_path_candidates"]

