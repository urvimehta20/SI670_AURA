"""
Cleanup utilities for removing temporary files and directories before runs.
"""

from __future__ import annotations

import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_runs_and_logs(project_root: str, keep_models: bool = False, keep_intermediate_data: bool = False) -> None:
    """
    Clean up runs/, logs/, models/, and intermediate_data/ directories before starting a new run.
    
    Args:
        project_root: Project root directory
        keep_models: If True, keep models/ directory (default: False, delete it too)
        keep_intermediate_data: If True, keep intermediate_data/ directory (default: False, delete it for fresh run)
    """
    project_path = Path(project_root)
    
    # Directories to clean
    dirs_to_clean = ["runs", "logs"]
    if not keep_models:
        dirs_to_clean.append("models")
    if not keep_intermediate_data:
        dirs_to_clean.append("intermediate_data")
    
    for dir_name in dirs_to_clean:
        dir_path = project_path / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                logger.info("✓ Deleted %s/ directory", dir_name)
            except Exception as e:
                logger.warning("Failed to delete %s/: %s", dir_name, str(e))
        else:
            logger.debug("Directory %s/ does not exist, skipping", dir_name)
    
    # Recreate empty directories (except intermediate_data which will be created by pipeline)
    for dir_name in dirs_to_clean:
        if dir_name != "intermediate_data":  # Don't recreate intermediate_data, pipeline will create it
            dir_path = project_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Created empty %s/ directory", dir_name)


def cleanup_intermediate_files(project_root: str, keep_augmentations: bool = True) -> None:
    """
    Clean up intermediate files (augmentations, splits, etc.).
    
    Args:
        project_root: Project root directory
        keep_augmentations: If True, keep augmented_clips/ (default: True)
    """
    project_path = Path(project_root)
    
    # Clean up intermediate directories in runs/ (if exists)
    runs_path = project_path / "runs"
    if runs_path.exists():
        for run_dir in runs_path.iterdir():
            if run_dir.is_dir():
                # Clean up intermediate files in each run
                intermediate_dirs = ["augmented_clips", "splits", "checkpoints"]
                if not keep_augmentations:
                    intermediate_dirs.append("augmented_clips")
                
                for intermediate_dir in intermediate_dirs:
                    intermediate_path = run_dir / intermediate_dir
                    if intermediate_path.exists():
                        try:
                            shutil.rmtree(intermediate_path)
                            logger.debug("Deleted %s from run %s", intermediate_dir, run_dir.name)
                        except Exception as e:
                            logger.warning("Failed to delete %s: %s", intermediate_path, str(e))


__all__ = ["cleanup_runs_and_logs", "cleanup_intermediate_files"]

