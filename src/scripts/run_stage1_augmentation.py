#!/usr/bin/env python3
"""
Stage 1: Video Augmentation Script

Generates multiple augmented versions of each video using spatial and temporal
transformations. Creates augmented clips for training data augmentation.

Usage:
    python src/scripts/run_stage1_augmentation.py
    python src/scripts/run_stage1_augmentation.py --num-augmentations 10
    python src/scripts/run_stage1_augmentation.py --project-root /path/to/project
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.augmentation import stage1_augment_videos
from lib.utils.memory import log_memory_stats

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for extensive logs
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.augmentation").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def main():
    """Run Stage 1: Video Augmentation."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate augmented versions of videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 10 augmentations per video
  python src/scripts/run_stage1_augmentation.py
  
  # Custom number of augmentations
  python src/scripts/run_stage1_augmentation.py --num-augmentations 5
  
  # Custom project root and output directory
  python src/scripts/run_stage1_augmentation.py --project-root /path/to/project --output-dir data/custom_augmented
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=10,
        help="Number of augmentations to generate per video (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented_videos",
        help="Output directory for augmented videos (default: data/augmented_videos)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing augmented metadata (skip already processed videos)"
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing augmentations before regenerating (default: False, preserves existing)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for video range (0-based, inclusive). If not specified, processes all videos from start."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for video range (0-based, exclusive). If not specified, processes all videos to end."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for video range (0-based, inclusive). If not specified, starts from 0."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for video range (0-based, exclusive). If not specified, processes all videos."
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage1_augmentation_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 1: VIDEO AUGMENTATION")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Number of augmentations per video: %d", args.num_augmentations)
    logger.info("Resume mode: %s", "Enabled" if args.resume else "Disabled")
    logger.info("Delete existing augmentations: %s", "Yes" if args.delete_existing else "No (preserved)")
    if args.start_idx is not None or args.end_idx is not None:
        logger.info("Video range: [%s, %s)", 
                   args.start_idx if args.start_idx is not None else "0",
                   args.end_idx if args.end_idx is not None else "all")
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    # Check metadata file - prefer FVC_dup.csv, fallback to video_index_input.csv
    metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            metadata_path = candidate_path
            logger.info("✓ Metadata file found: %s", metadata_path)
            break
    
    if metadata_path is None:
        logger.error("Metadata file not found. Expected:")
        logger.error("  - %s", project_root / "data" / "FVC_dup.csv")
        logger.error("  - %s", project_root / "data" / "video_index_input.csv")
        logger.error("Please run data preparation first: python src/setup_fvc_dataset.py")
        return 1
    
    # Check videos directory
    videos_dir = project_root / "videos"
    if not videos_dir.exists():
        logger.warning("Videos directory not found: %s", videos_dir)
        logger.warning("Video path resolution may fail if videos are in different location")
    else:
        logger.info("✓ Videos directory found: %s", videos_dir)
    
    # Log system information
    try:
        import psutil
        logger.debug("System information:")
        logger.debug("  CPU count: %d", psutil.cpu_count())
        logger.debug("  Total memory: %.2f GB", psutil.virtual_memory().total / 1e9)
        logger.debug("  Available memory: %.2f GB", psutil.virtual_memory().available / 1e9)
    except ImportError:
        logger.debug("psutil not available, skipping system info")
    
    # Log initial memory stats
    logger.info("=" * 80)
    logger.info("Initial memory statistics:")
    logger.info("=" * 80)
    log_memory_stats("Stage 1: before augmentation", detailed=True)
    
    # Run Stage 1
    logger.info("=" * 80)
    logger.info("Starting Stage 1: Video Augmentation")
    logger.info("=" * 80)
    logger.info("This may take a while depending on dataset size...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        result_df = stage1_augment_videos(
            project_root=str(project_root),
            num_augmentations=args.num_augmentations,
            output_dir=args.output_dir,
            delete_existing=args.delete_existing,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Metadata file: %s", output_dir / "augmented_metadata.csv")
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            logger.debug("Result DataFrame shape: %s", result_df.shape)
            
            # Log statistics
            try:
                import polars as pl
                if "is_original" in result_df.columns:
                    original_count = result_df.filter(pl.col("is_original") == True).height
                    augmented_count = result_df.filter(pl.col("is_original") == False).height
                    logger.info("Original videos: %d", original_count)
                    logger.info("Augmented videos: %d", augmented_count)
                    logger.info("Total videos (original + augmented): %d", result_df.height)
            except Exception as e:
                logger.debug("Could not compute statistics: %s", e)
        else:
            logger.warning("Result DataFrame is None or invalid")
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 1: after augmentation", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  - Run Stage 2: python src/scripts/run_stage2_features.py")
        logger.info("  - Or continue with full pipeline: python src/run_new_pipeline.py --skip-stage 1")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("AUGMENTATION INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 1 FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Full traceback:", exc_info=True)
        logger.error("Output directory: %s", output_dir)
        logger.error("Partial results may be available")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

