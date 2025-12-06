#!/usr/bin/env python3
"""
Stage 2: Handcrafted Feature Extraction Script

Extracts handcrafted features from original videos (M features).

Usage:
    python src/scripts/run_stage2_features.py
    python src/scripts/run_stage2_features.py --num-frames 8
    python src/scripts/run_stage2_features.py --augmented-metadata data/augmented_videos/augmented_metadata.csv
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

from lib.features import stage2_extract_features
from lib.utils.memory import log_memory_stats

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.features").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def main():
    """Run Stage 2: Handcrafted Feature Extraction."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Extract handcrafted features from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use augmented metadata from Stage 1
  python src/scripts/run_stage2_features.py
  
  # Custom number of frames
  python src/scripts/run_stage2_features.py --num-frames 6
  
  # Custom metadata path
  python src/scripts/run_stage2_features.py --augmented-metadata data/custom/augmented_metadata.csv
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--augmented-metadata",
        type=str,
        default="data/augmented_videos/augmented_metadata.arrow",
        help="Path to augmented metadata from Stage 1 (default: data/augmented_videos/augmented_metadata.arrow). "
             "Supports .arrow, .parquet, or .csv formats. Will auto-detect format if extension is omitted."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to sample per video (if provided, overrides percentage-based sampling). "
             "If not provided, uses percentage-based adaptive sampling (10% of frames, min=5, max=50)"
    )
    parser.add_argument(
        "--frame-percentage",
        type=float,
        default=0.10,
        help="Percentage of frames to sample per video (default: 0.10 = 10%). "
             "Only used if --num-frames is not provided."
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum frames to sample per video (for percentage-based sampling, default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum frames to sample per video (for percentage-based sampling, default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features_stage2",
        help="Output directory for features (default: data/features_stage2)"
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
        "--delete-existing",
        action="store_true",
        help="Delete existing feature files before regenerating (clean mode, default: False, preserves existing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing feature files (skip already processed videos, default: True)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(args.project_root).resolve()
    augmented_metadata_path = project_root / args.augmented_metadata
    output_dir = project_root / args.output_dir
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage2_features_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 2: HANDCRAFTED FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Augmented metadata: %s", augmented_metadata_path)
    logger.info("Output directory: %s", output_dir)
    if args.num_frames is not None:
        logger.info("Frame sampling: Fixed %d frames per video", args.num_frames)
    else:
        logger.info("Frame sampling: Adaptive (%.1f%% of frames, min=%d, max=%d)", 
                   args.frame_percentage * 100, args.min_frames, args.max_frames)
    if args.start_idx is not None or args.end_idx is not None:
        logger.info("Video range: [%s, %s)",
                   args.start_idx if args.start_idx is not None else "0",
                   args.end_idx if args.end_idx is not None else "all")
    logger.info("Delete existing: %s", args.delete_existing)
    logger.info("Resume mode: %s", args.resume)
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    if not augmented_metadata_path.exists():
        logger.error("Augmented metadata file not found: %s", augmented_metadata_path)
        logger.error("Please run Stage 1 first: python src/scripts/run_stage1_augmentation.py")
        return 1
    logger.info("✓ Augmented metadata file found: %s", augmented_metadata_path)
    
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
    log_memory_stats("Stage 2: before feature extraction", detailed=True)
    
    # Run Stage 2
    logger.info("=" * 80)
    logger.info("Starting Stage 2: Handcrafted Feature Extraction")
    logger.info("=" * 80)
    logger.info("This may take a while depending on dataset size...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        result_df = stage2_extract_features(
            project_root=str(project_root),
            augmented_metadata_path=str(augmented_metadata_path),
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            delete_existing=args.delete_existing,
            resume=args.resume
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 2 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Features metadata: %s (will be .arrow or .parquet)", output_dir / "features_metadata")
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            logger.debug("Result DataFrame shape: %s", result_df.shape)
        else:
            logger.warning("Result DataFrame is None or invalid")
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 2: after feature extraction", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  - Run Stage 3: python src/scripts/run_stage3_scaling.py")
        logger.info("  - Or continue with full pipeline: python src/run_new_pipeline.py --skip-stage 1,2")
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("FEATURE EXTRACTION INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 2 FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Full traceback:", exc_info=True)
        logger.error("Output directory: %s", output_dir)
        logger.error("Partial results may be available")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    # Ensure all output is flushed before exit
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)

