#!/usr/bin/env python3
"""
Stage 5: Model Training Script

Trains models using downscaled videos and extracted features.

Usage:
    python src/scripts/run_stage5_training.py
    python src/scripts/run_stage5_training.py --model-types logistic_regression svm
    python src/scripts/run_stage5_training.py --n-splits 5
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.training import stage5_train_models, list_available_models
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
logging.getLogger("lib.training").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.models").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def main():
    """Run Stage 5: Model Training."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Train models using downscaled videos and features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: train logistic_regression and svm with 5-fold CV
  python src/scripts/run_stage5_training.py
  
  # Train specific models
  python src/scripts/run_stage5_training.py --model-types logistic_regression svm naive_cnn
  
  # Custom k-fold splits
  python src/scripts/run_stage5_training.py --n-splits 10
  
  # Train all available models
  python src/scripts/run_stage5_training.py --model-types all
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--downscaled-metadata",
        type=str,
        default="data/downscaled_videos/downscaled_metadata.csv",
        help="Path to downscaled metadata CSV from Stage 3 (default: data/downscaled_videos/downscaled_metadata.csv)"
    )
    parser.add_argument(
        "--features-stage2",
        type=str,
        default="data/features_stage2/features_metadata.csv",
        help="Path to Stage 2 features metadata (default: data/features_stage2/features_metadata.csv)"
    )
    parser.add_argument(
        "--features-stage4",
        type=str,
        default="data/features_stage4/features_downscaled_metadata.csv",
        help="Path to Stage 4 features metadata (default: data/features_stage4/features_downscaled_metadata.csv)"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=["logistic_regression", "svm"],
        help="Model types to train (default: logistic_regression svm). Use 'all' for all available models."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of k-fold splits (default: 5)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Number of frames per video (default: 6, optimized for 80GB RAM)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_results",
        help="Output directory for training results (default: data/training_results)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable experiment tracking"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(args.project_root).resolve()
    downscaled_metadata_path = project_root / args.downscaled_metadata
    features_stage2_path = project_root / args.features_stage2
    features_stage4_path = project_root / args.features_stage4
    output_dir = project_root / args.output_dir
    
    # Handle "all" model types
    if "all" in args.model_types:
        available_models = list_available_models()
        model_types = available_models
        logger.info("Training all available models: %s", model_types)
    else:
        model_types = args.model_types
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage5_training_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 5: MODEL TRAINING")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Downscaled metadata: %s", downscaled_metadata_path)
    logger.info("Features Stage 2: %s", features_stage2_path)
    logger.info("Features Stage 4: %s", features_stage4_path)
    logger.info("Model types: %s", model_types)
    logger.info("K-fold splits: %d", args.n_splits)
    logger.info("Number of frames: %d", args.num_frames)
    logger.info("Output directory: %s", output_dir)
    logger.info("Experiment tracking: %s", "Disabled" if args.no_tracking else "Enabled")
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    if not downscaled_metadata_path.exists():
        logger.error("Downscaled metadata file not found: %s", downscaled_metadata_path)
        logger.error("Please run Stage 3 first: python src/scripts/run_stage3_downscaling.py")
        return 1
    logger.info("✓ Downscaled metadata file found: %s", downscaled_metadata_path)
    
    if not features_stage2_path.exists():
        logger.error("Stage 2 features metadata not found: %s", features_stage2_path)
        logger.error("Please run Stage 2 first: python src/scripts/run_stage2_features.py")
        return 1
    logger.info("✓ Stage 2 features metadata found: %s", features_stage2_path)
    
    if not features_stage4_path.exists():
        logger.error("Stage 4 features metadata not found: %s", features_stage4_path)
        logger.error("Please run Stage 4 first: python src/scripts/run_stage4_downscaled_features.py")
        return 1
    logger.info("✓ Stage 4 features metadata found: %s", features_stage4_path)
    
    # Validate model types
    available_models = list_available_models()
    invalid_models = [m for m in model_types if m not in available_models]
    if invalid_models:
        logger.error("Invalid model types: %s", invalid_models)
        logger.error("Available models: %s", available_models)
        return 1
    
    logger.info("✓ All model types are valid")
    logger.debug("Available models: %s", available_models)
    
    # Log system information
    try:
        import psutil
        import torch
        logger.debug("System information:")
        logger.debug("  CPU count: %d", psutil.cpu_count())
        logger.debug("  Total memory: %.2f GB", psutil.virtual_memory().total / 1e9)
        logger.debug("  Available memory: %.2f GB", psutil.virtual_memory().available / 1e9)
        logger.debug("  CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.debug("  GPU: %s", torch.cuda.get_device_name(0))
            logger.debug("  GPU memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    except ImportError:
        logger.debug("psutil/torch not available, skipping system info")
    
    # Log initial memory stats
    logger.info("=" * 80)
    logger.info("Initial memory statistics:")
    logger.info("=" * 80)
    log_memory_stats("Stage 5: before training", detailed=True)
    
    # Run Stage 5
    logger.info("=" * 80)
    logger.info("Starting Stage 5: Model Training")
    logger.info("=" * 80)
    logger.info("Training %d model(s) with %d-fold cross-validation", len(model_types), args.n_splits)
    logger.info("This may take a while depending on dataset size and model complexity...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        results = stage5_train_models(
            project_root=str(project_root),
            downscaled_metadata_path=str(downscaled_metadata_path),
            features_stage2_path=str(features_stage2_path),
            features_stage4_path=str(features_stage4_path),
            model_types=model_types,
            n_splits=args.n_splits,
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            use_tracking=not args.no_tracking
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 5 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Models trained: %s", model_types)
        logger.info("K-fold splits: %d", args.n_splits)
        
        if results:
            logger.debug("Training results: %s", results)
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 5: after training", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("Results saved to: %s", output_dir)
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("TRAINING INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 5 FAILED")
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

