#!/usr/bin/env python3
"""
Stage 5: Model Training Script

Trains models using scaled videos and extracted features.

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

from lib.training.stage5_feature_pipeline import stage5_train_all_models
from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
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
        description="Stage 5: Train models using scaled videos and features",
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
        "--scaled-metadata",
        type=str,
        default="data/scaled_videos/scaled_metadata.arrow",
        help="Path to scaled metadata from Stage 3 (default: data/scaled_videos/scaled_metadata.arrow). "
             "Also supports .parquet and .csv formats."
    )
    parser.add_argument(
        "--features-stage2",
        type=str,
        default="data/features_stage2/features_metadata.arrow",
        help="Path to Stage 2 features metadata (default: data/features_stage2/features_metadata.arrow). "
             "Also supports .parquet and .csv formats."
    )
    parser.add_argument(
        "--features-stage4",
        type=str,
        default="data/features_stage4/features_scaled_metadata.arrow",
        help="Path to Stage 4 features metadata (default: data/features_stage4/features_scaled_metadata.arrow). "
             "Also supports .parquet and .csv formats."
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
        default=8,
        help="Number of frames per video (default: 8, optimized for 256GB RAM)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage5",
        help="Output directory for training results (default: data/training_results)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable experiment tracking"
    )
    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        default=False,
        help="Train ensemble model after individual models (default: False)"
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        choices=["meta_learner", "weighted_average"],
        default="meta_learner",
        help="Ensemble method: 'meta_learner' (train MLP) or 'weighted_average' (simple average) (default: meta_learner)"
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=None,
        help="Model index for multi-node training (0-based). If specified, trains only this model from the list. "
             "For multi-node: each node trains one model."
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing model checkpoints/results before regenerating (clean mode, default: False, preserves existing)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(args.project_root).resolve()
    scaled_metadata_path = project_root / args.scaled_metadata
    features_stage2_path = project_root / args.features_stage2
    features_stage4_path = project_root / args.features_stage4
    output_dir = project_root / args.output_dir
    
    # Handle "all" model types
    if "all" in args.model_types:
        # Get all models (feature + video based)
        all_model_types = list(FEATURE_BASED_MODELS | VIDEO_BASED_MODELS)
        logger.info("Training all available models: %s", all_model_types)
    else:
        all_model_types = args.model_types
    
    # Handle model-idx for multi-node training
    if args.model_idx is not None:
        if args.model_idx < 0 or args.model_idx >= len(all_model_types):
            logger.error("Invalid model-idx %d. Must be between 0 and %d", args.model_idx, len(all_model_types) - 1)
            return 1
        model_types = [all_model_types[args.model_idx]]
        logger.info("Multi-node mode: Training model %d/%d: %s", args.model_idx + 1, len(all_model_types), model_types[0])
    else:
        model_types = all_model_types
    
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
    logger.info("Scaled metadata: %s", scaled_metadata_path)
    logger.info("Features Stage 2: %s", features_stage2_path)
    logger.info("Features Stage 4: %s", features_stage4_path)
    logger.info("Model types: %s", model_types)
    logger.info("K-fold splits: %d", args.n_splits)
    logger.info("Number of frames: %d", args.num_frames)
    logger.info("Output directory: %s", output_dir)
    logger.info("Experiment tracking: %s", "Disabled" if args.no_tracking else "Enabled")
    if args.model_idx is not None:
        logger.info("Model index: %d (multi-node mode)", args.model_idx)
    logger.info("Delete existing: %s", args.delete_existing)
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    if not scaled_metadata_path.exists():
        logger.error("Scaled metadata file not found: %s", scaled_metadata_path)
        logger.error("Please run Stage 3 first: python src/scripts/run_stage3_scaling.py")
        return 1
    logger.info("✓ Scaled metadata file found: %s", scaled_metadata_path)
    
    if not features_stage2_path.exists():
        logger.error("Stage 2 features metadata not found: %s", features_stage2_path)
        logger.error("Please run Stage 2 first: python src/scripts/run_stage2_features.py")
        return 1
    logger.info("✓ Stage 2 features metadata found: %s", features_stage2_path)
    
    if not features_stage4_path.exists():
        logger.error("Stage 4 features metadata not found: %s", features_stage4_path)
        logger.error("Please run Stage 4 first: python src/scripts/run_stage4_scaled_features.py")
        return 1
    logger.info("✓ Stage 4 features metadata found: %s", features_stage4_path)
    
    # Validate model types
    available_models = list(FEATURE_BASED_MODELS | VIDEO_BASED_MODELS)
    invalid_models = [m for m in model_types if m not in available_models]
    if invalid_models:
        logger.error("Invalid model types: %s", invalid_models)
        logger.error("Available models: %s", available_models)
        return 1
    
    logger.info("✓ All model types are valid")
    logger.debug("Available models: %s", available_models)
    logger.debug("Feature-based: %s", FEATURE_BASED_MODELS)
    logger.debug("Video-based: %s", VIDEO_BASED_MODELS)
    
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
        results = stage5_train_all_models(
            project_root=str(project_root),
            scaled_metadata_path=str(scaled_metadata_path),
            features_stage2_path=str(features_stage2_path),
            features_stage4_path=str(features_stage4_path),
            model_types=model_types,
            n_splits=args.n_splits,
            output_dir=args.output_dir,
            use_gpu=True,
            batch_size=32,
            epochs=100,
            num_frames=args.num_frames
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

