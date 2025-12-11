#!/usr/bin/env python3
"""
Validate Stage 5 imports before running expensive training jobs.

This script should be run before submitting SLURM jobs to catch import errors
and basic functionality issues early.

Usage:
    python src/scripts/validate_stage5_imports.py
"""

import sys
import os
import gc
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup comprehensive logging
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"validate_stage5_imports_{int(time.time())}.log"

# Create logger - prevent duplicate handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear existing handlers to prevent duplication
logger.handlers.clear()

# Console handler (INFO level for user visibility)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(console_formatter)

# File handler (DEBUG level for comprehensive logs)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Validation log file: {log_file}")
logger.debug("=" * 80)
logger.debug("Starting Stage 5 import validation")
logger.debug("=" * 80)

def validate_imports():
    """Validate all critical imports for Stage 5."""
    errors = []
    warnings = []
    
    logger.info("=" * 80)
    logger.info("Validating Stage 5 imports...")
    logger.info("=" * 80)
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Python path: {sys.path[:3]}")
    logger.debug(f"Python version: {sys.version}")
    
    # Test 1: lib.models imports
    logger.info("\n[1/8] Testing lib.models imports...")
    logger.debug("Attempting to import from lib.models...")
    try:
        logger.debug("Importing VideoConfig, VideoDataset, variable_ar_collate, PretrainedInceptionVideoModel, VariableARVideoModel")
        from lib.models import (
            VideoConfig,
            VideoDataset,
            variable_ar_collate,
            PretrainedInceptionVideoModel,
            VariableARVideoModel,
        )
        logger.debug(f"✓ VideoConfig imported: {VideoConfig}")
        logger.debug(f"✓ VideoDataset imported: {VideoDataset}")
        logger.debug(f"✓ variable_ar_collate imported: {variable_ar_collate}")
        logger.debug(f"✓ PretrainedInceptionVideoModel imported: {PretrainedInceptionVideoModel}")
        logger.debug(f"✓ VariableARVideoModel imported: {VariableARVideoModel}")
        logger.info("  ✓ All lib.models imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from lib.models: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 2: stage5_feature_pipeline imports
    logger.info("\n[2/8] Testing stage5_feature_pipeline imports...")
    logger.debug("Attempting to import stage5_train_all_models from lib.training.stage5_feature_pipeline")
    try:
        from lib.training.stage5_feature_pipeline import stage5_train_all_models
        logger.debug(f"✓ stage5_train_all_models imported: {stage5_train_all_models}")
        logger.info("  ✓ stage5_feature_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from stage5_feature_pipeline: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 3: video_training_pipeline imports
    logger.info("\n[3/8] Testing video_training_pipeline imports...")
    logger.debug("Attempting to import from lib.training.video_training_pipeline")
    try:
        from lib.training.video_training_pipeline import (
            train_video_model,
            is_feature_based,
            is_video_based,
            FEATURE_BASED_MODELS,
            VIDEO_BASED_MODELS,
        )
        logger.debug(f"✓ train_video_model imported: {train_video_model}")
        logger.debug(f"✓ is_feature_based imported: {is_feature_based}")
        logger.debug(f"✓ is_video_based imported: {is_video_based}")
        logger.debug(f"✓ FEATURE_BASED_MODELS: {FEATURE_BASED_MODELS}")
        logger.debug(f"✓ VIDEO_BASED_MODELS: {VIDEO_BASED_MODELS}")
        logger.info("  ✓ video_training_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from video_training_pipeline: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 4: feature_training_pipeline imports
    logger.info("\n[4/8] Testing feature_training_pipeline imports...")
    logger.debug("Attempting to import from lib.training.feature_training_pipeline")
    try:
        from lib.training.feature_training_pipeline import (
            train_feature_model,
            load_features_for_training,
        )
        logger.debug(f"✓ train_feature_model imported: {train_feature_model}")
        logger.debug(f"✓ load_features_for_training imported: {load_features_for_training}")
        logger.info("  ✓ feature_training_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from feature_training_pipeline: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 5: model_factory imports
    logger.info("\n[5/8] Testing model_factory imports...")
    logger.debug("Attempting to import create_model from lib.training.model_factory")
    try:
        from lib.training.model_factory import create_model
        logger.debug(f"✓ create_model imported: {create_model}")
        logger.info("  ✓ model_factory imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from model_factory: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 6: variable_ar_collate functionality
    logger.info("\n[6/8] Testing variable_ar_collate with dummy tensors...")
    logger.debug("Importing torch and variable_ar_collate")
    try:
        import torch
        logger.debug(f"✓ torch imported, version: {torch.__version__}")
        logger.debug(f"✓ CUDA available: {torch.cuda.is_available()}")
        from lib.models import variable_ar_collate
        logger.debug(f"✓ variable_ar_collate imported: {variable_ar_collate}")
        
        # Create dummy batch
        logger.debug("Creating dummy batch with variable aspect ratios")
        batch = [
            (torch.randn(8, 3, 256, 256), torch.tensor(0)),
            (torch.randn(8, 3, 240, 320), torch.tensor(1)),
        ]
        logger.debug(f"Batch[0] shape: {batch[0][0].shape}, label: {batch[0][1]}")
        logger.debug(f"Batch[1] shape: {batch[1][0].shape}, label: {batch[1][1]}")
        
        logger.debug("Calling variable_ar_collate...")
        clips_padded, labels = variable_ar_collate(batch)
        logger.debug(f"✓ variable_ar_collate returned clips_padded shape: {clips_padded.shape}, labels shape: {labels.shape}")
        
        logger.debug("Validating output shapes...")
        assert clips_padded.shape[0] == 2, f"Expected batch size 2, got {clips_padded.shape[0]}"
        assert clips_padded.shape[1] == 3, f"Expected 3 channels, got {clips_padded.shape[1]}"
        assert clips_padded.shape[2] == 8, f"Expected 8 frames, got {clips_padded.shape[2]}"
        assert labels.shape == (2,), f"Expected labels shape (2,), got {labels.shape}"
        logger.debug("✓ All shape assertions passed")
        
        logger.info("  ✓ variable_ar_collate works with dummy tensors")
    except Exception as e:
        error_msg = f"variable_ar_collate test failed: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Test 7: Model creation with dummy configs
    logger.info("\n[7/8] Testing model creation...")
    logger.debug("Testing model creation with variable_ar_cnn")
    try:
        from lib.training.model_factory import create_model
        from lib.mlops.config import RunConfig
        logger.debug("✓ create_model and RunConfig imported")
        
        logger.debug("Creating RunConfig for variable_ar_cnn model")
        model_config = RunConfig(
            run_id="test_validation",
            experiment_name="test",
            model_type="variable_ar_cnn",
            num_frames=8,
            model_specific_config={}
        )
        logger.debug(f"RunConfig created: model_type={model_config.model_type}, num_frames={model_config.num_frames}")
        
        logger.debug("Calling create_model...")
        model = create_model("variable_ar_cnn", model_config)
        logger.debug(f"✓ Model created: {type(model)}")
        assert model is not None, "Model is None"
        logger.debug("✓ Model is not None")
        logger.info("  ✓ Model creation works")
    except Exception as e:
        warning_msg = f"Model creation test failed (may require specific configs): {e}"
        warnings.append(warning_msg)
        logger.warning(f"  ⚠ {warning_msg}")
        logger.exception("Full traceback:")
    
    # Test 8: run_stage5_training.py imports
    logger.info("\n[8/8] Testing run_stage5_training.py imports...")
    logger.debug("Testing imports needed by run_stage5_training.py")
    try:
        from lib.training.stage5_feature_pipeline import stage5_train_all_models
        logger.debug("✓ stage5_train_all_models imported")
        from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
        logger.debug(f"✓ FEATURE_BASED_MODELS: {FEATURE_BASED_MODELS}")
        logger.debug(f"✓ VIDEO_BASED_MODELS: {VIDEO_BASED_MODELS}")
        from lib.utils.memory import log_memory_stats
        logger.debug(f"✓ log_memory_stats imported: {log_memory_stats}")
        
        logger.info("  ✓ run_stage5_training.py imports work")
    except ImportError as e:
        error_msg = f"Failed to import modules needed by run_stage5_training.py: {e}"
        errors.append(error_msg)
        logger.error(f"  ✗ {error_msg}")
        logger.exception("Full traceback:")
        import traceback
        traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    logger.debug(f"Total errors: {len(errors)}, Total warnings: {len(warnings)}")
    
    if errors:
        logger.error(f"\n✗ ERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")
        logger.error("\n❌ Validation FAILED. Fix errors before running training jobs.")
        logger.debug("Validation failed - returning exit code 1")
        return 1
    else:
        logger.info("\n✓ All critical imports validated successfully!")
        logger.debug("No errors found in validation")
    
    if warnings:
        logger.warning(f"\n⚠ WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warning}")
        logger.warning("\n⚠ Some warnings detected, but imports are OK.")
        logger.debug(f"Warnings present but non-critical: {warnings}")
    
    logger.info("\n✅ Validation PASSED. Safe to run training jobs.")
    logger.debug("Validation passed successfully - returning exit code 0")
    return 0


if __name__ == "__main__":
    try:
        logger.debug("Starting validation process...")
        exit_code = validate_imports()
        logger.debug(f"Validation completed with exit code: {exit_code}")
        
        # Explicit cleanup before exit to prevent segfaults from C extensions
        logger.debug("Performing explicit cleanup (gc.collect, logging.shutdown)...")
        gc.collect()
        logger.debug(f"GC collected, logging shutdown...")
        logging.shutdown()
        logger.debug("Cleanup complete, using os._exit() to bypass Python cleanup phase")
        
        # Use os._exit() to bypass Python's cleanup phase where C extensions can segfault
        # This is safe because we've already validated everything and cleaned up
        os._exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Validation interrupted by user")
        logger.debug("KeyboardInterrupt caught, exiting with code 130")
        gc.collect()
        logging.shutdown()
        os._exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.critical(f"\n\n✗ Unexpected error during validation: {e}")
        logger.exception("Unexpected exception during validation:")
        import traceback
        traceback.print_exc()
        logger.debug("Performing cleanup before exit...")
        gc.collect()
        logging.shutdown()
        logger.debug("Exiting with code 1 due to unexpected error")
        os._exit(1)
