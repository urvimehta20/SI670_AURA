#!/usr/bin/env python3
"""
Validate Stage 5 imports before running expensive training jobs.

This script should be run before submitting SLURM jobs to catch import errors
and basic functionality issues early.

Usage:
    python src/scripts/validate_stage5_imports.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def validate_imports():
    """Validate all critical imports for Stage 5."""
    errors = []
    warnings = []
    
    print("=" * 80)
    print("Validating Stage 5 imports...")
    print("=" * 80)
    
    # Test 1: lib.models imports
    print("\n[1/8] Testing lib.models imports...")
    try:
        from lib.models import (
            VideoConfig,
            VideoDataset,
            variable_ar_collate,
            PretrainedInceptionVideoModel,
            VariableARVideoModel,
        )
        print("  ✓ All lib.models imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from lib.models: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 2: stage5_feature_pipeline imports
    print("\n[2/8] Testing stage5_feature_pipeline imports...")
    try:
        from lib.training.stage5_feature_pipeline import stage5_train_all_models
        print("  ✓ stage5_feature_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from stage5_feature_pipeline: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 3: video_training_pipeline imports
    print("\n[3/8] Testing video_training_pipeline imports...")
    try:
        from lib.training.video_training_pipeline import (
            train_video_model,
            is_feature_based,
            is_video_based,
            FEATURE_BASED_MODELS,
            VIDEO_BASED_MODELS,
        )
        print("  ✓ video_training_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from video_training_pipeline: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 4: feature_training_pipeline imports
    print("\n[4/8] Testing feature_training_pipeline imports...")
    try:
        from lib.training.feature_training_pipeline import (
            train_feature_model,
            load_features_for_training,
        )
        print("  ✓ feature_training_pipeline imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from feature_training_pipeline: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 5: model_factory imports
    print("\n[5/8] Testing model_factory imports...")
    try:
        from lib.training.model_factory import create_model
        print("  ✓ model_factory imports successful")
    except ImportError as e:
        error_msg = f"Failed to import from model_factory: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 6: variable_ar_collate functionality
    print("\n[6/8] Testing variable_ar_collate with dummy tensors...")
    try:
        import torch
        from lib.models import variable_ar_collate
        
        # Create dummy batch
        batch = [
            (torch.randn(8, 3, 224, 256), torch.tensor(0)),
            (torch.randn(8, 3, 240, 320), torch.tensor(1)),
        ]
        
        clips_padded, labels = variable_ar_collate(batch)
        
        assert clips_padded.shape[0] == 2
        assert clips_padded.shape[1] == 3
        assert clips_padded.shape[2] == 8
        assert labels.shape == (2,)
        
        print("  ✓ variable_ar_collate works with dummy tensors")
    except Exception as e:
        error_msg = f"variable_ar_collate test failed: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Model creation with dummy configs
    print("\n[7/8] Testing model creation...")
    try:
        from lib.training.model_factory import create_model
        from lib.mlops.config import RunConfig
        
        model_config = RunConfig(
            run_id="test_validation",
            experiment_name="test",
            model_type="variable_ar_cnn",
            num_frames=8,
            model_specific_config={}
        )
        model = create_model("variable_ar_cnn", model_config)
        assert model is not None
        print("  ✓ Model creation works")
    except Exception as e:
        warning_msg = f"Model creation test failed (may require specific configs): {e}"
        warnings.append(warning_msg)
        print(f"  ⚠ {warning_msg}")
    
    # Test 8: run_stage5_training.py imports
    print("\n[8/8] Testing run_stage5_training.py imports...")
    try:
        from lib.training.stage5_feature_pipeline import stage5_train_all_models
        from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
        from lib.utils.memory import log_memory_stats
        
        print("  ✓ run_stage5_training.py imports work")
    except ImportError as e:
        error_msg = f"Failed to import modules needed by run_stage5_training.py: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    if errors:
        print(f"\n✗ ERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n❌ Validation FAILED. Fix errors before running training jobs.")
        return 1
    else:
        print("\n✓ All critical imports validated successfully!")
    
    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print("\n⚠ Some warnings detected, but imports are OK.")
    
    print("\n✅ Validation PASSED. Safe to run training jobs.")
    return 0


if __name__ == "__main__":
    exit_code = validate_imports()
    sys.exit(exit_code)
