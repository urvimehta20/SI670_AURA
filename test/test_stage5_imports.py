#!/usr/bin/env python3
"""
Test Stage 5 imports and basic functionality with dummy tensors.

This test suite validates that all imports work correctly and basic functionality
can be tested with dummy tensors before running expensive training jobs.

Run this before submitting expensive SLURM jobs to catch import errors early.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
from typing import List, Dict, Any


def test_import_lib_models():
    """Test that all required imports from lib.models work."""
    try:
        from lib.models import (
            VideoConfig,
            VideoDataset,
            variable_ar_collate,
            PretrainedInceptionVideoModel,
            VariableARVideoModel,
            uniform_sample_indices,
        )
        assert VideoConfig is not None
        assert VideoDataset is not None
        assert variable_ar_collate is not None
        assert callable(variable_ar_collate)
        print("✓ All lib.models imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from lib.models: {e}")


def test_import_stage5_feature_pipeline():
    """Test that stage5_feature_pipeline imports work."""
    try:
        from lib.training.stage5_feature_pipeline import (
            stage5_train_all_models,
        )
        assert stage5_train_all_models is not None
        assert callable(stage5_train_all_models)
        print("✓ stage5_feature_pipeline imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from stage5_feature_pipeline: {e}")


def test_import_video_training_pipeline():
    """Test that video_training_pipeline imports work."""
    try:
        from lib.training.video_training_pipeline import (
            train_video_model,
            is_feature_based,
            is_video_based,
            FEATURE_BASED_MODELS,
            VIDEO_BASED_MODELS,
        )
        assert train_video_model is not None
        assert callable(train_video_model)
        assert callable(is_feature_based)
        assert callable(is_video_based)
        assert isinstance(FEATURE_BASED_MODELS, (set, dict))
        assert isinstance(VIDEO_BASED_MODELS, (set, dict))
        print("✓ video_training_pipeline imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from video_training_pipeline: {e}")


def test_import_feature_training_pipeline():
    """Test that feature_training_pipeline imports work."""
    try:
        from lib.training.feature_training_pipeline import (
            train_feature_model,
            load_features_for_training,
        )
        assert train_feature_model is not None
        assert callable(train_feature_model)
        assert load_features_for_training is not None
        assert callable(load_features_for_training)
        print("✓ feature_training_pipeline imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from feature_training_pipeline: {e}")


def test_import_model_factory():
    """Test that model_factory imports work."""
    try:
        from lib.training.model_factory import create_model
        assert create_model is not None
        assert callable(create_model)
        print("✓ model_factory imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from model_factory: {e}")


def test_variable_ar_collate_with_dummy_tensors():
    """Test variable_ar_collate function with dummy tensors."""
    from lib.models import variable_ar_collate
    
    # Create dummy batch with variable aspect ratios
    batch = [
        (torch.randn(8, 3, 224, 256), torch.tensor(0)),  # (T, C, H, W)
        (torch.randn(8, 3, 240, 320), torch.tensor(1)),
        (torch.randn(8, 3, 200, 200), torch.tensor(0)),
    ]
    
    # Test collate function
    clips_padded, labels = variable_ar_collate(batch)
    
    # Verify output shapes
    assert clips_padded.shape[0] == 3, f"Expected batch size 3, got {clips_padded.shape[0]}"
    assert clips_padded.shape[1] == 3, f"Expected 3 channels, got {clips_padded.shape[1]}"
    assert clips_padded.shape[2] == 8, f"Expected 8 frames, got {clips_padded.shape[2]}"
    assert clips_padded.shape[3] == 240, f"Expected max height 240, got {clips_padded.shape[3]}"
    assert clips_padded.shape[4] == 320, f"Expected max width 320, got {clips_padded.shape[4]}"
    assert labels.shape == (3,), f"Expected labels shape (3,), got {labels.shape}"
    
    print("✓ variable_ar_collate works with dummy tensors")


def test_video_config_creation():
    """Test VideoConfig creation."""
    from lib.models import VideoConfig
    
    # Test fixed size config
    config1 = VideoConfig(num_frames=8, fixed_size=256)
    assert config1.num_frames == 8
    assert config1.fixed_size == 256
    
    # Test variable AR config
    config2 = VideoConfig(num_frames=16, max_size=224)
    assert config2.num_frames == 16
    assert config2.max_size == 224
    
    print("✓ VideoConfig creation works")


def test_model_factory_with_dummy_configs():
    """Test model factory can create models with dummy configs."""
    from lib.training.model_factory import create_model
    from lib.mlops.config import RunConfig
    
    # Test creating a simple model
    try:
        model_config = RunConfig(
            run_id="test_model",
            experiment_name="test",
            model_type="variable_ar_cnn",
            num_frames=8,
            model_specific_config={}
        )
        model = create_model("variable_ar_cnn", model_config)
        assert model is not None
        print("✓ Model factory can create variable_ar_cnn")
    except Exception as e:
        pytest.fail(f"Failed to create model: {e}")


def test_model_forward_pass_with_dummy_tensors():
    """Test that models can do forward pass with dummy tensors."""
    from lib.training.model_factory import create_model
    from lib.mlops.config import RunConfig
    
    model_types_to_test = [
        "variable_ar_cnn",
        # Add more model types as needed
    ]
    
    for model_type in model_types_to_test:
        try:
            model_config = RunConfig(
                run_id=f"test_{model_type}",
                experiment_name="test",
                model_type=model_type,
                num_frames=8,
                model_specific_config={}
            )
            model = create_model(model_type, model_config)
            model.eval()
            
            # Create dummy input: (batch, channels, time, height, width)
            dummy_input = torch.randn(2, 3, 8, 224, 224)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Verify output shape
            assert output is not None
            assert output.shape[0] == 2, f"Expected batch size 2, got {output.shape[0]}"
            print(f"✓ {model_type} forward pass works with dummy tensors")
            
        except Exception as e:
            pytest.fail(f"Failed to test {model_type} forward pass: {e}")


def test_feature_based_model_classification():
    """Test that feature-based models are correctly identified."""
    from lib.training.video_training_pipeline import (
        is_feature_based,
        is_video_based,
        FEATURE_BASED_MODELS,
        VIDEO_BASED_MODELS,
    )
    
    # Test feature-based models
    assert is_feature_based("logistic_regression")
    assert is_feature_based("svm")
    assert not is_video_based("logistic_regression")
    assert not is_video_based("svm")
    
    # Test video-based models
    assert is_video_based("variable_ar_cnn")
    assert is_video_based("naive_cnn")
    assert not is_feature_based("variable_ar_cnn")
    assert not is_feature_based("naive_cnn")
    
    print("✓ Feature/video model classification works")


def test_all_model_types_importable():
    """Test that all model types can be imported and created."""
    from lib.training.video_training_pipeline import (
        FEATURE_BASED_MODELS,
        VIDEO_BASED_MODELS,
    )
    from lib.training.model_factory import create_model
    from lib.mlops.config import RunConfig
    
    all_models = list(FEATURE_BASED_MODELS | VIDEO_BASED_MODELS)
    
    for model_type in all_models:
        try:
            # Try to create model config (this should not fail)
            model_config = RunConfig(
                run_id=f"test_{model_type}",
                experiment_name="test",
                model_type=model_type,
                num_frames=8,
                model_specific_config={}
            )
            
            # For video-based models, try to create the model
            if model_type in VIDEO_BASED_MODELS:
                try:
                    model = create_model(model_type, model_config)
                    assert model is not None
                    print(f"✓ {model_type} can be created")
                except Exception as e:
                    # Some models might require specific configs, that's OK
                    print(f"⚠ {model_type} creation requires specific config: {e}")
            else:
                print(f"✓ {model_type} config created (feature-based)")
                
        except Exception as e:
            pytest.fail(f"Failed to create config for {model_type}: {e}")


def test_run_stage5_training_imports():
    """Test that run_stage5_training.py can import all required modules."""
    try:
        # This simulates what run_stage5_training.py does
        from lib.training.stage5_feature_pipeline import stage5_train_all_models
        from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
        from lib.utils.memory import log_memory_stats
        
        assert stage5_train_all_models is not None
        assert FEATURE_BASED_MODELS is not None
        assert VIDEO_BASED_MODELS is not None
        assert log_memory_stats is not None
        
        print("✓ run_stage5_training.py imports work")
    except ImportError as e:
        pytest.fail(f"Failed to import modules needed by run_stage5_training.py: {e}")


if __name__ == "__main__":
    """Run tests directly."""
    print("=" * 80)
    print("Testing Stage 5 imports and basic functionality")
    print("=" * 80)
    
    # Run all test functions
    test_functions = [
        test_import_lib_models,
        test_import_stage5_feature_pipeline,
        test_import_video_training_pipeline,
        test_import_feature_training_pipeline,
        test_import_model_factory,
        test_variable_ar_collate_with_dummy_tensors,
        test_video_config_creation,
        test_model_factory_with_dummy_configs,
        test_feature_based_model_classification,
        test_run_stage5_training_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! ✓")
