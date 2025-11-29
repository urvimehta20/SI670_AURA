"""
Model factory and registry for centralized model creation.

Supports:
- Baseline models: logistic_regression, svm, naive_cnn
- Frame→temporal: vit_gru, vit_transformer
- Spatiotemporal: slowfast, x3d
- Existing: pretrained_inception
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
import torch
import torch.nn as nn

from .mlops_core import RunConfig

logger = logging.getLogger(__name__)

# Memory-optimized configurations for each model type
# NOTE: Baseline models use slightly larger batch sizes but remain conservative
# to avoid excessive memory usage during feature extraction.
MODEL_MEMORY_CONFIGS = {
    "logistic_regression": {
        "batch_size": 16,  # Ultra conservative: reduced from 32
        "num_workers": 1,  # Reduced from 2 for memory safety
        "num_frames": 8,
        "gradient_accumulation_steps": 1,
    },
    "svm": {
        "batch_size": 16,  # Ultra conservative: reduced from 32
        "num_workers": 1,  # Reduced from 2 for memory safety
        "num_frames": 8,
        "gradient_accumulation_steps": 1,
    },
    "naive_cnn": {
        "batch_size": 8,  # Ultra conservative: reduced from 16
        "num_workers": 1,  # Reduced from 2 for memory safety
        "num_frames": 8,
        "gradient_accumulation_steps": 2,  # Compensate with gradient accumulation
    },
    "vit_gru": {
        "batch_size": 2,  # Ultra conservative: reduced from 4
        "num_workers": 0,  # Reduced from 1 to avoid multiprocessing overhead
        "num_frames": 8,
        "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
    },
    "vit_transformer": {
        "batch_size": 1,  # Ultra conservative: reduced from 2
        "num_workers": 0,  # Reduced from 1 to avoid multiprocessing overhead
        "num_frames": 8,
        "gradient_accumulation_steps": 16,  # Increased to maintain effective batch size
    },
    "slowfast": {
        "batch_size": 1,  # Ultra conservative: reduced from 2
        "num_workers": 0,  # Reduced from 1 to avoid multiprocessing overhead
        "num_frames": 16,  # SlowFast needs more frames
        "gradient_accumulation_steps": 16,  # Increased to maintain effective batch size
    },
    "x3d": {
        "batch_size": 2,  # Ultra conservative: reduced from 4
        "num_workers": 0,  # Reduced from 1 to avoid multiprocessing overhead
        "num_frames": 16,
        "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
    },
    "pretrained_inception": {
        "batch_size": 4,  # Ultra conservative: reduced from 8
        "num_workers": 1,  # Reduced from 2 for memory safety
        "num_frames": 8,
        "gradient_accumulation_steps": 4,  # Increased to maintain effective batch size
    },
}


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get memory-optimized configuration for a model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Dictionary with batch_size, num_workers, etc.
    """
    if model_type not in MODEL_MEMORY_CONFIGS:
        logger.warning(f"Unknown model type: {model_type}. Using default config.")
        return {
            "batch_size": 8,
            "num_workers": 2,
            "num_frames": 8,
            "gradient_accumulation_steps": 2,
        }
    
    return MODEL_MEMORY_CONFIGS[model_type].copy()


def list_available_models() -> List[str]:
    """List all available model types."""
    return list(MODEL_MEMORY_CONFIGS.keys())


def create_model(model_type: str, config: RunConfig) -> Any:
    """
    Create a model instance based on model type and config.
    
    Args:
        model_type: Model type identifier
        config: RunConfig with model-specific settings
    
    Returns:
        Model instance (PyTorch nn.Module or sklearn-style model)
    """
    # Handle both RunConfig and dict
    if isinstance(config, dict):
        model_specific = config.get("model_specific_config", {})
        num_frames = config.get("num_frames", 8)
    else:
        # RunConfig object - model_specific_config is always a dict
        model_specific = getattr(config, 'model_specific_config', {})
        if not isinstance(model_specific, dict):
            model_specific = {}
        num_frames = getattr(config, 'num_frames', 8)
    
    # Helper to safely get parameter from model_specific dict
    def get_param(key, default):
        if isinstance(model_specific, dict):
            return model_specific.get(key, default)
        return default
    
    if model_type == "logistic_regression":
        from .baseline_models import LogisticRegressionBaseline
        return LogisticRegressionBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "svm":
        from .baseline_models import SVMBaseline
        return SVMBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "naive_cnn":
        from .baseline_models import NaiveCNNBaseline
        return NaiveCNNBaseline(
            num_frames=get_param("num_frames", num_frames),
            num_classes=2
        )
    
    elif model_type == "vit_gru":
        from .frame_temporal_models import ViTGRUModel
        return ViTGRUModel(
            num_frames=get_param("num_frames", num_frames),
            hidden_dim=get_param("hidden_dim", 256),
            num_layers=get_param("num_layers", 2),
            dropout=get_param("dropout", 0.5),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "vit_transformer":
        from .frame_temporal_models import ViTTransformerModel
        return ViTTransformerModel(
            num_frames=get_param("num_frames", num_frames),
            d_model=get_param("d_model", 768),
            nhead=get_param("nhead", 8),
            num_layers=get_param("num_layers", 2),
            dim_feedforward=get_param("dim_feedforward", 2048),
            dropout=get_param("dropout", 0.5),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "slowfast":
        from .spatiotemporal_models import SlowFastModel
        return SlowFastModel(
            slow_frames=get_param("slow_frames", 16),
            fast_frames=get_param("fast_frames", 64),
            alpha=get_param("alpha", 8),
            beta=get_param("beta", 1.0 / 8),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "x3d":
        from .spatiotemporal_models import X3DModel
        return X3DModel(
            variant=get_param("variant", "x3d_m"),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "pretrained_inception":
        from .video_modeling import PretrainedInceptionVideoModel
        return PretrainedInceptionVideoModel(
            freeze_backbone=get_param("freeze_backbone", False)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list_available_models()}")


def is_pytorch_model(model_type: str) -> bool:
    """
    Check if model type is a PyTorch model (vs sklearn baseline).
    
    Args:
        model_type: Model type identifier
    
    Returns:
        True if PyTorch model, False if sklearn baseline
    """
    sklearn_models = {"logistic_regression", "svm"}
    return model_type not in sklearn_models


def get_model_input_shape(model_type: str, config: RunConfig) -> tuple:
    """
    Get expected input shape for a model.
    
    Args:
        model_type: Model type identifier
        config: RunConfig
    
    Returns:
        Input shape tuple (C, T, H, W) or description
    """
    num_frames = config.num_frames
    fixed_size = config.fixed_size or 224
    
    if model_type in ["logistic_regression", "svm"]:
        return "features"  # Handcrafted features, not video
    
    elif model_type in ["naive_cnn", "vit_gru", "vit_transformer"]:
        return (3, num_frames, fixed_size, fixed_size)
    
    elif model_type in ["slowfast", "x3d", "pretrained_inception"]:
        return (3, num_frames, fixed_size, fixed_size)
    
    else:
        return (3, num_frames, fixed_size, fixed_size)


def download_pretrained_models(model_types: List[str]) -> Dict[str, bool]:
    """
    Download and verify pretrained models upfront as a prerequisite.
    
    This ensures all required pretrained weights are available before training starts,
    preventing delays and network issues during training.
    
    Args:
        model_types: List of model types to train
    
    Returns:
        Dictionary mapping model_type -> success (True if downloaded/verified, False otherwise)
    """
    results = {}
    
    for model_type in model_types:
        logger.info("Checking pretrained weights for model: %s", model_type)
        
        try:
            if model_type in ["vit_gru", "vit_transformer"]:
                # ViT models use timm, which downloads from Hugging Face
                try:
                    import timm
                    # Create model with pretrained=True to trigger download
                    model = timm.create_model(
                        'vit_base_patch16_224',
                        pretrained=True,
                        num_classes=0,
                        global_pool='',
                    )
                    # Delete model immediately to free memory
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    results[model_type] = True
                    logger.info("✓ Downloaded/verified ViT pretrained weights for %s", model_type)
                except Exception as e:
                    logger.error("✗ Failed to download ViT weights for %s: %s", model_type, e)
                    results[model_type] = False
            
            elif model_type == "slowfast":
                # SlowFast uses torchvision, but has fallback to r3d_18 pretrained weights
                try:
                    from torchvision.models.video import slowfast_r50, SlowFast_R50_Weights
                    try:
                        weights = SlowFast_R50_Weights.KINETICS400_V1
                        model = slowfast_r50(weights=weights)
                    except (AttributeError, ValueError):
                        model = slowfast_r50(pretrained=True)
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    results[model_type] = True
                    logger.info("✓ Downloaded/verified SlowFast pretrained weights")
                except ImportError:
                    # torchvision doesn't have SlowFast - verify r3d_18 is available as fallback
                    try:
                        from torchvision.models.video import r3d_18, R3D_18_Weights
                        try:
                            weights = R3D_18_Weights.KINETICS400_V1
                            model = r3d_18(weights=weights)
                        except (AttributeError, ValueError):
                            model = r3d_18(pretrained=True)
                        del model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        results[model_type] = True
                        logger.info("✓ Verified r3d_18 pretrained weights (will be used as SlowFast backbone)")
                    except Exception as e:
                        logger.warning("⚠ No pretrained weights available for SlowFast: %s", e)
                        results[model_type] = False
                except Exception as e:
                    logger.error("✗ Failed to download SlowFast weights: %s", e)
                    results[model_type] = False
            
            elif model_type == "x3d":
                # X3D uses torchvision, but has fallback to r3d_18 pretrained weights
                try:
                    from torchvision.models.video import x3d_m, X3D_M_Weights
                    try:
                        weights = X3D_M_Weights.KINETICS400_V1
                        model = x3d_m(weights=weights)
                    except (AttributeError, ValueError):
                        model = x3d_m(pretrained=True)
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    results[model_type] = True
                    logger.info("✓ Downloaded/verified X3D pretrained weights")
                except ImportError:
                    # torchvision doesn't have X3D - verify r3d_18 is available as fallback
                    try:
                        from torchvision.models.video import r3d_18, R3D_18_Weights
                        try:
                            weights = R3D_18_Weights.KINETICS400_V1
                            model = r3d_18(weights=weights)
                        except (AttributeError, ValueError):
                            model = r3d_18(pretrained=True)
                        del model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        results[model_type] = True
                        logger.info("✓ Verified r3d_18 pretrained weights (will be used as X3D backbone)")
                    except Exception as e:
                        logger.warning("⚠ No pretrained weights available for X3D: %s", e)
                        results[model_type] = False
                except Exception as e:
                    logger.error("✗ Failed to download X3D weights: %s", e)
                    results[model_type] = False
            
            elif model_type == "pretrained_inception":
                # PretrainedInceptionVideoModel uses torchvision r3d_18
                try:
                    from torchvision.models.video import r3d_18, R3D_18_Weights
                    try:
                        weights = R3D_18_Weights.KINETICS400_V1
                        model = r3d_18(weights=weights)
                    except (AttributeError, ValueError):
                        model = r3d_18(pretrained=True)
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    results[model_type] = True
                    logger.info("✓ Downloaded/verified R3D_18 pretrained weights")
                except Exception as e:
                    logger.error("✗ Failed to download R3D_18 weights: %s", e)
                    results[model_type] = False
            
            else:
                # Baseline models (logistic_regression, svm, naive_cnn) don't need pretrained weights
                results[model_type] = True
                logger.info("✓ No pretrained weights needed for %s", model_type)
        
        except Exception as e:
            logger.error("✗ Unexpected error checking pretrained weights for %s: %s", model_type, e)
            results[model_type] = False
    
    return results


__all__ = [
    "MODEL_MEMORY_CONFIGS",
    "get_model_config",
    "list_available_models",
    "create_model",
    "is_pytorch_model",
    "get_model_input_shape",
    "download_pretrained_models",
]

