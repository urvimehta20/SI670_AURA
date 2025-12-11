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

# Import RunConfig - using TYPE_CHECKING to avoid circular import at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lib.mlops.config import RunConfig

logger = logging.getLogger(__name__)

# Memory-optimized configurations for each model type
# NOTE: Optimized for 64GB RAM with 1000 frames per video - very conservative batch sizes
# Target: Keep memory usage under 55GB
MODEL_MEMORY_CONFIGS = {
    "logistic_regression": {
        "batch_size": 32,  # Feature-based, low memory
        "num_workers": 0,  # No multiprocessing for memory safety
        "num_frames": 1000,  # Not used (feature-based)
        "gradient_accumulation_steps": 1,
    },
    "logistic_regression_stage2": {
        "batch_size": 32,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "logistic_regression_stage2_stage4": {
        "batch_size": 32,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "svm": {
        "batch_size": 32,  # Feature-based, low memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "svm_stage2": {
        "batch_size": 32,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "svm_stage2_stage4": {
        "batch_size": 32,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "naive_cnn": {
        "batch_size": 1,  # 1000 frames = very memory intensive
        "num_workers": 0,  # No multiprocessing to save memory
        "num_frames": 1000,
        "gradient_accumulation_steps": 16,  # Effective batch size = 16
    },
    "vit_gru": {
        "batch_size": 1,  # 1000 frames + GRU = high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    "vit_transformer": {
        "batch_size": 1,  # 1000 frames + transformer = very high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    "slowfast": {
        "batch_size": 1,  # Dual pathway + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    "x3d": {
        "batch_size": 1,  # 1000 frames + 3D CNN = very high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    "pretrained_inception": {
        "batch_size": 1,  # 1000 frames = high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 16,  # Effective batch size = 16
    },
    "variable_ar_cnn": {
        "batch_size": 1,  # Variable AR + 1000 frames = high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 16,
    },
    "i3d": {
        "batch_size": 1,  # 1000 frames + I3D = very high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    "r2plus1d": {
        "batch_size": 1,  # 1000 frames + R2Plus1D = very high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,  # Effective batch size = 32
    },
    # XGBoost models use pretrained models for feature extraction
    "xgboost_i3d": {
        "batch_size": 1,  # Feature extraction with 1000 frames
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "xgboost_r2plus1d": {
        "batch_size": 1,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "xgboost_vit_gru": {
        "batch_size": 1,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "xgboost_vit_transformer": {
        "batch_size": 1,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    "xgboost_pretrained_inception": {
        "batch_size": 1,
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 1,
    },
    # Future models
    "timesformer": {
        "batch_size": 1,  # Transformer + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,
    },
    "vivit": {
        "batch_size": 1,  # Video transformer + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,
    },
    "two_stream": {
        "batch_size": 1,  # Dual streams + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,
    },
    "slowfast_attention": {
        "batch_size": 1,  # Attention + dual pathway + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,
    },
    "slowfast_multiscale": {
        "batch_size": 1,  # Multiple pathways + 1000 frames = extremely high memory
        "num_workers": 0,
        "num_frames": 1000,
        "gradient_accumulation_steps": 32,
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
            "batch_size": 1,  # Conservative default for 64GB RAM with 1000 frames
            "num_workers": 0,  # No multiprocessing to save memory
            "num_frames": 1000,
            "gradient_accumulation_steps": 16,  # Maintain effective batch size
        }
    
    return MODEL_MEMORY_CONFIGS[model_type].copy()


def list_available_models(include_xgboost: bool = True) -> List[str]:
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
        num_frames = config.get("num_frames", 1000)
    else:
        # RunConfig object - model_specific_config is always a dict
        model_specific = getattr(config, 'model_specific_config', {})
        if not isinstance(model_specific, dict):
            model_specific = {}
        num_frames = getattr(config, 'num_frames', 1000)
    
    # Helper to safely get parameter from model_specific dict
    def get_param(key, default):
        if isinstance(model_specific, dict):
            return model_specific.get(key, default)
        return default
    
    if model_type == "logistic_regression":
        from ._linear import LogisticRegressionBaseline
        return LogisticRegressionBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "logistic_regression_stage2":
        from ._linear import LogisticRegressionBaseline
        return LogisticRegressionBaseline(
            features_stage2_path=get_param("features_stage2_path", None),
            features_stage4_path=None,
            use_stage2_only=True,
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "logistic_regression_stage2_stage4":
        from ._linear import LogisticRegressionBaseline
        return LogisticRegressionBaseline(
            features_stage2_path=get_param("features_stage2_path", None),
            features_stage4_path=get_param("features_stage4_path", None),
            use_stage2_only=False,
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "svm":
        from ._svm import SVMBaseline
        return SVMBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "svm_stage2":
        from ._svm import SVMBaseline
        return SVMBaseline(
            features_stage2_path=get_param("features_stage2_path", None),
            features_stage4_path=None,
            use_stage2_only=True,
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "svm_stage2_stage4":
        from ._svm import SVMBaseline
        return SVMBaseline(
            features_stage2_path=get_param("features_stage2_path", None),
            features_stage4_path=get_param("features_stage4_path", None),
            use_stage2_only=False,
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "naive_cnn":
        from ._cnn import NaiveCNNBaseline
        return NaiveCNNBaseline(
            num_frames=get_param("num_frames", num_frames),
            num_classes=2
        )
    
    elif model_type == "vit_gru":
        from ._transformer_gru import ViTGRUModel
        return ViTGRUModel(
            num_frames=get_param("num_frames", num_frames),
            hidden_dim=get_param("hidden_dim", 256),
            num_layers=get_param("num_layers", 2),
            dropout=get_param("dropout", 0.5),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "vit_transformer":
        from ._transformer import ViTTransformerModel
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
        from .slowfast import SlowFastModel
        return SlowFastModel(
            slow_frames=get_param("slow_frames", 16),
            fast_frames=get_param("fast_frames", 64),
            alpha=get_param("alpha", 8),
            beta=get_param("beta", 1.0 / 8),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "x3d":
        from .x3d import X3DModel
        return X3DModel(
            variant=get_param("variant", "x3d_m"),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "pretrained_inception":
        from lib.models import PretrainedInceptionVideoModel
        return PretrainedInceptionVideoModel(
            freeze_backbone=get_param("freeze_backbone", False)
        )
    
    elif model_type == "variable_ar_cnn":
        from lib.models import VariableARVideoModel
        return VariableARVideoModel(
            in_channels=get_param("in_channels", 3),
            base_channels=get_param("base_channels", 32)
        )
    
    elif model_type == "i3d":
        from .i3d import I3DModel
        return I3DModel(
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "r2plus1d":
        from .r2plus1d import R2Plus1DModel
        return R2Plus1DModel(
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type.startswith("xgboost_"):
        # XGBoost with pretrained model features
        # Format: "xgboost_i3d", "xgboost_r2plus1d", "xgboost_vit_gru", etc.
        base_model_type = model_type.replace("xgboost_", "")
        
        # Validate base model type
        if base_model_type not in ["i3d", "r2plus1d", "vit_gru", "vit_transformer", "pretrained_inception"]:
            raise ValueError(
                f"Unsupported base model for XGBoost: {base_model_type}. "
                f"Supported: i3d, r2plus1d, vit_gru, vit_transformer, pretrained_inception"
            )
        
        from ._xgboost_pretrained import XGBoostPretrainedBaseline
        return XGBoostPretrainedBaseline(
            base_model_type=base_model_type,
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames),
            xgb_params=get_param("xgb_params", None)
        )
    
    elif model_type == "timesformer":
        from .timesformer import TimeSformerModel
        return TimeSformerModel(
            num_frames=get_param("num_frames", num_frames),
            img_size=get_param("img_size", 256),  # Match scaled video dimensions
            patch_size=get_param("patch_size", 16),
            embed_dim=get_param("embed_dim", 768),
            depth=get_param("depth", 12),
            num_heads=get_param("num_heads", 12),
            mlp_ratio=get_param("mlp_ratio", 4.0),
            qkv_bias=get_param("qkv_bias", True),
            dropout=get_param("dropout", 0.1),
            attn_drop=get_param("attn_drop", 0.0),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "vivit":
        from .vivit import ViViTModel
        return ViViTModel(
            num_frames=get_param("num_frames", num_frames),
            img_size=get_param("img_size", 256),  # Match scaled video dimensions
            tubelet_size=get_param("tubelet_size", (2, 16, 16)),
            embed_dim=get_param("embed_dim", 768),
            depth=get_param("depth", 12),
            num_heads=get_param("num_heads", 12),
            mlp_ratio=get_param("mlp_ratio", 4.0),
            qkv_bias=get_param("qkv_bias", True),
            dropout=get_param("dropout", 0.1),
            attn_drop=get_param("attn_drop", 0.0),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "two_stream":
        from .two_stream import TwoStreamModel
        return TwoStreamModel(
            num_frames=get_param("num_frames", num_frames),
            rgb_backbone=get_param("rgb_backbone", "resnet18"),
            flow_backbone=get_param("flow_backbone", "resnet18"),
            fusion_method=get_param("fusion_method", "concat"),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "slowfast_attention":
        from .slowfast_advanced import SlowFastAttentionModel
        return SlowFastAttentionModel(
            slow_frames=get_param("slow_frames", 16),
            fast_frames=get_param("fast_frames", 64),
            alpha=get_param("alpha", 8),
            beta=get_param("beta", 1.0 / 8),
            pretrained=get_param("pretrained", True),
            attention_type=get_param("attention_type", "cross")
        )
    
    elif model_type == "slowfast_multiscale":
        from .slowfast_advanced import MultiScaleSlowFastModel
        return MultiScaleSlowFastModel(
            num_frames=get_param("num_frames", num_frames),
            scales=get_param("scales", [1, 2, 4, 8]),
            pretrained=get_param("pretrained", True)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list_available_models()}")


def is_xgboost_model(model_type: str) -> bool:
    """Check if model type is XGBoost-based."""
    return model_type.startswith("xgboost_")


def is_pytorch_model(model_type: str) -> bool:
    """
    Check if model type is a PyTorch model (vs sklearn/XGBoost baseline).
    
    Args:
        model_type: Model type identifier
    
    Returns:
        True if PyTorch model, False if sklearn/XGBoost baseline
    """
    sklearn_models = {
        "logistic_regression", "svm",
        "logistic_regression_stage2", "logistic_regression_stage2_stage4",
        "svm_stage2", "svm_stage2_stage4"
    }
    # XGBoost models are not PyTorch models (they use PyTorch for feature extraction only)
    if is_xgboost_model(model_type):
        return False
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
    fixed_size = config.fixed_size or 256
    
    if model_type in [
        "logistic_regression", "svm",
        "logistic_regression_stage2", "logistic_regression_stage2_stage4",
        "svm_stage2", "svm_stage2_stage4"
    ]:
        return "features"  # Stage 2/4 features, not video
    
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

