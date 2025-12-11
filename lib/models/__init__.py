"""
Video models and datasets module.

Provides:
- Video models (Inception-based, pretrained)
- Video datasets
- Video configuration
- Collate functions
"""

try:
    from .video import (
        VideoConfig,
        VideoDataset,
        variable_ar_collate,
        PretrainedInceptionVideoModel,
        VariableARVideoModel,
        uniform_sample_indices,
        _read_video_wrapper,
    )
except ImportError as e:
    # Re-raise with more context
    raise ImportError(
        f"Failed to import from lib.models.video: {e}. "
        "This may be due to missing dependencies (polars, torch, etc.) or a syntax error in video.py"
    ) from e

# Explicitly verify that required names are available
_required_names = [
    "VideoConfig",
    "VideoDataset", 
    "variable_ar_collate",
    "PretrainedInceptionVideoModel",
    "VariableARVideoModel",
    "uniform_sample_indices",
    "_read_video_wrapper",
]

_missing_names = [name for name in _required_names if name not in globals()]
if _missing_names:
    raise ImportError(
        f"Failed to import required names from lib.models.video: {_missing_names}. "
        "This indicates a problem with the video.py module or its dependencies."
    )

__all__ = [
    "VideoConfig",
    "VideoDataset",
    "variable_ar_collate",
    "PretrainedInceptionVideoModel",
    "VariableARVideoModel",
    "uniform_sample_indices",
    "_read_video_wrapper",
]

