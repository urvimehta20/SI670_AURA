"""
Video augmentation module.

Provides:
- Spatial and temporal augmentation transforms
- Video I/O utilities (frame loading/saving)
- Pre-generation pipeline for augmented clips
- Stage 1 augmentation pipeline
"""

from .io import load_frames, save_frames
from .transforms import (
    RandomRotation,
    RandomAffine,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomCutout,
    LetterboxResize,
    apply_simple_augmentation,
    temporal_frame_drop,
    temporal_frame_duplicate,
    temporal_reverse,
)
# Pregenerate imports are optional - only import if needed
# These are used for pre-generation pipeline, not Stage 1 augmentation
try:
    from .pregenerate import (
        generate_augmented_clips,
        pregenerate_augmented_dataset,
        load_precomputed_clip,
        build_comprehensive_frame_transforms,
        apply_temporal_augmentations,
    )
    PREGENERATE_AVAILABLE = True
except ImportError:
    # If pregenerate can't be imported (e.g., missing lib.models), make functions unavailable
    PREGENERATE_AVAILABLE = False
    generate_augmented_clips = None
    pregenerate_augmented_dataset = None
    load_precomputed_clip = None
    build_comprehensive_frame_transforms = None
    apply_temporal_augmentations = None
from .pipeline import stage1_augment_videos

__all__ = [
    # Transforms
    "build_comprehensive_frame_transforms",
    "apply_temporal_augmentations",
    "RandomRotation",
    "RandomAffine",
    "RandomGaussianNoise",
    "RandomGaussianBlur",
    "RandomCutout",
    "LetterboxResize",
    "apply_simple_augmentation",
    "temporal_frame_drop",
    "temporal_frame_duplicate",
    "temporal_reverse",
    # I/O
    "load_frames",
    "save_frames",
    # Pre-generation
    "generate_augmented_clips",
    "pregenerate_augmented_dataset",
    "load_precomputed_clip",
    # Stage 1
    "stage1_augment_videos",
]

