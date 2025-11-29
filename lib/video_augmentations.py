"""
Comprehensive video data augmentation utilities.

Includes:
- Spatial augmentations (geometric, color, noise, blur, cutout)
- Temporal augmentations (frame dropping, duplication, temporal jittering)
- Consistent application across frames for temporal coherence
"""

from __future__ import annotations

import random
import logging
from typing import List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial Augmentations
# ---------------------------------------------------------------------------


class RandomRotation:
    """Random rotation with small angles (maintains temporal coherence)."""
    
    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        self.degrees = degrees
        self.p = p
        self.angle = 0.0
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            self.angle = random.uniform(-self.degrees, self.degrees)
            return img.rotate(self.angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        return img
    
    def get_params(self) -> float:
        """Get the rotation angle for consistent application."""
        return self.angle


class RandomAffine:
    """Random affine transformation (translation, scale, shear)."""
    
    def __init__(self, translate: Tuple[float, float] = (0.1, 0.1), 
                 scale: Tuple[float, float] = (0.9, 1.1),
                 shear: float = 5.0, p: float = 0.5):
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p
        self.params = None
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            translate_x = random.uniform(-self.translate[0], self.translate[0]) * img.width
            translate_y = random.uniform(-self.translate[1], self.translate[1]) * img.height
            scale = random.uniform(self.scale[0], self.scale[1])
            shear_x = random.uniform(-self.shear, self.shear)
            shear_y = random.uniform(-self.shear, self.shear)
            
            self.params = (translate_x, translate_y, scale, shear_x, shear_y)
            # torchvision.transforms.functional.affine uses 'fill' (not 'fillcolor')
            # in recent versions. Using 'fill' keeps this compatible across versions.
            return transforms.functional.affine(
                img,
                angle=0,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=(shear_x, shear_y),
                fill=0,
            )
        return img


class RandomGaussianNoise:
    """Add Gaussian noise to image."""
    
    def __init__(self, std: float = 0.1, p: float = 0.3):
        self.std = std
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply after ToTensor (img is already a tensor)."""
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, 0.0, 1.0)
        return img


class RandomGaussianBlur:
    """Apply Gaussian blur."""
    
    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class RandomCutout:
    """Random cutout/erasing (masks rectangular regions)."""
    
    def __init__(self, num_holes: int = 1, max_h_size: int = 32, max_w_size: int = 32, 
                 fill_value: float = 0.0, p: float = 0.5):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply after ToTensor (img is already a tensor: C, H, W)."""
        if random.random() < self.p:
            h = img.shape[1]
            w = img.shape[2]
            
            for _ in range(self.num_holes):
                y = random.randint(0, h)
                x = random.randint(0, w)
                hole_h = min(self.max_h_size, h - y)
                hole_w = min(self.max_w_size, w - x)
                
                img[:, y:y+hole_h, x:x+hole_w] = self.fill_value
        
        return img


class RandomElasticTransform:
    """Elastic deformation (simplified version)."""
    
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply after ToTensor (img is already a tensor: C, H, W)."""
        if random.random() < self.p:
            # Simplified elastic transform using grid sampling
            C, H, W = img.shape
            device = img.device
            
            # Generate random displacement field
            dx = torch.randn(1, H, W, device=device) * self.alpha
            dy = torch.randn(1, H, W, device=device) * self.alpha
            
            # Smooth the displacement field
            kernel_size = int(6 * self.sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            dx = F.conv2d(dx.unsqueeze(0), 
                         torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2),
                         padding=kernel_size // 2).squeeze(0)
            dy = F.conv2d(dy.unsqueeze(0),
                         torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2),
                         padding=kernel_size // 2).squeeze(0)
            
            # Create coordinate grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            
            # Apply displacement
            grid_x = grid_x + dx.squeeze(0)
            grid_y = grid_y + dy.squeeze(0)
            
            # Normalize to [-1, 1]
            grid_x = 2.0 * grid_x / (W - 1) - 1.0
            grid_y = 2.0 * grid_y / (H - 1) - 1.0
            
            grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
            
            # Apply grid sampling
            img = img.unsqueeze(0)  # Add batch dimension
            img = F.grid_sample(img, grid.permute(0, 2, 3, 1), mode='bilinear', 
                              padding_mode='zeros', align_corners=True)
            return img.squeeze(0)
        
        return img


# ---------------------------------------------------------------------------
# Temporal Augmentations
# ---------------------------------------------------------------------------


def temporal_frame_drop(frames: List[torch.Tensor], drop_prob: float = 0.1) -> List[torch.Tensor]:
    """Randomly drop frames from the sequence."""
    if random.random() < drop_prob and len(frames) > 1:
        num_to_drop = random.randint(1, max(1, len(frames) // 4))  # Drop up to 25%
        indices_to_drop = random.sample(range(len(frames)), num_to_drop)
        frames = [f for i, f in enumerate(frames) if i not in indices_to_drop]
    return frames


def temporal_frame_duplicate(frames: List[torch.Tensor], dup_prob: float = 0.1) -> List[torch.Tensor]:
    """Randomly duplicate frames (simulates slow motion)."""
    if random.random() < dup_prob and len(frames) > 1:
        num_to_dup = random.randint(1, max(1, len(frames) // 4))  # Duplicate up to 25%
        indices_to_dup = random.sample(range(len(frames)), num_to_dup)
        new_frames = []
        for i, f in enumerate(frames):
            new_frames.append(f)
            if i in indices_to_dup:
                new_frames.append(f)  # Duplicate
        frames = new_frames
    return frames


def temporal_reverse(frames: List[torch.Tensor], reverse_prob: float = 0.1) -> List[torch.Tensor]:
    """Reverse the temporal order of frames."""
    if random.random() < reverse_prob:
        return list(reversed(frames))
    return frames


# ---------------------------------------------------------------------------
# Consistent Augmentation Wrapper
# ---------------------------------------------------------------------------


class ConsistentAugmentation:
    """
    Apply the same spatial augmentation to all frames in a clip.
    This maintains temporal coherence.
    """
    
    def __init__(self, transform: Callable):
        self.transform = transform
        self.params = None
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Store parameters on first call, reuse for subsequent frames
        if hasattr(self.transform, 'get_params'):
            if self.params is None:
                result = self.transform(img)
                self.params = self.transform.get_params() if hasattr(self.transform, 'get_params') else None
                return result
            else:
                # Reuse stored parameters (for consistent application)
                return self.transform(img)
        return self.transform(img)
    
    def reset(self):
        """Reset parameters for a new clip."""
        self.params = None
        if hasattr(self.transform, 'params'):
            self.transform.params = None


# ---------------------------------------------------------------------------
# Comprehensive Augmentation Pipeline
# ---------------------------------------------------------------------------


class LetterboxResize:
    """
    Deterministic letterbox resize to a fixed square size.
    Implemented as a top-level class so it is picklable for
    multiprocessing DataLoader workers.
    """

    def __init__(self, fixed_size: int):
        self.fixed_size = fixed_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.fixed_size / w, self.fixed_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.fixed_size, self.fixed_size), (0, 0, 0))
        paste_x = (self.fixed_size - new_w) // 2
        paste_y = (self.fixed_size - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        return canvas


def build_comprehensive_frame_transforms(
    train: bool = True,
    fixed_size: Optional[int] = None,
    max_size: Optional[int] = None,
    augmentation_config: Optional[dict] = None,
) -> Tuple[transforms.Compose, Optional[Callable]]:
    """
    Build comprehensive augmentation pipeline for video frames.
    
    Args:
        train: If True, apply augmentations
        fixed_size: Fixed size for both dimensions (letterboxing)
        max_size: Maximum size for longer edge (variable size)
        augmentation_config: Dict with augmentation parameters:
            - rotation_degrees: float (default: 10.0)
            - rotation_p: float (default: 0.5)
            - affine_p: float (default: 0.3)
            - gaussian_noise_std: float (default: 0.1)
            - gaussian_noise_p: float (default: 0.3)
            - gaussian_blur_p: float (default: 0.3)
            - cutout_p: float (default: 0.5)
            - cutout_max_size: int (default: 32)
            - elastic_transform_p: float (default: 0.2)
            - color_jitter_brightness: float (default: 0.2)
            - color_jitter_contrast: float (default: 0.2)
            - color_jitter_saturation: float (default: 0.2)
            - color_jitter_hue: float (default: 0.05)
    
    Returns:
        Tuple of (spatial_transform, post_tensor_transform)
        - spatial_transform: Applied to PIL Images
        - post_tensor_transform: Applied after ToTensor (for tensor-based augs)
    """
    from .video_modeling import IMG_MEAN, IMG_STD
    
    if augmentation_config is None:
        augmentation_config = {}
    
    # Default augmentation parameters
    defaults = {
        'rotation_degrees': 10.0,
        'rotation_p': 0.5,
        'affine_p': 0.3,
        'gaussian_noise_std': 0.1,
        'gaussian_noise_p': 0.3,
        'gaussian_blur_p': 0.3,
        'cutout_p': 0.5,
        'cutout_max_size': 32,
        'elastic_transform_p': 0.2,
        'color_jitter_brightness': 0.2,
        'color_jitter_contrast': 0.2,
        'color_jitter_saturation': 0.2,
        'color_jitter_hue': 0.05,
    }
    defaults.update(augmentation_config)
    cfg = defaults
    
    transform_list = [transforms.functional.to_pil_image]
    
    # Resize strategy
    if fixed_size is not None:
        # Use a top-level, picklable transform to avoid multiprocessing
        # pickling errors like:
        # "Can't get local object 'build_comprehensive_frame_transforms.<locals>.letterbox_resize'"
        transform_list.append(LetterboxResize(fixed_size))
    elif max_size is not None:
        transform_list.append(transforms.Resize(max_size, antialias=True))
    
    # Training augmentations (spatial, applied to PIL Images)
    if train:
        # Geometric augmentations (consistent across frames)
        if cfg['rotation_p'] > 0:
            transform_list.append(
                ConsistentAugmentation(
                    RandomRotation(degrees=cfg['rotation_degrees'], p=cfg['rotation_p'])
                )
            )
        
        if cfg['affine_p'] > 0:
            transform_list.append(
                ConsistentAugmentation(
                    RandomAffine(p=cfg['affine_p'])
                )
            )
        
        # Color augmentations
        transform_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )
        
        transform_list.append(
            transforms.ColorJitter(
                brightness=cfg['color_jitter_brightness'],
                contrast=cfg['color_jitter_contrast'],
                saturation=cfg['color_jitter_saturation'],
                hue=cfg['color_jitter_hue']
            )
        )
        
        # Blur (applied to PIL)
        if cfg['gaussian_blur_p'] > 0:
            transform_list.append(
                RandomGaussianBlur(p=cfg['gaussian_blur_p'])
            )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    
    spatial_transform = transforms.Compose(transform_list)
    
    # Post-tensor augmentations (applied after ToTensor)
    post_tensor_augs = []
    if train:
        if cfg['gaussian_noise_p'] > 0:
            post_tensor_augs.append(
                RandomGaussianNoise(std=cfg['gaussian_noise_std'], p=cfg['gaussian_noise_p'])
            )
        
        if cfg['cutout_p'] > 0:
            post_tensor_augs.append(
                RandomCutout(p=cfg['cutout_p'], max_h_size=cfg['cutout_max_size'], 
                           max_w_size=cfg['cutout_max_size'])
            )
        
        if cfg['elastic_transform_p'] > 0:
            post_tensor_augs.append(
                RandomElasticTransform(p=cfg['elastic_transform_p'])
            )
    
    if post_tensor_augs:
        def post_tensor_transform(tensor):
            for aug in post_tensor_augs:
                tensor = aug(tensor)
            return tensor
    else:
        post_tensor_transform = None
    
    return spatial_transform, post_tensor_transform


def apply_temporal_augmentations(
    frames: List[torch.Tensor],
    train: bool = True,
    frame_drop_prob: float = 0.1,
    frame_dup_prob: float = 0.1,
    reverse_prob: float = 0.1,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Apply temporal augmentations to a list of frames.
    
    Args:
        frames: List of frame tensors (each is C, H, W)
        train: If True, apply augmentations
        frame_drop_prob: Probability of dropping frames
        frame_dup_prob: Probability of duplicating frames
        reverse_prob: Probability of reversing temporal order
        seed: Random seed for deterministic augmentations
    
    Returns:
        Augmented list of frames
    """
    if not train:
        return frames
    
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Apply temporal augmentations
    frames = temporal_reverse(frames, reverse_prob=reverse_prob)
    frames = temporal_frame_drop(frames, drop_prob=frame_drop_prob)
    frames = temporal_frame_duplicate(frames, dup_prob=frame_dup_prob)
    
    return frames


__all__ = [
    "build_comprehensive_frame_transforms",
    "apply_temporal_augmentations",
    "RandomRotation",
    "RandomAffine",
    "RandomGaussianNoise",
    "RandomGaussianBlur",
    "RandomCutout",
    "RandomElasticTransform",
    "ConsistentAugmentation",
    "temporal_frame_drop",
    "temporal_frame_duplicate",
    "temporal_reverse",
]

