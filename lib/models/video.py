"""
Core video modeling utilities for FVC:

- Inception-like 3D CNN (`VariableARVideoModel`) that supports variable aspect ratios
  and arbitrary spatial resolutions via global pooling.
- Video dataset and augmentations that:
    * sample a fixed number of frames per clip
    * preserve original spatial resolution (no resize/crop) for the variable-AR pipeline
    * support batching of mixed aspect ratios via padding in a custom collate_fn.

This is a pure Python module (no notebooks) intended for use in training / eval scripts.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any, Union

import numpy as np
import polars as pl
import torch

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

# Try to import modern video reading libraries
TORCHCODEC_AVAILABLE = False
try:
    # Try different possible import paths for TorchCodec
    try:
        from torchcodec.decoders import VideoDecoder as TorchCodecDecoder
        TORCHCODEC_AVAILABLE = True
    except ImportError:
        try:
            from torchcodec import VideoDecoder as TorchCodecDecoder
            TORCHCODEC_AVAILABLE = True
        except ImportError:
            pass
except Exception:
    pass

# Fallback to torchvision with warning suppression
try:
    # Suppress the deprecation warning at import time
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                              message=".*video decoding.*deprecated.*")
        from torchvision.io import read_video as torchvision_read_video
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    torchvision_read_video = None


# ---------------------------------------------------------------------------
# Configuration and helpers
# ---------------------------------------------------------------------------

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def _read_video_wrapper(video_path: str) -> torch.Tensor:
    """
    Unified video reading wrapper that tries TorchCodec first, then torchvision.
    Returns video tensor of shape (T, H, W, C) where T is number of frames.
    """
    try:
        # Try TorchCodec first (modern, no deprecation warnings)
        if TORCHCODEC_AVAILABLE:
            try:
                decoder = TorchCodecDecoder(video_path)
                # TorchCodec API may vary - try different methods
                try:
                    frames_list = decoder.get_all_frames()  # Returns list of tensors
                except AttributeError:
                    # Alternative API: decode all frames
                    frames_list = [decoder.get_frame(i) for i in range(decoder.num_frames())]
                
                if frames_list:
                    # Stack frames: each frame is (H, W, C), stack to (T, H, W, C)
                    video = torch.stack(frames_list, dim=0)
                    return video
                else:
                    raise RuntimeError(f"No frames decoded from {video_path}")
            except Exception as torchcodec_error:
                # Fall back to torchvision if TorchCodec fails
                if TORCHVISION_AVAILABLE:
                    # Suppress deprecation warning
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, 
                                              message=".*video decoding.*deprecated.*")
                        warnings.filterwarnings("ignore", category=UserWarning,
                                              module="torchvision.io._video_deprecation_warning")
                        video, _, _ = torchvision_read_video(video_path, pts_unit="sec")
                    return video
                else:
                    raise RuntimeError(
                        f"TorchCodec failed and torchvision not available. "
                        f"TorchCodec error: {str(torchcodec_error)}"
                    ) from torchcodec_error
        elif TORCHVISION_AVAILABLE:
            # Use torchvision with comprehensive warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                      message=".*video decoding.*deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning,
                                      module="torchvision.io._video_deprecation_warning")
                video, _, _ = torchvision_read_video(video_path, pts_unit="sec")
            return video
        else:
            raise RuntimeError(
                "Neither TorchCodec nor torchvision.io available for video reading. "
                "Install torchcodec (recommended) or ensure torchvision is installed."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to read video: {video_path}. Error: {str(e)}"
        ) from e


@dataclass
class VideoConfig:
    """Configuration for video sampling and preprocessing.
    
    The model architecture (AdaptiveAvgPool3d) can handle variable temporal dimensions,
    so num_frames is flexible. However, for efficiency, we typically sample a fixed number.
    
    For rolling window inference, use rolling_window=True and specify window_size.
    
    For memory efficiency, use fixed_size for consistent batch dimensions (recommended),
    or max_size for variable aspect ratios (requires padding in collate function).
    """

    num_frames: int = 1000
    # Fixed size for both dimensions (e.g., 256) - uses letterboxing to maintain aspect ratio
    # This ensures all videos have the same dimensions after preprocessing, avoiding padding issues
    # Default: 256 (matches scaled video dimensions from Stage 3)
    # None = no resizing (preserves original resolution, memory-intensive)
    fixed_size: Optional[int] = None  # e.g., 256 (matches scaled videos)
    # Maximum size for the longer edge when resizing (maintains aspect ratio, but results in variable sizes)
    # Use this only if you want variable aspect ratios (requires padding in batches)
    # None = no resizing (preserves original resolution, memory-intensive)
    max_size: Optional[int] = None  # e.g., 256 (matches scaled videos)
    # Legacy: kept for compatibility, but use fixed_size instead
    img_size: Optional[int] = None  # Deprecated, use fixed_size
    # Rolling window options for inference
    rolling_window: bool = False  # If True, use rolling windows instead of uniform sampling
    window_size: Optional[int] = None  # Size of rolling window (defaults to num_frames)
    window_stride: Optional[int] = None  # Stride between windows (defaults to window_size // 2)
    # Augmentation configuration (see lib.augmentation.transforms for details)
    augmentation_config: Optional[dict] = None  # Spatial augmentation parameters
    temporal_augmentation_config: Optional[dict] = None  # Temporal augmentation parameters


def uniform_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    """Uniformly sample `num_frames` indices from [0, total_frames-1].

    If `total_frames` < `num_frames`, frames are repeated.
    """
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.extend(indices)
        return indices[:num_frames]

    step = total_frames / float(num_frames)
    return [int(step * i) for i in range(num_frames)]


def rolling_window_indices(
    total_frames: int, 
    window_size: int, 
    stride: Optional[int] = None
) -> List[List[int]]:
    """
    Generate rolling window indices for temporal analysis.
    
    Args:
        total_frames: Total number of frames in video
        window_size: Number of frames per window
        stride: Stride between windows (defaults to window_size // 2 for 50% overlap)
    
    Returns:
        List of window index lists, each containing window_size frame indices
    """
    if total_frames <= 0 or window_size <= 0:
        return []
    
    if stride is None:
        stride = max(1, window_size // 2)  # 50% overlap by default
    
    if total_frames <= window_size:
        # If video is shorter than window, return single window with all frames
        return [list(range(total_frames))]
    
    windows = []
    start = 0
    while start + window_size <= total_frames:
        windows.append(list(range(start, start + window_size)))
        start += stride
    
    # Always include the last window (even if it overlaps)
    if windows and windows[-1][-1] < total_frames - 1:
        last_start = max(0, total_frames - window_size)
        windows.append(list(range(last_start, total_frames)))
    
    return windows


def build_frame_transforms(
    train: bool = True, 
    fixed_size: Optional[int] = None,
    max_size: Optional[int] = None
) -> transforms.Compose:
    """Build a per-frame transform pipeline.

    Args:
        train: If True, apply data augmentations (flip, color jitter)
        fixed_size: Fixed size for both dimensions (e.g., 256). Uses letterboxing to maintain aspect ratio.
                    Ensures all videos have the same dimensions, avoiding padding issues.
        max_size: Maximum size for longer edge when resizing (maintains aspect ratio, but variable output size).
                  Only used if fixed_size is None. Results in variable dimensions that need padding in batches.
    
    Returns:
        Transform pipeline that optionally resizes and applies augmentations.
    """
    transform_list = [transforms.functional.to_pil_image]
    
    # Resize strategy: prefer fixed_size (letterboxing) over max_size (variable)
    if fixed_size is not None:
        # Fixed size with letterboxing: resize to fit within fixed_size x fixed_size while maintaining aspect ratio
        # This ensures all frames have the same dimensions (fixed_size x fixed_size)
        def letterbox_resize(img):
            """Resize image to fit within fixed_size x fixed_size, adding black bars if needed.
            
            img is already a PIL Image from transforms.functional.to_pil_image.
            PIL/Pillow is typically available via torchvision dependencies.
            """
            from PIL import Image  # PIL is available via torchvision
            
            w, h = img.size
            scale = min(fixed_size / w, fixed_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Use LANCZOS resampling for high quality (resample=3)
            img_resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            
            # Create a black canvas of fixed_size x fixed_size
            canvas = Image.new('RGB', (fixed_size, fixed_size), (0, 0, 0))
            # Center the resized image on the canvas
            paste_x = (fixed_size - new_w) // 2
            paste_y = (fixed_size - new_h) // 2
            canvas.paste(img_resized, (paste_x, paste_y))
            return canvas
        
        transform_list.append(letterbox_resize)
    elif max_size is not None:
        # Variable size: resize longer edge to max_size (maintains aspect ratio)
        # This results in variable dimensions that need padding in batches
        transform_list.append(transforms.Resize(max_size, antialias=True))
    
    # Training augmentations
    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
        ])
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    
    return transforms.Compose(transform_list)


# ---------------------------------------------------------------------------
# Dataset and batching for variable-AR pipeline
# ---------------------------------------------------------------------------


class VideoDataset(Dataset):
    """Dataset over videos described in a DataFrame.

    Expects columns at least:
      - video_path
      - label
    Optionally:
      - subset (train/val/test) for the caller to filter before constructing.
    """

    def __init__(
        self,
        df: Union[pl.DataFrame, Any],
        project_root: str,
        config: VideoConfig,
        train: bool = True,
        max_videos: Optional[int] = None,
    ) -> None:
        # Prefer Polars; support pandas-like objects minimally for compatibility.
        if isinstance(df, pl.DataFrame):
            self.df_pl: pl.DataFrame = df.clone()
            if max_videos is not None:
                self.df_pl = self.df_pl[:max_videos]
            self._use_polars = True
        else:
            # Fallback: treat as pandas-like
            self.df_pl = None  # type: ignore[assignment]
            self.df_pd = df.reset_index(drop=True)  # type: ignore[attr-defined]
            if max_videos is not None:
                self.df_pd = self.df_pd.iloc[:max_videos].reset_index(drop=True)  # type: ignore[attr-defined]
            self._use_polars = False

        self.project_root = project_root
        self.config = config
        self.train = train

        # Build label mapping (handles string labels as well).
        if self._use_polars:
            # Polars .to_list() method - compatible with all Polars versions
            labels = sorted(self.df_pl["label"].unique().to_list())
        else:
            labels = sorted(self.df_pd["label"].unique())  # type: ignore[attr-defined]
        self.label_to_idx: Dict[Any, int] = {lbl: i for i, lbl in enumerate(labels)}

        # Determine resize strategy: prefer fixed_size, then max_size, then img_size (legacy)
        fixed_size = self.config.fixed_size
        max_size = self.config.max_size
        
        # Legacy support: if img_size is set, use it as fixed_size
        if fixed_size is None and self.config.img_size is not None:
            fixed_size = self.config.img_size
            logger.warning(
                "Using deprecated img_size=%d. Please use fixed_size instead.",
                self.config.img_size
            )
        
        # Use comprehensive augmentations if available, otherwise fallback to basic
        try:
            from lib.augmentation.transforms import build_comprehensive_frame_transforms
            self._frame_transform, self._post_tensor_transform = build_comprehensive_frame_transforms(
                train=train,
                fixed_size=fixed_size,
                max_size=max_size,
                augmentation_config=getattr(self.config, 'augmentation_config', None),
            )
        except ImportError:
            # Fallback to basic augmentations
            logger.warning("Comprehensive augmentations not available, using basic augmentations")
            self._frame_transform = build_frame_transforms(
                train=train, 
                fixed_size=fixed_size,
                max_size=max_size
            )
            self._post_tensor_transform = None

    def __len__(self) -> int:
        if self._use_polars:
            return self.df_pl.height
        return len(self.df_pd)  # type: ignore[arg-type]

    def _get_row(self, idx: int) -> Dict[str, Any]:
        if self._use_polars:
            return self.df_pl.row(idx, named=True)
        # pandas-like
        row = self.df_pd.iloc[idx]  # type: ignore[attr-defined]
        return row.to_dict()  # type: ignore[no-any-return]

    def _get_video_path(self, row: Dict[str, Any]) -> str:
        from lib.utils.paths import resolve_video_path
        video_rel = row["video_path"]
        return resolve_video_path(video_rel, self.project_root)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._get_row(idx)
        video_path = self._get_video_path(row)
        label_value = row["label"]
        label_idx = self.label_to_idx[label_value]
        
        # Check if this is a pre-computed augmented clip (.pt or .pth file)
        if video_path.endswith('.pt') or video_path.endswith('.pth'):
            try:
                from lib.augmentation.pregenerate import load_precomputed_clip
                clip = load_precomputed_clip(video_path)
                # Ensure clip has correct shape: (T, C, H, W)
                if clip.dim() == 4:
                    # Verify frame count matches config (allow some flexibility)
                    if clip.shape[0] == self.config.num_frames:
                        return clip, torch.tensor(label_idx, dtype=torch.long)
                    else:
                        logger.warning(
                            "Pre-computed clip has wrong frame count: %d. Expected %d. Resampling...",
                            clip.shape[0], self.config.num_frames
                        )
                        # Resample frames to match config
                        if clip.shape[0] > self.config.num_frames:
                            # Uniformly sample (uniform_sample_indices is already imported at module level)
                            indices = uniform_sample_indices(clip.shape[0], self.config.num_frames)
                            clip = clip[indices]
                        else:
                            # Pad with last frame
                            last_frame = clip[-1:]
                            while clip.shape[0] < self.config.num_frames:
                                clip = torch.cat([clip, last_frame], dim=0)
                        return clip, torch.tensor(label_idx, dtype=torch.long)
                else:
                    logger.warning("Pre-computed clip has wrong dimensions: %s. Expected 4D (T, C, H, W). Falling back.",
                                 str(clip.shape))
            except Exception as e:
                logger.warning("Failed to load pre-computed clip %s: %s. Falling back to video loading.", 
                             video_path, str(e))
                # Fall through to regular video loading

        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found at index {idx}: {video_path}. "
                "Use filter_existing_videos() before creating the dataset."
            )
        
        if not os.path.isfile(video_path):
            raise ValueError(
                f"Video path is not a file at index {idx}: {video_path}"
            )

        # Use unified video reading wrapper
        try:
            video = _read_video_wrapper(video_path)
        except Exception as e:
            error_msg = str(e).lower()
            # Handle corrupted videos (moov atom not found, etc.)
            if 'moov atom' in error_msg or 'invalid data' in error_msg or 'corrupt' in error_msg:
                logger.warning(
                    f"Corrupted video at index {idx}: {video_path}. Error: {str(e)}. "
                    f"Skipping this video. Consider filtering corrupted videos upfront."
                )
                # Return a dummy sample (all zeros) with the correct label
                default_size = self.config.fixed_size or self.config.max_size or self.config.img_size or 256
                dummy_clip = torch.zeros(
                    (self.config.num_frames, 3, default_size, default_size),
                    dtype=torch.float32
                )
                return dummy_clip, torch.tensor(label_idx, dtype=torch.long)
            else:
                raise RuntimeError(
                    f"Failed to read video at index {idx}: {video_path}. "
                    f"Error: {str(e)}"
                ) from e
        
        # Check if video has frames
        # If filter_existing_videos was called with check_frames=True, this should be rare
        if video.shape[0] == 0:
            # Instead of crashing, log a warning and skip this video by returning a dummy sample
            # This allows training to continue even if some videos are empty
            logger.warning(
                "Video has no frames at index %d: %s. Skipping this video. "
                "Consider running filter_existing_videos(..., check_frames=True) to filter these upfront.",
                idx, video_path
            )
            # Return a dummy sample (all zeros) with the correct label
            # This is a workaround - ideally these should be filtered out
            # Use a reasonable default size if fixed_size/max_size/img_size not set
            default_size = self.config.fixed_size or self.config.max_size or self.config.img_size or 256
            dummy_clip = torch.zeros(
                (self.config.num_frames, 3, default_size, default_size),
                dtype=torch.float32
            )
            return dummy_clip, torch.tensor(label_idx, dtype=torch.long)
        
        total_frames = video.shape[0]

        # Support both uniform sampling and rolling windows
        if self.config.rolling_window:
            window_size = self.config.window_size or self.config.num_frames
            stride = self.config.window_stride
            windows = rolling_window_indices(total_frames, window_size, stride)
            
            if not windows:
                raise RuntimeError(
                    f"No windows generated from video at index {idx}: {video_path}"
                )
            
            # For training, randomly select one window; for inference, use first window
            # (Inference can aggregate multiple windows later)
            if self.train:
                import random
                selected_window = random.choice(windows)
            else:
                selected_window = windows[0]  # Use first window for deterministic inference
            
            indices = selected_window
        else:
            # Uniform sampling (default)
            indices = uniform_sample_indices(total_frames, self.config.num_frames)
        
        frames: List[torch.Tensor] = []

        # Apply temporal augmentations if available
        try:
            from lib.augmentation.transforms import apply_temporal_augmentations
            use_temporal_aug = True
            temporal_config = getattr(self.config, 'temporal_augmentation_config', None) or {}
        except ImportError:
            use_temporal_aug = False
            temporal_config = {}

        for i in indices:
            frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
            frame_tensor = self._frame_transform(frame)  # (C, H, W)
            
            # Apply post-tensor augmentations if available
            if self._post_tensor_transform is not None:
                frame_tensor = self._post_tensor_transform(frame_tensor)
            
            frames.append(frame_tensor)
        
        # Apply temporal augmentations (frame dropping, duplication, reversal)
        if use_temporal_aug and self.train:
            frames = apply_temporal_augmentations(
                frames,
                train=self.train,
                frame_drop_prob=temporal_config.get('frame_drop_prob', 0.1),
                frame_dup_prob=temporal_config.get('frame_dup_prob', 0.1),
                reverse_prob=temporal_config.get('reverse_prob', 0.1),
            )

        # Check if we have frames to stack
        if len(frames) == 0:
            raise RuntimeError(
                f"No frames extracted from video at index {idx}: {video_path}"
            )

        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        label = torch.tensor(label_idx, dtype=torch.long)
        return clip, label


def variable_ar_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad clips in batch to max H and W, preserving content.

    batch: list of (clip, label), where clip is (T, C, H, W).
    Returns:
      clips_padded: (N, C, T, H_max, W_max)
      labels: (N,)
    """
    clips, labels = zip(*batch)

    # Determine max sizes.
    T = clips[0].shape[0]
    C = clips[0].shape[1]
    H_max = max(c.shape[2] for c in clips)
    W_max = max(c.shape[3] for c in clips)

    padded_clips: List[torch.Tensor] = []
    for c in clips:
        _, _, h, w = c.shape
        pad_h = H_max - h
        pad_w = W_max - w
        # Pad (left, right, top, bottom) -> (w_left, w_right, h_top, h_bottom)
        # Here we pad at the bottom and right sides only.
        c_padded = F.pad(c, (0, pad_w, 0, pad_h))  # (T, C, H_max, W_max)
        padded_clips.append(c_padded)

    clips_tensor = torch.stack(padded_clips, dim=0)  # (N, T, C, H_max, W_max)
    clips_tensor = clips_tensor.permute(0, 2, 1, 3, 4)  # (N, C, T, H_max, W_max)
    labels_tensor = torch.stack(labels, dim=0)
    return clips_tensor, labels_tensor


def build_sample_loader(
    meta_df: Any,
    project_root: str,
    config: Optional[VideoConfig] = None,
    num_videos: int = 10,
    batch_size: int = 2,
) -> DataLoader:
    """Build a small DataLoader on a subset of videos for interim testing."""
    if config is None:
        config = VideoConfig()

    # Prefer train-like subset if available.
    if isinstance(meta_df, pl.DataFrame):
        if "subset" in meta_df.columns:
            train_df = meta_df.filter(pl.col("subset").is_in(["train", "Train", "TRAIN"]))
            if train_df.height == 0:
                train_df = meta_df
        else:
            train_df = meta_df
        train_df = train_df.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=42)
        sample_df = train_df[:num_videos]
    else:
        # pandas-like fallback
        if "subset" in meta_df.columns:
            train_df = meta_df[meta_df["subset"].isin(["train", "Train", "TRAIN"])]
            if len(train_df) == 0:
                train_df = meta_df
        else:
            train_df = meta_df
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        sample_df = train_df.iloc[:num_videos].reset_index(drop=True)

    dataset = VideoDataset(sample_df, project_root, config=config, train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=variable_ar_collate,
    )
    return loader


# ---------------------------------------------------------------------------
# Inception-like 3D CNN for variable aspect ratios
# ---------------------------------------------------------------------------


class Inception3DBlock(nn.Module):
    """Simple Inception-style 3D block with multiple kernel sizes.

    Expects input of shape (N, C, T, H, W) and returns same T, H, W with more channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Split output channels across branches.
        b = out_channels // 4
        self.branch1 = nn.Conv3d(in_channels, b, kernel_size=1, padding=0)
        self.branch3 = nn.Conv3d(in_channels, b, kernel_size=3, padding=1)
        self.branch5 = nn.Conv3d(in_channels, b, kernel_size=5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, b, kernel_size=1, padding=0),
        )

        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        return out


class VariableARVideoModel(nn.Module):
    """Inception-like 3D CNN that supports variable aspect ratios via global pooling.

    Optimized for efficiency with proper initialization and BatchNorm momentum.
    
    - Input: (N, C, T, H, W) with arbitrary T, H, W.
    - Output: (N, 1) logits for binary classification.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels, momentum=0.1),  # Lower momentum for small batches
            nn.ReLU(inplace=True),
        )

        self.incept1 = Inception3DBlock(base_channels, base_channels * 2)
        self.incept2 = Inception3DBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # (N, C, 1, 1, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(base_channels * 4, 1)  # binary logit
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, H, W)
        x = self.stem(x)
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.pool(x)  # (N, C, 1, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        x = self.dropout(x)
        logits = self.fc(x)  # (N, 1)
        return logits


class PretrainedInceptionVideoModel(nn.Module):
    """Pretrained 3D ResNet backbone + Inception block head.

    - Backbone: torchvision.models.video.r3d_18 pretrained on Kinetics.
    - Head: Inception3DBlock + global average pooling + linear binary head.
    - Supports variable aspect ratios because everything is convolutional
      followed by adaptive pooling.
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            # Try to use the new weights API (torchvision 0.13+)
            try:
                weights = R3D_18_Weights.KINETICS400_V1
                backbone = r3d_18(weights=weights)
            except (AttributeError, ValueError):
                # Fallback: load without weights (pretrained=True for older torchvision)
                backbone = r3d_18(pretrained=True)
        except ImportError:
            raise ImportError(
                "torchvision.models.video.r3d_18 not available. "
                "Ensure torchvision is installed and supports video models."
            )

        # Extract convolutional backbone (stem + layers)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_backbone:
            for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in m.parameters():
                    p.requires_grad = False

        # r3d_18 final conv channels = 512
        in_channels = 512
        self.incept = Inception3DBlock(in_channels, in_channels)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels, 1)
        
        # Initialize the head properly for binary classification
        # Initialize fc layer with small weights to avoid saturation
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.incept(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


__all__ = [
    "VideoConfig",
    "VideoDataset",
    "variable_ar_collate",
    "build_sample_loader",
    "Inception3DBlock",
    "VariableARVideoModel",
    "PretrainedInceptionVideoModel",
    "uniform_sample_indices",
    "rolling_window_indices",
    "_read_video_wrapper",
]


