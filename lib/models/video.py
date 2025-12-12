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
    
    For OOM prevention, use chunk_size to process frames in chunks (e.g., 200 frames per chunk).
    When chunk_size is set, frames are loaded and processed in chunks, then concatenated.
    This reduces peak memory usage while ensuring num_frames total frames per video.
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
    # Chunked frame loading for OOM prevention (e.g., chunk_size=200 for 1000 total frames = 5 chunks)
    # When set, frames are loaded and processed in chunks, then concatenated to get num_frames total
    # This reduces peak memory usage by processing frames incrementally
    chunk_size: Optional[int] = None  # If set, process frames in chunks of this size
    # Augmentation configuration (DEPRECATED: Augmentation done in Stage 1, not Stage 5)
    # These are kept for backward compatibility but ignored when use_scaled_videos=True
    augmentation_config: Optional[dict] = None  # Spatial augmentation parameters (not used in Stage 5)
    temporal_augmentation_config: Optional[dict] = None  # Temporal augmentation parameters (not used in Stage 5)
    # If True, videos are already scaled/processed in Stage 3 - skip all transforms, only apply normalization
    # Stage 5 should ALWAYS use scaled videos (use_scaled_videos=True)
    use_scaled_videos: bool = False


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
    
    NOTE: This is LEGACY code. In Stage 5, videos should already be scaled (use_scaled_videos=True),
    so this function should not be called. Augmentation is done in Stage 1, scaling in Stage 3.

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


class AdaptiveChunkSizeManager:
    """
    Manages adaptive chunk sizes using AIMD (Additive Increase Multiplicative Decrease).
    
    - On OOM: Multiplicative decrease (chunk_size *= 0.5)
    - On success: Additive increase (chunk_size += increment)
    - Tracks optimal chunk sizes per video or globally
    """
    
    def __init__(
        self,
        initial_chunk_size: int = 200,
        min_chunk_size: int = 10,
        max_chunk_size: int = 500,
        decrease_factor: float = 0.5,
        increase_increment: int = 10,
        success_threshold: int = 3,  # Number of successes before increasing
    ):
        self.initial_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.decrease_factor = decrease_factor
        self.increase_increment = increase_increment
        self.success_threshold = success_threshold
        
        # Global chunk size cache (can be per-video in future)
        self._global_chunk_size: Optional[int] = None
        self._success_count: int = 0
        self._video_chunk_sizes: Dict[str, int] = {}  # Per-video optimal sizes
        
    def get_chunk_size(self, video_path: Optional[str] = None, base_chunk_size: Optional[int] = None) -> int:
        """
        Get current chunk size for a video.
        
        Args:
            video_path: Optional video path for per-video tracking
            base_chunk_size: Base chunk size from config (if provided, use as starting point)
        
        Returns:
            Current chunk size to use
        """
        # Use base_chunk_size if provided, otherwise use cached or initial
        if base_chunk_size is not None:
            current_size = base_chunk_size
        elif video_path and video_path in self._video_chunk_sizes:
            current_size = self._video_chunk_sizes[video_path]
        elif self._global_chunk_size is not None:
            current_size = self._global_chunk_size
        else:
            current_size = self.initial_chunk_size
        
        # Ensure within bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, current_size))
    
    def on_oom(self, video_path: Optional[str] = None) -> int:
        """
        Handle OOM error by reducing chunk size (multiplicative decrease).
        
        Args:
            video_path: Optional video path for per-video tracking
        
        Returns:
            New reduced chunk size
        """
        current_size = self.get_chunk_size(video_path)
        new_size = max(self.min_chunk_size, int(current_size * self.decrease_factor))
        
        # Update cache
        if video_path:
            self._video_chunk_sizes[video_path] = new_size
        else:
            self._global_chunk_size = new_size
        
        # Reset success count on OOM
        self._success_count = 0
        
        logger.warning(
            f"OOM detected: reducing chunk size from {current_size} to {new_size} "
            f"(factor: {self.decrease_factor})"
        )
        
        return new_size
    
    def on_success(self, video_path: Optional[str] = None) -> int:
        """
        Handle successful processing by gradually increasing chunk size (additive increase).
        
        Args:
            video_path: Optional video path for per-video tracking
        
        Returns:
            Current chunk size (may be increased if threshold met)
        """
        self._success_count += 1
        current_size = self.get_chunk_size(video_path)
        
        # Only increase after success_threshold consecutive successes
        if self._success_count >= self.success_threshold:
            new_size = min(self.max_chunk_size, current_size + self.increase_increment)
            
            if new_size > current_size:
                # Update cache
                if video_path:
                    self._video_chunk_sizes[video_path] = new_size
                else:
                    self._global_chunk_size = new_size
                
                logger.info(
                    f"Chunk size increased from {current_size} to {new_size} "
                    f"(after {self._success_count} successes, increment: {self.increase_increment})"
                )
                # Reset success count after increase
                self._success_count = 0
                return new_size
        
        return current_size


# Global adaptive chunk size manager (shared across all VideoDataset instances)
_adaptive_chunk_manager = AdaptiveChunkSizeManager(
    initial_chunk_size=200,
    min_chunk_size=10,
    max_chunk_size=500,
    decrease_factor=0.5,
    increase_increment=10,
    success_threshold=3,
)


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
        adaptive_chunk_size: bool = True,  # Enable adaptive chunk sizing
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
        self.adaptive_chunk_size = adaptive_chunk_size and getattr(config, 'chunk_size', None) is not None

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
        
        # Check if using scaled videos (already processed) - skip transforms except normalization
        use_scaled_videos = getattr(self.config, 'use_scaled_videos', False)
        
        if use_scaled_videos:
            # Scaled videos are already processed in Stage 3 - only apply normalization
            # No resizing, no augmentation (augmentation done in Stage 1, scaling done in Stage 3)
            logger.debug("Using scaled videos: skipping all transforms, applying normalization only")
            self._frame_transform = transforms.Compose([
                transforms.functional.to_pil_image,
                transforms.ToTensor(),
            ])
            self._post_tensor_transform = transforms.Compose([
                transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
            ])
        else:
            # Legacy path: Use comprehensive augmentations if available, otherwise fallback to basic
            # This path should not be used in Stage 5 (all videos should be scaled)
            logger.warning("Not using scaled videos - this should not happen in Stage 5. Augmentation should be done in Stage 1.")
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
    
    def _load_frames_chunked(
        self,
        video: torch.Tensor,
        total_frames: int,
        chunk_size: int,
        video_path: str,
        idx: int
    ) -> List[torch.Tensor]:
        """
        Load frames in chunks with the specified chunk size.
        
        This is separated from __getitem__ to enable retry logic with different chunk sizes.
        Raises RuntimeError on OOM to trigger adaptive chunk size reduction.
        """
        # Chunked loading: process frames in chunks to reduce peak memory
        # Calculate number of chunks needed
        num_chunks = (self.config.num_frames + chunk_size - 1) // chunk_size  # Ceiling division
        frames_per_chunk = self.config.num_frames // num_chunks
        remainder = self.config.num_frames % num_chunks
        
        logger.debug(
            f"Chunked loading: {self.config.num_frames} frames in {num_chunks} chunks "
            f"(chunk_size={chunk_size}, frames_per_chunk={frames_per_chunk}, remainder={remainder})"
        )
        
        all_frames: List[torch.Tensor] = []
        # use_scaled_videos is passed from __getitem__ or defined here if called directly
        # This is a local variable in _load_frames_chunked, not used in this method but kept for consistency
        
        # Process each chunk sequentially to minimize peak memory
        for chunk_idx in range(num_chunks):
            try:
                # Calculate frames for this chunk (distribute remainder across first chunks)
                chunk_frames = frames_per_chunk + (1 if chunk_idx < remainder else 0)
                
                # Uniformly sample indices for this chunk from the video
                # Each chunk samples from the entire video to ensure good coverage
                chunk_indices = uniform_sample_indices(total_frames, chunk_frames)
                
                # Load and process frames for this chunk
                chunk_frame_tensors: List[torch.Tensor] = []
                for i in chunk_indices:
                    frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
                    frame_tensor = self._frame_transform(frame)  # (C, H, W)
                    
                    # Apply post-tensor augmentations if available (only normalization for scaled videos)
                    if self._post_tensor_transform is not None:
                        frame_tensor = self._post_tensor_transform(frame_tensor)
                    
                    chunk_frame_tensors.append(frame_tensor)
                
                # Add chunk frames to all_frames
                all_frames.extend(chunk_frame_tensors)
                
                # Clear chunk data to free memory immediately after processing
                del chunk_frame_tensors, chunk_indices
                import gc
                gc.collect()
                
                # Clear CUDA cache if available (helps with GPU memory)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Check if it's an OOM error - if so, re-raise to trigger adaptive reduction
                try:
                    from lib.utils.memory import check_oom_error
                    if check_oom_error(e):
                        # Clear memory before re-raising
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(f"OOM during chunk {chunk_idx + 1}/{num_chunks}: {e}") from e
                except ImportError:
                    # Fallback: check error message
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "cuda" in error_msg and "memory" in error_msg:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(f"OOM during chunk {chunk_idx + 1}/{num_chunks}: {e}") from e
                # Not OOM: re-raise original error
                raise
        
        # Ensure we have exactly num_frames (critical for training consistency)
        if len(all_frames) > self.config.num_frames:
            all_frames = all_frames[:self.config.num_frames]
            logger.debug(f"Trimmed frames from {len(all_frames)} to {self.config.num_frames}")
        elif len(all_frames) < self.config.num_frames:
            # Repeat frames if we're short (shouldn't happen with correct chunk calculation)
            logger.warning(
                f"Only got {len(all_frames)} frames, expected {self.config.num_frames}. "
                f"Repeating last frame to reach target."
            )
            while len(all_frames) < self.config.num_frames:
                all_frames.append(all_frames[-1] if all_frames else all_frames[0])
        
        # Final verification
        if len(all_frames) != self.config.num_frames:
            raise RuntimeError(
                f"Chunked loading failed: got {len(all_frames)} frames, expected {self.config.num_frames}"
            )
        
        return all_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._get_row(idx)
        video_path = self._get_video_path(row)
        label_value = row["label"]
        label_idx = self.label_to_idx[label_value]
        
        # DEAD CODE (Stage 5): Pre-computed clip loading not used in Stage 5
        # Stage 5 uses scaled videos from Stage 3, not pre-computed clips
        # This code path is kept for backward compatibility but is not executed in Stage 5 pipeline
        # Removed: 32 lines of pre-computed clip loading logic (lines 417-449)

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

        # Define use_scaled_videos early so it's available in both chunked and non-chunked paths
        use_scaled_videos = getattr(self.config, 'use_scaled_videos', False)

        # Support chunked frame loading for OOM prevention with adaptive sizing
        base_chunk_size = getattr(self.config, 'chunk_size', None)
        use_chunked_loading = base_chunk_size is not None and base_chunk_size > 0
        
        if use_chunked_loading:
            # Get adaptive chunk size if enabled, otherwise use base chunk size
            if self.adaptive_chunk_size:
                chunk_size = _adaptive_chunk_manager.get_chunk_size(
                    video_path=video_path,
                    base_chunk_size=base_chunk_size
                )
            else:
                chunk_size = base_chunk_size
            
            # Retry logic with adaptive chunk sizing on OOM
            max_retries = 5  # Maximum retries with reduced chunk size
            retry_count = 0
            frames: List[torch.Tensor] = []
            
            while retry_count <= max_retries:
                try:
                    frames = self._load_frames_chunked(
                        video, total_frames, chunk_size, video_path, idx
                    )
                    # Success: report to adaptive manager
                    if self.adaptive_chunk_size:
                        _adaptive_chunk_manager.on_success(video_path=video_path)
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # Use utility function for better OOM detection
                    try:
                        from lib.utils.memory import check_oom_error
                        is_oom = check_oom_error(e)
                    except ImportError:
                        # Fallback to string matching if utility not available
                        error_msg = str(e).lower()
                        is_oom = (
                            "out of memory" in error_msg or
                            "cuda" in error_msg and "memory" in error_msg or
                            "oom" in error_msg or
                            "outofmemoryerror" in error_msg
                        )
                    
                    if is_oom and self.adaptive_chunk_size and retry_count < max_retries:
                        # OOM detected: reduce chunk size and retry
                        chunk_size = _adaptive_chunk_manager.on_oom(video_path=video_path)
                        retry_count += 1
                        
                        # Clear memory before retry
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        logger.info(
                            f"OOM detected: retrying with reduced chunk size {chunk_size} "
                            f"(attempt {retry_count}/{max_retries})"
                        )
                        continue
                    else:
                        # Not OOM or max retries reached: re-raise
                        raise
            
            # Clear video tensor to free memory (chunks already processed)
            del video
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Original non-chunked loading
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

            # use_scaled_videos already defined above, no need to redefine

            for i in indices:
                frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
                frame_tensor = self._frame_transform(frame)  # (C, H, W)
                
                # Apply post-tensor augmentations if available (only normalization for scaled videos)
                if self._post_tensor_transform is not None:
                    frame_tensor = self._post_tensor_transform(frame_tensor)
                
                frames.append(frame_tensor)
        
        # No temporal augmentations when using scaled videos (augmentation done in Stage 1)
        # Temporal augmentations are only applied if NOT using scaled videos
        if not use_scaled_videos and self.train:
            try:
                from lib.augmentation.transforms import apply_temporal_augmentations
                # Get temporal config from VideoConfig, or use defaults
                base_temporal_config = getattr(self.config, 'temporal_augmentation_config', None) or {}
                # Build proper temporal_config dict matching function signature
                # Function expects: frame_drop, frame_duplicate, reverse (boolean flags)
                # Also accepts probabilities if needed, but function uses hardcoded 0.1
                temporal_config = {
                    'frame_drop': base_temporal_config.get('frame_drop', False) or base_temporal_config.get('frame_drop_prob', 0.0) > 0,
                    'frame_duplicate': base_temporal_config.get('frame_duplicate', False) or base_temporal_config.get('frame_dup_prob', 0.0) > 0,
                    'reverse': base_temporal_config.get('reverse', False) or base_temporal_config.get('reverse_prob', 0.0) > 0,
                }
                # Only apply if at least one augmentation is enabled
                if any(temporal_config.values()):
                    # CRITICAL: Function signature is: apply_temporal_augmentations(frames, temporal_config=dict)
                    # DO NOT pass train=, frame_drop_prob=, etc. as separate parameters
                    frames = apply_temporal_augmentations(frames, temporal_config=temporal_config)
            except ImportError as e:
                # Temporal augmentations module not available, skip silently
                logger.debug(f"Temporal augmentations not available: {e}")
            except TypeError as e:
                # Function signature mismatch - log clearly for debugging
                error_msg = str(e)
                if "unexpected keyword argument" in error_msg or "got an unexpected keyword argument" in error_msg:
                    logger.error(
                        f"CRITICAL: Function signature mismatch in apply_temporal_augmentations: {e}. "
                        f"Expected signature: apply_temporal_augmentations(frames, temporal_config=dict). "
                        f"Please check lib/augmentation/transforms.py for correct signature."
                    )
                # Skip temporal augmentations on error
                logger.warning(f"Skipping temporal augmentations due to error: {e}")
            except Exception as e:
                # Catch any other unexpected errors
                logger.warning(f"Unexpected error applying temporal augmentations: {e}, skipping")

        # Check if we have frames to stack
        if len(frames) == 0:
            raise RuntimeError(
                f"No frames extracted from video at index {idx}: {video_path}"
            )

        # Stack frames into clip tensor
        # For chunked loading, this happens after all chunks are processed
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        
        # Clear frames list to free memory (clip tensor now holds the data)
        del frames
        if not use_chunked_loading:
            # For non-chunked loading, also clear video tensor here
            del video
        import gc
        gc.collect()
        
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


