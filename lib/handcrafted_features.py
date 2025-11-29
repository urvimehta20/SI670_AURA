"""
Handcrafted feature extraction for baseline models.

Extracts features like:
- Noise residual energy
- DCT band statistics
- Local blur/sharpness
- Block boundary inconsistency
- Codec-centric cues
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import cv2
from scipy.fft import dct
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_noise_residual(frame: np.ndarray) -> np.ndarray:
    """
    Extract noise residual using high-pass filtering.
    
    Args:
        frame: Input frame (H, W, C) in uint8 format
    
    Returns:
        Noise residual array
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Convert to float64 to match CV_64F operations below.
    # Some OpenCV builds do not support mixing float32 input with CV_64F output
    # for linear filters (see "Unsupported combination of source format (=5) and
    # destination format (=6)" errors). Using float64 consistently avoids this.
    gray_float = gray.astype(np.float64)
    
    # Apply Gaussian blur to get smooth version
    blurred = cv2.GaussianBlur(gray_float, (5, 5), 1.0)
    
    # Compute residual (high-frequency component)
    residual = gray_float - blurred
    
    return residual


def extract_dct_statistics(frame: np.ndarray, block_size: int = 8) -> Dict[str, float]:
    """
    Extract DCT band statistics from frame.
    
    Args:
        frame: Input frame (H, W, C) in uint8 format
        block_size: DCT block size (typically 8 for JPEG)
    
    Returns:
        Dictionary of DCT statistics
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Convert to float and normalize
    gray_float = gray.astype(np.float32) / 255.0
    
    # Pad to multiple of block_size
    h, w = gray_float.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        gray_float = np.pad(gray_float, ((0, pad_h), (0, pad_w)), mode='edge')
    
    # Compute DCT for each block
    h_blocks = gray_float.shape[0] // block_size
    w_blocks = gray_float.shape[1] // block_size
    
    dct_coeffs = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = gray_float[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            dct_coeffs.append(dct_block)
    
    dct_coeffs = np.array(dct_coeffs)
    
    # Extract statistics
    # DC coefficient (0,0) - average intensity
    dc_coeffs = dct_coeffs[:, 0, 0]
    
    # AC coefficients (excluding DC)
    ac_coeffs = dct_coeffs[:, :, :].flatten()
    ac_coeffs = np.delete(ac_coeffs, np.arange(0, len(ac_coeffs), block_size * block_size))  # Remove DC
    
    # Low-frequency AC (first row/column, excluding DC)
    low_freq = []
    for block in dct_coeffs:
        # First row (excluding DC)
        low_freq.extend(block[0, 1:min(4, block_size)])
        # First column (excluding DC)
        low_freq.extend(block[1:min(4, block_size), 0])
    low_freq = np.array(low_freq)
    
    # High-frequency AC (corner region)
    high_freq = []
    for block in dct_coeffs:
        high_freq.extend(block[block_size//2:, block_size//2:].flatten())
    high_freq = np.array(high_freq)
    
    stats = {
        'dct_dc_mean': float(np.mean(dc_coeffs)),
        'dct_dc_std': float(np.std(dc_coeffs)),
        'dct_ac_mean': float(np.mean(np.abs(ac_coeffs))),
        'dct_ac_std': float(np.std(ac_coeffs)),
        'dct_low_freq_mean': float(np.mean(np.abs(low_freq))) if len(low_freq) > 0 else 0.0,
        'dct_high_freq_mean': float(np.mean(np.abs(high_freq))) if len(high_freq) > 0 else 0.0,
        'dct_energy_ratio': float(np.mean(np.abs(low_freq)) / (np.mean(np.abs(high_freq)) + 1e-6)) if len(high_freq) > 0 and len(low_freq) > 0 else 0.0,
    }
    
    return stats


def extract_blur_sharpness(frame: np.ndarray) -> Dict[str, float]:
    """
    Extract blur and sharpness metrics.
    
    Args:
        frame: Input frame (H, W, C) in uint8 format
    
    Returns:
        Dictionary of blur/sharpness statistics
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Ensure grayscale is uint8; many OpenCV builds are best-behaved when using
    # uint8 input with CV_64F output for linear filters. Using float32 input
    # with CV_64F can trigger "Unsupported combination of source format (=5)
    # and destination format (=6)" on some platforms.
    gray_u8 = gray.astype(np.uint8, copy=False)
    
    # Laplacian variance (measure of sharpness)
    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    laplacian_var = float(np.var(laplacian))
    
    # Gradient magnitude statistics
    grad_x = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    grad_mean = float(np.mean(gradient_magnitude))
    grad_std = float(np.std(gradient_magnitude))
    grad_max = float(np.max(gradient_magnitude))
    
    # Tenengrad (another sharpness measure)
    tenengrad = float(np.sum(gradient_magnitude**2))
    
    # Brenner gradient (simple sharpness measure) – operate on float for stability
    gray_float = gray_u8.astype(np.float32)
    brenner = float(np.sum((gray_float[2:, :] - gray_float[:-2, :])**2))
    
    stats = {
        'laplacian_variance': laplacian_var,
        'gradient_mean': grad_mean,
        'gradient_std': grad_std,
        'gradient_max': grad_max,
        'tenengrad': tenengrad,
        'brenner_gradient': brenner,
    }
    
    return stats


def extract_boundary_inconsistency(frame: np.ndarray, block_size: int = 8) -> float:
    """
    Detect block boundary inconsistency (compression artifacts).
    
    Args:
        frame: Input frame (H, W, C) in uint8 format
        block_size: Expected block size (8 for JPEG, 16 for H.264)
    
    Returns:
        Boundary inconsistency score
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Convert to float
    gray_float = gray.astype(np.float32)
    
    h, w = gray_float.shape
    
    # Compute gradients
    grad_x = np.abs(np.diff(gray_float, axis=1))
    grad_y = np.abs(np.diff(gray_float, axis=0))
    
    # Check for high gradients at block boundaries
    boundary_scores = []
    
    # Vertical boundaries
    for i in range(block_size, h, block_size):
        if i < grad_y.shape[0]:
            boundary_scores.append(np.mean(grad_y[i, :]))
    
    # Horizontal boundaries
    for j in range(block_size, w, block_size):
        if j < grad_x.shape[1]:
            boundary_scores.append(np.mean(grad_x[:, j]))
    
    # Compare boundary gradients to non-boundary gradients
    if len(boundary_scores) > 0:
        boundary_mean = np.mean(boundary_scores)
        # Average gradient in non-boundary regions
        non_boundary_grad = np.mean(grad_x) + np.mean(grad_y)
        inconsistency = float(boundary_mean / (non_boundary_grad + 1e-6))
    else:
        inconsistency = 0.0
    
    return inconsistency


def extract_codec_cues(video_path: str) -> Dict[str, float]:
    """
    Extract codec-centric cues from video metadata.
    
    Note: This is a simplified version. Full implementation would require
    deep video analysis or metadata extraction.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary of codec-related features
    """
    import subprocess
    import json
    
    cues = {
        'has_metadata': 0.0,
        'estimated_bitrate': 0.0,
        'estimated_fps': 0.0,
    }
    
    # Try to extract metadata using ffprobe if available
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            cues['has_metadata'] = 1.0
            
            # Extract bitrate
            if 'format' in metadata and 'bit_rate' in metadata['format']:
                try:
                    cues['estimated_bitrate'] = float(metadata['format']['bit_rate']) / 1e6  # Convert to Mbps
                except (ValueError, TypeError):
                    pass
            
            # Extract FPS
            if 'streams' in metadata and len(metadata['streams']) > 0:
                for stream in metadata['streams']:
                    if stream.get('codec_type') == 'video':
                        if 'r_frame_rate' in stream:
                            try:
                                num, den = map(int, stream['r_frame_rate'].split('/'))
                                cues['estimated_fps'] = float(num) / den if den > 0 else 0.0
                            except (ValueError, AttributeError):
                                pass
                        break
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        # ffprobe not available or failed
        pass
    
    return cues


def extract_all_features(
    video_path: str,
    num_frames: int = 8,
    project_root: Optional[str] = None
) -> np.ndarray:
    """
    Extract all handcrafted features from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        project_root: Project root for path resolution
    
    Returns:
        Feature vector as numpy array
    """
    from .video_modeling import _read_video_wrapper, uniform_sample_indices
    from .video_paths import resolve_video_path
    
    # Resolve video path
    if project_root:
        video_path = resolve_video_path(video_path, project_root)
    
    # Read video
    try:
        video = _read_video_wrapper(video_path)
        if video.shape[0] == 0:
            logger.warning(f"Video has no frames: {video_path}")
            return np.zeros(50)  # Return zero vector if video is empty
    except Exception as e:
        logger.warning(f"Failed to read video {video_path}: {e}")
        return np.zeros(50)
    
    total_frames = video.shape[0]
    
    # Sample frames uniformly
    indices = uniform_sample_indices(total_frames, num_frames)
    
    # Extract features from each frame
    all_features = []
    
    for idx in indices:
        frame = video[idx].numpy()  # (H, W, C) in uint8
        
        # Noise residual
        noise_residual = extract_noise_residual(frame)
        noise_features = [
            np.mean(noise_residual),
            np.std(noise_residual),
            np.max(np.abs(noise_residual)),
        ]
        
        # DCT statistics
        dct_stats = extract_dct_statistics(frame)
        
        # Blur/sharpness
        blur_stats = extract_blur_sharpness(frame)
        
        # Boundary inconsistency
        boundary_score = extract_boundary_inconsistency(frame)
        
        # Combine features for this frame
        frame_features = (
            noise_features +
            list(dct_stats.values()) +
            list(blur_stats.values()) +
            [boundary_score]
        )
        
        all_features.append(frame_features)
    
    # Average features across frames
    avg_features = np.mean(all_features, axis=0)
    
    # Add codec cues (once per video, not per frame)
    codec_cues = extract_codec_cues(video_path)
    codec_features = list(codec_cues.values())
    
    # Combine all features
    final_features = np.concatenate([avg_features, codec_features])
    
    return final_features


class HandcraftedFeatureExtractor:
    """
    Feature extractor that caches results to disk.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, num_frames: int = 8):
        """
        Initialize feature extractor.
        
        Args:
            cache_dir: Directory to cache extracted features
            num_frames: Number of frames to sample per video
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
    
    def _get_cache_path(self, video_path: str) -> Optional[Path]:
        """Get cache path for a video."""
        if not self.cache_dir:
            return None
        
        import hashlib
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
        return self.cache_dir / f"{video_hash}.npy"
    
    def extract(self, video_path: str, project_root: Optional[str] = None) -> np.ndarray:
        """
        Extract features for a video, using cache if available.
        
        Args:
            video_path: Path to video file
            project_root: Project root for path resolution
        
        Returns:
            Feature vector
        """
        cache_path = self._get_cache_path(video_path)
        
        # Check cache
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached features: {e}")
        
        # Extract features
        features = extract_all_features(video_path, self.num_frames, project_root)
        
        # Save to cache
        if cache_path:
            try:
                np.save(cache_path, features)
            except Exception as e:
                logger.warning(f"Failed to cache features: {e}")
        
        return features
    
    def extract_batch(
        self,
        video_paths: List[str],
        project_root: Optional[str] = None,
        batch_size: int = 10
    ) -> np.ndarray:
        """
        Extract features for multiple videos in batches.
        
        Args:
            video_paths: List of video paths
            project_root: Project root for path resolution
            batch_size: Number of videos to process before GC
        
        Returns:
            Feature matrix (n_videos, n_features)
        """
        all_features = []
        
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i+batch_size]
            
            for video_path in batch_paths:
                features = self.extract(video_path, project_root)
                all_features.append(features)
            
            # Aggressive GC after each batch
            import gc
            gc.collect()
        
        return np.array(all_features)


__all__ = [
    "extract_noise_residual",
    "extract_dct_statistics",
    "extract_blur_sharpness",
    "extract_boundary_inconsistency",
    "extract_codec_cues",
    "extract_all_features",
    "HandcraftedFeatureExtractor",
]

