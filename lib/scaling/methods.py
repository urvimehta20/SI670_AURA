"""
Video scaling methods.

Provides:
- Resolution-based scaling (letterbox resize)
- Autoencoder-based scaling using pretrained Hugging Face models (optional)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Import OOM handling utilities
try:
    from lib.utils.memory import check_oom_error, handle_oom_error, get_memory_stats, aggressive_gc
except ImportError:
    # Fallback if memory utils not available
    def check_oom_error(e): return False
    def handle_oom_error(e, ctx=""): pass
    def get_memory_stats(): return {}
    def aggressive_gc(clear_cuda=True): pass

# Try to import torch for autoencoder support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Try to import Hugging Face transformers/diffusers for pretrained autoencoders
try:
    from diffusers import AutoencoderKL
    from transformers import AutoImageProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoencoderKL = None
    AutoImageProcessor = None


def letterbox_resize(
    frame: np.ndarray,
    target_size: int = 224
) -> np.ndarray:
    """
    Resize frame with letterboxing to maintain aspect ratio.
    
    Args:
        frame: Input frame (H, W, 3)
        target_size: Target size for both dimensions
    
    Returns:
        Resized frame (target_size, target_size, 3)
    """
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize using INTER_AREA for scaling (better quality, avoids aliasing)
    # INTER_AREA is specifically designed for image reduction and provides better results
    # than INTER_LINEAR when scaling down
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create letterbox
    result = np.zeros((target_size, target_size, 3), dtype=frame.dtype)
    paste_y = (target_size - new_h) // 2
    paste_x = (target_size - new_w) // 2
    result[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
    
    return result


def load_hf_autoencoder(
    model_name: str = "stabilityai/sd-vae-ft-mse",
    device: Optional[str] = None
) -> object:
    """
    Load a pretrained autoencoder from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier (default: Stable Diffusion VAE)
        device: Device to load model on ("cuda", "cpu", or None for auto-detect)
    
    Returns:
        Loaded autoencoder model
    
    Examples:
        - "stabilityai/sd-vae-ft-mse" - Stable Diffusion VAE (recommended)
        - "stabilityai/sd-vae-ft-ema" - Stable Diffusion VAE (EMA version)
        - "CompVis/stable-diffusion-v1-4" - Full Stable Diffusion (use .vae)
    """
    if not HF_AVAILABLE:
        raise RuntimeError(
            "Hugging Face diffusers library is required. Install with: pip install diffusers transformers"
        )
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for Hugging Face autoencoder")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading Hugging Face autoencoder: {model_name} on {device}")
    
    # Determine if this is likely a standalone VAE model
    # Standalone VAE models (e.g., "stabilityai/sd-vae-ft-mse") don't have a subfolder
    # Full Stable Diffusion models (e.g., "CompVis/stable-diffusion-v1-4") have VAE in "vae/" subfolder
    # Reference: https://huggingface.co/stabilityai/sd-vae-ft-mse
    is_likely_standalone = (
        "sd-vae" in model_name.lower() or 
        model_name.lower().endswith("-vae") or
        "/vae" in model_name  # Explicit path like "model/vae"
    )
    
    # Try loading with appropriate strategy based on model type
    # Use OSError for file-not-found errors (recommended by Hugging Face)
    try:
        if is_likely_standalone:
            # Standalone VAE models: try without subfolder first (correct for sd-vae-ft-mse)
            logger.debug(f"Attempting to load as standalone VAE (no subfolder)")
            try:
                vae = AutoencoderKL.from_pretrained(
                    model_name,
                    subfolder=None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
            except (OSError, FileNotFoundError) as e1:
                # If standalone load fails (file not found), try with subfolder as fallback
                logger.debug(f"Standalone load failed (file not found), trying with subfolder='vae': {e1}")
                vae = AutoencoderKL.from_pretrained(
                    model_name,
                    subfolder="vae",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
        else:
            # Full Stable Diffusion models: try with subfolder first
            logger.debug(f"Attempting to load from full model (subfolder='vae')")
            try:
                vae = AutoencoderKL.from_pretrained(
                    model_name,
                    subfolder="vae",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
            except (OSError, FileNotFoundError) as e2:
                # If subfolder load fails, try without subfolder as fallback
                logger.debug(f"Subfolder load failed (file not found), trying without subfolder: {e2}")
                vae = AutoencoderKL.from_pretrained(
                    model_name,
                    subfolder=None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
        
        vae = vae.to(device)
        vae.eval()
        logger.info(f"✓ Loaded autoencoder: {model_name}")
        return vae
    except Exception as e:
        logger.error(f"Failed to load Hugging Face autoencoder {model_name} after trying both subfolder options: {e}")
        raise


def _autoencoder_scale(
    frames: list,
    autoencoder: object,
    target_size: int = 256,
    preserve_aspect_ratio: bool = True
) -> list:
    """
    Scale frames using an autoencoder model, preserving aspect ratio.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size).
    
    Args:
        frames: List of frames (each is H, W, 3) as numpy arrays
        autoencoder: Autoencoder model (PyTorch nn.Module, Hugging Face VAE, or object with encode/decode methods)
        target_size: Target max dimension (max(width, height) will be target_size). Default: 256
        preserve_aspect_ratio: If True, maintain original aspect ratio with max dimension = target_size; if False, force square output
    
    Returns:
        List of scaled frames with max(width, height) = target_size, preserving aspect ratio
    """
    if not frames:
        return []
    
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required for autoencoder scaling")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if autoencoder is a Hugging Face VAE (AutoencoderKL)
    is_hf_vae = HF_AVAILABLE and hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode') and hasattr(autoencoder, 'latent_channels')
    
    # Check if autoencoder is a PyTorch model
    is_torch_model = TORCH_AVAILABLE and isinstance(autoencoder, nn.Module) and not is_hf_vae
    
    if is_torch_model or is_hf_vae:
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
    
    scaled_frames = []
    
    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            original_h, original_w = frame.shape[:2]
            original_aspect = original_w / original_h if original_h > 0 else 1.0
            
            # Proactive memory check every 10 frames
            if frame_idx > 0 and frame_idx % 10 == 0:
                try:
                    mem_stats = get_memory_stats()
                    gpu_allocated_gb = mem_stats.get("gpu_allocated_gb", 0)
                    if gpu_allocated_gb > 8:  # Warn if GPU memory > 8GB
                        logger.warning(f"High GPU memory usage at frame {frame_idx}: {gpu_allocated_gb:.2f}GB")
                        aggressive_gc(clear_cuda=True)
                except Exception:
                    pass
            
            try:
                if is_hf_vae:
                    # Hugging Face Stable Diffusion VAE
                    # VAE expects input in range [-1, 1] and shape (B, C, H, W)
                    # VAE has a downscale factor of 8, so we need to pad to multiples of 8
                    
                    # Get the dtype of the autoencoder model (float16 or float32)
                    model_dtype = next(autoencoder.parameters()).dtype
                    
                    # First, resize frame so that max(width, height) = target_size (preserving aspect ratio)
                    # This ensures the output will have max dimension = target_size
                    h, w = frame.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > target_size:
                        scale = target_size / max_dim
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        # Resize before encoding to reduce computation
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        original_h, original_w = new_h, new_w
                    
                    # Convert frame to tensor: (H, W, 3) -> (1, 3, H, W), normalize to [-1, 1]
                    # Match the model's dtype (float16 or float32)
                    frame_tensor = _frame_to_tensor_hf_vae(frame, device, dtype=model_dtype)
                    
                    # Pad to multiple of 8 (VAE requirement)
                    frame_tensor = _pad_to_multiple_of_8(frame_tensor)
                    
                    # Encode to latent space
                    with torch.no_grad():
                        latent = autoencoder.encode(frame_tensor).latent_dist.sample()
                    
                    # Decode back to pixel space
                    with torch.no_grad():
                        decoded = autoencoder.decode(latent).sample
                    
                    # Remove padding and convert back to numpy
                    decoded = _unpad_from_multiple_of_8(decoded, original_h, original_w)
                    scaled_frame = _tensor_to_frame_hf_vae(decoded)
                    
                    # Ensure max dimension is exactly target_size (may need slight adjustment after VAE)
                    h_out, w_out = scaled_frame.shape[:2]
                    max_dim_out = max(h_out, w_out)
                    if max_dim_out != target_size:
                        # Resize to ensure max dimension is exactly target_size
                        scale = target_size / max_dim_out
                        new_w = int(w_out * scale)
                        new_h = int(h_out * scale)
                        scaled_frame = cv2.resize(
                            scaled_frame, 
                            (new_w, new_h), 
                            interpolation=cv2.INTER_AREA
                        )
                    
                    # Verify max dimension is target_size
                    final_h, final_w = scaled_frame.shape[:2]
                    final_max = max(final_h, final_w)
                    if final_max != target_size:
                        logger.warning(
                            f"Frame max dimension is {final_max}, expected {target_size}. "
                            f"Original: {original_h}x{original_w}, Output: {final_h}x{final_w}"
                        )
                    
                elif is_torch_model:
                    # Standard PyTorch model
                    frame_tensor = _frame_to_tensor(frame, device)
                    output = autoencoder(frame_tensor)
                    scaled_frame = _tensor_to_frame(output, target_size)
                    
                elif hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode'):
                    # Custom encode-decode interface
                    frame_tensor = _frame_to_tensor(frame, device)
                    encoded = autoencoder.encode(frame_tensor)
                    output = autoencoder.decode(encoded)
                    scaled_frame = _tensor_to_frame(output, target_size)
                    
                elif hasattr(autoencoder, '__call__'):
                    # Callable interface
                    frame_tensor = _frame_to_tensor(frame, device)
                    output = autoencoder(frame_tensor)
                    scaled_frame = _tensor_to_frame(output, target_size)
                else:
                    raise ValueError(
                        "Autoencoder must be a PyTorch nn.Module, Hugging Face VAE, "
                        "have encode/decode methods, or be callable"
                    )
                
                scaled_frames.append(scaled_frame)
                
            except Exception as e:
                if check_oom_error(e):
                    handle_oom_error(e, f"autoencoder frame {frame_idx}")
                    logger.warning(
                        f"OOM during autoencoder scaling for frame {frame_idx} ({original_h}x{original_w}), "
                        f"falling back to letterbox"
                    )
                    # Fallback to letterbox resize
                    try:
                        scaled_frame = letterbox_resize(frame, target_size)
                        scaled_frames.append(scaled_frame)
                    except Exception as e2:
                        if check_oom_error(e2):
                            handle_oom_error(e2, f"letterbox fallback frame {frame_idx}")
                            logger.error(f"OOM even with letterbox fallback for frame {frame_idx}, skipping frame")
                            aggressive_gc(clear_cuda=True)
                            continue
                        raise
                else:
                    logger.warning(
                        f"Autoencoder scaling failed for frame ({original_h}x{original_w}), "
                        f"falling back to letterbox: {e}"
                    )
                    # Fallback to letterbox resize
                    scaled_frame = letterbox_resize(frame, target_size)
                    scaled_frames.append(scaled_frame)
    
    return scaled_frames


def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert numpy frame (H, W, 3) to PyTorch tensor (1, 3, H, W).
    
    Args:
        frame: Input frame as numpy array (H, W, 3), uint8 [0, 255] or float [0, 1]
        device: Target device (CPU or CUDA)
    
    Returns:
        Tensor of shape (1, 3, H, W), float32, normalized to [0, 1]
    """
    # Ensure frame is float32 and in [0, 1] range
    if frame.dtype == np.uint8:
        frame_float = frame.astype(np.float32) / 255.0
    else:
        frame_float = frame.astype(np.float32)
        if frame_float.max() > 1.0:
            frame_float = frame_float / 255.0
    
    # Convert (H, W, 3) to (3, H, W) then add batch dimension (1, 3, H, W)
    frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)
    return frame_tensor.to(device)


def _frame_to_tensor_hf_vae(frame: np.ndarray, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Convert numpy frame to tensor for Hugging Face VAE.
    VAE expects input in range [-1, 1] and shape (B, C, H, W).
    
    Args:
        frame: Input frame as numpy array (H, W, 3), uint8 [0, 255]
        device: Target device (CPU or CUDA)
        dtype: Target dtype (float16 or float32). If None, uses float32.
    
    Returns:
        Tensor of shape (1, 3, H, W), matching model dtype, normalized to [-1, 1]
    """
    # Ensure frame is float32 and normalize to [-1, 1]
    if frame.dtype == np.uint8:
        frame_float = frame.astype(np.float32) / 255.0
    else:
        frame_float = frame.astype(np.float32)
        if frame_float.max() > 1.0:
            frame_float = frame_float / 255.0
    
    # Normalize to [-1, 1]
    frame_float = frame_float * 2.0 - 1.0
    
    # Convert (H, W, 3) to (3, H, W) then add batch dimension (1, 3, H, W)
    frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)
    
    # Convert to target dtype if specified (to match model dtype)
    if dtype is not None:
        frame_tensor = frame_tensor.to(dtype=dtype)
    
    return frame_tensor.to(device)


def _tensor_to_frame_hf_vae(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert Hugging Face VAE output tensor to numpy frame.
    VAE outputs in range [-1, 1], convert back to [0, 255].
    
    Args:
        tensor: Input tensor, shape (B, C, H, W) or (C, H, W)
    
    Returns:
        Numpy array (H, W, 3), uint8 [0, 255]
    """
    # Move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert float16 to float32 for numpy compatibility (numpy doesn't support float16 well)
    if tensor.dtype == torch.float16:
        tensor = tensor.float()
    
    # Handle different tensor shapes
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]  # Take first batch item
    if tensor.dim() == 3 and tensor.shape[0] == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    
    # Convert to numpy
    frame = tensor.numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    frame = (frame + 1.0) / 2.0
    
    # Clip and convert to [0, 255]
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    
    return frame


def _pad_to_multiple_of_8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pad tensor to be multiple of 8 (required by Stable Diffusion VAE).
    
    Args:
        tensor: Input tensor (B, C, H, W)
    
    Returns:
        Padded tensor (B, C, H', W') where H' and W' are multiples of 8
    """
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h == 0 and pad_w == 0:
        return tensor
    
    # Pad with reflection to avoid black borders
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')


def _unpad_from_multiple_of_8(tensor: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
    """
    Remove padding to restore original dimensions.
    
    Args:
        tensor: Padded tensor (B, C, H', W')
        original_h: Original height
        original_w: Original width
    
    Returns:
        Unpadded tensor (B, C, H, W)
    """
    _, _, h, w = tensor.shape
    
    # Calculate how much was padded
    pad_h = h - original_h
    pad_w = w - original_w
    
    if pad_h == 0 and pad_w == 0:
        return tensor
    
    # Crop to original size
    return tensor[:, :, :original_h, :original_w]


def _tensor_to_frame(tensor: torch.Tensor, target_size: int) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy frame.
    
    Args:
        tensor: Input tensor, shape (B, C, H, W) or (C, H, W) or (H, W, C)
        target_size: Target size for output frame
    
    Returns:
        Numpy array (target_size, target_size, 3), uint8 [0, 255]
    """
    # Move to CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle different tensor shapes
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]  # Take first batch item
    if tensor.dim() == 3 and tensor.shape[0] == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    
    # Convert to numpy
    frame = tensor.numpy()
    
    # Ensure values are in [0, 1] range, then convert to [0, 255]
    frame = np.clip(frame, 0, 1)
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # Resize to target size if needed
    h, w = frame.shape[:2]
    if h != target_size or w != target_size:
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return frame


class _nullcontext:
    """Context manager that does nothing (for non-torch case)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def scale_video_frames(
    frames: list,
    method: str = "letterbox",
    target_size: int = 224,
    autoencoder: Optional[object] = None,
    preserve_aspect_ratio: bool = True
) -> list:
    """
    Scale a list of video frames to target max dimension.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size).
    
    Args:
        frames: List of frames (each is H, W, 3)
        method: Scaling method ("letterbox" or "autoencoder")
        target_size: For letterbox: square size. For autoencoder: max(width, height) = target_size (default: 256)
        autoencoder: Optional autoencoder model for autoencoder method
        preserve_aspect_ratio: If True, maintain original aspect ratio (default: True)
    
    Returns:
        List of scaled frames. For autoencoder: max(width, height) = target_size. For letterbox: (target_size, target_size)
    """
    if method == "letterbox":
        return [letterbox_resize(frame, target_size) for frame in frames]
    elif method == "autoencoder":
        if autoencoder is None:
            raise ValueError("Autoencoder model required for autoencoder method")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for autoencoder scaling")
        
        return _autoencoder_scale(frames, autoencoder, target_size, preserve_aspect_ratio)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

