"""
Naive CNN baseline model that processes frames independently.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Import aggressive GC for memory management
try:
    from lib.utils.memory import aggressive_gc
except ImportError:
    # Fallback if not available
    def aggressive_gc(clear_cuda: bool = False):
        import gc
        gc.collect()
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


class NaiveCNNBaseline(nn.Module):
    """
    Naive CNN baseline that processes frames independently and averages predictions.
    """
    
    def __init__(self, num_frames: int = 1000, num_classes: int = 2):
        """
        Initialize CNN model.
        
        Args:
            num_frames: Number of frames to process
            num_classes: Number of classes (2 for binary)
        """
        super().__init__()
        
        # Simple 2D CNN for per-frame processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.num_frames = num_frames
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, num_classes)
        """
        # Input validation
        if x is None or x.numel() == 0:
            raise ValueError("Input tensor is empty or None")
        
        if x.dim() not in [4, 5]:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D tensor with shape {x.shape}")
        
        # Handle different input formats with validation
        original_shape = x.shape
        try:
            if x.dim() == 5:
                if x.shape[1] == 3:  # (N, C, T, H, W)
                    # Rearrange to (N*T, C, H, W) for per-frame processing
                    N, C, T, H, W = x.shape
                    if C != 3:
                        raise ValueError(f"Expected 3 channels, got {C} channels")
                    x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                    x = x.view(N * T, C, H, W)
                else:  # (N, T, C, H, W)
                    N, T, C, H, W = x.shape
                    if C != 3:
                        raise ValueError(f"Expected 3 channels, got {C} channels")
                    x = x.view(N * T, C, H, W)
            else:
                # Already in (N*T, C, H, W) format
                if x.shape[1] != 3:
                    raise ValueError(f"Expected 3 channels, got {x.shape[1]} channels")
            
            # Validate spatial dimensions
            if x.shape[2] < 8 or x.shape[3] < 8:
                raise ValueError(f"Input spatial dimensions too small: {x.shape[2]}x{x.shape[3]}, minimum 8x8 required")
            
            # Process frames in chunks to avoid OOM when processing many frames (e.g., 1000)
            # Chunk size: process up to 100 frames at a time to limit memory usage
            chunk_size = 100
            total_frames = x.size(0)
            num_chunks = (total_frames + chunk_size - 1) // chunk_size
            
            all_logits = []
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_frames)
                chunk = x[start_idx:end_idx]
                
                # Process chunk through CNN
                chunk = F.relu(self.bn1(self.conv1(chunk)))
                chunk = F.max_pool2d(chunk, 2)
                chunk = F.relu(self.bn2(self.conv2(chunk)))
                chunk = F.max_pool2d(chunk, 2)
                chunk = F.relu(self.bn3(self.conv3(chunk)))
                chunk = self.pool(chunk)
                chunk = chunk.view(chunk.size(0), -1)  # Flatten
                
                # Validate flattened size
                if chunk.shape[1] != 128:
                    raise ValueError(f"Unexpected feature size after pooling: {chunk.shape[1]}, expected 128")
                
                # Classification
                chunk = F.relu(self.fc1(chunk))
                chunk = self.dropout(chunk)
                chunk_logits = self.fc2(chunk)  # (chunk_size, num_classes)
                
                all_logits.append(chunk_logits)
                
                # Aggressive GC after each chunk to free memory (except last chunk)
                if torch.cuda.is_available() and chunk_idx < num_chunks - 1:  # Don't clear on last chunk
                    del chunk, chunk_logits
                    torch.cuda.empty_cache()
                    aggressive_gc(clear_cuda=True)
            
            # Concatenate all chunk logits (all should already be on same device as input)
            logits = torch.cat(all_logits, dim=0)  # (N*T, num_classes)
            
            # Clean up intermediate tensors
            del all_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                aggressive_gc(clear_cuda=True)
            
            # Reshape back to (N, T, num_classes) and average over frames
            # Fix: check logits.dim() instead of x.dim() (x is already flattened)
            if logits.dim() == 2 and logits.shape[0] % self.num_frames == 0:
                N = logits.shape[0] // self.num_frames
                logits = logits.view(N, self.num_frames, -1)
                logits = logits.mean(dim=1)  # Average over frames
            elif logits.dim() == 2:
                # If we can't reshape properly, just return the logits as-is
                # This handles edge cases where num_frames doesn't match
                logger.warning(
                    f"Cannot reshape logits: shape {logits.shape}, num_frames {self.num_frames}. "
                    f"Returning logits without temporal averaging."
                )
            
            return logits
            
        except RuntimeError as e:
            logger.error(
                f"Runtime error in forward pass: {e}. "
                f"Input shape: {original_shape}, Model num_frames: {self.num_frames}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in forward pass: {e}. "
                f"Input shape: {original_shape}, Model num_frames: {self.num_frames}"
            )
            raise


__all__ = ["NaiveCNNBaseline"]

