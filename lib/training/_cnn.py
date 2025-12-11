"""
Naive CNN baseline model that processes frames independently.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
            
            # Process each frame independently
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            
            # Validate flattened size
            if x.shape[1] != 128:
                raise ValueError(f"Unexpected feature size after pooling: {x.shape[1]}, expected 128")
            
            # Classification
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x)  # (N*T, num_classes)
            
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

