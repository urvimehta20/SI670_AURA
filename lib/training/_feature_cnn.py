"""
Feature-based 1D Convolutional Neural Network models.

GPU-first with CPU fallback, designed to work with extracted features from Stage 2/4.
"""

from __future__ import annotations

import logging
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureCNN1D(nn.Module):
    """
    1D CNN for feature-based classification.
    
    Treats features as a sequence and applies 1D convolutions.
    Supports GPU-first with automatic CPU fallback.
    """
    
    def __init__(
        self,
        input_dim: int,
        conv_channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3],
        num_classes: int = 2,
        dropout: float = 0.5,
        pool_type: str = "max",
        batch_norm: bool = True
    ):
        """
        Initialize 1D CNN model.
        
        Args:
            input_dim: Number of input features
            conv_channels: List of convolutional channel sizes
            kernel_sizes: List of kernel sizes (must match conv_channels length)
            num_classes: Number of output classes
            dropout: Dropout probability
            pool_type: Pooling type ("max", "avg", "adaptive")
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if len(kernel_sizes) != len(conv_channels):
            raise ValueError("kernel_sizes must match conv_channels length")
        
        # Build convolutional layers
        conv_layers = []
        prev_channels = 1  # Treat features as 1-channel sequence
        seq_length = input_dim
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.append(nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2))
            if batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            
            if pool_type == "max":
                conv_layers.append(nn.MaxPool1d(2))
            elif pool_type == "avg":
                conv_layers.append(nn.AvgPool1d(2))
            # adaptive doesn't pool here
            
            prev_channels = out_channels
            seq_length = seq_length // 2 if pool_type != "adaptive" else seq_length
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global pooling
        if pool_type == "adaptive":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.global_pool = None
        
        # Calculate flattened size after conv layers
        # Approximate: after pooling, sequence length reduces
        with torch.no_grad():
            test_input = torch.zeros(1, 1, input_dim)
            test_output = self.conv_layers(test_input)
            if self.global_pool:
                test_output = self.global_pool(test_output)
            flattened_size = test_output.numel()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Reshape to (batch_size, 1, input_dim) for 1D conv
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global pooling if needed
        if self.global_pool:
            x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def to_device(self, device: Optional[torch.device] = None) -> 'FeatureCNN1D':
        """
        Move model to device with GPU-first, CPU fallback.
        
        Args:
            device: Target device (None = auto-detect)
        
        Returns:
            Self for chaining
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using GPU for FeatureCNN1D")
            else:
                device = torch.device("cpu")
                logger.info("GPU not available, using CPU for FeatureCNN1D")
        
        try:
            self.to(device)
            # Test if model works on device
            test_input = torch.zeros(1, self.input_dim, device=device)
            _ = self.forward(test_input)
            logger.debug(f"Successfully moved FeatureCNN1D to {device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU OOM, falling back to CPU: {e}")
                device = torch.device("cpu")
                self.to(device)
            else:
                raise
        
        return self

