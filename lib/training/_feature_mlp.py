"""
Feature-based Multi-Layer Perceptron (MLP) models.

GPU-first with CPU fallback, designed to work with extracted features from Stage 2/4.
"""

from __future__ import annotations

import logging
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureMLP(nn.Module):
    """
    Multi-Layer Perceptron for feature-based classification.
    
    Supports GPU-first with automatic CPU fallback.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "tanh")
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def to_device(self, device: Optional[torch.device] = None) -> 'FeatureMLP':
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
                logger.info("Using GPU for FeatureMLP")
            else:
                device = torch.device("cpu")
                logger.info("GPU not available, using CPU for FeatureMLP")
        
        try:
            self.to(device)
            # Test if model works on device
            test_input = torch.zeros(1, self.input_dim, device=device)
            _ = self.forward(test_input)
            logger.debug(f"Successfully moved FeatureMLP to {device}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU OOM, falling back to CPU: {e}")
                device = torch.device("cpu")
                self.to(device)
            else:
                raise
        
        return self

