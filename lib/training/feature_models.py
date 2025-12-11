"""
Feature-based PyTorch models using Stage 2/4 features.

All models are GPU-first with CPU fallback.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureMLP(nn.Module):
    """Multi-layer perceptron for features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.5,
        num_classes: int = 1
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FeatureCNN1D(nn.Module):
    """1D CNN for features (treats features as sequence)."""
    
    def __init__(
        self,
        input_dim: int,
        num_filters: list[int] = [64, 128, 256],
        kernel_sizes: list[int] = [3, 3, 3],
        dropout: float = 0.5,
        num_classes: int = 1
    ):
        """
        Initialize 1D CNN.
        
        Args:
            input_dim: Input feature dimension
            num_filters: List of filter numbers
            kernel_sizes: List of kernel sizes
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Reshape input: (batch, features) -> (batch, 1, features)
        self.input_dim = input_dim
        
        layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(num_filters, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Linear(num_filters[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class FeatureTransformer(nn.Module):
    """Transformer encoder for features."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        """
        Initialize Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and add positional encoding
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.classifier(x)
        return x


class FeatureLSTM(nn.Module):
    """LSTM for features (treats features as sequence)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 1
    ):
        """
        Initialize LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        output, (hidden, cell) = self.lstm(x)
        # Use last output
        x = output[:, -1, :]
        x = self.classifier(x)
        return x


class FeatureResNet(nn.Module):
    """ResNet-style architecture for features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 512, 256],
        dropout: float = 0.5,
        num_classes: int = 1
    ):
        """
        Initialize ResNet-style model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(self._make_residual_block(hidden_dims[i], hidden_dims[i+1], dropout))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def _make_residual_block(self, in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            # Skip connection (with projection if dimensions differ)
            if x.shape[1] != residual.shape[1]:
                residual = nn.Linear(residual.shape[1], x.shape[1]).to(x.device)(residual)
            x = F.relu(x + residual)
        
        x = self.classifier(x)
        return x


# Model factory functions
def create_feature_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a feature-based model.
    
    Args:
        model_type: Model type name
        input_dim: Input feature dimension
        **kwargs: Model-specific arguments
    
    Returns:
        PyTorch model
    """
    if model_type == "mlp" or model_type == "logistic_regression":
        hidden_dims = kwargs.get("hidden_dims", [256, 128, 64])
        dropout = kwargs.get("dropout", 0.5)
        return FeatureMLP(input_dim, hidden_dims, dropout)
    
    elif model_type == "cnn1d" or model_type == "naive_cnn":
        num_filters = kwargs.get("num_filters", [64, 128, 256])
        kernel_sizes = kwargs.get("kernel_sizes", [3, 3, 3])
        dropout = kwargs.get("dropout", 0.5)
        return FeatureCNN1D(input_dim, num_filters, kernel_sizes, dropout)
    
    elif model_type == "transformer" or model_type == "vit_transformer":
        d_model = kwargs.get("d_model", 256)
        nhead = kwargs.get("nhead", 8)
        num_layers = kwargs.get("num_layers", 4)
        dim_feedforward = kwargs.get("dim_feedforward", 1024)
        dropout = kwargs.get("dropout", 0.1)
        return FeatureTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout)
    
    elif model_type == "lstm" or model_type == "vit_gru":
        hidden_dim = kwargs.get("hidden_dim", 256)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.5)
        bidirectional = kwargs.get("bidirectional", True)
        return FeatureLSTM(input_dim, hidden_dim, num_layers, dropout, bidirectional)
    
    elif model_type == "resnet":
        hidden_dims = kwargs.get("hidden_dims", [256, 512, 256])
        dropout = kwargs.get("dropout", 0.5)
        return FeatureResNet(input_dim, hidden_dims, dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

