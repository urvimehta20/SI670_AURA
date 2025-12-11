"""
ViT-GRU model: Vision Transformer backbone with GRU temporal head.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. ViT models will not work.")


class ViTGRUModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) backbone with GRU temporal head.
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize ViT+GRU model.
        
        Args:
            num_frames: Number of frames to process
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
            pretrained: Use pretrained ViT weights
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT models. Install with: pip install timm")
        
        # ViT backbone (extract features, not classification head)
        # Specify img_size=256 to interpolate positional embeddings for 256x256 input
        self.vit_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # No global pooling
            img_size=256,  # Match scaled video dimensions (timm will interpolate pos embeddings)
        )
        
        # Get feature dimension from ViT
        # ViT-B/16 outputs [CLS] token of size 768
        feature_dim = self.vit_backbone.embed_dim  # 768 for ViT-B
        
        # GRU temporal head
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
        self.num_frames = num_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, T, C, H, W) or (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # Handle input format
        if x.dim() == 5:
            if x.shape[1] == 3:  # (N, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now x is (N, T, C, H, W)
        
        N, T, C, H, W = x.shape
        
        # Process each frame through ViT
        # Reshape to (N*T, C, H, W)
        x = x.view(N * T, C, H, W)
        
        # Extract features using ViT
        # ViT expects (N, C, H, W) and outputs (N, num_patches+1, embed_dim)
        # We use the [CLS] token (first token)
        vit_output = self.vit_backbone.forward_features(x)
        # vit_output shape: (N*T, num_patches+1, embed_dim)
        # Extract [CLS] token (first token)
        frame_features = vit_output[:, 0, :]  # (N*T, embed_dim)
        
        # Reshape back to (N, T, embed_dim)
        frame_features = frame_features.view(N, T, -1)
        
        # Process through GRU
        gru_out, _ = self.gru(frame_features)  # (N, T, hidden_dim)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # (N, hidden_dim)
        
        # Classification
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (N, 1)
        
        return logits


__all__ = ["ViTGRUModel"]

