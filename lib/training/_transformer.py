"""
ViT-Transformer model: Vision Transformer backbone with Transformer encoder temporal head.
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


class ViTTransformerModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) backbone with Transformer encoder temporal head.
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize ViT+Transformer model.
        
        Args:
            num_frames: Number of frames to process
            d_model: Transformer model dimension (should match ViT embed_dim)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            pretrained: Use pretrained ViT weights
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT models. Install with: pip install timm")
        
        # ViT backbone - specify img_size=256 to interpolate positional embeddings for 256x256 input
        self.vit_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=256,  # Match scaled video dimensions (timm will interpolate pos embeddings)
        )
        
        # Get feature dimension from ViT
        feature_dim = self.vit_backbone.embed_dim  # 768 for ViT-B
        
        # Ensure d_model matches feature_dim
        if d_model != feature_dim:
            logger.warning(f"d_model ({d_model}) != ViT embed_dim ({feature_dim}). Using {feature_dim}.")
            d_model = feature_dim
        
        # Transformer encoder temporal head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        
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
        x = x.view(N * T, C, H, W)
        
        # Extract features using ViT
        vit_output = self.vit_backbone.forward_features(x)
        # vit_output shape: (N*T, num_patches+1, embed_dim)
        # Extract [CLS] token
        frame_features = vit_output[:, 0, :]  # (N*T, embed_dim)
        
        # Reshape to (N, T, embed_dim)
        frame_features = frame_features.view(N, T, -1)
        
        # Process through Transformer encoder
        # Transformer expects (N, seq_len, d_model)
        transformer_out = self.transformer(frame_features)  # (N, T, d_model)
        
        # Use mean pooling over temporal dimension
        pooled = transformer_out.mean(dim=1)  # (N, d_model)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # (N, 1)
        
        return logits


__all__ = ["ViTTransformerModel"]

