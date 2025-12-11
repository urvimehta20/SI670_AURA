"""
ViViT (Video Vision Transformer) model.
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
    logger.warning("timm not available. ViViT will not work.")


class ViViTModel(nn.Module):
    """
    ViViT (Video Vision Transformer) model.
    
    Uses tubelet embedding (3D patches) and standard transformer encoder.
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        img_size: int = 256,  # Match scaled video dimensions
        tubelet_size: tuple = (2, 16, 16),  # (temporal, spatial, spatial)
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        pretrained: bool = True
    ):
        """
        Initialize ViViT model.
        
        Args:
            num_frames: Number of frames to process
            img_size: Input image size
            tubelet_size: Size of 3D patches (temporal, height, width)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Use bias in QKV projection
            dropout: Dropout probability
            attn_drop: Attention dropout probability
            pretrained: Use pretrained weights (if available)
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViViT. Install with: pip install timm")
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        
        # Calculate number of tubelets
        t_patches = num_frames // tubelet_size[0]
        h_patches = img_size // tubelet_size[1]
        w_patches = img_size // tubelet_size[2]
        self.num_tubelets = t_patches * h_patches * w_patches
        
        # Tubelet embedding: 3D convolution
        self.tubelet_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tubelets + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder
        # Use ViT's transformer blocks as reference
        # Note: We use ViT blocks but with our own tubelet embedding, so img_size affects
        # our tubelet calculation, not the ViT blocks directly
        vit_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=256,  # Match scaled video dimensions (for consistency)
        )
        
        # Extract transformer blocks (these are size-agnostic)
        self.blocks = vit_backbone.blocks
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # Handle input format
        if x.dim() == 5:
            if x.shape[1] == 3:  # (N, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now x is (N, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W)
        
        N, C, T, H, W = x.shape
        
        # Tubelet embedding: (N, C, T, H, W) -> (N, embed_dim, T', H', W')
        x = self.tubelet_embed(x)  # (N, embed_dim, t_patches, h_patches, w_patches)
        
        # Flatten spatial and temporal dimensions
        N, D, T_p, H_p, W_p = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, T_p, H_p, W_p, D)
        x = x.view(N, T_p * H_p * W_p, D)  # (N, num_tubelets, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(N, -1, -1)  # (N, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (N, num_tubelets+1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Extract CLS token
        cls_token = x[:, 0]  # (N, embed_dim)
        
        # Classification
        logits = self.head(cls_token)  # (N, 1)
        
        return logits


__all__ = ["ViViTModel"]

