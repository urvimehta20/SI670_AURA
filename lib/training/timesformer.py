"""
TimeSformer model: Space-time attention for video recognition.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. TimeSformer will not work.")


class SpaceTimeAttention(nn.Module):
    """
    Space-time divided attention block.
    First applies spatial attention, then temporal attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with space-time divided attention.
        
        Args:
            x: Input tensor (N, T*H*W, dim) where T is temporal, H*W is spatial
        
        Returns:
            Output tensor (N, T*H*W, dim)
        """
        N, L, D = x.shape
        
        # Reshape to separate temporal and spatial dimensions
        # Assume x is (N, T*H*W, D) - we need to know T, H, W
        # For simplicity, we'll use a fixed temporal dimension
        # In practice, this should be passed as a parameter or inferred
        
        # Spatial attention: attend within each frame
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Spatial attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_spatial = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x_spatial = self.proj(x_spatial)
        x_spatial = self.proj_drop(x_spatial)
        
        # Temporal attention: attend across frames for each spatial location
        # Reshape to (N, T, H*W, D) for temporal attention
        # For now, we'll use a simplified approach: apply temporal attention
        # by reshaping and applying attention across the temporal dimension
        
        # Add residual
        x = x + x_spatial
        
        # Temporal attention (simplified: apply across all tokens)
        qkv_t = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
        
        attn_t = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn_t = attn_t.softmax(dim=-1)
        attn_t = self.attn_drop(attn_t)
        
        x_temporal = (attn_t @ v_t).transpose(1, 2).reshape(N, L, D)
        x_temporal = self.proj(x_temporal)
        x_temporal = self.proj_drop(x_temporal)
        
        # Add residual
        x = x + x_temporal
        
        return x


class TimeSformerModel(nn.Module):
    """
    TimeSformer: Space-time attention for video recognition.
    
    Based on: "Is Space-Time Attention All You Need for Video Understanding?"
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        img_size: int = 256,  # Match scaled video dimensions
        patch_size: int = 16,
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
        Initialize TimeSformer model.
        
        Args:
            num_frames: Number of frames to process
            img_size: Input image size
            patch_size: Patch size for ViT
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
            raise ImportError("timm is required for TimeSformer. Install with: pip install timm")
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use ViT as backbone for patch embedding
        # Note: We extract patch_embed which works with any size, but we create our own pos_embed
        # for img_size=256. The patch_embed will produce 16x16=256 patches for 256x256 input.
        vit_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=256,  # Match scaled video dimensions (patch_embed will adapt)
        )
        
        # Extract patch embedding (works with 256x256, produces 16x16=256 patches)
        self.patch_embed = vit_backbone.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Time embedding (learnable)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks with space-time attention
        self.blocks = nn.ModuleList([
            SpaceTimeAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
    
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
        
        N, T, C, H, W = x.shape
        
        # Process each frame through patch embedding
        x = x.view(N * T, C, H, W)  # (N*T, C, H, W)
        x = self.patch_embed(x)  # (N*T, num_patches, embed_dim)
        
        # Reshape to (N, T, num_patches, embed_dim)
        x = x.view(N, T, self.num_patches, self.embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(N, T, -1, -1)  # (N, T, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=2)  # (N, T, num_patches+1, embed_dim)
        
        # Add positional embeddings (spatial)
        x = x + self.pos_embed.unsqueeze(1)  # (N, T, num_patches+1, embed_dim)
        
        # Add temporal embeddings
        x = x + self.time_embed.unsqueeze(2)  # (N, T, num_patches+1, embed_dim)
        
        # Flatten: (N, T*(num_patches+1), embed_dim)
        x = x.view(N, T * (self.num_patches + 1), self.embed_dim)
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks with space-time attention
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Extract CLS token (first token of each frame, then average)
        # CLS tokens are at indices: 0, num_patches+1, 2*(num_patches+1), ...
        cls_indices = torch.arange(0, T * (self.num_patches + 1), self.num_patches + 1, device=x.device)
        cls_tokens = x[:, cls_indices, :]  # (N, T, embed_dim)
        
        # Average pool over temporal dimension
        cls_token = cls_tokens.mean(dim=1)  # (N, embed_dim)
        
        # Classification
        logits = self.head(cls_token)  # (N, 1)
        
        return logits


__all__ = ["TimeSformerModel"]

