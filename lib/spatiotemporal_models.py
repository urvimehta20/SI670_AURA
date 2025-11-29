"""
Spatiotemporal models:
- SlowFast network
- X3D network
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SlowFastModel(nn.Module):
    """
    SlowFast network for video recognition.
    
    Implements a simplified SlowFast architecture:
    - Slow pathway: processes frames at low temporal rate (2 fps)
    - Fast pathway: processes frames at high temporal rate (8 fps)
    - Fusion: combines features from both pathways
    """
    
    def __init__(
        self,
        slow_frames: int = 16,
        fast_frames: int = 64,
        alpha: int = 8,  # Temporal ratio between fast and slow
        beta: float = 1.0 / 8,  # Channel ratio between fast and slow
        pretrained: bool = True
    ):
        """
        Initialize SlowFast model.
        
        Args:
            slow_frames: Number of frames for slow pathway
            fast_frames: Number of frames for fast pathway
            alpha: Temporal ratio (fast_fps / slow_fps)
            beta: Channel ratio (fast_channels / slow_channels)
            pretrained: Use pretrained weights if available
        """
        super().__init__()
        
        # Try to use torchvision's SlowFast if available
        try:
            from torchvision.models.video import slowfast_r50, SlowFast_R50_Weights
            if pretrained:
                try:
                    weights = SlowFast_R50_Weights.KINETICS400_V1
                    self.backbone = slowfast_r50(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = slowfast_r50(pretrained=True)
            else:
                self.backbone = slowfast_r50(pretrained=False)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            self.use_r3d_fallback = False
            
        except (ImportError, AttributeError):
            # Fallback: use r3d_18 as pretrained backbone (similar to X3D)
            logger.warning("torchvision SlowFast not available. Using r3d_18 as pretrained backbone.")
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                if pretrained:
                    try:
                        weights = R3D_18_Weights.KINETICS400_V1
                        self.backbone = r3d_18(weights=weights)
                    except (AttributeError, ValueError):
                        self.backbone = r3d_18(pretrained=True)
                else:
                    self.backbone = r3d_18(pretrained=False)
                
                # Replace classification head for binary classification
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_torchvision = True  # Use backbone directly in forward
                self.use_r3d_fallback = True  # Mark that we're using r3d_18
            except (ImportError, AttributeError):
                # Final fallback: simplified SlowFast without pretrained weights
                logger.warning("r3d_18 also not available. Using simplified SlowFast (no pretrained weights).")
                self.use_torchvision = False
                self.use_r3d_fallback = False
                self._build_simplified_slowfast(slow_frames, fast_frames, alpha, beta)
        
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.alpha = alpha
    
    def _build_simplified_slowfast(
        self,
        slow_frames: int,
        fast_frames: int,
        alpha: int,
        beta: float
    ):
        """Build simplified SlowFast architecture."""
        # Slow pathway: 3D ResNet-like
        slow_channels = 64
        self.slow_pathway = nn.Sequential(
            # Stem
            nn.Conv3d(3, slow_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(slow_channels, slow_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(inplace=True),
            
            # Res blocks (simplified)
            self._make_res_block(slow_channels, slow_channels * 2, stride=2),
            self._make_res_block(slow_channels * 2, slow_channels * 4, stride=2),
            self._make_res_block(slow_channels * 4, slow_channels * 8, stride=2),
        )
        
        # Fast pathway: fewer channels, more frames
        fast_channels = int(slow_channels * beta)
        self.fast_pathway = nn.Sequential(
            # Stem
            nn.Conv3d(3, fast_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(fast_channels, fast_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(fast_channels),
            nn.ReLU(inplace=True),
            
            # Res blocks
            self._make_res_block(fast_channels, fast_channels * 2, stride=2),
            self._make_res_block(fast_channels * 2, fast_channels * 4, stride=2),
            self._make_res_block(fast_channels * 4, fast_channels * 8, stride=2),
        )
        
        # Lateral connections (simplified: just concatenate)
        # In real SlowFast, there are lateral connections between pathways
        
        # Fusion and classification
        fusion_dim = slow_channels * 8 + fast_channels * 8
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def _make_res_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Make a simplified 3D ResNet block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        if self.use_torchvision:
            # Use torchvision's SlowFast
            return self.backbone(x)
        
        # Simplified SlowFast
        N, C, T, H, W = x.shape
        
        # Sample frames for slow and fast pathways
        # Slow: sample every alpha frames
        slow_indices = torch.arange(0, T, self.alpha, device=x.device)
        slow_x = x[:, :, slow_indices, :, :]  # (N, C, T_slow, H, W)
        
        # Fast: use all frames (or sample at higher rate)
        fast_x = x  # (N, C, T, H, W)
        
        # Process through pathways
        slow_features = self.slow_pathway(slow_x)  # (N, C_slow, T', H', W')
        fast_features = self.fast_pathway(fast_x)  # (N, C_fast, T'', H'', W'')
        
        # Temporal alignment (simplified: just pool)
        slow_features = F.adaptive_avg_pool3d(slow_features, (1, 1, 1))  # (N, C_slow, 1, 1, 1)
        fast_features = F.adaptive_avg_pool3d(fast_features, (1, 1, 1))  # (N, C_fast, 1, 1, 1)
        
        # Concatenate
        combined = torch.cat([slow_features, fast_features], dim=1)  # (N, C_slow+C_fast, 1, 1, 1)
        
        # Classification
        logits = self.fusion(combined)  # (N, 1)
        
        return logits


class X3DModel(nn.Module):
    """
    X3D (Expanding Architectures for Efficient Video Recognition) model.
    """
    
    def __init__(
        self,
        variant: str = "x3d_m",  # "x3d_s", "x3d_m", "x3d_l", "x3d_xl"
        pretrained: bool = True
    ):
        """
        Initialize X3D model.
        
        Args:
            variant: X3D variant (x3d_s, x3d_m, x3d_l, x3d_xl)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Try to use torchvision's X3D if available
        try:
            from torchvision.models.video import x3d_m, X3D_M_Weights
            
            if variant == "x3d_m":
                if pretrained:
                    try:
                        weights = X3D_M_Weights.KINETICS400_V1
                        self.backbone = x3d_m(weights=weights)
                    except (AttributeError, ValueError):
                        self.backbone = x3d_m(pretrained=True)
                else:
                    self.backbone = x3d_m(pretrained=False)
            else:
                # For other variants, try to load or use x3d_m as fallback
                logger.warning(f"X3D variant {variant} not available. Using x3d_m.")
                self.backbone = x3d_m(pretrained=pretrained)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            
        except (ImportError, AttributeError):
            # Fallback: use r3d_18 as approximation
            logger.warning("torchvision X3D not available. Using r3d_18 as approximation.")
            from torchvision.models.video import r3d_18, R3D_18_Weights
            try:
                weights = R3D_18_Weights.KINETICS400_V1
                self.backbone = r3d_18(weights=weights)
            except (AttributeError, ValueError):
                self.backbone = r3d_18(pretrained=pretrained)
            
            # Replace classification head
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = False
        
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        return self.backbone(x)


__all__ = [
    "SlowFastModel",
    "X3DModel",
]

