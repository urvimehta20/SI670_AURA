"""
Two-Stream Networks: RGB stream + Optical flow stream for video recognition.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available. Two-Stream will use ResNet backbone.")


class TwoStreamModel(nn.Module):
    """
    Two-Stream Network: RGB stream + Optical flow stream.
    
    Architecture:
    - RGB stream: Processes RGB frames
    - Optical flow stream: Processes optical flow frames
    - Fusion: Combines features from both streams
    """
    
    def __init__(
        self,
        num_frames: int = 1000,
        rgb_backbone: str = "resnet18",  # "resnet18", "resnet50", "vit"
        flow_backbone: str = "resnet18",
        fusion_method: str = "concat",  # "concat", "weighted", "attention"
        pretrained: bool = True
    ):
        """
        Initialize Two-Stream model.
        
        Args:
            num_frames: Number of frames to process
            rgb_backbone: Backbone for RGB stream
            flow_backbone: Backbone for optical flow stream
            fusion_method: How to fuse RGB and flow features
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        
        # RGB stream backbone
        if rgb_backbone == "vit" and TIMM_AVAILABLE:
            self.rgb_backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0,
                global_pool='cls',
                img_size=256,  # Match scaled video dimensions (timm will interpolate pos embeddings)
            )
            rgb_feature_dim = 768
        else:
            # Use ResNet
            try:
                import torchvision.models as models
                if rgb_backbone == "resnet18":
                    resnet = models.resnet18(pretrained=pretrained)
                elif rgb_backbone == "resnet50":
                    resnet = models.resnet50(pretrained=pretrained)
                else:
                    logger.warning(f"Unknown RGB backbone {rgb_backbone}, using resnet18")
                    resnet = models.resnet18(pretrained=pretrained)
                
                # Remove final FC layer
                self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
                rgb_feature_dim = resnet.fc.in_features
            except ImportError:
                raise ImportError("torchvision is required for ResNet backbones")
        
        # Optical flow stream backbone (same architecture as RGB)
        if flow_backbone == "vit" and TIMM_AVAILABLE:
            self.flow_backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                img_size=256,  # Match scaled video dimensions (timm will interpolate pos embeddings)
                num_classes=0,
                global_pool='cls'
            )
            flow_feature_dim = 768
        else:
            try:
                import torchvision.models as models
                if flow_backbone == "resnet18":
                    resnet = models.resnet18(pretrained=pretrained)
                elif flow_backbone == "resnet50":
                    resnet = models.resnet50(pretrained=pretrained)
                else:
                    logger.warning(f"Unknown flow backbone {flow_backbone}, using resnet18")
                    resnet = models.resnet18(pretrained=pretrained)
                
                # Remove final FC layer
                self.flow_backbone = nn.Sequential(*list(resnet.children())[:-1])
                flow_feature_dim = resnet.fc.in_features
            except ImportError:
                raise ImportError("torchvision is required for ResNet backbones")
        
        # Fusion layer
        if fusion_method == "concat":
            fusion_dim = rgb_feature_dim + flow_feature_dim
            self.fusion = nn.Identity()  # Just concatenate
        elif fusion_method == "weighted":
            fusion_dim = rgb_feature_dim  # Assume same dimension
            self.rgb_weight = nn.Parameter(torch.tensor(0.5))
            self.flow_weight = nn.Parameter(torch.tensor(0.5))
            self.fusion = nn.Identity()
        elif fusion_method == "attention":
            fusion_dim = rgb_feature_dim + flow_feature_dim
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=rgb_feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(rgb_feature_dim + flow_feature_dim, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Temporal aggregation (average pool over frames)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(fusion_dim, 1)
    
    def _compute_optical_flow(self, rgb_frames: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow from RGB frames on-the-fly.
        
        Args:
            rgb_frames: RGB frames (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Optical flow frames (N, C, T, H, W) - converted to RGB visualization
        """
        # Handle input format
        if rgb_frames.dim() == 5:
            if rgb_frames.shape[1] == 3:  # (N, C, T, H, W)
                rgb_frames = rgb_frames.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now rgb_frames is (N, T, C, H, W)
            N, T, C, H, W = rgb_frames.shape
            rgb_frames = rgb_frames.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W)
        
        N, C, T, H, W = rgb_frames.shape
        
        # Convert to numpy for OpenCV processing
        rgb_np = rgb_frames.permute(0, 2, 3, 4, 1).cpu().numpy()  # (N, T, H, W, C)
        rgb_np = (rgb_np * 255).astype(np.uint8)  # Denormalize if needed
        
        flow_frames_list = []
        
        for n in range(N):
            video_flows = []
            for t in range(T - 1):
                frame1 = rgb_np[n, t]  # (H, W, C)
                frame2 = rgb_np[n, t + 1]  # (H, W, C)
                
                # Compute optical flow
                try:
                    from lib.utils.optical_flow import extract_optical_flow, flow_to_rgb
                    flow = extract_optical_flow(frame1, frame2, method="farneback")
                    flow_rgb = flow_to_rgb(flow)  # (H, W, 3)
                    video_flows.append(flow_rgb)
                except Exception as e:
                    logger.warning(f"Failed to compute optical flow: {e}. Using zero flow.")
                    # Fallback: zero flow
                    flow_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                    video_flows.append(flow_rgb)
            
            # For last frame, duplicate previous flow
            if len(video_flows) > 0:
                video_flows.append(video_flows[-1])
            else:
                # If only one frame, use zero flow
                video_flows.append(np.zeros((H, W, 3), dtype=np.uint8))
            
            # Stack flows: (T, H, W, 3)
            video_flows = np.stack(video_flows, axis=0)
            flow_frames_list.append(video_flows)
        
        # Convert back to tensor: (N, T, H, W, 3) -> (N, C, T, H, W)
        flow_frames = np.stack(flow_frames_list, axis=0)  # (N, T, H, W, 3)
        flow_frames = torch.from_numpy(flow_frames).float()  # (N, T, H, W, 3)
        flow_frames = flow_frames / 255.0  # Normalize to [0, 1]
        flow_frames = flow_frames.permute(0, 4, 1, 2, 3).contiguous()  # (N, C, T, H, W)
        
        # Move to same device as rgb_frames
        flow_frames = flow_frames.to(rgb_frames.device)
        
        return flow_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: RGB frames (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # x contains RGB frames - compute optical flow on-the-fly
        rgb_frames = x
        flow_frames = self._compute_optical_flow(rgb_frames)
        
        # Handle input format for RGB
        if rgb_frames.dim() == 5:
            if rgb_frames.shape[1] == 3:  # (N, C, T, H, W)
                rgb_frames = rgb_frames.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now rgb_frames is (N, T, C, H, W)
            N, T, C, H, W = rgb_frames.shape
            rgb_frames = rgb_frames.view(N * T, C, H, W)  # (N*T, C, H, W)
        
        # flow_frames is already (N, C, T, H, W) from _compute_optical_flow
        # Convert to (N, T, C, H, W) for consistency
        if flow_frames.dim() == 5:
            if flow_frames.shape[1] == 3:  # (N, C, T, H, W)
                flow_frames = flow_frames.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
            # Now flow_frames is (N, T, C, H, W)
            N, T, C, H, W = flow_frames.shape
            flow_frames = flow_frames.view(N * T, C, H, W)  # (N*T, C, H, W)
        
        # Process RGB frames
        rgb_features = self.rgb_backbone(rgb_frames)  # (N*T, feature_dim)
        
        # Process optical flow frames
        flow_features = self.flow_backbone(flow_frames)  # (N*T, feature_dim)
        
        # Reshape to (N, T, feature_dim)
        rgb_features = rgb_features.view(N, T, -1)
        flow_features = flow_features.view(N, T, -1)
        
        # Fuse features
        if self.fusion_method == "concat":
            # Concatenate along feature dimension
            fused = torch.cat([rgb_features, flow_features], dim=2)  # (N, T, rgb_dim + flow_dim)
        elif self.fusion_method == "weighted":
            # Weighted combination
            fused = self.rgb_weight * rgb_features + self.flow_weight * flow_features
        elif self.fusion_method == "attention":
            # Attention-based fusion
            # Use RGB as query, flow as key/value
            attn_out, _ = self.attention(rgb_features, flow_features, flow_features)
            # Concatenate RGB and attended flow
            fused = torch.cat([rgb_features, attn_out], dim=2)
            fused = self.fusion_proj(fused)
        
        # Temporal aggregation: average pool over time
        fused = fused.transpose(1, 2)  # (N, feature_dim, T)
        fused = self.temporal_pool(fused)  # (N, feature_dim, 1)
        fused = fused.squeeze(-1)  # (N, feature_dim)
        
        # Classification
        fused = self.dropout(fused)
        logits = self.fc(fused)  # (N, 1)
        
        return logits


__all__ = ["TwoStreamModel"]

