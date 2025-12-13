"""
X3D model: Expanding Architectures for Efficient Video Recognition.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        
        # Try PyTorchVideo first (recommended method for X3D)
        backbone_loaded = False
        try:
            # Map variant names to PyTorchVideo model names
            pytorchvideo_model_map = {
                "x3d_xs": "x3d_xs",
                "x3d_s": "x3d_s",
                "x3d_m": "x3d_m",
                "x3d_l": "x3d_l",
            }
            
            pytorchvideo_model_name = pytorchvideo_model_map.get(variant, "x3d_m")
            
            if pretrained:
                logger.info(f"Loading X3D model from PyTorchVideo: {pytorchvideo_model_name} (pretrained=True)")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    pytorchvideo_model_name,
                    pretrained=True
                )
            else:
                logger.info(f"Loading X3D model from PyTorchVideo: {pytorchvideo_model_name} (pretrained=False)")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    pytorchvideo_model_name,
                    pretrained=False
                )
            
            # Replace classification head for binary classification
            # PyTorchVideo X3D models typically have a head with a projection layer
            # Try multiple strategies to find and replace the final classification layer
            head_replaced = False
            
            # Strategy 1: Direct head.proj access (most common for PyTorchVideo X3D)
            if hasattr(self.backbone, 'head'):
                if hasattr(self.backbone.head, 'proj') and isinstance(self.backbone.head.proj, nn.Linear):
                    in_features = self.backbone.head.proj.in_features
                    self.backbone.head.proj = nn.Linear(in_features, 1)
                    head_replaced = True
                elif isinstance(self.backbone.head, nn.Linear):
                    # Head is directly a Linear layer
                    in_features = self.backbone.head.in_features
                    self.backbone.head = nn.Linear(in_features, 1)
                    head_replaced = True
            
            # Strategy 2: Find the last Linear layer in the model
            if not head_replaced:
                last_linear_name = None
                last_linear_in_features = None
                # First pass: find the last Linear layer
                for name, module in self.backbone.named_modules():
                    if isinstance(module, nn.Linear):
                        last_linear_name = name
                        last_linear_in_features = module.in_features
                
                # Second pass: replace it
                if last_linear_name is not None:
                    parts = last_linear_name.split('.')
                    if len(parts) > 1:
                        parent = self.backbone
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], nn.Linear(last_linear_in_features, 1))
                    else:
                        setattr(self.backbone, last_linear_name, nn.Linear(last_linear_in_features, 1))
                    head_replaced = True
                    logger.debug(f"Replaced classification head at: {last_linear_name}")
            
            if not head_replaced:
                logger.warning("Could not find classification head in PyTorchVideo X3D model. Model may not work correctly.")
            
            self.use_pytorchvideo = True
            backbone_loaded = True
            logger.info(f"✓ Successfully loaded X3D model from PyTorchVideo: {pytorchvideo_model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load X3D from PyTorchVideo: {e}. Trying torchvision...")
            backbone_loaded = False
        
        # Fallback to torchvision X3D if PyTorchVideo failed
        if not backbone_loaded:
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
                    logger.warning(f"X3D variant {variant} not available in torchvision. Using x3d_m.")
                    self.backbone = x3d_m(pretrained=pretrained)
                
                # Replace classification head for binary classification
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_pytorchvideo = False
                self.use_torchvision = True
                backbone_loaded = True
                logger.info("✓ Successfully loaded X3D model from torchvision")
                
            except (ImportError, AttributeError) as e:
                logger.warning(f"torchvision X3D not available: {e}. Using r3d_18 as approximation.")
                backbone_loaded = False
        
        # Final fallback: use r3d_18 as approximation
        if not backbone_loaded:
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                try:
                    weights = R3D_18_Weights.KINETICS400_V1
                    self.backbone = r3d_18(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = r3d_18(pretrained=pretrained)
                
                # Replace classification head
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_pytorchvideo = False
                self.use_torchvision = False
                logger.warning("⚠ Using r3d_18 as X3D approximation (X3D not available)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load X3D model or fallback. "
                    f"Tried PyTorchVideo, torchvision, and r3d_18. Last error: {e}. "
                    f"Please install pytorchvideo: pip install pytorchvideo"
                ) from e
        
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


__all__ = ["X3DModel"]

