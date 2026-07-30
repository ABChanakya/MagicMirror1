"""
VideoSwinBackbone: wraps torchvision's Swin3D-B with Kinetics-400 weights.

Input:  (B, 3, T, H, W)  — T=30 frames, H=W=224, values in [0,1]-normalised
Output: (B, 1024)         — spatial-temporal feature vector
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torchvision.models.video import swin3d_b, Swin3D_B_Weights


class VideoSwinBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        weights = Swin3D_B_Weights.KINETICS400_V1 if pretrained else None
        backbone = swin3d_b(weights=weights)
        backbone.head = nn.Identity()
        self.backbone = backbone

    def freeze_all(self):
        """Freeze entire backbone — Stages 1-4 all locked."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def freeze_except_last_stage(self):
        """Freeze Stages 1-3, unfreeze Stage 4 + LayerNorm only.
        Kinetics-400 pretrained features are preserved; only the last
        stage is nudged with a tiny LR to adapt to gestures."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # features[6] = Stage 4 (last Swin stage)
        for p in self.backbone.features[6].parameters():
            p.requires_grad_(True)
        # backbone.norm = LayerNorm after all stages
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        """Call before Phase 3 to trade compute for memory."""
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → (B, 1024)"""
        return self.backbone(x)
