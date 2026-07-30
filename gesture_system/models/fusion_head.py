"""
GestureRecognitionModel: fuses LandmarkTransformer + VideoSwinBackbone.

forward(landmarks, frames)        → logits (B, num_classes)
forward_landmark_only(landmarks)  → logits (B, num_classes)  [no video branch]
"""

import torch
import torch.nn as nn

from models.landmark_transformer import LandmarkTransformer
from models.video_swin import VideoSwinBackbone


class FusionHead(nn.Module):
    """
    Concat-fuses a 256-d landmark embedding and a 1024-d video embedding.

    Input:  two tensors of shape (B, 256) and (B, 1024) — passed as a tuple
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        landmark_dim: int = 256,
        video_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        fused_dim = landmark_dim + video_dim  # 1280

        self.net = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, landmark_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmark_feat: (B, 256)
            video_feat:    (B, 1024)
        Returns:
            logits: (B, num_classes)
        """
        fused = torch.cat([landmark_feat, video_feat], dim=-1)  # (B, 1280)
        return self.net(fused)


class LandmarkOnlyHead(nn.Module):
    """
    Lightweight head used when no video frames are available (e.g. --landmark-only).
    Maps 256-d landmark embedding directly to logits.
    """

    def __init__(
        self,
        landmark_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, landmark_feat: torch.Tensor) -> torch.Tensor:
        return self.net(landmark_feat)


class GestureRecognitionModel(nn.Module):
    """
    Full two-stream gesture model.

    Components
    ----------
    landmark_encoder : LandmarkTransformer   (B,30,63) → (B,256)
    video_encoder    : VideoSwinBackbone      (B,3,30,224,224) → (B,1024)
    fusion_head      : FusionHead             (B,256)+(B,1024) → (B,5)
    lm_only_head     : LandmarkOnlyHead       (B,256) → (B,5)  [fallback]
    """

    def __init__(
        self,
        landmark_d_model: int = 256,
        landmark_nhead: int = 8,
        landmark_num_layers: int = 6,
        landmark_dim_feedforward: int = 1024,
        landmark_dropout: float = 0.1,
        fusion_hidden: int = 512,
        fusion_dropout: float = 0.3,
        num_classes: int = 5,
        pretrained_swin: bool = True,
    ):
        super().__init__()

        self.landmark_encoder = LandmarkTransformer(
            input_dim=63,
            d_model=landmark_d_model,
            nhead=landmark_nhead,
            num_layers=landmark_num_layers,
            dim_feedforward=landmark_dim_feedforward,
            dropout=landmark_dropout,
        )

        self.video_encoder = VideoSwinBackbone(pretrained=pretrained_swin)

        self.fusion_head = FusionHead(
            landmark_dim=landmark_d_model,
            video_dim=1024,
            hidden_dim=fusion_hidden,
            num_classes=num_classes,
            dropout=fusion_dropout,
        )

        self.lm_only_head = LandmarkOnlyHead(
            landmark_dim=landmark_d_model,
            hidden_dim=fusion_hidden,
            num_classes=num_classes,
            dropout=fusion_dropout,
        )

    # ------------------------------------------------------------------
    # Parameter group helpers (used by train.py for differential LRs)
    # ------------------------------------------------------------------

    def swin_parameters(self):
        """All parameters belonging to the Video Swin backbone."""
        return list(self.video_encoder.parameters())

    def non_swin_parameters(self):
        """Everything except Video Swin (landmark encoder + heads)."""
        swin_ids = {id(p) for p in self.video_encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in swin_ids]

    def fusion_linear_parameters(self):
        """Only the Linear layers inside fusion_head and lm_only_head."""
        params = []
        for module in [self.fusion_head, self.lm_only_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    params.extend(list(layer.parameters()))
        return params

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        landmarks: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full two-stream forward pass.

        Args:
            landmarks : (B, 30, 63)          — normalised landmark sequences
            frames    : (B, 3, 30, 224, 224)  — normalised video clips (C,T,H,W)
        Returns:
            logits: (B, num_classes)
        """
        lm_feat    = self.landmark_encoder(landmarks)   # (B, 256)
        video_feat = self.video_encoder(frames)          # (B, 1024)
        logits     = self.fusion_head(lm_feat, video_feat)
        return logits

    def forward_landmark_only(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Landmark-only forward pass — no video branch.
        Useful when camera frames are unavailable or for ablation studies.

        Args:
            landmarks: (B, 30, 63)
        Returns:
            logits: (B, num_classes)
        """
        lm_feat = self.landmark_encoder(landmarks)  # (B, 256)
        logits  = self.lm_only_head(lm_feat)
        return logits
