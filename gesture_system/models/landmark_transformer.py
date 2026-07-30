"""
LandmarkTransformer: encodes a sequence of 30 hand landmark frames.

Input:  (batch, 30, 63)   — 21 landmarks × 3 coords, pre-normalised
Output: (batch, 256)       — sequence-pooled, layer-normed embedding
"""

import math
import torch
import torch.nn as nn


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 30,
    ):
        super().__init__()

        # Project raw landmark coords into model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable positional encoding — one vector per temporal position
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, C) convention
            norm_first=True,   # Pre-LN for stability with deep stacks
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # keeps bf16 path clean
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 63)  — normalised landmark sequences
        Returns:
            (B, 256)        — clip-level embedding
        """
        B, T, _ = x.shape

        # (B, T, 63) → (B, T, 256)
        x = self.input_projection(x)

        # Add learnable positional encoding (broadcast over batch)
        x = x + self.pos_encoding[:T].unsqueeze(0)

        # Transformer encoder: (B, T, 256) → (B, T, 256)
        x = self.transformer(x)

        # Mean-pool over the time dimension → (B, 256)
        x = x.mean(dim=1)

        # Final layer norm
        x = self.norm(x)

        return x
