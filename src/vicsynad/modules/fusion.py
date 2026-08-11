"""
Multi-Modal Fusion Perceiver (Layer 2 of ViCSynAD)

Combines frozen Qwen2-VL vision features with trainable numeric embeddings
via cross-modal attention and gated fusion for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class NumericEncoder(nn.Module):
    """
    Encode statistical features + raw time series into a joint numeric embedding.

    Uses 1D convolutions for local temporal pattern capture + statistical
    feature projection.
    """

    def __init__(
        self,
        n_sensors: int,
        n_stat_features: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim

        # Temporal encoder: 1D CNN over sensor dimensions
        conv_layers = []
        in_ch = 1  # Treat each sensor as 1 channel initially

        for i in range(num_layers):
            out_ch = hidden_dim // (2 ** (num_layers - i - 1))
            # Use LayerNorm instead of BatchNorm1d for BF16 compatibility
            # and to avoid in-place op conflicts
            conv_layers.extend([
                nn.Conv1d(in_ch if i == 0 else hidden_dim // (2 ** (num_layers - i)),
                          out_ch, kernel_size,
                          padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.conv_net = nn.Sequential(*conv_layers)

        # Statistical feature projector
        self.stat_proj = nn.Sequential(
            nn.Linear(n_sensors * n_stat_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion of temporal + statistical
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.output_dim = hidden_dim

    def forward(
        self, x_window: torch.Tensor, x_stat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_window: (B, T, D) time series window
            x_stat:   (B, D, S) statistical features (S=6)

        Returns:
            embedding: (B, hidden_dim)
        """
        B, T, D = x_window.shape

        # Temporal encoding via 1D CNN
        # Reshape: (B, T, D) → (B, 1, D*T) — treat all sensors as channels... no
        # Better: (B, T, D) → (B*D, 1, T) → conv → pool → (B, D, H) → mean → (B, H)
        # Simplest: project (B, T, D) → (B, D, hidden_dim) then pool
        # Using 1D conv over time axis:
        x_t = x_window.permute(0, 2, 1)  # (B, D, T)
        x_t = x_t.reshape(B * D, 1, T)   # (B*D, 1, T)
        x_t = self.conv_net(x_t)          # (B*D, H_conv, T')
        x_t = x_t.mean(dim=-1)            # (B*D, H_conv) global average pool
        x_t = x_t.reshape(B, D, -1)       # (B, D, H_conv)
        x_t = x_t.mean(dim=1)             # (B, H_conv) average over sensors
        temporal_emb = x_t                # (B, hidden_dim)

        # Statistical encoding
        x_s = x_stat.reshape(B, -1)       # (B, D * S)
        stat_emb = self.stat_proj(x_s)    # (B, hidden_dim)

        # Fuse
        fused = self.fusion_proj(
            torch.cat([temporal_emb, stat_emb], dim=-1)
        )
        return fused


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: vision features attend to numeric embeddings
    and vice versa, producing a unified representation.
    """

    def __init__(
        self,
        vision_dim: int = 3584,  # Qwen2-VL-7B vision encoder output
        numeric_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both modalities to same dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Cross-attention: vision → numeric
        self.vision_to_numeric = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: numeric → vision
        self.numeric_to_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, vision_emb: torch.Tensor, numeric_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_emb:  (B, vision_dim)
            numeric_emb: (B, numeric_dim)

        Returns:
            fused: (B, hidden_dim)
        """
        B = vision_emb.shape[0]

        # Project
        v = self.vision_proj(vision_emb).unsqueeze(1)   # (B, 1, H)
        n = self.numeric_proj(numeric_emb).unsqueeze(1)  # (B, 1, H)

        # Cross-attend
        v2n, _ = self.vision_to_numeric(v, n, n)  # Vision attending to numeric
        n2v, _ = self.numeric_to_vision(n, v, v)  # Numeric attending to vision

        v2n = v2n.squeeze(1)  # (B, H)
        n2v = n2v.squeeze(1)  # (B, H)

        # Gated fusion
        gate = self.gate(torch.cat([v2n, n2v], dim=-1))
        fused = gate * v2n + (1 - gate) * n2v

        # Final projection
        concat = torch.cat([fused, gate * v2n + (1 - gate) * n2v], dim=-1)
        output = self.output_proj(concat)

        return output


class AnomalyDetectionHead(nn.Module):
    """MLP head for binary anomaly classification."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),  # LayerNorm for BF16 stability
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,) logits


class ViCSynADModel(nn.Module):
    """
    Complete ViCSynAD model: Vision + Numeric → Cross-Modal Fusion → Anomaly Score.

    Does NOT include the VLM vision encoder — that's frozen and handled
    by the training pipeline separately for VRAM efficiency.
    """

    def __init__(
        self,
        n_sensors: int,
        n_stat_features: int = 6,
        vision_dim: int = 3584,
        numeric_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        fusion_num_heads: int = 4,
        anomaly_head_dims: list = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.numeric_encoder = NumericEncoder(
            n_sensors=n_sensors,
            n_stat_features=n_stat_features,
            hidden_dim=numeric_hidden_dim,
            dropout=dropout,
        )

        self.cross_modal = CrossModalAttention(
            vision_dim=vision_dim,
            numeric_dim=numeric_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            dropout=dropout,
        )

        self.anomaly_head = AnomalyDetectionHead(
            input_dim=fusion_hidden_dim,
            hidden_dims=anomaly_head_dims,
            dropout=dropout,
        )

        self.fusion_hidden_dim = fusion_hidden_dim

        # For contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(
        self,
        vision_emb: torch.Tensor,
        x_window: torch.Tensor,
        x_stat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_emb: (B, vision_dim) pre-extracted vision features
            x_window:   (B, T, D) raw time series window
            x_stat:     (B, D, S) statistical features

        Returns:
            dict with logits, embeddings, projection
        """
        numeric_emb = self.numeric_encoder(x_window, x_stat)
        fused_emb = self.cross_modal(vision_emb, numeric_emb)
        logits = self.anomaly_head(fused_emb)
        proj = self.projection_head(fused_emb)

        return {
            "logits": logits,
            "fused_emb": fused_emb,
            "numeric_emb": numeric_emb,
            "projection": proj,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
