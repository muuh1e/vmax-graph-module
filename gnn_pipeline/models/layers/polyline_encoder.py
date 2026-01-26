"""Polyline encoder for lane geometry encoding."""

from typing import Optional

import torch
import torch.nn as nn


class PolylineEncoder(nn.Module):
    """
    Encodes lane polylines (sequences of points) into embeddings.

    Supports multiple encoding strategies:
    - MLP: Flatten all points and project
    - CNN: 1D convolution over points
    - Transformer: Attention over points
    - PointNet: Per-point MLP + max pooling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        num_points: int = 10,
        encoder_type: str = "pointnet",  # "mlp", "cnn", "transformer", "pointnet"
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Features per point (e.g., x, y, dx, dy)
            out_channels: Output embedding dimension
            num_points: Number of sampled points per polyline
            encoder_type: Encoding strategy
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.out_channels = out_channels

        if encoder_type == "mlp":
            # Flatten and project
            self.encoder = nn.Sequential(
                nn.Linear(in_channels * num_points, out_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels * 2, out_channels),
            )

        elif encoder_type == "cnn":
            # 1D convolution over points
            hidden = out_channels // 2
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveMaxPool1d(1)

        elif encoder_type == "transformer":
            # Small transformer over points
            from .temporal_encoder import TemporalTransformerEncoder
            self.encoder = TemporalTransformerEncoder(
                in_channels=in_channels,
                d_model=out_channels,
                num_layers=num_layers,
                num_heads=4,
                dropout=dropout,
                pooling="mean",
            )

        else:  # pointnet
            # Per-point MLP + max pooling (simple but effective)
            self.point_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, out_channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Polyline points [batch, num_points, in_channels]
               OR [batch, in_channels] if already flattened for MLP
            mask: Optional validity mask [batch, num_points]

        Returns:
            Polyline embeddings [batch, out_channels]
        """
        if self.encoder_type == "mlp":
            if x.dim() == 3:
                x = x.view(x.size(0), -1)
            return self.encoder(x)

        elif self.encoder_type == "cnn":
            x = x.transpose(1, 2)  # [batch, channels, points]
            if mask is not None:
                x = x * mask.unsqueeze(1).float()
            x = self.encoder(x)
            x = self.pool(x).squeeze(-1)
            return x

        elif self.encoder_type == "transformer":
            return self.encoder(x, mask)

        else:  # pointnet
            # Per-point features
            point_feats = self.point_mlp(x)  # [batch, points, out_channels]

            if mask is not None:
                point_feats = point_feats.masked_fill(~mask.unsqueeze(-1), float('-inf'))

            # Max pooling over points
            max_feat = point_feats.max(dim=1)[0]  # [batch, out_channels]

            # Mean pooling over points
            if mask is not None:
                point_feats_for_mean = point_feats.masked_fill(~mask.unsqueeze(-1), 0)
                mean_feat = point_feats_for_mean.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                mean_feat = point_feats.mean(dim=1)

            # Combine and project
            combined = torch.cat([max_feat, mean_feat], dim=-1)
            return self.global_mlp(combined)


class SubgraphPolylineEncoder(nn.Module):
    """
    Encodes sets of polylines (e.g., all lanes in a subgraph).

    Two-stage encoding:
    1. Encode each polyline to an embedding
    2. Aggregate all polyline embeddings (attention or pooling)

    This is useful for hierarchical encoding where we first encode
    individual lanes, then aggregate at the subgraph/cluster level.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        num_points: int = 10,
        polyline_encoder_type: str = "pointnet",
        aggregation: str = "attention",  # "attention", "mean", "max"
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Polyline-level encoder
        self.polyline_encoder = PolylineEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            num_points=num_points,
            encoder_type=polyline_encoder_type,
            dropout=dropout,
        )

        self.aggregation = aggregation

        if aggregation == "attention":
            # Self-attention over polyline embeddings
            self.attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        polylines: torch.Tensor,
        polyline_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            polylines: [batch, num_polylines, num_points, in_channels]
            polyline_mask: [batch, num_polylines] which polylines are valid
            point_mask: [batch, num_polylines, num_points] which points are valid

        Returns:
            Subgraph embedding [batch, out_channels]
        """
        batch_size, num_polylines, num_points, in_channels = polylines.shape

        # Reshape for polyline encoding
        polylines_flat = polylines.view(batch_size * num_polylines, num_points, in_channels)

        if point_mask is not None:
            point_mask_flat = point_mask.view(batch_size * num_polylines, num_points)
        else:
            point_mask_flat = None

        # Encode each polyline
        poly_embeds = self.polyline_encoder(polylines_flat, point_mask_flat)
        poly_embeds = poly_embeds.view(batch_size, num_polylines, -1)

        # Aggregate
        if self.aggregation == "attention":
            key_padding_mask = ~polyline_mask if polyline_mask is not None else None
            attended, _ = self.attention(
                poly_embeds, poly_embeds, poly_embeds,
                key_padding_mask=key_padding_mask,
            )
            poly_embeds = self.norm(poly_embeds + attended)

            if polyline_mask is not None:
                poly_embeds = poly_embeds.masked_fill(~polyline_mask.unsqueeze(-1), 0)
                out = poly_embeds.sum(dim=1) / polyline_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                out = poly_embeds.mean(dim=1)

        elif self.aggregation == "max":
            if polyline_mask is not None:
                poly_embeds = poly_embeds.masked_fill(~polyline_mask.unsqueeze(-1), float('-inf'))
            out = poly_embeds.max(dim=1)[0]

        else:  # mean
            if polyline_mask is not None:
                poly_embeds = poly_embeds.masked_fill(~polyline_mask.unsqueeze(-1), 0)
                out = poly_embeds.sum(dim=1) / polyline_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                out = poly_embeds.mean(dim=1)

        return out
