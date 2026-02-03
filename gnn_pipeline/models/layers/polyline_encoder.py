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


# =============================================================================
# Phase 2C: New Polyline Encoder Classes
# =============================================================================

class PointNetEncoder(nn.Module):
    """
    PointNet-style polyline encoder.

    Uses shared MLP + max pooling for permutation invariance.
    For ordered polylines, we add positional encoding to preserve order.
    """

    def __init__(
        self,
        input_dim: int = 2,       # x, y per point
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        use_position_encoding: bool = True,
    ):
        super().__init__()

        self.use_position_encoding = use_position_encoding
        effective_input_dim = input_dim + 1 if use_position_encoding else input_dim

        layers = []
        dims = [effective_input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i+1]))

        self.mlp = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: [batch, N_points, input_dim] polyline points
            mask: [batch, N_points] validity mask (True = valid)

        Returns:
            [batch, output_dim] polyline embedding
        """
        batch, N, D = points.shape

        if self.use_position_encoding:
            # Add normalized position index
            pos_idx = torch.linspace(0, 1, N, device=points.device)
            pos_idx = pos_idx.view(1, N, 1).expand(batch, -1, -1)
            points = torch.cat([points, pos_idx], dim=-1)

        # Shared MLP
        h = self.mlp(points)  # [batch, N_points, output_dim]

        # Apply mask before pooling
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        # Max pooling over points
        h = h.max(dim=1)[0]  # [batch, output_dim]

        # Handle all-masked case
        h = torch.where(
            torch.isinf(h),
            torch.zeros_like(h),
            h
        )

        return self.output_norm(h)


class TransformerPolylineEncoder(nn.Module):
    """
    Transformer-based polyline encoder.

    Uses self-attention to capture point relationships.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable position embedding (max 50 points)
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: [batch, N_points, input_dim]
            mask: [batch, N_points] validity mask (True = valid)

        Returns:
            [batch, output_dim]
        """
        batch, N, _ = points.shape

        # Project input
        h = self.input_proj(points)  # [batch, N, hidden_dim]

        # Add position embedding
        h = h + self.pos_embedding[:, :N, :]

        # Create attention mask (True = ignore for TransformerEncoder)
        if mask is not None:
            attn_mask = ~mask  # True = ignore
        else:
            attn_mask = None

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Attention pooling
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool_attn(query, h, h, key_padding_mask=attn_mask)
        pooled = pooled.squeeze(1)  # [batch, hidden_dim]

        # Output projection
        out = self.output_proj(pooled)
        return self.output_norm(out)


class Conv1DPolylineEncoder(nn.Module):
    """
    1D CNN polyline encoder (VectorNet-style).

    Treats polyline as 1D sequence and applies convolutions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
        self.convs = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: [batch, N_points, input_dim]
            mask: [batch, N_points] validity mask (True = valid)

        Returns:
            [batch, output_dim]
        """
        # Project input
        h = self.input_proj(points)  # [batch, N, hidden_dim]

        # Transpose for conv1d
        h = h.transpose(1, 2)  # [batch, hidden_dim, N]

        # Apply convolutions
        h = self.convs(h)  # [batch, hidden_dim, N]

        # Apply mask and pool
        h = h.transpose(1, 2)  # [batch, N, hidden_dim]
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
            # Mean pooling over valid points
            h = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-6)
        else:
            h = h.mean(dim=1)

        # Output projection
        out = self.output_proj(h)
        return self.output_norm(out)


def get_polyline_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for polyline encoders.

    Args:
        encoder_type: "pointnet", "transformer", or "conv1d"
        input_dim: Input dimension per point (e.g., 2 for x, y)
        hidden_dim: Hidden dimension for encoder
        output_dim: Output embedding dimension
        **kwargs: Additional encoder-specific arguments

    Returns:
        Polyline encoder module
    """
    if encoder_type == "pointnet":
        return PointNetEncoder(input_dim, hidden_dim, output_dim, **kwargs)
    elif encoder_type == "transformer":
        return TransformerPolylineEncoder(input_dim, hidden_dim, output_dim, **kwargs)
    elif encoder_type == "conv1d":
        return Conv1DPolylineEncoder(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown polyline encoder type: {encoder_type}")
