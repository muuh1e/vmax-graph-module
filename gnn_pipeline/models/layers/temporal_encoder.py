"""
Temporal encoders for agent trajectory history.

Encodes [batch, T, feat_dim] -> [batch, hidden_dim]

Provides three encoder types:
- TemporalTransformerEncoder: HiVT-style transformer with positional encoding
- TemporalConv1DEncoder: LaneGCN-style 1D CNN
- TemporalGRUEncoder: GRU-based recurrent encoder
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal sequences (agent history).

    Encodes a sequence of timesteps into a single embedding via:
    1. Linear projection to d_model
    2. Positional encoding
    3. Transformer encoder layers
    4. Pooling (mean, max, or [CLS] token)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        pooling: str = "last",  # "mean", "max", "cls", "last"
    ):
        """
        Args:
            input_dim: Input feature dimension per timestep
            hidden_dim: Transformer hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            pooling: Pooling strategy ("mean", "max", "cls", "last")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            mask: Optional validity mask [batch, seq_len], True = valid

        Returns:
            Encoded sequence [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Add CLS token if needed
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, hidden_dim]
            if mask is not None:
                # Prepend True for CLS token
                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask (True = ignore)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Transformer expects True = ignore

        # Transform
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Pool to single vector
        if self.pooling == "cls":
            out = x[:, 0, :]  # CLS token
        elif self.pooling == "last":
            if mask is not None:
                # Get last valid position
                lengths = mask.sum(dim=1).long()
                batch_indices = torch.arange(batch_size, device=x.device)
                out = x[batch_indices, lengths - 1, :]
            else:
                out = x[:, -1, :]
        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            out = x.max(dim=1)[0]
        else:  # mean
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
                out = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                out = x.mean(dim=1)

        # Output projection
        out = self.output_proj(out)
        return self.norm(out)


class TemporalConv1DEncoder(nn.Module):
    """
    1D CNN encoder for temporal sequences (LaneGCN-style).

    Faster alternative to transformer for simpler patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.conv = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: Optional [batch, seq_len]

        Returns:
            [batch, output_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Conv expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        x = self.conv(x)
        
        # Pool: use last timestep
        x = x[:, :, -1]  # [batch, hidden_dim]
        
        # Output projection
        out = self.output_proj(x)
        return self.norm(out)


class TemporalGRUEncoder(nn.Module):
    """
    GRU-based temporal encoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(gru_out_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] validity mask

        Returns:
            [batch, output_dim]
        """
        h = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # GRU encoding
        output, h_n = self.gru(h)  # output: [batch, seq_len, hidden_dim]

        # Use final hidden state
        if self.gru.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_final = h_n[-1]

        # Output projection
        out = self.output_proj(h_final)
        return self.norm(out)


def get_temporal_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for temporal encoders.

    Args:
        encoder_type: "transformer", "conv1d", or "gru"
        input_dim: Per-timestep feature dimension
        hidden_dim: Hidden dimension for encoder
        output_dim: Output embedding dimension
        **kwargs: Additional encoder-specific arguments

    Returns:
        Temporal encoder module
    """
    if encoder_type == "transformer":
        return TemporalTransformerEncoder(input_dim, hidden_dim, output_dim, **kwargs)
    elif encoder_type == "conv1d":
        # Filter out transformer-specific kwargs
        conv_kwargs = {k: v for k, v in kwargs.items()
                      if k in ['num_layers', 'kernel_size', 'dropout']}
        return TemporalConv1DEncoder(input_dim, hidden_dim, output_dim, **conv_kwargs)
    elif encoder_type == "gru":
        # Filter out transformer-specific kwargs
        gru_kwargs = {k: v for k, v in kwargs.items()
                     if k in ['num_layers', 'dropout', 'bidirectional']}
        return TemporalGRUEncoder(input_dim, hidden_dim, output_dim, **gru_kwargs)
    else:
        raise ValueError(f"Unknown temporal encoder type: {encoder_type}. "
                         f"Choose from: 'transformer', 'conv1d', 'gru'")


# Backward compatibility alias
TemporalCNNEncoder = TemporalConv1DEncoder


