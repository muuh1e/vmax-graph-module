"""Temporal Transformer Encoder for agent history encoding."""

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
        pe[:, 1::2] = torch.cos(position * div_term)
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
        in_channels: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        pooling: str = "mean",  # "mean", "max", "cls", "last"
    ):
        """
        Args:
            in_channels: Input feature dimension per timestep
            d_model: Transformer hidden dimension
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            pooling: Pooling strategy ("mean", "max", "cls", "last")
        """
        super().__init__()

        self.d_model = d_model
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, in_channels]
            mask: Optional validity mask [batch, seq_len], True = valid

        Returns:
            Encoded sequence [batch, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add CLS token if needed
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]
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

        return self.norm(out)


class TemporalCNNEncoder(nn.Module):
    """
    1D CNN encoder for temporal sequences.

    Faster alternative to transformer for simpler patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels

            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, in_channels]
            mask: Optional [batch, seq_len]

        Returns:
            [batch, out_channels]
        """
        # Conv expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        x = self.conv(x)
        x = self.pool(x).squeeze(-1)

        return x
