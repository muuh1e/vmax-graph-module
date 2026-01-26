"""Multi-modal trajectory decoder for uncertainty estimation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalDecoder(nn.Module):
    """
    Decodes multiple possible future trajectories with confidence scores.

    Uses learnable mode queries that attend to context and generate
    diverse predictions. Training uses winner-take-all loss (only
    backprop through closest mode).
    """

    def __init__(
        self,
        in_channels: int,
        num_modes: int = 6,
        num_future_steps: int = 80,
        hidden_channels: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Input context dimension
            num_modes: Number of trajectory modes to predict (K)
            num_future_steps: Number of future timesteps (T)
            hidden_channels: Hidden layer dimension
            num_heads: Attention heads for mode queries
            dropout: Dropout rate
        """
        super().__init__()

        self.num_modes = num_modes
        self.num_future_steps = num_future_steps
        self.in_channels = in_channels

        # Learnable mode queries
        self.mode_queries = nn.Parameter(torch.randn(num_modes, in_channels) * 0.02)

        # Mode query to context attention
        self.mode_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Trajectory decoder (per mode)
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_future_steps * 2),
        )

        # Confidence decoder
        self.confidence_decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [batch, context_len, in_channels] scene context
                    OR [batch, in_channels] single ego embedding
            context_mask: [batch, context_len] valid context (True = valid)

        Returns:
            trajectories: [batch, num_modes, num_future_steps, 2]
            confidences: [batch, num_modes] (softmax normalized)
        """
        batch_size = context.size(0)

        # Handle single vector vs sequence
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [batch, 1, in_channels]
            context_mask = None

        # Expand mode queries for batch
        mode_q = self.mode_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch, num_modes, in_channels]

        # Mode queries attend to context
        if context_mask is not None:
            key_padding_mask = ~context_mask
        else:
            key_padding_mask = None

        mode_embeds, _ = self.mode_attention(
            mode_q, context, context,
            key_padding_mask=key_padding_mask,
        )
        # [batch, num_modes, in_channels]

        # Add residual
        mode_embeds = mode_embeds + mode_q

        # Decode trajectories
        traj_flat = self.trajectory_decoder(mode_embeds)
        # [batch, num_modes, num_future_steps * 2]
        trajectories = traj_flat.view(batch_size, self.num_modes, self.num_future_steps, 2)

        # Decode confidences
        conf_logits = self.confidence_decoder(mode_embeds).squeeze(-1)
        # [batch, num_modes]
        confidences = F.softmax(conf_logits, dim=-1)

        return trajectories, confidences


class GRUDecoder(nn.Module):
    """
    Autoregressive GRU decoder for trajectory prediction.

    Generates trajectory one step at a time, potentially more
    accurate for long horizons but slower.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_future_steps: int = 80,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_future_steps = num_future_steps
        self.num_layers = num_layers

        # Initial hidden state projection
        self.init_h = nn.Linear(in_channels, hidden_channels * num_layers)

        # GRU
        self.gru = nn.GRU(
            input_size=2,  # x, y position
            hidden_size=hidden_channels,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_channels, 2)

    def forward(
        self,
        context: torch.Tensor,
        initial_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, in_channels] context embedding
            initial_position: [batch, 2] starting position (default 0,0)

        Returns:
            [batch, num_future_steps, 2] predicted trajectory
        """
        batch_size = context.size(0)

        # Initialize hidden state from context
        h = self.init_h(context)
        h = h.view(batch_size, self.num_layers, self.hidden_channels)
        h = h.permute(1, 0, 2).contiguous()  # [num_layers, batch, hidden]

        # Initial position (typically 0,0 in ego frame)
        if initial_position is None:
            pos = torch.zeros(batch_size, 1, 2, device=context.device)
        else:
            pos = initial_position.unsqueeze(1)

        # Autoregressive decoding
        outputs = []
        for t in range(self.num_future_steps):
            out, h = self.gru(pos, h)
            delta = self.output_proj(out)  # Predict displacement
            pos = pos + delta  # Update position
            outputs.append(pos)

        trajectory = torch.cat(outputs, dim=1)
        return trajectory


class AnchorBasedDecoder(nn.Module):
    """
    Anchor-based decoder using predefined trajectory templates.

    Predicts refinements to anchor trajectories rather than
    absolute positions. Common in production systems.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 64,
        num_future_steps: int = 80,
        hidden_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_anchors = num_anchors
        self.num_future_steps = num_future_steps

        # Learnable anchor trajectories
        self.anchors = nn.Parameter(
            torch.zeros(num_anchors, num_future_steps, 2)
        )
        # Initialize with diverse patterns
        self._init_anchors()

        # Classification head (which anchor)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_anchors),
        )

        # Refinement head (per anchor)
        self.refiner = nn.Sequential(
            nn.Linear(in_channels + num_anchors, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_future_steps * 2),
        )

    def _init_anchors(self):
        """Initialize anchors with diverse motion patterns."""
        with torch.no_grad():
            for i in range(self.num_anchors):
                # Create diverse patterns: straight, curves, stops
                t = torch.arange(self.num_future_steps).float() / 10  # 0-8 seconds

                # Vary speed and curvature
                speed = 5 + 10 * (i % 8) / 7  # 5-15 m/s
                curvature = (i // 8 - 4) * 0.02  # -0.08 to 0.08

                self.anchors.data[i, :, 0] = speed * t  # x
                self.anchors.data[i, :, 1] = curvature * speed * t ** 2  # y

    def forward(
        self,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [batch, in_channels]

        Returns:
            trajectories: [batch, num_anchors, num_future_steps, 2]
            confidences: [batch, num_anchors]
            best_trajectory: [batch, num_future_steps, 2]
        """
        batch_size = context.size(0)

        # Classify anchor
        logits = self.classifier(context)  # [batch, num_anchors]
        confidences = F.softmax(logits, dim=-1)

        # Expand anchors
        anchors = self.anchors.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # [batch, num_anchors, T, 2]

        # Per-anchor refinement
        anchor_one_hot = F.one_hot(
            torch.arange(self.num_anchors, device=context.device),
            self.num_anchors
        ).float()  # [num_anchors, num_anchors]
        anchor_one_hot = anchor_one_hot.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch, num_anchors, num_anchors]

        # Concatenate context with anchor indicator
        context_expanded = context.unsqueeze(1).expand(-1, self.num_anchors, -1)
        refine_input = torch.cat([context_expanded, anchor_one_hot], dim=-1)

        refinements = self.refiner(refine_input)
        refinements = refinements.view(batch_size, self.num_anchors, self.num_future_steps, 2)

        # Add refinements to anchors
        trajectories = anchors + refinements

        # Best trajectory
        best_idx = confidences.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=context.device)
        best_trajectory = trajectories[batch_indices, best_idx]

        return trajectories, confidences, best_trajectory
