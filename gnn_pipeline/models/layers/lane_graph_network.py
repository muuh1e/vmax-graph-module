"""Lane Graph Network layers (LaneGCN-style typed convolutions)."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class TypedLaneConv(MessagePassing):
    """
    Single convolution for one type of lane relationship.

    Performs message passing along a specific lane topology edge type
    (e.g., successor, predecessor, left_neighbor, right_neighbor).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        use_edge_attr: bool = False,
        edge_attr_dim: int = 7,
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_attr = use_edge_attr

        # Message MLP
        if use_edge_attr:
            self.message_mlp = nn.Sequential(
                nn.Linear(in_channels * 2 + edge_attr_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(in_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] optional edge features

        Returns:
            [N, out_channels] updated features
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        # x_j: source node features
        # x_i: target node features
        if self.use_edge_attr and edge_attr is not None:
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(msg_input)


class DilatedLaneConv(nn.Module):
    """
    Dilated convolution over lane graph for multi-hop reasoning.

    Uses pre-computed edges at different hop distances (1, 2, 4, 8).
    This allows the model to "see" far ahead along the road without
    adding dense all-to-all edges.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = [1, 2, 4, 8],
        aggr: str = "mean",
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            dilations: List of dilation factors (hop distances)
            aggr: Aggregation method
        """
        super().__init__()

        self.dilations = dilations

        # One conv per dilation
        self.convs = nn.ModuleList([
            TypedLaneConv(in_channels, out_channels, aggr=aggr)
            for _ in dilations
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * len(dilations), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_indices_by_hop: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels] node features
            edge_indices_by_hop: Dict mapping hop count to edge_index

        Returns:
            [N, out_channels] updated features
        """
        outputs = []

        for dilation, conv in zip(self.dilations, self.convs):
            if dilation in edge_indices_by_hop:
                edge_index = edge_indices_by_hop[dilation]
                out = conv(x, edge_index)
            else:
                # No edges at this dilation, use zeros
                out = torch.zeros(x.size(0), conv.out_channels, device=x.device)
            outputs.append(out)

        # Fuse all dilations
        combined = torch.cat(outputs, dim=-1)
        return self.fusion(combined)


class LaneGraphNetwork(nn.Module):
    """
    Full Lane Graph Network (LaneGCN-style) with typed convolutions.

    Processes lane nodes through separate convolutions for each
    semantic edge type:
    - Successor: lanes that follow this lane
    - Predecessor: lanes that precede this lane
    - Left neighbor: lanes to the left
    - Right neighbor: lanes to the right

    Each edge type captures different driving semantics.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_dilated: bool = True,
        dilated_scales: List[int] = [1, 2, 4, 8],
    ):
        """
        Args:
            in_channels: Input lane feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate
            use_dilated: Whether to use dilated convolutions
            dilated_scales: Dilation factors for multi-hop
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_dilated = use_dilated

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Typed convolutions for each layer
        self.successor_convs = nn.ModuleList()
        self.predecessor_convs = nn.ModuleList()
        self.left_neighbor_convs = nn.ModuleList()
        self.right_neighbor_convs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels

            self.successor_convs.append(
                TypedLaneConv(in_ch, out_ch)
            )
            self.predecessor_convs.append(
                TypedLaneConv(in_ch, out_ch)
            )
            self.left_neighbor_convs.append(
                TypedLaneConv(in_ch, out_ch)
            )
            self.right_neighbor_convs.append(
                TypedLaneConv(in_ch, out_ch)
            )

        # Dilated convolutions (optional)
        if use_dilated:
            self.dilated_convs = nn.ModuleList([
                DilatedLaneConv(
                    hidden_channels if i < num_layers - 1 else out_channels,
                    hidden_channels if i < num_layers - 1 else out_channels,
                    dilations=dilated_scales,
                )
                for i in range(num_layers)
            ])

        # Fusion after typed convolutions
        num_types = 4 + (1 if use_dilated else 0)
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    (hidden_channels if i < num_layers - 1 else out_channels) * num_types,
                    hidden_channels if i < num_layers - 1 else out_channels
                ),
                nn.ReLU(),
            )
            for i in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        typed_edges: Dict[str, torch.Tensor],
        dilated_edges: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N_lanes, in_channels] lane features
            typed_edges: Dict with keys:
                - 'successor': [2, E1] successor edges
                - 'predecessor': [2, E2] predecessor edges
                - 'left_of': [2, E3] left neighbor edges
                - 'right_of': [2, E4] right neighbor edges
            dilated_edges: Dict mapping hop count to edge_index

        Returns:
            [N_lanes, out_channels] lane embeddings
        """
        # Input projection
        h = self.input_proj(x)

        for i in range(self.num_layers):
            # Apply typed convolutions
            outputs = []

            # Successor
            if 'successor' in typed_edges and typed_edges['successor'].numel() > 0:
                out_succ = self.successor_convs[i](h, typed_edges['successor'])
            else:
                out_succ = torch.zeros_like(h)
            outputs.append(out_succ)

            # Predecessor
            if 'predecessor' in typed_edges and typed_edges['predecessor'].numel() > 0:
                out_pred = self.predecessor_convs[i](h, typed_edges['predecessor'])
            else:
                out_pred = torch.zeros_like(h)
            outputs.append(out_pred)

            # Left neighbor
            if 'left_of' in typed_edges and typed_edges['left_of'].numel() > 0:
                out_left = self.left_neighbor_convs[i](h, typed_edges['left_of'])
            else:
                out_left = torch.zeros_like(h)
            outputs.append(out_left)

            # Right neighbor
            if 'right_of' in typed_edges and typed_edges['right_of'].numel() > 0:
                out_right = self.right_neighbor_convs[i](h, typed_edges['right_of'])
            else:
                out_right = torch.zeros_like(h)
            outputs.append(out_right)

            # Dilated (optional)
            if self.use_dilated and dilated_edges is not None:
                out_dilated = self.dilated_convs[i](h, dilated_edges)
                outputs.append(out_dilated)

            # Fuse all edge types
            combined = torch.cat(outputs, dim=-1)
            h_new = self.fusions[i](combined)

            # Residual + norm + dropout
            if h.shape == h_new.shape:
                h = h + self.dropout(h_new)
            else:
                h = self.dropout(h_new)
            h = self.norms[i](h)

        return h


class AgentLaneFusion(nn.Module):
    """
    Bidirectional attention fusion between agents and lanes.

    Agents learn from lanes (road context) and lanes learn from agents
    (occupancy awareness).
    """

    def __init__(
        self,
        agent_dim: int,
        lane_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        from .attention import CrossAttention

        # Agent attends to lanes
        self.agent_to_lane = CrossAttention(
            query_dim=agent_dim,
            key_dim=lane_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Lane attends to agents
        self.lane_to_agent = CrossAttention(
            query_dim=lane_dim,
            key_dim=agent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        agent_embeds: torch.Tensor,
        lane_embeds: torch.Tensor,
        a2l_edge_index: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_embeds: [batch, N_agents, agent_dim] or [N_agents, agent_dim]
            lane_embeds: [batch, N_lanes, lane_dim] or [N_lanes, lane_dim]
            a2l_edge_index: Optional [2, E] for sparse attention
            agent_mask: [batch, N_agents] valid agents
            lane_mask: [batch, N_lanes] valid lanes

        Returns:
            updated_agents: Same shape as agent_embeds
            updated_lanes: Same shape as lane_embeds
        """
        # Add batch dimension if needed
        if agent_embeds.dim() == 2:
            agent_embeds = agent_embeds.unsqueeze(0)
            lane_embeds = lane_embeds.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Bidirectional attention
        agents_updated = self.agent_to_lane(
            query=agent_embeds,
            context=lane_embeds,
            context_mask=lane_mask,
        )

        lanes_updated = self.lane_to_agent(
            query=lane_embeds,
            context=agent_embeds,
            context_mask=agent_mask,
        )

        if squeeze_output:
            agents_updated = agents_updated.squeeze(0)
            lanes_updated = lanes_updated.squeeze(0)

        return agents_updated, lanes_updated
