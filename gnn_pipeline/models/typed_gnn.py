"""
TypedHeteroGNN: LaneGCN-inspired heterogeneous GNN with typed convolutions.

Key innovations over SimpleHeteroGNN:
1. Semantically-aware convolutions per edge relationship (successor, predecessor, neighbor)
2. Dilated lane graph for multi-hop lane reasoning
3. Agent-Lane fusion with bidirectional cross-attention
4. Typed A2A convolutions for different interaction types
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData

from . import register_model
from .base_model import BaseMotionPredictor
from .layers.lane_graph_network import LaneGraphNetwork, AgentLaneFusion
from .layers.attention import CrossAttention


class TypedA2AConv(nn.Module):
    """
    Typed agent-to-agent convolution with different weights per interaction type.

    Interaction types:
    - following: Same lane, behind target
    - leading: Same lane, ahead of target
    - adjacent: Neighboring lane
    - crossing: Conflicting paths (intersection)
    - approaching: Closing distance fast
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_interaction_types: int = 5,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 20,
        aggr: str = "mean",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = num_interaction_types

        # Type-specific message MLPs
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels * 2 + (edge_attr_dim if use_edge_attr else 0), out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            for _ in range(num_interaction_types)
        ])

        # Edge type classifier (from edge attributes)
        if use_edge_attr:
            self.type_classifier = nn.Sequential(
                nn.Linear(edge_attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_interaction_types),
            )
        else:
            self.type_classifier = None

        # Aggregation
        self.aggr = aggr

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels] agent features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_attr_dim] edge features
            edge_types: [E] pre-computed edge type indices (0 to num_types-1)

        Returns:
            [N, out_channels] updated features
        """
        if edge_index.numel() == 0:
            return torch.zeros(x.size(0), self.out_channels, device=x.device)

        src, dst = edge_index
        N = x.size(0)
        E = edge_index.size(1)

        # Determine edge types
        if edge_types is not None:
            type_weights = torch.zeros(E, self.num_types, device=x.device)
            type_weights.scatter_(1, edge_types.unsqueeze(1), 1.0)
        elif self.type_classifier is not None and edge_attr is not None:
            type_logits = self.type_classifier(edge_attr)
            type_weights = torch.softmax(type_logits, dim=-1)
        else:
            # Uniform across types
            type_weights = torch.ones(E, self.num_types, device=x.device) / self.num_types

        # Compute messages for each type
        x_src = x[src]
        x_dst = x[dst]

        if edge_attr is not None:
            msg_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_src, x_dst], dim=-1)

        # Weighted combination of type-specific messages
        messages = torch.zeros(E, self.out_channels, device=x.device)
        for t, mlp in enumerate(self.message_mlps):
            msg_t = mlp(msg_input)  # [E, out_channels]
            messages = messages + type_weights[:, t:t+1] * msg_t

        # Aggregate using PyTorch Geometric's scatter
        from torch_geometric.utils import scatter
        if self.aggr == "mean":
            out = scatter(messages, dst, dim=0, dim_size=N, reduce="mean")
        else:
            out = scatter(messages, dst, dim=0, dim_size=N, reduce="sum")

        return out


@register_model("typed_gnn")
class TypedHeteroGNN(BaseMotionPredictor):
    """
    LaneGCN-inspired heterogeneous GNN with typed convolutions.

    Architecture:
    1. Node encoders (MLP per node type)
    2. Lane Graph Network (typed L2L convolutions)
    3. Agent-Lane Fusion (bidirectional cross-attention)
    4. Typed A2A Convolutions (different conv per interaction type)
    5. Ego extraction + MLP decoder

    Args:
        agent_in_channels: Agent input features
        lane_in_channels: Lane input features
        tl_in_channels: Traffic light input features
        hidden_channels: Hidden embedding dimension
        num_layers: Number of GNN layers
        num_future_steps: Prediction horizon
        dropout: Dropout rate
        conv_type: Base convolution type ("sage" or "gat")
        edge_types: List of edge types for HeteroConv
        use_edge_attr: Whether to use edge attributes
        use_dilated_lanes: Whether to use dilated lane convolutions
        dilated_scales: Dilation factors for lane graph
        num_interaction_types: Number of A2A interaction types
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        tl_in_channels: int = 12,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
        conv_type: str = "sage",
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_edge_attr: bool = False,
        use_dilated_lanes: bool = True,
        dilated_scales: List[int] = [1, 2, 4, 8],
        num_interaction_types: int = 5,
        a2a_edge_attr_dim: int = 20,  # NEW: configurable (20 for legacy, 32 for enhanced)
    ):
        super().__init__(num_future_steps=num_future_steps)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr
        self.use_dilated_lanes = use_dilated_lanes

        # Default edge types
        if edge_types is None:
            edge_types = [
                ("agent", "to", "agent"),
                ("agent", "to", "lane"),
                ("lane", "rev_to", "agent"),
            ]
        self.edge_types = edge_types
        self.has_tl = any("tl" in et for et in edge_types)
        self.has_l2l = any(et[0] == "lane" and et[2] == "lane" for et in edge_types)

        # ============ Node Encoders ============
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if self.has_tl:
            self.tl_encoder = nn.Sequential(
                nn.Linear(tl_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

        # ============ Lane Graph Network ============
        if self.has_l2l:
            self.lane_gnn = LaneGraphNetwork(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                use_dilated=use_dilated_lanes,
                dilated_scales=dilated_scales,
            )
        else:
            # Fallback: simple MLP layers for lanes
            self.lane_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_channels),
                )
                for _ in range(num_layers)
            ])

        # ============ Agent-Lane Fusion ============
        self.agent_lane_fusion = AgentLaneFusion(
            agent_dim=hidden_channels,
            lane_dim=hidden_channels,
            hidden_dim=hidden_channels,
            num_heads=4,
            dropout=dropout,
        )

        # ============ Typed A2A Convolutions ============
        self.a2a_convs = nn.ModuleList([
            TypedA2AConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_interaction_types=num_interaction_types,
                use_edge_attr=use_edge_attr,
                edge_attr_dim=a2a_edge_attr_dim,  # Use configurable dimension
            )
            for _ in range(num_layers)
        ])

        # ============ Layer Norms ============
        self.agent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.lane_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # ============ Decoder ============
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_future_steps * 2),
        )

    def _get_typed_l2l_edges(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Extract typed L2L edges from data if available."""
        typed_edges = {}

        # Try to get typed edges from data
        edge_type_names = ['successor', 'predecessor', 'left_of', 'right_of']

        for name in edge_type_names:
            edge_key = ('lane', name, 'lane')
            if edge_key in data.edge_types:
                try:
                    typed_edges[name] = data[edge_key].edge_index
                except (KeyError, AttributeError):
                    pass

        # Fallback: use generic L2L edge and infer types from attributes
        if not typed_edges and ('lane', 'to', 'lane') in data.edge_types:
            try:
                l2l_edge = data['lane', 'to', 'lane']
                edge_index = l2l_edge.edge_index
                edge_attr = l2l_edge.edge_attr if hasattr(l2l_edge, 'edge_attr') else None

                if edge_attr is not None and edge_attr.size(1) >= 5:
                    # Parse edge attributes: [dist, is_succ, is_pred, is_left, is_right, ...]
                    is_succ = edge_attr[:, 1] > 0.5
                    is_pred = edge_attr[:, 2] > 0.5
                    is_left = edge_attr[:, 3] > 0.5
                    is_right = edge_attr[:, 4] > 0.5

                    if is_succ.any():
                        typed_edges['successor'] = edge_index[:, is_succ]
                    if is_pred.any():
                        typed_edges['predecessor'] = edge_index[:, is_pred]
                    if is_left.any():
                        typed_edges['left_of'] = edge_index[:, is_left]
                    if is_right.any():
                        typed_edges['right_of'] = edge_index[:, is_right]
                else:
                    # Use all L2L edges as generic
                    typed_edges['successor'] = edge_index
            except (KeyError, AttributeError):
                pass

        return typed_edges

    def _get_dilated_edges(self, typed_edges: Dict[str, torch.Tensor], num_lanes: int) -> Dict[int, torch.Tensor]:
        """Compute dilated edges by following successor chains."""
        if 'successor' not in typed_edges or typed_edges['successor'].numel() == 0:
            return {}

        succ_edge = typed_edges['successor']

        # Build adjacency dict
        adj = {}
        for i in range(succ_edge.size(1)):
            src = succ_edge[0, i].item()
            dst = succ_edge[1, i].item()
            if src not in adj:
                adj[src] = []
            adj[src].append(dst)

        dilated_edges = {1: succ_edge}

        # Compute multi-hop edges
        for hop in [2, 4, 8]:
            prev_hop = hop // 2
            if prev_hop not in dilated_edges:
                continue

            prev_edge = dilated_edges[prev_hop]
            new_src, new_dst = [], []

            for i in range(prev_edge.size(1)):
                mid = prev_edge[1, i].item()
                src = prev_edge[0, i].item()
                if mid in adj:
                    for dst in adj[mid]:
                        new_src.append(src)
                        new_dst.append(dst)

            if new_src:
                dilated_edges[hop] = torch.tensor(
                    [new_src, new_dst],
                    dtype=torch.long,
                    device=succ_edge.device
                )

        return dilated_edges

    def forward(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: HeteroData batch
            ego_indices: Optional ego agent indices

        Returns:
            [batch_size, num_future_steps, 2] trajectory predictions
        """
        # ============ Encode Nodes ============
        h_agent = self.agent_encoder(data['agent'].x)
        h_lane = self.lane_encoder(data['lane'].x)

        if self.has_tl and 'tl' in data.node_types and data['tl'].x.size(0) > 0:
            h_tl = self.tl_encoder(data['tl'].x)
        else:
            h_tl = None

        # ============ Lane Graph Network ============
        if self.has_l2l:
            typed_l2l = self._get_typed_l2l_edges(data)
            dilated_l2l = self._get_dilated_edges(typed_l2l, h_lane.size(0))
            h_lane = self.lane_gnn(h_lane, typed_l2l, dilated_l2l)
        else:
            for mlp in self.lane_mlps:
                h_lane = mlp(h_lane)

        # ============ Message Passing Layers ============
        # Get edge indices
        a2a_edge_index = data['agent', 'to', 'agent'].edge_index if ('agent', 'to', 'agent') in data.edge_types else None
        a2a_edge_attr = None
        if a2a_edge_index is not None and self.use_edge_attr:
            try:
                a2a_edge_attr = data['agent', 'to', 'agent'].edge_attr
            except (KeyError, AttributeError):
                pass

        for i in range(self.num_layers):
            # A2A convolution
            if a2a_edge_index is not None and a2a_edge_index.numel() > 0:
                h_a2a = self.a2a_convs[i](h_agent, a2a_edge_index, a2a_edge_attr)
                h_agent = self.agent_norms[i](h_agent + self.dropout(h_a2a))

            # Agent-Lane Fusion (every other layer to reduce compute)
            if i % 2 == 0:
                h_agent_new, h_lane_new = self.agent_lane_fusion(
                    h_agent.unsqueeze(0),
                    h_lane.unsqueeze(0),
                )
                h_agent = h_agent_new.squeeze(0)
                h_lane = h_lane_new.squeeze(0)

        # ============ Extract Ego Embeddings ============
        if ego_indices is not None:
            ego_idx = ego_indices
        elif hasattr(data, 'ego_indices_global'):
            ego_idx = data.ego_indices_global
        else:
            ptr = data['agent'].ptr
            ego_local = data.ego_idx_tensor.squeeze(-1)
            ego_idx = ego_local + ptr[:-1]

        ego_embeddings = h_agent[ego_idx]

        # ============ Decode Trajectory ============
        pred_flat = self.decoder(ego_embeddings)
        pred = pred_flat.view(-1, self.num_future_steps, 2)

        return pred
