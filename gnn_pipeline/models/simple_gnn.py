"""
Simple heterogeneous GNN for ego trajectory prediction.

Uses PyTorch Geometric's HeteroConv for message passing between agents and lanes.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData

from . import register_model
from .base_model import BaseMotionPredictor


@register_model("simple_gnn")
class SimpleHeteroGNN(BaseMotionPredictor):
    """
    Simple heterogeneous GNN for motion prediction.

    Architecture:
    1. Linear projection for each node type
    2. N HeteroConv layers with configurable convolution type
    3. Extract ego node embedding
    4. MLP decoder: hidden -> num_future_steps * 2

    Args:
        agent_in_channels: Number of agent input features
        lane_in_channels: Number of lane input features
        tl_in_channels: Number of traffic light input features
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        num_future_steps: Number of future timesteps to predict
        dropout: Dropout rate
        conv_type: Convolution type ("sage" or "gat")
        edge_types: List of edge types to use in message passing
        use_edge_attr: Whether to use edge attributes (not implemented for SAGEConv)
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        tl_in_channels: int = 12,
        goal_in_channels: int = 10,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
        conv_type: str = "sage",
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_edge_attr: bool = False,
        **kwargs,  # Accept and ignore extra params (e.g., temporal/polyline encoder settings)
    ):
        super().__init__(num_future_steps=num_future_steps)

        self.agent_in_channels = agent_in_channels
        self.lane_in_channels = lane_in_channels
        self.tl_in_channels = tl_in_channels
        self.goal_in_channels = goal_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.use_edge_attr = use_edge_attr

        # Default edge types (baseline)
        if edge_types is None:
            edge_types = [
                ("agent", "to", "agent"),
                ("agent", "to", "lane"),
                ("lane", "rev_to", "agent"),
            ]
        self.edge_types = edge_types

        # Check which node types are needed
        self.has_tl = any("tl" in et for et in edge_types)
        self.has_goal = any("goal" in et for et in edge_types)

        # Input projections
        self.agent_proj = Linear(agent_in_channels, hidden_channels)
        self.lane_proj = Linear(lane_in_channels, hidden_channels)
        if self.has_tl:
            self.tl_proj = Linear(tl_in_channels, hidden_channels)
        if self.has_goal:
            self.goal_proj = Linear(goal_in_channels, hidden_channels)

        # Build heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = self._build_conv_dict(hidden_channels, conv_type)
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Layer norms for stability
        self.agent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.lane_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        if self.has_tl:
            self.tl_norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels) for _ in range(num_layers)
            ])
        if self.has_goal:
            self.goal_norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels) for _ in range(num_layers)
            ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # MLP decoder for trajectory prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_future_steps * 2),
        )

    def _build_conv_dict(
        self,
        hidden_channels: int,
        conv_type: str,
    ) -> dict:
        """Build convolution dictionary for HeteroConv."""
        conv_dict = {}

        for edge_type in self.edge_types:
            if conv_type == "sage":
                conv = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            elif conv_type == "gat":
                conv = GATConv(
                    hidden_channels,
                    hidden_channels // 4,
                    heads=4,
                    concat=True,
                    dropout=0.1,
                    add_self_loops=False, 
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            conv_dict[edge_type] = conv

        return conv_dict

    def _get_edge_index_dict(self, data: HeteroData) -> dict:
        """Build edge_index_dict from data based on configured edge types."""
        edge_index_dict = {}

        for edge_type in self.edge_types:
            src, rel, dst = edge_type

            # Handle reverse edges
            if rel == "rev_to":
                # Reverse of agent->lane
                if src == "lane" and dst == "agent":
                    orig_edge = data['agent', 'to', 'lane'].edge_index
                    edge_index_dict[edge_type] = torch.stack(
                        [orig_edge[1], orig_edge[0]], dim=0
                    )
                # Reverse of agent->tl
                elif src == "tl" and dst == "agent":
                    if hasattr(data, ('agent', 'to', 'tl')):
                        orig_edge = data['agent', 'to', 'tl'].edge_index
                        edge_index_dict[edge_type] = torch.stack(
                            [orig_edge[1], orig_edge[0]], dim=0
                        )
            else:
                # Regular edge - access from data
                try:
                    edge_index_dict[edge_type] = data[src, rel, dst].edge_index
                except KeyError:
                    # Edge type not in data, skip
                    pass

        return edge_index_dict

    def forward(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: HeteroData batch with node features and edge indices
            ego_indices: Optional tensor of ego agent indices

        Returns:
            Predicted trajectories of shape [batch_size, num_future_steps, 2]
        """
        # Get node features
        x_agent = data['agent'].x
        x_lane = data['lane'].x

        # Input projection
        h_agent = self.agent_proj(x_agent)
        h_lane = self.lane_proj(x_lane)

        # Build x_dict
        x_dict = {'agent': h_agent, 'lane': h_lane}

        # Handle traffic lights if present
        if self.has_tl and 'tl' in data.node_types and data['tl'].x.shape[0] > 0:
            x_dict['tl'] = self.tl_proj(data['tl'].x)

        # Handle goal nodes if present
        if self.has_goal and 'goal' in data.node_types and data['goal'].x.shape[0] > 0:
            x_dict['goal'] = self.goal_proj(data['goal'].x)

        # Build edge_index_dict
        edge_index_dict = self._get_edge_index_dict(data)

        # Apply heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            # Filter edge_index_dict to only include edges we have
            valid_edges = {k: v for k, v in edge_index_dict.items() if k in conv.convs}
            
            if not valid_edges:
                continue
                
            x_dict_new = conv(x_dict, valid_edges)

            # Apply normalization and residual connection for agents
            if 'agent' in x_dict_new:
                x_dict_new['agent'] = self.agent_norms[i](x_dict_new['agent'])
                x_dict_new['agent'] = x_dict['agent'] + self.dropout(x_dict_new['agent'])
            else:
                x_dict_new['agent'] = x_dict['agent']

            # Apply normalization and residual connection for lanes
            if 'lane' in x_dict_new:
                x_dict_new['lane'] = self.lane_norms[i](x_dict_new['lane'])
                x_dict_new['lane'] = x_dict['lane'] + self.dropout(x_dict_new['lane'])
            else:
                x_dict_new['lane'] = x_dict['lane']

            # Apply normalization and residual connection for TLs
            if self.has_tl and 'tl' in x_dict_new:
                x_dict_new['tl'] = self.tl_norms[i](x_dict_new['tl'])
                x_dict_new['tl'] = x_dict['tl'] + self.dropout(x_dict_new['tl'])
            elif self.has_tl and 'tl' in x_dict:
                x_dict_new['tl'] = x_dict['tl']

            # Apply normalization/residual for goal
            if self.has_goal and 'goal' in x_dict_new:
                x_dict_new['goal'] = self.goal_norms[i](x_dict_new['goal'])
                x_dict_new['goal'] = x_dict['goal'] + self.dropout(x_dict_new['goal'])
            elif self.has_goal and 'goal' in x_dict:
                x_dict_new['goal'] = x_dict['goal']

            x_dict = x_dict_new

        # Extract ego embeddings
        if ego_indices is not None:
            ego_idx = ego_indices
        elif hasattr(data, 'ego_indices_global'):
            ego_idx = data.ego_indices_global
        else:
            # Fallback: compute from ptr and ego_idx_tensor
            ptr = data['agent'].ptr
            ego_local = data.ego_idx_tensor.squeeze(-1)
            ego_idx = ego_local + ptr[:-1]

        ego_embeddings = x_dict['agent'][ego_idx]  # [batch_size, hidden_channels]

        # Decode to trajectory predictions
        pred_flat = self.decoder(ego_embeddings)  # [batch_size, num_future_steps * 2]
        pred = pred_flat.view(-1, self.num_future_steps, 2)  # [batch_size, 80, 2]

        return pred


# Keep the edge-attr variant as well for future use
class SimpleHeteroGNNWithEdgeAttr(BaseMotionPredictor):
    """
    Heterogeneous GNN with edge attribute handling.

    Similar to SimpleHeteroGNN but uses edge attributes in message passing.
    This is a more advanced variant that can leverage edge features.
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        tl_in_channels: int = 12,
        a2a_edge_channels: int = 20,
        a2l_edge_channels: int = 9,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_future_steps=num_future_steps)

        self.hidden_channels = hidden_channels

        # Input projections
        self.agent_proj = Linear(agent_in_channels, hidden_channels)
        self.lane_proj = Linear(lane_in_channels, hidden_channels)

        # Edge projections
        self.a2a_edge_proj = Linear(a2a_edge_channels, hidden_channels // 4)
        self.a2l_edge_proj = Linear(a2l_edge_channels, hidden_channels // 4)

        # Message passing layers
        self.agent_update_layers = nn.ModuleList()
        self.lane_update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.agent_update_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2 + hidden_channels // 4, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))
            self.lane_update_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2 + hidden_channels // 4, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))

        # Layer norms
        self.agent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.lane_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_future_steps * 2),
        )

    def aggregate_messages(
        self,
        src_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_proj: nn.Module,
        num_dst_nodes: int,
    ) -> torch.Tensor:
        """Aggregate messages from source to destination nodes."""
        if edge_index.shape[1] == 0:
            return torch.zeros(
                num_dst_nodes,
                src_features.shape[1] + self.hidden_channels // 4,
                device=src_features.device,
                dtype=src_features.dtype
            )

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_feats = src_features[src_idx]
        edge_feats = edge_proj(edge_attr)

        messages = torch.cat([src_feats, edge_feats], dim=-1)

        from torch_scatter import scatter_mean
        aggregated = scatter_mean(messages, dst_idx, dim=0, dim_size=num_dst_nodes)

        return aggregated

    def forward(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with edge attributes."""
        h_agent = self.agent_proj(data['agent'].x)
        h_lane = self.lane_proj(data['lane'].x)

        a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        a2a_edge_attr = data['agent', 'to', 'agent'].edge_attr
        a2l_edge_index = data['agent', 'to', 'lane'].edge_index
        a2l_edge_attr = data['agent', 'to', 'lane'].edge_attr

        l2a_edge_index = torch.stack([a2l_edge_index[1], a2l_edge_index[0]], dim=0)

        for i in range(len(self.agent_update_layers)):
            # Agent receives from agents
            a2a_msg = self.aggregate_messages(
                h_agent, a2a_edge_index, a2a_edge_attr,
                self.a2a_edge_proj, h_agent.shape[0]
            )

            # Agent receives from lanes
            l2a_msg = self.aggregate_messages(
                h_lane, l2a_edge_index, a2l_edge_attr,
                self.a2l_edge_proj, h_agent.shape[0]
            )

            # Update agent
            agent_input = torch.cat([
                h_agent,
                a2a_msg,
                l2a_msg[:, :self.hidden_channels // 4]
            ], dim=-1)

            if agent_input.shape[1] < self.hidden_channels * 2 + self.hidden_channels // 4:
                pad_size = self.hidden_channels * 2 + self.hidden_channels // 4 - agent_input.shape[1]
                agent_input = torch.cat([
                    agent_input,
                    torch.zeros(agent_input.shape[0], pad_size, device=agent_input.device)
                ], dim=-1)

            h_agent_new = self.agent_update_layers[i](
                agent_input[:, :self.hidden_channels * 2 + self.hidden_channels // 4]
            )
            h_agent_new = self.agent_norms[i](h_agent_new)
            h_agent = h_agent + self.dropout(h_agent_new)

            # Lane receives from agents
            a2l_msg = self.aggregate_messages(
                h_agent, a2l_edge_index, a2l_edge_attr,
                self.a2l_edge_proj, h_lane.shape[0]
            )

            lane_input = torch.cat([h_lane, a2l_msg], dim=-1)
            if lane_input.shape[1] < self.hidden_channels * 2 + self.hidden_channels // 4:
                pad_size = self.hidden_channels * 2 + self.hidden_channels // 4 - lane_input.shape[1]
                lane_input = torch.cat([
                    lane_input,
                    torch.zeros(lane_input.shape[0], pad_size, device=lane_input.device)
                ], dim=-1)

            h_lane_new = self.lane_update_layers[i](
                lane_input[:, :self.hidden_channels * 2 + self.hidden_channels // 4]
            )
            h_lane_new = self.lane_norms[i](h_lane_new)
            h_lane = h_lane + self.dropout(h_lane_new)

        # Extract ego embeddings
        if ego_indices is not None:
            ego_idx = ego_indices
        elif hasattr(data, 'ego_indices_global'):
            ego_idx = data.ego_indices_global
        else:
            ptr = data['agent'].ptr
            ego_local = data.ego_idx_tensor.squeeze(-1)
            ego_idx = ego_local + ptr[:-1]

        ego_embeddings = h_agent[ego_idx]

        pred_flat = self.decoder(ego_embeddings)
        pred = pred_flat.view(-1, self.num_future_steps, 2)

        return pred
