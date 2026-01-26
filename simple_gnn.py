#!/usr/bin/env python3
"""
simple_gnn.py

Simple heterogeneous GNN for ego trajectory prediction.
Uses PyTorch Geometric's HeteroConv for message passing between agents and lanes.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData


class SimpleHeteroGNN(nn.Module):
    """
    Simple heterogeneous GNN for motion prediction.

    Architecture:
    1. Linear projection for each node type (agent: 66->hidden, lane: 40->hidden)
    2. 2-3 HeteroConv layers with SAGEConv for each edge type
    3. Extract ego node embedding
    4. MLP decoder: hidden -> 80*2 (predict 80 timesteps, x and y)

    Input: HeteroData with 'agent' and 'lane' nodes, A2A and A2L edges
    Output: [batch, 80, 2] predicted future positions
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.agent_in_channels = agent_in_channels
        self.lane_in_channels = lane_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_future_steps = num_future_steps

        # Input projections
        self.agent_proj = Linear(agent_in_channels, hidden_channels)
        self.lane_proj = Linear(lane_in_channels, hidden_channels)

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # Agent-to-agent message passing
                ('agent', 'to', 'agent'): SAGEConv(
                    hidden_channels, hidden_channels, aggr='mean'
                ),
                # Agent-to-lane message passing (agents receive info from lanes)
                ('agent', 'to', 'lane'): SAGEConv(
                    hidden_channels, hidden_channels, aggr='mean'
                ),
                # Lane-to-agent message passing (reverse direction)
                ('lane', 'rev_to', 'agent'): SAGEConv(
                    hidden_channels, hidden_channels, aggr='mean'
                ),
            }, aggr='sum')
            self.convs.append(conv)

        # Layer norms for stability
        self.agent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.lane_norms = nn.ModuleList([
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

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: HeteroData batch with:
                - data['agent'].x: [N_agents_total, agent_in_channels]
                - data['lane'].x: [N_lanes_total, lane_in_channels]
                - data['agent', 'to', 'agent'].edge_index
                - data['agent', 'to', 'lane'].edge_index
                - data.ego_indices_global: [batch_size] global ego indices

        Returns:
            pred: [batch_size, num_future_steps, 2] predicted trajectories
        """
        # Get node features
        x_agent = data['agent'].x
        x_lane = data['lane'].x

        # Input projection
        h_agent = self.agent_proj(x_agent)
        h_lane = self.lane_proj(x_lane)

        # Build x_dict for HeteroConv
        x_dict = {'agent': h_agent, 'lane': h_lane}

        # Build edge_index_dict
        edge_index_dict = {
            ('agent', 'to', 'agent'): data['agent', 'to', 'agent'].edge_index,
            ('agent', 'to', 'lane'): data['agent', 'to', 'lane'].edge_index,
        }

        # Add reverse edges for lane->agent message passing
        a2l_edge_index = data['agent', 'to', 'lane'].edge_index
        l2a_edge_index = torch.stack([a2l_edge_index[1], a2l_edge_index[0]], dim=0)
        edge_index_dict[('lane', 'rev_to', 'agent')] = l2a_edge_index

        # Apply heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)

            # Apply normalization and residual connection
            if 'agent' in x_dict_new:
                x_dict_new['agent'] = self.agent_norms[i](x_dict_new['agent'])
                x_dict_new['agent'] = x_dict['agent'] + self.dropout(x_dict_new['agent'])
            else:
                x_dict_new['agent'] = x_dict['agent']

            if 'lane' in x_dict_new:
                x_dict_new['lane'] = self.lane_norms[i](x_dict_new['lane'])
                x_dict_new['lane'] = x_dict['lane'] + self.dropout(x_dict_new['lane'])
            else:
                x_dict_new['lane'] = x_dict['lane']

            x_dict = x_dict_new

        # Extract ego embeddings
        if hasattr(data, 'ego_indices_global'):
            ego_indices = data.ego_indices_global
        else:
            # Fallback: compute from ptr and ego_idx_tensor
            ptr = data['agent'].ptr
            ego_idx = data.ego_idx_tensor.squeeze(-1)
            ego_indices = ego_idx + ptr[:-1]

        # ego_indices = data.ego_indices_global  # [batch_size]
        ego_embeddings = x_dict['agent'][ego_indices]  # [batch_size, hidden_channels]

        # Decode to trajectory predictions
        pred_flat = self.decoder(ego_embeddings)  # [batch_size, num_future_steps * 2]
        pred = pred_flat.view(-1, self.num_future_steps, 2)  # [batch_size, 80, 2]

        return pred


class SimpleHeteroGNNWithEdgeAttr(nn.Module):
    """
    Heterogeneous GNN with edge attribute handling.

    Similar to SimpleHeteroGNN but uses edge attributes in message passing.
    Uses a simple approach: concatenate edge features to node features before aggregation.
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        a2a_edge_channels: int = 20,
        a2l_edge_channels: int = 9,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_future_steps = num_future_steps

        # Input projections
        self.agent_proj = Linear(agent_in_channels, hidden_channels)
        self.lane_proj = Linear(lane_in_channels, hidden_channels)

        # Edge projections
        self.a2a_edge_proj = Linear(a2a_edge_channels, hidden_channels // 4)
        self.a2l_edge_proj = Linear(a2l_edge_channels, hidden_channels // 4)

        # Message passing layers (simple MLP-based)
        self.agent_update_layers = nn.ModuleList()
        self.lane_update_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Agent update: aggregate from other agents + lanes
            self.agent_update_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2 + hidden_channels // 4, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))
            # Lane update: aggregate from agents
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
        """
        Aggregate messages from source to destination nodes.

        Args:
            src_features: [N_src, hidden] source node features
            edge_index: [2, E] edge indices (src -> dst)
            edge_attr: [E, edge_dim] edge attributes
            edge_proj: Linear layer to project edge attributes
            num_dst_nodes: Number of destination nodes

        Returns:
            aggregated: [N_dst, hidden + hidden//4] aggregated messages
        """
        if edge_index.shape[1] == 0:
            # No edges - return zeros
            return torch.zeros(
                num_dst_nodes,
                src_features.shape[1] + self.hidden_channels // 4,
                device=src_features.device,
                dtype=src_features.dtype
            )

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Get source features and edge features
        src_feats = src_features[src_idx]  # [E, hidden]
        edge_feats = edge_proj(edge_attr)  # [E, hidden//4]

        # Concatenate for message
        messages = torch.cat([src_feats, edge_feats], dim=-1)  # [E, hidden + hidden//4]

        # Scatter mean aggregation
        from torch_scatter import scatter_mean
        aggregated = scatter_mean(messages, dst_idx, dim=0, dim_size=num_dst_nodes)

        return aggregated

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass with edge attributes."""
        # Input projection
        h_agent = self.agent_proj(data['agent'].x)
        h_lane = self.lane_proj(data['lane'].x)

        # Get edge indices and attributes
        a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        a2a_edge_attr = data['agent', 'to', 'agent'].edge_attr
        a2l_edge_index = data['agent', 'to', 'lane'].edge_index
        a2l_edge_attr = data['agent', 'to', 'lane'].edge_attr

        # Reverse edges for lane->agent
        l2a_edge_index = torch.stack([a2l_edge_index[1], a2l_edge_index[0]], dim=0)

        for i in range(len(self.agent_update_layers)):
            # Agent receives from agents (A2A)
            a2a_msg = self.aggregate_messages(
                h_agent, a2a_edge_index, a2a_edge_attr,
                self.a2a_edge_proj, h_agent.shape[0]
            )

            # Agent receives from lanes (L2A, reverse of A2L)
            l2a_msg = self.aggregate_messages(
                h_lane, l2a_edge_index, a2l_edge_attr,
                self.a2l_edge_proj, h_agent.shape[0]
            )

            # Update agent
            agent_input = torch.cat([h_agent, a2a_msg, l2a_msg[:, :self.hidden_channels // 4]], dim=-1)
            # Pad if needed
            if agent_input.shape[1] < self.hidden_channels * 2 + self.hidden_channels // 4:
                pad_size = self.hidden_channels * 2 + self.hidden_channels // 4 - agent_input.shape[1]
                agent_input = torch.cat([
                    agent_input,
                    torch.zeros(agent_input.shape[0], pad_size, device=agent_input.device)
                ], dim=-1)

            h_agent_new = self.agent_update_layers[i](agent_input[:, :self.hidden_channels * 2 + self.hidden_channels // 4])
            h_agent_new = self.agent_norms[i](h_agent_new)
            h_agent = h_agent + self.dropout(h_agent_new)

            # Lane receives from agents (A2L)
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

            h_lane_new = self.lane_update_layers[i](lane_input[:, :self.hidden_channels * 2 + self.hidden_channels // 4])
            h_lane_new = self.lane_norms[i](h_lane_new)
            h_lane = h_lane + self.dropout(h_lane_new)

        # Extract ego embeddings and decode
        ego_indices = data.ego_indices_global
        ego_embeddings = h_agent[ego_indices]

        pred_flat = self.decoder(ego_embeddings)
        pred = pred_flat.view(-1, self.num_future_steps, 2)

        return pred


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing SimpleHeteroGNN...")

    model = SimpleHeteroGNN(
        agent_in_channels=66,
        lane_in_channels=40,
        hidden_channels=128,
        num_layers=3,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dummy batch
    from torch_geometric.data import HeteroData, Batch

    def create_dummy_graph():
        data = HeteroData()
        n_agents = 10
        n_lanes = 50

        data['agent'].x = torch.randn(n_agents, 66)
        data['lane'].x = torch.randn(n_lanes, 40)

        # A2A edges (ego -> others)
        data['agent', 'to', 'agent'].edge_index = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4]
        ], dtype=torch.long)
        data['agent', 'to', 'agent'].edge_attr = torch.randn(4, 20)

        # A2L edges
        n_a2l = n_agents * 3
        src = torch.repeat_interleave(torch.arange(n_agents), 3)
        dst = torch.randint(0, n_lanes, (n_a2l,))
        data['agent', 'to', 'lane'].edge_index = torch.stack([src, dst])
        data['agent', 'to', 'lane'].edge_attr = torch.randn(n_a2l, 9)

        data.ego_idx_tensor = torch.tensor([0])
        data.ego_future_target = torch.randn(80, 2)
        data.ego_future_valid = torch.ones(80)

        return data

    # Create batch of 4 graphs
    graphs = [create_dummy_graph() for _ in range(4)]
    batch = Batch.from_data_list(graphs)

    # Compute ego_indices_global
    ptr = batch['agent'].ptr
    ego_indices = batch.ego_idx_tensor.squeeze(-1)
    batch.ego_indices_global = ego_indices + ptr[:-1]

    print(f"\nBatch statistics:")
    print(f"  Total agent nodes: {batch['agent'].x.shape[0]}")
    print(f"  Total lane nodes: {batch['lane'].x.shape[0]}")
    print(f"  Ego indices (global): {batch.ego_indices_global}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(batch)

    print(f"\nOutput shape: {pred.shape}")  # Should be [4, 80, 2]
    print(f"Output sample (first graph, first 5 steps):\n{pred[0, :5, :]}")

    print("\nSimpleHeteroGNN test passed!")
