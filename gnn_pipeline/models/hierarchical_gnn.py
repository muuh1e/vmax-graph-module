"""
HierarchicalGNN: HiVT-inspired hierarchical encoder for motion prediction.

Key innovations over SimpleHeteroGNN:
1. Local encoding with temporal transformers (agent history) and polyline encoders (lanes)
2. Local interaction: k-nearest neighbor attention for nearby entities
3. Global interaction: sparse attention for scene-level reasoning
4. Ego-centric aggregation: ego attends to all relevant context
5. Multi-modal decoding: predicts K possible futures with confidences
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from . import register_model
from .base_model import BaseMotionPredictor
from .layers.temporal_encoder import TemporalTransformerEncoder, TemporalCNNEncoder
from .layers.attention import CrossAttention, SparseAttention, LocalGlobalAttention
from .layers.multi_modal_decoder import MultiModalDecoder


class LocalInteractionModule(nn.Module):
    """
    Local interaction via k-nearest neighbor attention.

    Each agent attends to its k nearest agents and lanes.
    Uses relative position encoding for translation invariance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        k_agents: int = 10,
        k_lanes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.k_agents = k_agents
        self.k_lanes = k_lanes

        # Relative position encoder
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(4, embed_dim // 4),  # dx, dy, dist, angle
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

        # Agent-agent local attention
        self.agent_local_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Agent-lane local attention
        self.lane_local_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def _get_k_nearest(
        self,
        query_pos: torch.Tensor,  # [N, 2]
        key_pos: torch.Tensor,    # [M, 2]
        k: int,
        exclude_self: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get k nearest neighbors.

        Returns:
            indices: [N, k] indices of nearest neighbors
            distances: [N, k] distances
            rel_pos: [N, k, 4] relative position features
        """
        N = query_pos.size(0)
        M = key_pos.size(0)
        k = min(k, M - (1 if exclude_self else 0))

        if k <= 0:
            return (
                torch.zeros(N, 0, dtype=torch.long, device=query_pos.device),
                torch.zeros(N, 0, device=query_pos.device),
                torch.zeros(N, 0, 4, device=query_pos.device),
            )

        # Pairwise distances
        dist = torch.cdist(query_pos.float(), key_pos.float())  # [N, M]

        if exclude_self and N == M:
            dist = dist + torch.eye(N, device=dist.device) * 1e6

        # Top-k nearest
        distances, indices = torch.topk(dist, k, dim=-1, largest=False)

        # Compute relative positions
        query_expanded = query_pos.unsqueeze(1).expand(-1, k, -1)  # [N, k, 2]
        key_gathered = key_pos[indices]  # [N, k, 2]

        dx = key_gathered[:, :, 0] - query_expanded[:, :, 0]
        dy = key_gathered[:, :, 1] - query_expanded[:, :, 1]
        dist_feat = distances
        angle = torch.atan2(dy, dx)

        rel_pos = torch.stack([dx, dy, dist_feat, angle], dim=-1)

        return indices, distances, rel_pos

    def forward(
        self,
        agent_embeds: torch.Tensor,  # [N_agents, embed_dim]
        lane_embeds: torch.Tensor,   # [N_lanes, embed_dim]
        agent_pos: torch.Tensor,     # [N_agents, 2]
        lane_pos: torch.Tensor,      # [N_lanes, 2]
    ) -> torch.Tensor:
        """
        Args:
            agent_embeds: Agent embeddings
            lane_embeds: Lane embeddings
            agent_pos: Agent positions
            lane_pos: Lane centroid positions

        Returns:
            Updated agent embeddings [N_agents, embed_dim]
        """
        N = agent_embeds.size(0)
        embed_dim = agent_embeds.size(1)

        # Get k nearest agents
        agent_indices, _, agent_rel_pos = self._get_k_nearest(
            agent_pos, agent_pos, self.k_agents, exclude_self=True
        )
        k_a = agent_indices.size(1)

        # Get k nearest lanes
        lane_indices, _, lane_rel_pos = self._get_k_nearest(
            agent_pos, lane_pos, self.k_lanes, exclude_self=False
        )
        k_l = lane_indices.size(1)

        # Agent-agent local attention
        if k_a > 0:
            agent_neighbors = agent_embeds[agent_indices]  # [N, k_a, embed_dim]
            agent_rel_embed = self.rel_pos_encoder(agent_rel_pos)  # [N, k_a, embed_dim]
            agent_neighbors = agent_neighbors + agent_rel_embed

            agent_query = agent_embeds.unsqueeze(1)  # [N, 1, embed_dim]
            agent_context, _ = self.agent_local_attn(
                agent_query, agent_neighbors, agent_neighbors
            )
            agent_context = agent_context.squeeze(1)  # [N, embed_dim]
        else:
            agent_context = torch.zeros_like(agent_embeds)

        # Agent-lane local attention
        if k_l > 0:
            lane_neighbors = lane_embeds[lane_indices]  # [N, k_l, embed_dim]
            lane_rel_embed = self.rel_pos_encoder(lane_rel_pos)
            lane_neighbors = lane_neighbors + lane_rel_embed

            lane_query = agent_embeds.unsqueeze(1)
            lane_context, _ = self.lane_local_attn(
                lane_query, lane_neighbors, lane_neighbors
            )
            lane_context = lane_context.squeeze(1)
        else:
            lane_context = torch.zeros_like(agent_embeds)

        # Fuse contexts
        fused = self.fusion(torch.cat([agent_embeds, agent_context, lane_context], dim=-1))

        return self.norm(agent_embeds + fused)


class GlobalInteractionModule(nn.Module):
    """
    Global interaction via sparse attention over all entities.

    Allows agents to reason about distant but relevant entities
    (e.g., vehicles at an upcoming intersection).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        global_k: int = 20,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Agent global self-attention
        self.agent_self_attn = nn.ModuleList([
            SparseAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                top_k=global_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Cross-modal attention (agents ↔ lanes)
        self.agent_to_lane = CrossAttention(
            query_dim=embed_dim,
            key_dim=embed_dim,
            hidden_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.lane_to_agent = CrossAttention(
            query_dim=embed_dim,
            key_dim=embed_dim,
            hidden_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        agent_embeds: torch.Tensor,  # [N_agents, embed_dim]
        lane_embeds: torch.Tensor,   # [N_lanes, embed_dim]
        agent_pos: torch.Tensor,     # [N_agents, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_embeds: Agent embeddings
            lane_embeds: Lane embeddings
            agent_pos: Agent positions for sparse attention

        Returns:
            Updated agent and lane embeddings
        """
        # Add batch dimension
        agent_embeds_b = agent_embeds.unsqueeze(0)
        lane_embeds_b = lane_embeds.unsqueeze(0)
        agent_pos_b = agent_pos.unsqueeze(0)

        # Agent self-attention
        for attn in self.agent_self_attn:
            agent_embeds_b = attn(agent_embeds_b, agent_pos_b)

        # Cross-modal attention
        agent_embeds_b = self.agent_to_lane(agent_embeds_b, lane_embeds_b)
        lane_embeds_b = self.lane_to_agent(lane_embeds_b, agent_embeds_b)

        return agent_embeds_b.squeeze(0), lane_embeds_b.squeeze(0)


class EgoAggregationModule(nn.Module):
    """
    Ego-centric aggregation: ego attends to all relevant context.

    The ego agent gets special treatment with dedicated attention
    to extract driving-relevant features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Ego queries attend to agents
        self.ego_agent_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Ego queries attend to lanes
        self.ego_lane_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ego_embed: torch.Tensor,      # [batch, embed_dim]
        agent_embeds: torch.Tensor,   # [N_agents, embed_dim]
        lane_embeds: torch.Tensor,    # [N_lanes, embed_dim]
        ego_indices: torch.Tensor,    # [batch] indices into agent_embeds
        agent_batch: torch.Tensor,    # [N_agents] batch assignment
        lane_batch: torch.Tensor,     # [N_lanes] batch assignment
    ) -> torch.Tensor:
        """
        Args:
            ego_embed: Ego embeddings
            agent_embeds: All agent embeddings
            lane_embeds: All lane embeddings
            ego_indices: Global indices of ego agents
            agent_batch: Batch assignment for agents
            lane_batch: Batch assignment for lanes

        Returns:
            Ego context embeddings [batch, embed_dim]
        """
        batch_size = ego_embed.size(0)
        embed_dim = ego_embed.size(1)
        device = ego_embed.device

        ego_contexts = []

        for b in range(batch_size):
            # Get agents and lanes for this batch
            agent_mask = agent_batch == b
            lane_mask = lane_batch == b

            agents_b = agent_embeds[agent_mask]  # [N_b, embed_dim]
            lanes_b = lane_embeds[lane_mask]     # [M_b, embed_dim]
            ego_b = ego_embed[b:b+1]             # [1, embed_dim]

            # Ego attends to agents
            if agents_b.size(0) > 0:
                agent_ctx, _ = self.ego_agent_attn(
                    ego_b.unsqueeze(1),
                    agents_b.unsqueeze(0),
                    agents_b.unsqueeze(0),
                )
                agent_ctx = agent_ctx.squeeze(1)
            else:
                agent_ctx = torch.zeros_like(ego_b)

            # Ego attends to lanes
            if lanes_b.size(0) > 0:
                lane_ctx, _ = self.ego_lane_attn(
                    ego_b.unsqueeze(1),
                    lanes_b.unsqueeze(0),
                    lanes_b.unsqueeze(0),
                )
                lane_ctx = lane_ctx.squeeze(1)
            else:
                lane_ctx = torch.zeros_like(ego_b)

            # Fuse
            fused = self.fusion(torch.cat([ego_b, agent_ctx, lane_ctx], dim=-1))
            ego_contexts.append(self.norm(ego_b + fused))

        return torch.cat(ego_contexts, dim=0)


@register_model("hierarchical_gnn")
class HierarchicalGNN(BaseMotionPredictor):
    """
    HiVT-inspired hierarchical GNN for motion prediction.

    Architecture:
    1. Local Encoding: Temporal transformer (agents), Polyline encoder (lanes)
    2. Local Interaction: K-nearest neighbor attention
    3. Global Interaction: Sparse self-attention + cross-modal attention
    4. Ego Aggregation: Ego-centric context extraction
    5. Trajectory Decoding: Multi-modal MLP decoder

    Args:
        agent_in_channels: Agent input features (per timestep)
        lane_in_channels: Lane input features
        tl_in_channels: Traffic light features
        hidden_channels: Hidden embedding dimension
        num_local_layers: Local encoder transformer layers
        num_global_layers: Global attention layers
        num_heads: Attention heads
        local_k_agents: K nearest agents for local interaction
        local_k_lanes: K nearest lanes for local interaction
        global_k: K for sparse global attention
        num_modes: Number of trajectory modes for multi-modal prediction
        num_future_steps: Prediction horizon
        dropout: Dropout rate
        temporal_encoder_type: "transformer" or "cnn"
        use_multi_modal: Whether to use multi-modal prediction
    """

    def __init__(
        self,
        agent_in_channels: int = 66,
        lane_in_channels: int = 40,
        tl_in_channels: int = 12,
        hidden_channels: int = 128,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        num_heads: int = 4,
        local_k_agents: int = 10,
        local_k_lanes: int = 5,
        global_k: int = 20,
        num_modes: int = 6,
        num_future_steps: int = 80,
        dropout: float = 0.1,
        # Additional args for compatibility
        num_layers: int = 3,  # Ignored, uses local/global layers
        conv_type: str = "sage",  # Ignored
        edge_types: Optional[List[Tuple[str, str, str]]] = None,  # Ignored
        use_edge_attr: bool = False,  # Ignored
        temporal_encoder_type: str = "transformer",
        use_multi_modal: bool = True,
        **kwargs,  # Accept and ignore extra params (e.g., polyline encoder settings)
    ):
        super().__init__(num_future_steps=num_future_steps)

        self.hidden_channels = hidden_channels
        self.num_modes = num_modes
        self.use_multi_modal = use_multi_modal

        # ============ Local Encoders ============
        # Agent encoder (handles full feature vector, may include history)
        if temporal_encoder_type == "transformer":
            # If agent features include temporal history, use full encoder
            # Otherwise just project
            self.agent_local_encoder = nn.Sequential(
                nn.Linear(agent_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.agent_local_encoder = nn.Sequential(
                nn.Linear(agent_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )

        # Lane encoder (lanes come pre-encoded, not as polylines)
        self.lane_local_encoder = nn.Sequential(
            nn.Linear(lane_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # TL encoder
        self.tl_encoder = nn.Sequential(
            nn.Linear(tl_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ============ Local Interaction ============
        self.local_interaction = LocalInteractionModule(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            k_agents=local_k_agents,
            k_lanes=local_k_lanes,
            dropout=dropout,
        )

        # ============ Global Interaction ============
        self.global_interaction = GlobalInteractionModule(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            global_k=global_k,
            num_layers=num_global_layers,
            dropout=dropout,
        )

        # ============ Ego Aggregation ============
        self.ego_aggregation = EgoAggregationModule(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ============ Decoder ============
        if use_multi_modal:
            self.decoder = MultiModalDecoder(
                in_channels=hidden_channels,
                num_modes=num_modes,
                num_future_steps=num_future_steps,
                hidden_channels=hidden_channels * 2,
                dropout=dropout,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, num_future_steps * 2),
            )

        # Store multi-modal outputs for loss computation
        self.last_confidences = None

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
            (best mode if multi-modal)
        """
        # ============ Extract Data ============
        agent_x = data['agent'].x
        lane_x = data['lane'].x

        # Get positions (first 2 features are x, y)
        agent_pos = agent_x[:, :2]
        lane_pos = lane_x[:, :2]  # Centroid

        # Get batch assignments
        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch

        # Get ego indices
        if ego_indices is not None:
            ego_idx = ego_indices
        elif hasattr(data, 'ego_indices_global'):
            ego_idx = data.ego_indices_global
        else:
            ptr = data['agent'].ptr
            ego_local = data.ego_idx_tensor.squeeze(-1)
            ego_idx = ego_local + ptr[:-1]

        batch_size = ego_idx.size(0)

        # ============ Local Encoding ============
        h_agent = self.agent_local_encoder(agent_x)
        h_lane = self.lane_local_encoder(lane_x)

        # ============ Local Interaction ============
        h_agent = self.local_interaction(
            h_agent, h_lane, agent_pos, lane_pos
        )

        # ============ Global Interaction ============
        h_agent, h_lane = self.global_interaction(
            h_agent, h_lane, agent_pos
        )

        # ============ Ego Aggregation ============
        ego_embed = h_agent[ego_idx]  # [batch, hidden]
        ego_context = self.ego_aggregation(
            ego_embed, h_agent, h_lane, ego_idx, agent_batch, lane_batch
        )

        # ============ Decode ============
        if self.use_multi_modal:
            trajectories, confidences = self.decoder(ego_context)
            # trajectories: [batch, num_modes, T, 2]
            # confidences: [batch, num_modes]

            self.last_confidences = confidences

            # Return best mode
            best_idx = confidences.argmax(dim=-1)  # [batch]
            batch_indices = torch.arange(batch_size, device=ego_context.device)
            pred = trajectories[batch_indices, best_idx]  # [batch, T, 2]
        else:
            pred_flat = self.decoder(ego_context)
            pred = pred_flat.view(batch_size, self.num_future_steps, 2)

        return pred

    def forward_multi_modal(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning all modes.

        Returns:
            trajectories: [batch, num_modes, num_future_steps, 2]
            confidences: [batch, num_modes]
        """
        if not self.use_multi_modal:
            pred = self.forward(data, ego_indices)
            batch_size = pred.size(0)
            trajectories = pred.unsqueeze(1)  # [batch, 1, T, 2]
            confidences = torch.ones(batch_size, 1, device=pred.device)
            return trajectories, confidences

        # Full forward pass
        agent_x = data['agent'].x
        lane_x = data['lane'].x
        agent_pos = agent_x[:, :2]
        lane_pos = lane_x[:, :2]
        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch

        if ego_indices is not None:
            ego_idx = ego_indices
        elif hasattr(data, 'ego_indices_global'):
            ego_idx = data.ego_indices_global
        else:
            ptr = data['agent'].ptr
            ego_local = data.ego_idx_tensor.squeeze(-1)
            ego_idx = ego_local + ptr[:-1]

        h_agent = self.agent_local_encoder(agent_x)
        h_lane = self.lane_local_encoder(lane_x)
        h_agent = self.local_interaction(h_agent, h_lane, agent_pos, lane_pos)
        h_agent, h_lane = self.global_interaction(h_agent, h_lane, agent_pos)
        ego_embed = h_agent[ego_idx]
        ego_context = self.ego_aggregation(
            ego_embed, h_agent, h_lane, ego_idx, agent_batch, lane_batch
        )

        trajectories, confidences = self.decoder(ego_context)
        return trajectories, confidences
