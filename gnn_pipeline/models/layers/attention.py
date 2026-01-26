"""Attention mechanisms for GNN architectures."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with optional relative position encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_relative_distance: int = 50,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_relative_pos = use_relative_pos

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Relative position bias
        if use_relative_pos:
            num_positions = 2 * max_relative_distance + 1
            self.relative_pos_bias = nn.Embedding(num_positions, num_heads)
            self.max_relative_distance = max_relative_distance

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, query_len, embed_dim]
            key: [batch, key_len, embed_dim]
            value: [batch, key_len, embed_dim]
            mask: [batch, query_len, key_len] or [batch, key_len], True = attend
            relative_positions: [query_len, key_len] integer distances

        Returns:
            output: [batch, query_len, embed_dim]
            attention_weights: [batch, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Add relative position bias if used
        if self.use_relative_pos and relative_positions is not None:
            # Clamp to valid range
            rel_pos_clipped = relative_positions.clamp(
                -self.max_relative_distance, self.max_relative_distance
            ) + self.max_relative_distance  # Shift to [0, 2*max]
            bias = self.relative_pos_bias(rel_pos_clipped)  # [query_len, key_len, num_heads]
            bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, query_len, key_len]
            scores = scores + bias

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                # [batch, key_len] -> expand to [batch, 1, 1, key_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [batch, query_len, key_len] -> [batch, 1, query_len, key_len]
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights


class CrossAttention(nn.Module):
    """
    Cross-attention layer for attending between two different sets of entities.

    Example: Agents attending to lanes, or ego attending to all context.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key/value vectors
            hidden_dim: Internal dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
            residual: Whether to use residual connection
        """
        super().__init__()

        self.residual = residual
        self.hidden_dim = hidden_dim

        # Project query and key to same dimension if different
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.kv_proj = nn.Linear(key_dim, hidden_dim) if key_dim != hidden_dim else nn.Identity()

        # Attention
        self.attention = MultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        # Layer norm
        self.norm = nn.LayerNorm(query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, query_len, query_dim] - entities doing the attending
            context: [batch, context_len, key_dim] - entities being attended to
            context_mask: [batch, context_len] - True = valid context element

        Returns:
            Updated query [batch, query_len, query_dim]
        """
        # Project
        q = self.q_proj(query)
        kv = self.kv_proj(context)

        # Attend
        attended, _ = self.attention(q, kv, kv, mask=context_mask)

        # Project back
        attended = self.out_proj(attended)

        # Residual and norm
        if self.residual:
            out = self.norm(query + self.dropout(attended))
        else:
            out = self.norm(self.dropout(attended))

        return out


class SparseAttention(nn.Module):
    """
    Sparse attention that only attends to top-k most relevant entities.

    Reduces complexity from O(N^2) to O(N*k) for large scenes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        top_k: int = 20,
        dropout: float = 0.1,
        use_distance_bias: bool = True,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            top_k: Number of neighbors to attend to per query
            dropout: Dropout rate
            use_distance_bias: Whether to use distance-based attention bias
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.use_distance_bias = use_distance_bias

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Distance embedding for bias
        if use_distance_bias:
            self.distance_embedding = nn.Sequential(
                nn.Linear(1, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, num_heads),
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _get_top_k_indices(
        self,
        positions: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k nearest neighbors for each entity.

        Args:
            positions: [batch, N, 2] entity positions

        Returns:
            indices: [batch, N, k] indices of k-nearest neighbors
            distances: [batch, N, k] distances to neighbors
        """
        batch_size, N, _ = positions.shape
        k = min(k, N)

        # Compute pairwise distances
        # [batch, N, N]
        dist_matrix = torch.cdist(positions, positions)

        # Set self-distance to inf
        diag_mask = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
        dist_matrix = dist_matrix.masked_fill(diag_mask, float('inf'))

        # Get top-k nearest (smallest distances)
        distances, indices = torch.topk(dist_matrix, k, dim=-1, largest=False)

        return indices, distances

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, N, embed_dim] entity embeddings
            positions: [batch, N, 2] entity positions for sparse selection
            mask: [batch, N] valid entities

        Returns:
            Updated embeddings [batch, N, embed_dim]
        """
        batch_size, N, _ = x.shape

        if positions is None or N <= self.top_k + 1:
            # Fall back to full attention if no positions or small N
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            out, _ = self.attention(q, k, v, mask=mask)
            return self.norm(x + self.dropout(out))

        # Get top-k neighbors
        k = min(self.top_k, N - 1)
        neighbor_indices, neighbor_distances = self._get_top_k_indices(positions, k)

        # Gather neighbor embeddings
        # neighbor_indices: [batch, N, k]
        neighbor_indices_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        x_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)

        # Use gather to get neighbors
        neighbor_embeds = torch.gather(
            x.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=neighbor_indices_expanded,
        )  # [batch, N, k, embed_dim]

        # Attention query is each entity, keys/values are neighbors
        q = self.q_proj(x).unsqueeze(2)  # [batch, N, 1, embed_dim]
        k = self.k_proj(neighbor_embeds)  # [batch, N, k, embed_dim]
        v = self.v_proj(neighbor_embeds)  # [batch, N, k, embed_dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        # [batch, N, 1, k]

        # Add distance bias
        if self.use_distance_bias:
            dist_bias = self.distance_embedding(neighbor_distances.unsqueeze(-1))
            # [batch, N, k, num_heads] -> [batch, N, num_heads, k] -> mean over heads
            dist_bias = dist_bias.mean(dim=-1).unsqueeze(2)  # [batch, N, 1, k]
            scores = scores - dist_bias * 0.1  # Closer = higher attention

        # Apply neighbor mask if needed
        if mask is not None:
            neighbor_mask = torch.gather(
                mask.unsqueeze(1).expand(-1, N, -1),
                dim=2,
                index=neighbor_indices,
            )  # [batch, N, k]
            scores = scores.masked_fill(~neighbor_mask.unsqueeze(2), float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, v).squeeze(2)  # [batch, N, embed_dim]
        out = self.out_proj(out)

        return self.norm(x + self.dropout(out))


class LocalGlobalAttention(nn.Module):
    """
    Two-stage attention: local (k-nearest) then global (full but sparse).

    Combines efficiency of local attention with global context.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        local_k: int = 10,
        global_k: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.local_attn = SparseAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            top_k=local_k,
            dropout=dropout,
        )

        self.global_attn = SparseAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            top_k=global_k,
            dropout=dropout,
            use_distance_bias=False,  # Global uses learned attention
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, N, embed_dim]
            positions: [batch, N, 2]
            mask: [batch, N]

        Returns:
            [batch, N, embed_dim]
        """
        local_out = self.local_attn(x, positions, mask)
        global_out = self.global_attn(x, positions, mask)

        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))
        return self.norm(x + fused)
