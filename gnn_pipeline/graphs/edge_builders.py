"""
Edge building functions for heterogeneous graph construction.

Provides functions for building filtered and typed edges:
- A2A (agent-to-agent) with various filtering modes
- L2L (lane-to-lane) with typed semantic edges
- TL (traffic light) filtering
- Lane filtering
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs import GraphConfig


# =============================================================================
# A2A Edge Builders
# =============================================================================

def build_a2a_edges_ego_only(
    agent_positions: torch.Tensor,  # [N, 2]
    ego_idx: int,
    max_distance: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Original baseline: only ego -> other agent edges.
    
    Args:
        agent_positions: [N, 2] tensor of agent positions
        ego_idx: Index of ego agent
        max_distance: Maximum distance for edges
        
    Returns:
        (src, dst) tensors for edge index
    """
    N = agent_positions.shape[0]
    src_list, dst_list = [], []
    
    ego_pos = agent_positions[ego_idx]
    for i in range(N):
        if i != ego_idx:
            dist = torch.norm(agent_positions[i] - ego_pos)
            if dist < max_distance:
                src_list.append(ego_idx)
                dst_list.append(i)
    
    if len(src_list) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


def build_a2a_edges_k_nearest(
    agent_positions: torch.Tensor,  # [N, 2]
    ego_idx: int,
    k: int = 5,
    max_distance: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each agent connects to k nearest other agents.
    
    Args:
        agent_positions: [N, 2] tensor of agent positions
        ego_idx: Index of ego agent (unused but kept for API consistency)
        k: Number of nearest neighbors per agent
        max_distance: Maximum distance for edges
        
    Returns:
        (src, dst) tensors for edge index
    """
    N = agent_positions.shape[0]
    if N < 2:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    src_list, dst_list = [], []
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(agent_positions.float(), agent_positions.float())
    
    for i in range(N):
        # Get k nearest neighbors (excluding self)
        distances = dist_matrix[i].clone()
        distances[i] = float('inf')  # Exclude self
        
        # Filter by max distance first
        valid_mask = distances < max_distance
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            valid_distances = distances[valid_indices]
            # Get top-k from valid
            k_actual = min(k, len(valid_indices))
            _, top_k_local = torch.topk(valid_distances, k_actual, largest=False)
            top_k_global = valid_indices[top_k_local]
            
            for j in top_k_global:
                src_list.append(i)
                dst_list.append(j.item())
    
    if len(src_list) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


def build_a2a_edges_same_lane(
    agent_positions: torch.Tensor,  # [N, 2]
    agent_lane_ids: torch.Tensor,   # [N] - lane ID for each agent (-1 if none)
    ego_idx: int,
    max_distance: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Only connect agents that share the same lane (leading/following).
    
    Args:
        agent_positions: [N, 2] tensor of agent positions
        agent_lane_ids: [N] tensor of lane IDs per agent
        ego_idx: Index of ego agent
        max_distance: Maximum distance for edges
        
    Returns:
        (src, dst) tensors for edge index
    """
    N = agent_positions.shape[0]
    src_list, dst_list = [], []
    
    for i in range(N):
        if agent_lane_ids[i] < 0:  # No lane assigned
            continue
        for j in range(N):
            if i != j and agent_lane_ids[j] >= 0:
                # Check same lane
                if agent_lane_ids[i] == agent_lane_ids[j]:
                    dist = torch.norm(agent_positions[i] - agent_positions[j])
                    if dist < max_distance:
                        src_list.append(i)
                        dst_list.append(j)
    
    if len(src_list) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


def build_a2a_edges_directional(
    agent_positions: torch.Tensor,  # [N, 2]
    agent_headings: torch.Tensor,   # [N] - heading angle for each agent
    ego_idx: int,
    k: int = 5,
    max_distance: float = 50.0,
    fov_angle: float = 120.0,  # Field of view in degrees
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Only connect agents that are ahead (in field of view) of each agent.
    
    Args:
        agent_positions: [N, 2] tensor of agent positions
        agent_headings: [N] tensor of heading angles (radians)
        ego_idx: Index of ego agent
        k: Number of nearest neighbors to keep per agent
        max_distance: Maximum distance for edges
        fov_angle: Field of view angle in degrees
        
    Returns:
        (src, dst) tensors for edge index
    """
    N = agent_positions.shape[0]
    src_list, dst_list = [], []
    fov_rad = np.radians(fov_angle / 2)
    cos_fov = np.cos(fov_rad)
    
    for i in range(N):
        candidates = []
        pos_i = agent_positions[i]
        heading_i = agent_headings[i]
        
        # Direction vector for agent i
        dir_i = torch.tensor([torch.cos(heading_i), torch.sin(heading_i)], dtype=torch.float32)
        
        for j in range(N):
            if i != j:
                pos_j = agent_positions[j]
                vec_ij = pos_j - pos_i
                dist = torch.norm(vec_ij)
                
                if dist < max_distance and dist > 0:
                    # Check if j is in front of i (within FOV)
                    vec_ij_norm = vec_ij / dist
                    cos_angle = torch.dot(dir_i, vec_ij_norm.float())
                    
                    if cos_angle > cos_fov:  # Within FOV
                        candidates.append((j, dist.item()))
        
        # Take k nearest within FOV
        candidates.sort(key=lambda x: x[1])
        for j, _ in candidates[:k]:
            src_list.append(i)
            dst_list.append(j)
    
    if len(src_list) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


# =============================================================================
# L2L Edge Builders
# =============================================================================

def build_typed_l2l_edges(
    lane_polylines: List[np.ndarray],  # List of [P, 2] polylines
    lane_headings: np.ndarray,  # [N_lanes] heading angles
    max_successor_dist: float = 3.0,
    max_neighbor_dist: float = 5.0,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build typed lane-to-lane edges (LaneGCN style).
    
    Edge types:
    - successor: end of lane_i close to start of lane_j, same direction
    - predecessor: reverse of successor
    - left_of: parallel lane to the left
    - right_of: parallel lane to the right
    
    Args:
        lane_polylines: List of lane polylines (each [P, 2])
        lane_headings: [N_lanes] heading angles (radians)
        max_successor_dist: Max gap for successor edges
        max_neighbor_dist: Max lateral distance for neighbor edges
        
    Returns:
        Dict mapping edge type name to (src, dst) tensors
    """
    edges = {
        'successor': ([], []),
        'predecessor': ([], []),
        'left_of': ([], []),
        'right_of': ([], []),
    }
    
    N_lanes = len(lane_polylines)
    if N_lanes < 2:
        return {k: (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)) 
                for k in edges}
    
    # Precompute start/end points and midpoints
    start_points = []
    end_points = []
    mid_points = []
    
    for poly in lane_polylines:
        if len(poly) >= 2:
            start_points.append(poly[0])
            end_points.append(poly[-1])
            mid_points.append(poly[len(poly) // 2])
        else:
            # Degenerate polyline
            pt = poly[0] if len(poly) > 0 else np.zeros(2)
            start_points.append(pt)
            end_points.append(pt)
            mid_points.append(pt)
    
    start_points = np.array(start_points)  # [N, 2]
    end_points = np.array(end_points)      # [N, 2]
    mid_points = np.array(mid_points)      # [N, 2]
    
    for i in range(N_lanes):
        heading_i = lane_headings[i]
        dir_i = np.array([np.cos(heading_i), np.sin(heading_i)])
        perp_left_i = np.array([-np.sin(heading_i), np.cos(heading_i)])
        
        for j in range(N_lanes):
            if i == j:
                continue
            
            heading_j = lane_headings[j]
            
            # Check successor: end of i close to start of j, same direction
            gap_end_i_start_j = np.linalg.norm(end_points[i] - start_points[j])
            heading_diff = np.cos(heading_j - heading_i)
            
            if gap_end_i_start_j < max_successor_dist and heading_diff > 0.7:
                edges['successor'][0].append(i)
                edges['successor'][1].append(j)
                edges['predecessor'][0].append(j)
                edges['predecessor'][1].append(i)
            
            # Check lateral neighbors
            vec_ij = mid_points[j] - mid_points[i]
            dist = np.linalg.norm(vec_ij)
            
            if dist < max_neighbor_dist and heading_diff > 0.7:  # Same direction
                lateral_offset = np.dot(vec_ij, perp_left_i)
                longitudinal_offset = np.dot(vec_ij, dir_i)
                
                # Must be roughly parallel (small longitudinal offset)
                if abs(longitudinal_offset) < 5.0:
                    if lateral_offset > 1.5:  # Left
                        edges['left_of'][0].append(i)
                        edges['left_of'][1].append(j)
                    elif lateral_offset < -1.5:  # Right
                        edges['right_of'][0].append(i)
                        edges['right_of'][1].append(j)
    
    # Convert to tensors
    result = {}
    for edge_type, (src_list, dst_list) in edges.items():
        if len(src_list) > 0:
            result[edge_type] = (
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long)
            )
        else:
            result[edge_type] = (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long)
            )
    
    return result


# =============================================================================
# TL Filtering
# =============================================================================

def filter_relevant_traffic_lights(
    tl_positions: torch.Tensor,       # [N_tl, 2]
    ego_position: torch.Tensor,       # [2]
    ego_heading: float,               # radians
    max_distance: float = 100.0,
    max_relevant: int = 4,
    ahead_only: bool = True,
    filter_mode: str = "relevant",    # "all" | "relevant" | "controlling"
    tl_controlled_lane_ids: Optional[List[List[int]]] = None,
    ego_lane_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Filter traffic lights to only relevant ones.
    
    Args:
        tl_positions: [N_tl, 2] TL positions
        ego_position: [2] ego position
        ego_heading: Ego heading angle (radians)
        max_distance: Max distance from ego
        max_relevant: Max number of TLs to keep
        ahead_only: Only keep TLs ahead of ego
        filter_mode: Filtering mode
        tl_controlled_lane_ids: Which lanes each TL controls
        ego_lane_id: Lane ID the ego is on
        
    Returns:
        mask: Boolean tensor of which TLs to keep
        metadata: Dict with filtering info
    """
    N = tl_positions.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool), {'original_count': 0, 'filtered_count': 0}
    
    scores = torch.zeros(N)
    ego_dir = torch.tensor([np.cos(ego_heading), np.sin(ego_heading)], dtype=torch.float32)
    
    for i in range(N):
        tl_pos = tl_positions[i]
        vec_to_tl = tl_pos - ego_position
        dist = torch.norm(vec_to_tl)
        
        # Distance score (closer = higher)
        if dist < max_distance:
            dist_score = 1.0 - (dist / max_distance)
        else:
            scores[i] = -float('inf')
            continue
        
        # Ahead score (in front of ego = higher)
        if ahead_only and dist > 0:
            ahead_score = torch.dot(vec_to_tl / dist, ego_dir).item()
            if ahead_score < 0:  # Behind ego
                scores[i] = -float('inf')
                continue
        else:
            ahead_score = 1.0
        
        # Controlling score (controls ego's lane = much higher)
        control_score = 0.0
        if tl_controlled_lane_ids is not None and ego_lane_id is not None:
            controlled = tl_controlled_lane_ids[i] if i < len(tl_controlled_lane_ids) else []
            if ego_lane_id in controlled:
                control_score = 5.0  # High bonus
        
        scores[i] = dist_score + ahead_score + control_score
    
    # Apply filtering mode
    if filter_mode == "controlling":
        # Only keep TLs that control ego's lane
        mask = scores > 4.0
    elif filter_mode == "relevant":
        # Keep top-k by score
        valid_mask = scores > -float('inf')
        n_valid = valid_mask.sum().item()
        if n_valid > max_relevant:
            _, top_k_indices = torch.topk(scores, max_relevant)
            mask = torch.zeros(N, dtype=torch.bool)
            mask[top_k_indices] = True
        else:
            mask = valid_mask
    else:  # "all"
        mask = torch.ones(N, dtype=torch.bool)
    
    metadata = {
        'original_count': N,
        'filtered_count': mask.sum().item(),
        'scores': scores,
    }
    
    return mask, metadata


def build_tl_lane_edges(
    tl_mask: torch.Tensor,  # Which TLs are kept
    tl_controlled_lane_ids: List[List[int]],
    lane_id_to_idx: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build TL -> Lane edges (TL controls lane).
    
    Args:
        tl_mask: Boolean mask of which TLs to include
        tl_controlled_lane_ids: Which lanes each TL controls
        lane_id_to_idx: Mapping from lane ID to index in graph
        
    Returns:
        (src, dst) tensors for edge index (TL idx -> lane idx)
    """
    src_list, dst_list = [], []
    
    kept_indices = torch.where(tl_mask)[0]
    
    for new_idx, old_idx in enumerate(kept_indices):
        old_idx = old_idx.item()
        if old_idx < len(tl_controlled_lane_ids):
            controlled = tl_controlled_lane_ids[old_idx]
            for lane_id in controlled:
                if lane_id in lane_id_to_idx:
                    src_list.append(new_idx)
                    dst_list.append(lane_id_to_idx[lane_id])
    
    if len(src_list) == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


# =============================================================================
# Lane Filtering
# =============================================================================

def filter_lanes(
    lane_positions: torch.Tensor,  # [N_lanes, 2] - centroids
    ego_position: torch.Tensor,    # [2]
    ego_heading: float,            # radians
    lane_types: Optional[List[str]] = None,  # e.g., 'driving', 'parking'
    lane_ids: Optional[List[int]] = None,
    ego_path_lane_ids: Optional[List[int]] = None,
    max_count: int = 80,
    max_distance: float = 50.0,
    filter_mode: str = "all",      # "all" | "drivable" | "ego_relevant" | "ego_path"
    drivable_only: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    Filter lanes to only relevant ones.
    
    Args:
        lane_positions: [N_lanes, 2] lane centroid positions
        ego_position: [2] ego position
        ego_heading: Ego heading angle (radians)
        lane_types: List of lane type strings
        lane_ids: List of lane IDs
        ego_path_lane_ids: Lane IDs on ego's planned path
        max_count: Max number of lanes to keep
        max_distance: Max distance from ego
        filter_mode: Filtering mode
        drivable_only: Whether to exclude non-drivable lanes
        
    Returns:
        mask: Boolean tensor of which lanes to keep
        metadata: Dict with filtering info
    """
    N = lane_positions.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool), {'original_count': 0, 'filtered_count': 0}
    
    scores = torch.zeros(N)
    valid = torch.ones(N, dtype=torch.bool)
    
    # Drivable types filter
    drivable_types = {'driving', 'highway', 'urban', 'freeway', 'residential'}
    
    # Step 1: Filter by drivable
    if drivable_only and lane_types is not None:
        for i, lt in enumerate(lane_types):
            if lt.lower() not in drivable_types:
                valid[i] = False
    
    # Step 2: Filter by distance
    distances = torch.norm(lane_positions - ego_position, dim=1)
    valid = valid & (distances < max_distance)
    
    # Step 3: Score remaining lanes
    ego_dir = torch.tensor([np.cos(ego_heading), np.sin(ego_heading)], dtype=torch.float32)
    perp = torch.tensor([-np.sin(ego_heading), np.cos(ego_heading)], dtype=torch.float32)
    
    for i in range(N):
        if not valid[i]:
            scores[i] = -float('inf')
            continue
        
        dist = distances[i].item()
        
        # Distance score (closer = higher)
        dist_score = 1.0 - (dist / max_distance)
        
        # Ahead score (lanes ahead of ego are more relevant)
        vec_to_lane = lane_positions[i] - ego_position
        if dist > 0:
            ahead_score = torch.dot(vec_to_lane / dist, ego_dir).item()
            ahead_score = (ahead_score + 1) / 2  # Normalize to [0, 1]
        else:
            ahead_score = 0.5
        
        # Path score (on ego's planned path = much higher)
        path_score = 0.0
        if ego_path_lane_ids and lane_ids:
            if lane_ids[i] in ego_path_lane_ids:
                path_score = 5.0
        
        # Lateral score (lanes laterally close to ego = higher)
        lateral_dist = abs(torch.dot(vec_to_lane, perp).item())
        lateral_score = max(0, 1.0 - lateral_dist / 20.0)
        
        scores[i] = dist_score + ahead_score + path_score + lateral_score
    
    # Apply filtering mode
    if filter_mode == "ego_path" and ego_path_lane_ids and lane_ids:
        mask = torch.tensor([lid in ego_path_lane_ids for lid in lane_ids], dtype=torch.bool)
    elif filter_mode == "ego_relevant":
        # Top-k by relevance score
        valid_mask = scores > -float('inf')
        n_valid = valid_mask.sum().item()
        if n_valid > max_count:
            _, top_k = torch.topk(scores, max_count)
            mask = torch.zeros(N, dtype=torch.bool)
            mask[top_k] = True
        else:
            mask = valid_mask
    elif filter_mode == "drivable":
        mask = valid.clone()
        # Limit to max_count by distance
        if mask.sum() > max_count:
            valid_indices = torch.where(mask)[0]
            valid_distances = distances[valid_indices]
            _, closest = torch.topk(valid_distances, max_count, largest=False)
            mask = torch.zeros(N, dtype=torch.bool)
            mask[valid_indices[closest]] = True
    else:  # "all"
        mask = torch.ones(N, dtype=torch.bool)
    
    metadata = {
        'original_count': N,
        'filtered_count': mask.sum().item(),
        'scores': scores,
    }
    
    return mask, metadata
