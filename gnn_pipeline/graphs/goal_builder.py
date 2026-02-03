#!/usr/bin/env python3
"""
goal_builder.py

Goal node and goal-conditioned edge construction for trajectory prediction.

Adds SDC path (goal/route) as explicit nodes in the heterogeneous graph,
enabling goal-conditioned reasoning to reduce FDE.

Goal node features (10 dims):
    [0-1]: x, y position in ego frame
    [2]: distance from ego
    [3-4]: direction to waypoint (normalized)
    [5-6]: path direction at waypoint
    [7]: progress along path [0,1]
    [8]: is_final_goal flag
    [9]: waypoint index (normalized)

Edge features for agent→goal (5 dims):
    [0]: distance to goal waypoint
    [1-2]: direction to goal (dx, dy normalized)
    [3]: progress along path
    [4]: is_final_goal flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class GoalConfig:
    """Configuration for goal node construction."""
    
    goal_mode: str = "waypoints"      # "endpoint" or "waypoints"
    num_waypoints: int = 10           # Number of waypoints to keep
    waypoint_spacing: float = 5.0     # Meters between waypoints (for resampling)
    max_distance: float = 80.0        # Max distance from ego to include waypoints


# -----------------------------
# Coordinate Transforms
# -----------------------------

def world_to_ego_frame(
    points: np.ndarray,
    ego_pos: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """
    Transform points from world frame to ego-centric frame.
    
    Args:
        points: [N, 2] array of world coordinates
        ego_pos: [2] ego position (x, y)
        ego_heading: Ego heading in radians
        
    Returns:
        [N, 2] array of points in ego frame
    """
    if points.shape[0] == 0:
        return points
    
    # Translate to ego origin
    translated = points - ego_pos
    
    # Rotate to ego heading (ego heading becomes x-axis)
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    
    return translated @ rotation.T


# -----------------------------
# Path Resampling
# -----------------------------

def resample_path(
    path: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """
    Resample path to uniform spacing.
    
    Args:
        path: [N, 2] array of waypoints
        spacing: Target spacing between waypoints (meters)
        
    Returns:
        [M, 2] array of resampled waypoints
    """
    if path.shape[0] < 2:
        return path
    
    # Compute cumulative distance along path
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_dist[-1]
    
    if total_length < spacing:
        return path
    
    # Interpolate at uniform spacing
    num_points = int(total_length / spacing) + 1
    target_dists = np.linspace(0, total_length, num_points)
    
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    for i, d in enumerate(target_dists):
        # Find segment containing this distance
        idx = np.searchsorted(cumulative_dist, d, side='right') - 1
        idx = np.clip(idx, 0, len(path) - 2)
        
        # Interpolate within segment
        seg_start_dist = cumulative_dist[idx]
        seg_length = segment_lengths[idx] if idx < len(segment_lengths) else 1e-6
        if seg_length > 1e-6:
            t = (d - seg_start_dist) / seg_length
        else:
            t = 0.0
        t = np.clip(t, 0, 1)
        
        resampled[i] = path[idx] * (1 - t) + path[idx + 1] * t
    
    return resampled


def compute_path_direction(
    path: np.ndarray,
    idx: int,
) -> np.ndarray:
    """
    Compute path direction at a given waypoint.
    
    Args:
        path: [N, 2] array of waypoints
        idx: Waypoint index
        
    Returns:
        [2] normalized direction vector
    """
    if path.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    
    if idx < path.shape[0] - 1:
        # Direction to next waypoint
        direction = path[idx + 1] - path[idx]
    else:
        # Use previous segment for last point
        direction = path[idx] - path[idx - 1]
    
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        return direction / norm
    return np.array([1.0, 0.0], dtype=np.float32)


# -----------------------------
# Goal Node Construction
# -----------------------------

def build_goal_nodes(
    goal_points: np.ndarray,
    ego_pos: np.ndarray,
    ego_heading: float,
    config: GoalConfig,
) -> Tuple[torch.Tensor, Dict]:
    """
    Build goal node features from goal waypoints.
    
    Args:
        goal_points: [N, 2] goal waypoints in world frame
        ego_pos: [2] ego position
        ego_heading: Ego heading (radians)
        config: Goal configuration
    
    Returns:
        goal_features: [N_goal, 10] tensor
        goal_info: Dict with metadata for edge building
    """
    N_FEATURES = 10
    
    if goal_points is None or len(goal_points) == 0:
        return torch.zeros(0, N_FEATURES, dtype=torch.float32), {
            'num_goals': 0,
            'positions': np.array([], dtype=np.float32).reshape(0, 2),
        }
    
    # Transform to ego frame
    path_ego = world_to_ego_frame(goal_points, ego_pos, ego_heading)
    
    if config.goal_mode == "endpoint":
        # Single goal node at path endpoint
        goal_pos = path_ego[-1]
        dist = np.linalg.norm(goal_pos)
        
        # Direction to goal
        if dist > 1e-6:
            dir_vec = goal_pos / dist
        else:
            dir_vec = np.array([1.0, 0.0])
        
        # Path direction at endpoint
        path_dir = compute_path_direction(path_ego, len(path_ego) - 1)
        
        features = np.array([[
            goal_pos[0], goal_pos[1],    # [0-1]: position
            dist,                         # [2]: distance
            dir_vec[0], dir_vec[1],       # [3-4]: direction to goal
            path_dir[0], path_dir[1],     # [5-6]: path direction
            1.0,                          # [7]: progress (100%)
            1.0,                          # [8]: is_final_goal
            0.0,                          # [9]: waypoint index
        ]], dtype=np.float32)
        
        goal_info = {
            'num_goals': 1,
            'positions': goal_pos.reshape(1, 2),
        }
        
    elif config.goal_mode == "waypoints":
        # Multiple waypoint nodes along path
        # Resample to uniform spacing
        resampled = resample_path(path_ego, config.waypoint_spacing)
        
        # Keep up to num_waypoints (evenly spaced selection)
        if len(resampled) > config.num_waypoints:
            indices = np.linspace(0, len(resampled) - 1, config.num_waypoints).astype(int)
            resampled = resampled[indices]
        
        # Filter by max distance from ego
        distances = np.linalg.norm(resampled, axis=1)
        mask = distances < config.max_distance
        resampled = resampled[mask]
        distances = distances[mask]
        
        if len(resampled) == 0:
            return torch.zeros(0, N_FEATURES, dtype=torch.float32), {
                'num_goals': 0,
                'positions': np.array([], dtype=np.float32).reshape(0, 2),
            }
        
        # Build features for each waypoint
        n_waypoints = len(resampled)
        features = np.zeros((n_waypoints, N_FEATURES), dtype=np.float32)
        
        for i in range(n_waypoints):
            pos = resampled[i]
            dist = distances[i]
            
            # Direction from ego to waypoint
            if dist > 1e-6:
                dir_vec = pos / dist
            else:
                dir_vec = np.array([1.0, 0.0])
            
            # Path direction at waypoint
            path_dir = compute_path_direction(resampled, i)
            
            # Progress and flags
            progress = (i + 1) / n_waypoints
            is_final = 1.0 if i == n_waypoints - 1 else 0.0
            waypoint_idx = i / max(n_waypoints - 1, 1)  # Normalized index
            
            features[i] = [
                pos[0], pos[1],               # [0-1]: position
                dist,                          # [2]: distance
                dir_vec[0], dir_vec[1],        # [3-4]: direction to waypoint
                path_dir[0], path_dir[1],      # [5-6]: path direction
                progress,                      # [7]: progress
                is_final,                      # [8]: is_final_goal
                waypoint_idx,                  # [9]: waypoint index
            ]
        
        goal_info = {
            'num_goals': n_waypoints,
            'positions': resampled,
        }
        
    else:
        raise ValueError(f"Unknown goal_mode: {config.goal_mode}")
    
    return torch.tensor(features, dtype=torch.float32), goal_info


# -----------------------------
# Goal Edge Construction
# -----------------------------

def build_agent_to_goal_edges(
    agent_features: np.ndarray,
    ego_idx: int,
    goal_info: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges from ego agent to goal nodes.
    
    Only the ego agent connects to goals (other agents don't know ego's destination).
    
    Args:
        agent_features: [N_agents, F] agent node features
        ego_idx: Index of ego agent
        goal_info: Dict from build_goal_nodes containing positions
        
    Returns:
        edge_index: [2, E] tensor (ego → goals)
        edge_attr: [E, 5] tensor with edge features
    """
    N_EDGE_FEATURES = 5
    num_goals = goal_info['num_goals']
    
    if num_goals == 0:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, N_EDGE_FEATURES, dtype=torch.float32))
    
    # Ego position (already in ego frame, so [0, 0])
    ego_pos = agent_features[ego_idx, :2]
    
    # Goal positions
    goal_positions = goal_info['positions']
    
    # Build edges: ego connects to all goal nodes
    src = torch.full((num_goals,), ego_idx, dtype=torch.long)
    dst = torch.arange(num_goals, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    # Build edge features
    edge_attr = np.zeros((num_goals, N_EDGE_FEATURES), dtype=np.float32)
    
    for i in range(num_goals):
        pos = goal_positions[i]
        dx = pos[0] - ego_pos[0]
        dy = pos[1] - ego_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Normalized direction
        if dist > 1e-6:
            dir_x, dir_y = dx / dist, dy / dist
        else:
            dir_x, dir_y = 1.0, 0.0
        
        progress = (i + 1) / num_goals
        is_final = 1.0 if i == num_goals - 1 else 0.0
        
        edge_attr[i] = [dist, dir_x, dir_y, progress, is_final]
    
    return edge_index, torch.tensor(edge_attr, dtype=torch.float32)


# -----------------------------
# Goal Extraction Helpers
# -----------------------------

def extract_goal_from_future_trajectory(
    future_xy: np.ndarray,
    future_valid: np.ndarray,
    ego_idx: int,
    num_waypoints: int = 10,
) -> np.ndarray:
    """
    Extract goal waypoints from ego's future trajectory.
    
    Uses the logged future trajectory as a proxy for the intended goal/path.
    This works with existing TFRecord data that may not include SDC paths.
    
    Args:
        future_xy: [N_agents, T, 2] future positions
        future_valid: [N_agents, T] validity mask
        ego_idx: Index of ego agent
        num_waypoints: Number of waypoints to sample
        
    Returns:
        [M, 2] array of goal waypoints (in original frame, not ego frame)
    """
    ego_future = future_xy[ego_idx]  # [T, 2]
    ego_valid = future_valid[ego_idx]  # [T]
    
    # Get valid future points
    valid_mask = ego_valid > 0.5
    if not valid_mask.any():
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    valid_points = ego_future[valid_mask]
    
    if len(valid_points) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    # Sample waypoints (evenly spaced through valid future)
    if len(valid_points) <= num_waypoints:
        return valid_points
    
    indices = np.linspace(0, len(valid_points) - 1, num_waypoints).astype(int)
    return valid_points[indices]


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":
    print("Testing goal_builder.py...")
    
    # Create mock data
    np.random.seed(42)
    
    # Mock path in world frame
    t = np.linspace(0, 50, 20)
    path_world = np.stack([t * 0.8, np.sin(t * 0.1) * 2], axis=1).astype(np.float32)
    
    # Ego state
    ego_pos = np.array([0.0, 0.0], dtype=np.float32)
    ego_heading = 0.0
    
    # Test goal node construction
    config = GoalConfig(goal_mode="waypoints", num_waypoints=10)
    goal_features, goal_info = build_goal_nodes(path_world, ego_pos, ego_heading, config)
    
    print(f"\n=== Goal Node Features ===")
    print(f"Shape: {goal_features.shape}")
    print(f"Num goals: {goal_info['num_goals']}")
    
    # Test edge construction
    agent_features = np.random.randn(5, 66).astype(np.float32)
    agent_features[0, :2] = [0, 0]  # Ego at origin
    ego_idx = 0
    
    edge_index, edge_attr = build_agent_to_goal_edges(agent_features, ego_idx, goal_info)
    
    print(f"\n=== Agent→Goal Edges ===")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    
    # Test endpoint mode
    config_endpoint = GoalConfig(goal_mode="endpoint")
    goal_features_ep, goal_info_ep = build_goal_nodes(path_world, ego_pos, ego_heading, config_endpoint)
    
    print(f"\n=== Endpoint Mode ===")
    print(f"Shape: {goal_features_ep.shape}")
    print(f"Num goals: {goal_info_ep['num_goals']}")
    
    # Test empty path
    empty_path = np.array([], dtype=np.float32).reshape(0, 2)
    goal_empty, info_empty = build_goal_nodes(empty_path, ego_pos, ego_heading, config)
    
    print(f"\n=== Empty Path ===")
    print(f"Shape: {goal_empty.shape}")
    print(f"Num goals: {info_empty['num_goals']}")
    
    print("\n✓ All tests passed!")
