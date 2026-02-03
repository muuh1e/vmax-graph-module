#!/usr/bin/env python
"""
Test script for Phase 2B goal builder functions.

Verifies:
1. Goal node feature dimensions are correct (10 dims)
2. Edge construction produces correct shapes (5 dims)
3. No NaN/Inf values in features
4. Path resampling produces uniform spacing
5. Empty path handling is graceful

Usage:
    python gnn_pipeline/tests/test_goal_builder.py --all
    python gnn_pipeline/tests/test_goal_builder.py --check-dims
    python gnn_pipeline/tests/test_goal_builder.py --check-values
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct import to avoid torch_geometric dependency from package __init__
from gnn_pipeline.graphs.goal_builder import (
    GoalConfig,
    build_goal_nodes,
    build_agent_to_goal_edges,
    extract_goal_from_future_trajectory,
    resample_path,
    world_to_ego_frame,
)


def create_mock_path(n_points: int = 20, length: float = 50.0) -> np.ndarray:
    """Create a mock path for testing."""
    t = np.linspace(0, length, n_points)
    # Slight curve
    path = np.stack([t * 0.9, np.sin(t * 0.1) * 3], axis=1).astype(np.float32)
    return path


def create_mock_future_trajectory(
    n_agents: int = 5,
    n_timesteps: int = 80,
) -> tuple:
    """Create mock future trajectory data."""
    future_xy = np.random.randn(n_agents, n_timesteps, 2).astype(np.float32)
    
    # Make ego trajectory realistic (smooth path forward)
    t = np.linspace(0, 40, n_timesteps)  # 8 seconds at 10Hz
    future_xy[0, :, 0] = t  # Move forward
    future_xy[0, :, 1] = np.sin(t * 0.1) * 2  # Slight curve
    
    future_valid = np.ones((n_agents, n_timesteps), dtype=np.float32)
    # Some invalid at the end
    future_valid[:, -10:] = 0
    
    return future_xy, future_valid


def test_goal_node_dimensions(goal_mode: str = "waypoints") -> bool:
    """Test goal node feature dimensions."""
    print(f"\n[TEST] Goal node dimensions (mode={goal_mode})")
    
    path = create_mock_path()
    ego_pos = np.array([0.0, 0.0], dtype=np.float32)
    ego_heading = 0.0
    
    config = GoalConfig(goal_mode=goal_mode, num_waypoints=10)
    goal_features, goal_info = build_goal_nodes(path, ego_pos, ego_heading, config)
    
    expected_dim = 10
    actual_dim = goal_features.shape[1] if goal_features.shape[0] > 0 else 0
    
    if goal_mode == "endpoint":
        expected_nodes = 1
    else:
        expected_nodes = min(config.num_waypoints, len(path))
    
    dim_ok = actual_dim == expected_dim
    nodes_ok = goal_features.shape[0] > 0
    
    print(f"  Feature dims: {actual_dim} (expected {expected_dim}) - {'PASS' if dim_ok else 'FAIL'}")
    print(f"  Num nodes: {goal_features.shape[0]} (expected >0) - {'PASS' if nodes_ok else 'FAIL'}")
    
    return dim_ok and nodes_ok


def test_edge_dimensions() -> bool:
    """Test agent→goal edge dimensions."""
    print("\n[TEST] Agent→Goal edge dimensions")
    
    # Create mock agent features
    n_agents = 5
    agent_features = np.random.randn(n_agents, 66).astype(np.float32)
    agent_features[0, :2] = [0, 0]  # Ego at origin
    ego_idx = 0
    
    # Create goal info
    path = create_mock_path()
    config = GoalConfig(goal_mode="waypoints", num_waypoints=10)
    _, goal_info = build_goal_nodes(
        path, np.array([0.0, 0.0]), 0.0, config
    )
    
    edge_index, edge_attr = build_agent_to_goal_edges(agent_features, ego_idx, goal_info)
    
    expected_edge_dim = 5
    actual_edge_dim = edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0
    
    dim_ok = actual_edge_dim == expected_edge_dim
    edges_ok = edge_index.shape[1] > 0
    
    # Check all edges come from ego
    if edges_ok:
        all_from_ego = (edge_index[0] == ego_idx).all().item()
    else:
        all_from_ego = True
    
    print(f"  Edge attr dims: {actual_edge_dim} (expected {expected_edge_dim}) - {'PASS' if dim_ok else 'FAIL'}")
    print(f"  Num edges: {edge_index.shape[1]} - {'PASS' if edges_ok else 'FAIL'}")
    print(f"  All edges from ego: {all_from_ego} - {'PASS' if all_from_ego else 'FAIL'}")
    
    return dim_ok and edges_ok and all_from_ego


def test_no_nan_inf() -> bool:
    """Test that there are no NaN or Inf values."""
    print("\n[TEST] No NaN/Inf values")
    
    path = create_mock_path(n_points=30, length=80)
    ego_pos = np.array([0.0, 0.0], dtype=np.float32)
    ego_heading = 0.3  # Non-zero heading
    
    config = GoalConfig(goal_mode="waypoints", num_waypoints=15)
    goal_features, goal_info = build_goal_nodes(path, ego_pos, ego_heading, config)
    
    # Agent features
    agent_features = np.random.randn(8, 66).astype(np.float32)
    agent_features[0, :2] = [0, 0]
    
    edge_index, edge_attr = build_agent_to_goal_edges(agent_features, 0, goal_info)
    
    goal_nan = np.isnan(goal_features.numpy()).any()
    goal_inf = np.isinf(goal_features.numpy()).any()
    edge_nan = np.isnan(edge_attr.numpy()).any()
    edge_inf = np.isinf(edge_attr.numpy()).any()
    
    print(f"  Goal NaN: {goal_nan}, Inf: {goal_inf}")
    print(f"  Edge NaN: {edge_nan}, Inf: {edge_inf}")
    
    success = not (goal_nan or goal_inf or edge_nan or edge_inf)
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success


def test_path_resampling() -> bool:
    """Test path resampling produces uniform spacing."""
    print("\n[TEST] Path resampling")
    
    # Non-uniform path
    path = np.array([
        [0, 0], [1, 0], [2, 0], [5, 0], [10, 0], [20, 0]
    ], dtype=np.float32)
    
    resampled = resample_path(path, spacing=2.0)
    
    # Check spacing is approximately uniform
    diffs = np.diff(resampled, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    
    # All segments should be close to 2.0
    target_spacing = 2.0
    tolerance = 0.5  # Allow some tolerance
    
    spacing_ok = np.all(np.abs(segment_lengths - target_spacing) < tolerance)
    
    print(f"  Original points: {len(path)}")
    print(f"  Resampled points: {len(resampled)}")
    print(f"  Segment lengths: {segment_lengths[:5]}...")
    print(f"  Uniform spacing: {spacing_ok} - {'PASS' if spacing_ok else 'FAIL'}")
    
    return spacing_ok


def test_coordinate_transform() -> bool:
    """Test world-to-ego coordinate transform."""
    print("\n[TEST] Coordinate transform")
    
    # Points in world frame
    points = np.array([
        [10, 0],
        [0, 10],
        [10, 10],
    ], dtype=np.float32)
    
    ego_pos = np.array([5, 5], dtype=np.float32)
    ego_heading = np.pi / 4  # 45 degrees
    
    transformed = world_to_ego_frame(points, ego_pos, ego_heading)
    
    # First point (10, 0) relative to ego (5, 5) is (5, -5)
    # After 45 degree rotation, should be roughly (7.07, 0) in ego frame
    
    # Check that origin is at (0, 0) after transform
    ego_in_world = np.array([[5, 5]], dtype=np.float32)
    ego_transformed = world_to_ego_frame(ego_in_world, ego_pos, ego_heading)
    
    origin_ok = np.allclose(ego_transformed[0], [0, 0], atol=1e-5)
    
    print(f"  Ego at origin: {ego_transformed[0]} - {'PASS' if origin_ok else 'FAIL'}")
    print(f"  Transformed sample: {transformed[0]}")
    
    return origin_ok


def test_empty_path_handling() -> bool:
    """Test handling of empty paths."""
    print("\n[TEST] Empty path handling")
    
    empty_path = np.array([], dtype=np.float32).reshape(0, 2)
    ego_pos = np.array([0.0, 0.0], dtype=np.float32)
    ego_heading = 0.0
    
    config = GoalConfig()
    goal_features, goal_info = build_goal_nodes(empty_path, ego_pos, ego_heading, config)
    
    empty_ok = goal_features.shape[0] == 0
    info_ok = goal_info['num_goals'] == 0
    
    print(f"  Empty features shape: {goal_features.shape} - {'PASS' if empty_ok else 'FAIL'}")
    print(f"  Goal info num_goals: {goal_info['num_goals']} - {'PASS' if info_ok else 'FAIL'}")
    
    # Test edge building with empty goal info
    agent_features = np.random.randn(5, 66).astype(np.float32)
    edge_index, edge_attr = build_agent_to_goal_edges(agent_features, 0, goal_info)
    
    edge_ok = edge_index.shape[1] == 0
    print(f"  Empty edges shape: {edge_index.shape} - {'PASS' if edge_ok else 'FAIL'}")
    
    return empty_ok and info_ok and edge_ok


def test_future_trajectory_extraction() -> bool:
    """Test goal extraction from future trajectory."""
    print("\n[TEST] Future trajectory extraction")
    
    future_xy, future_valid = create_mock_future_trajectory(n_agents=5, n_timesteps=80)
    ego_idx = 0
    
    goal_waypoints = extract_goal_from_future_trajectory(
        future_xy, future_valid, ego_idx, num_waypoints=10
    )
    
    has_waypoints = goal_waypoints.shape[0] > 0
    shape_ok = len(goal_waypoints.shape) == 2 and goal_waypoints.shape[1] == 2
    
    print(f"  Extracted waypoints: {goal_waypoints.shape}")
    print(f"  Has waypoints: {has_waypoints} - {'PASS' if has_waypoints else 'FAIL'}")
    print(f"  Shape correct: {shape_ok} - {'PASS' if shape_ok else 'FAIL'}")
    
    return has_waypoints and shape_ok


def main():
    parser = argparse.ArgumentParser(description="Test goal builder functions")
    parser.add_argument("--check-dims", action="store_true", help="Check feature dimensions")
    parser.add_argument("--check-values", action="store_true", help="Check for NaN/Inf values")
    parser.add_argument("--check-resample", action="store_true", help="Check path resampling")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    if not any([args.check_dims, args.check_values, args.check_resample, args.all]):
        args.all = True
    
    all_passed = True
    
    if args.check_dims or args.all:
        all_passed &= test_goal_node_dimensions("waypoints")
        all_passed &= test_goal_node_dimensions("endpoint")
        all_passed &= test_edge_dimensions()
    
    if args.check_values or args.all:
        all_passed &= test_no_nan_inf()
    
    if args.check_resample or args.all:
        all_passed &= test_path_resampling()
        all_passed &= test_coordinate_transform()
    
    if args.all:
        all_passed &= test_empty_path_handling()
        all_passed &= test_future_trajectory_extraction()
    
    print("\n" + "=" * 50)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
