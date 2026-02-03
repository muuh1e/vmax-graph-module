#!/usr/bin/env python
"""
Test script for Phase 1 enhanced edge features.

Verifies:
1. Feature dimensions are correct (A2A: 32, A2L: 15 when enhanced)
2. No NaN/Inf values in edge features
3. Backward compatibility: default config produces original dimensions
4. Feature statistics are in reasonable ranges

Usage:
    python gnn_pipeline/tests/test_enhanced_edge_features.py --check-dims
    python gnn_pipeline/tests/test_enhanced_edge_features.py --backward-compat
    python gnn_pipeline/tests/test_enhanced_edge_features.py --check-values
    python gnn_pipeline/tests/test_enhanced_edge_features.py --stats
    python gnn_pipeline/tests/test_enhanced_edge_features.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import directly without going through __init__
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct imports to avoid torch_geometric dependency in tests
from gnn_pipeline.graphs.hetero_graph import (
    _compute_a2a_edge_attrs,
    build_a2l_edges,
    compute_lateral_velocity,
    compute_yaw_rate,
)


def create_mock_agent_features(n_agents: int = 5, k_past: int = 10) -> np.ndarray:
    """Create mock agent features for testing."""
    # Agent feature layout: [16 base + 5*k_past]
    n_features = 16 + 5 * k_past
    features = np.random.randn(n_agents, n_features).astype(np.float32)
    
    # Set reasonable values for key features
    features[:, 0] = np.random.uniform(-30, 30, n_agents)  # x
    features[:, 1] = np.random.uniform(-10, 10, n_agents)  # y
    features[:, 2] = np.random.uniform(-10, 20, n_agents)  # vx
    features[:, 3] = np.random.uniform(-5, 5, n_agents)    # vy
    features[:, 4] = np.random.uniform(-np.pi, np.pi, n_agents)  # yaw
    features[:, 5] = np.random.uniform(0, 25, n_agents)    # speed
    
    # Past trajectory - create smooth trajectory history
    past_x_start = 16
    past_y_start = 16 + k_past
    past_vx_start = 16 + 2 * k_past
    past_vy_start = 16 + 3 * k_past
    past_valid_start = 16 + 4 * k_past
    
    for i in range(n_agents):
        # Create a realistic trajectory (moving backward in time)
        x_now = features[i, 0]
        y_now = features[i, 1]
        vx = features[i, 2]
        vy = features[i, 3]
        dt = 0.1
        
        for t in range(k_past):
            features[i, past_x_start + t] = x_now - (k_past - t) * vx * dt
            features[i, past_y_start + t] = y_now - (k_past - t) * vy * dt
            features[i, past_vx_start + t] = vx + np.random.randn() * 0.1
            features[i, past_vy_start + t] = vy + np.random.randn() * 0.1
            features[i, past_valid_start + t] = 1.0
    
    return features


def create_mock_lane_features(n_lanes: int = 10) -> tuple:
    """Create mock lane features and polylines for testing."""
    n_features = 40
    features = np.zeros((n_lanes, n_features), dtype=np.float32)
    polylines = []
    
    for i in range(n_lanes):
        # Create lane centroid
        cx = np.random.uniform(-50, 50)
        cy = np.random.uniform(-20, 20)
        features[i, 0] = cx
        features[i, 1] = cy
        
        # Speed limit (normalized by 40.0)
        features[i, 9] = np.random.uniform(0.3, 1.0)
        
        # Create polyline (10-20 points)
        n_pts = np.random.randint(10, 20)
        heading = np.random.uniform(-np.pi/4, np.pi/4)
        length = np.random.uniform(20, 50)
        
        t = np.linspace(0, length, n_pts)
        x = cx + t * np.cos(heading) - length/2 * np.cos(heading)
        y = cy + t * np.sin(heading) - length/2 * np.sin(heading)
        
        polylines.append(np.stack([x, y], axis=1).astype(np.float32))
    
    return features, polylines


def test_a2a_dimensions(use_enhanced: bool = True) -> bool:
    """Test A2A edge feature dimensions."""
    print(f"\n[TEST] A2A dimensions (enhanced={use_enhanced})")
    
    agent_features = create_mock_agent_features(n_agents=5)
    
    # Create test edge index
    edge_index = np.array([
        [0, 0, 1, 1, 2],
        [1, 2, 0, 2, 3],
    ], dtype=np.int64)
    
    edge_attr = _compute_a2a_edge_attrs(
        agent_features, edge_index,
        use_enhanced=use_enhanced,
    )
    
    expected_dim = 32 if use_enhanced else 20
    actual_dim = edge_attr.shape[1]
    
    success = actual_dim == expected_dim
    print(f"  Expected: {expected_dim}, Got: {actual_dim} - {'PASS' if success else 'FAIL'}")
    
    return success


def test_a2l_dimensions(use_frenet: bool = True) -> bool:
    """Test A2L edge feature dimensions."""
    print(f"\n[TEST] A2L dimensions (use_frenet={use_frenet})")
    
    agent_features = create_mock_agent_features(n_agents=3)
    lane_features, lane_polylines = create_mock_lane_features(n_lanes=5)
    
    edge_index, edge_attr = build_a2l_edges(
        agent_features, lane_features, lane_polylines,
        k=3, use_frenet=use_frenet,
    )
    
    expected_dim = 15 if use_frenet else 9
    actual_dim = edge_attr.shape[1] if edge_attr.size > 0 else 0
    
    success = actual_dim == expected_dim
    print(f"  Expected: {expected_dim}, Got: {actual_dim} - {'PASS' if success else 'FAIL'}")
    
    return success


def test_no_nan_inf() -> bool:
    """Test that there are no NaN or Inf values in edge features."""
    print("\n[TEST] No NaN/Inf values")
    
    agent_features = create_mock_agent_features(n_agents=8)
    lane_features, lane_polylines = create_mock_lane_features(n_lanes=10)
    
    # A2A edges
    edge_index = np.array([
        [0, 0, 1, 1, 2, 3, 4],
        [1, 2, 0, 3, 4, 5, 6],
    ], dtype=np.int64)
    
    a2a_attr = _compute_a2a_edge_attrs(
        agent_features, edge_index, use_enhanced=True,
    )
    
    # A2L edges
    _, a2l_attr = build_a2l_edges(
        agent_features, lane_features, lane_polylines,
        k=3, use_frenet=True,
    )
    
    a2a_nan = np.isnan(a2a_attr).any()
    a2a_inf = np.isinf(a2a_attr).any()
    a2l_nan = np.isnan(a2l_attr).any()
    a2l_inf = np.isinf(a2l_attr).any()
    
    print(f"  A2A NaN: {a2a_nan}, Inf: {a2a_inf}")
    print(f"  A2L NaN: {a2l_nan}, Inf: {a2l_inf}")
    
    success = not (a2a_nan or a2a_inf or a2l_nan or a2l_inf)
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success


def test_backward_compatibility() -> bool:
    """Test backward compatibility when enhanced features are disabled."""
    print("\n[TEST] Backward compatibility")
    
    agent_features = create_mock_agent_features(n_agents=5)
    lane_features, lane_polylines = create_mock_lane_features(n_lanes=5)
    
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    
    # Original dimensions
    a2a_legacy = _compute_a2a_edge_attrs(agent_features, edge_index, use_enhanced=False)
    _, a2l_legacy = build_a2l_edges(
        agent_features, lane_features, lane_polylines, k=3, use_frenet=False
    )
    
    a2a_ok = a2a_legacy.shape[1] == 20
    a2l_ok = a2l_legacy.shape[1] == 9
    
    print(f"  A2A legacy dim: {a2a_legacy.shape[1]} (expected 20) - {'PASS' if a2a_ok else 'FAIL'}")
    print(f"  A2L legacy dim: {a2l_legacy.shape[1]} (expected 9) - {'PASS' if a2l_ok else 'FAIL'}")
    
    return a2a_ok and a2l_ok


def print_feature_stats() -> None:
    """Print statistics for edge features."""
    print("\n[STATS] Edge Feature Statistics")
    
    agent_features = create_mock_agent_features(n_agents=10)
    lane_features, lane_polylines = create_mock_lane_features(n_lanes=15)
    
    # Create more diverse edges
    edge_index = np.array([
        [0, 0, 0, 1, 1, 2, 2, 3, 4, 5],
        [1, 2, 3, 2, 4, 3, 5, 6, 7, 8],
    ], dtype=np.int64)
    
    a2a_attr = _compute_a2a_edge_attrs(agent_features, edge_index, use_enhanced=True)
    _, a2l_attr = build_a2l_edges(
        agent_features, lane_features, lane_polylines, k=3, use_frenet=True
    )
    
    # A2A feature names
    a2a_names = [
        "dx", "dy", "dist",
        "rel_vx", "rel_vy", "closing_speed",
        "ttc", "has_ttc",
        "is_ahead", "is_behind", "is_left", "is_right",
        "dist_very_close", "dist_close", "dist_medium", "dist_far",
        "is_approaching", "is_moving_away",
        "is_leading", "is_following",
        # Enhanced features
        "lateral_vel_i", "lateral_vel_j",
        "lateral_movement_i", "lateral_movement_j",
        "yaw_rate_i", "yaw_rate_j",
        "pred_dist_1s", "pred_dist_2s",
        "will_collide",
        "merging_score", "yielding_score", "cutting_in_score",
    ]
    
    print("\n  A2A Features (32 dims):")
    print(f"  {'Feature':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("  " + "-" * 65)
    for i, name in enumerate(a2a_names):
        if i < a2a_attr.shape[1]:
            col = a2a_attr[:, i]
            print(f"  {name:<25} {col.min():>10.3f} {col.max():>10.3f} {col.mean():>10.3f} {col.std():>10.3f}")
    
    # A2L feature names
    a2l_names = [
        "dx", "dy", "dist",
        "lateral_offset", "heading_alignment",
        "is_on_lane", "is_approaching_lane",
        "progress", "angle_to_lane",
        # Frenet features
        "s", "d", "d_dot",
        "v_longitudinal", "v_lateral", "v_lon_normalized",
    ]
    
    print("\n  A2L Features (15 dims):")
    print(f"  {'Feature':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("  " + "-" * 65)
    for i, name in enumerate(a2l_names):
        if i < a2l_attr.shape[1]:
            col = a2l_attr[:, i]
            print(f"  {name:<25} {col.min():>10.3f} {col.max():>10.3f} {col.mean():>10.3f} {col.std():>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test enhanced edge features")
    parser.add_argument("--check-dims", action="store_true", help="Check feature dimensions")
    parser.add_argument("--backward-compat", action="store_true", help="Test backward compatibility")
    parser.add_argument("--check-values", action="store_true", help="Check for NaN/Inf values")
    parser.add_argument("--stats", action="store_true", help="Print feature statistics")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    if not any([args.check_dims, args.backward_compat, args.check_values, args.stats, args.all]):
        args.all = True
    
    all_passed = True
    
    if args.check_dims or args.all:
        all_passed &= test_a2a_dimensions(use_enhanced=True)
        all_passed &= test_a2l_dimensions(use_frenet=True)
    
    if args.backward_compat or args.all:
        all_passed &= test_backward_compatibility()
    
    if args.check_values or args.all:
        all_passed &= test_no_nan_inf()
    
    if args.stats or args.all:
        print_feature_stats()
    
    print("\n" + "=" * 50)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
