#!/usr/bin/env python3
"""
viz_hetero_graph.py

Visualize and validate the HeteroData graph built by hetero_graph.py.
Uses ONLY data from HeteroData object - no TFRecord re-reading required.

Usage:
    python viz_hetero_graph.py --tfrecord path/to/file.tfrecord --record 0

Optional comparison with old semantic graph:
    python viz_hetero_graph.py --tfrecord path/to/file.tfrecord --record 0 --old_graphs semantic_graphs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

# Import our hetero graph builder
from hetero_graph import build_hetero_graph


def load_old_graph(jsonl_path: str, record_index: int) -> Optional[Dict]:
    """Load graph from old JSONL format for comparison."""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == record_index:
                    return json.loads(line)
    except FileNotFoundError:
        return None
    return None


def validate_relations(relations: List[Dict], verbose: bool = True) -> List[tuple]:
    """
    Validate that semantic relations are consistent with geometric data.
    Returns list of (dst, label, x_local, y_local, closing) mismatches.
    """
    mismatches = []

    for rel in relations:
        x_local = rel["x_local_m"]
        y_local = rel["y_local_m"]
        closing = rel["closing_speed"]
        dst = rel["dst"]
        labels = set(rel["relations"])

        # Check ahead_of / behind_of consistency
        if "ahead_of" in labels and not (x_local > 0):
            mismatches.append((dst, "ahead_of", x_local, y_local, closing))
        if "behind_of" in labels and not (x_local < 0):
            mismatches.append((dst, "behind_of", x_local, y_local, closing))

        # Check left_of / right_of consistency
        if "left_of" in labels and not (y_local > 0):
            mismatches.append((dst, "left_of", x_local, y_local, closing))
        if "right_of" in labels and not (y_local < 0):
            mismatches.append((dst, "right_of", x_local, y_local, closing))

        # Check approaching / moving_away consistency
        if "approaching" in labels and not (closing > 0.5):
            mismatches.append((dst, "approaching", x_local, y_local, closing))
        if "moving_away" in labels and not (closing < -0.5):
            mismatches.append((dst, "moving_away", x_local, y_local, closing))

        # Check distance bucket consistency
        dist = rel["distance_m"]
        if "very_close" in labels and not (dist < 5.0):
            mismatches.append((dst, "very_close", x_local, y_local, dist))
        if "close" in labels and not (5.0 <= dist < 15.0):
            mismatches.append((dst, "close", x_local, y_local, dist))
        if "medium" in labels and not (15.0 <= dist < 30.0):
            mismatches.append((dst, "medium", x_local, y_local, dist))
        if "far" in labels and not (dist >= 30.0):
            mismatches.append((dst, "far", x_local, y_local, dist))

    if verbose:
        print(f"\nValidation: {len(mismatches)} mismatches found")
        for m in mismatches[:10]:
            dst, lab, xl, yl, val = m
            print(f"  dst={dst:3d} label={lab:15s} x_local={xl:7.2f} y_local={yl:7.2f} val={val:6.2f}")
        if len(mismatches) > 10:
            print("  ... (truncated)")

    return mismatches


def validate_hetero_graph(data, a2l_k: int = 3, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of HeteroData graph structure.

    Checks:
    1. Ego agent is at position (0, 0) in ego frame
    2. Ego agent has is_ego=1 flag set
    3. All A2A edges originate from ego (src == ego_idx)
    4. Number of A2L edges == num_agents * k
    5. Semantic labels match geometric data
    6. No NaN or Inf values in any tensor

    Returns:
        (all_passed, list_of_errors)
    """
    errors = []
    ego_idx = data.ego_idx

    # 1. Check ego is at origin
    ego_x = float(data['agent'].x[ego_idx, 0])
    ego_y = float(data['agent'].x[ego_idx, 1])
    if abs(ego_x) > 1e-4 or abs(ego_y) > 1e-4:
        errors.append(f"Ego not at origin: ({ego_x:.4f}, {ego_y:.4f})")

    # 2. Check ego has is_ego flag set
    is_ego_flag = float(data['agent'].x[ego_idx, 15])
    if is_ego_flag != 1.0:
        errors.append(f"Ego is_ego flag not set: got {is_ego_flag}")

    # Check no other agent has is_ego flag
    for i in range(data['agent'].x.shape[0]):
        if i != ego_idx:
            flag = float(data['agent'].x[i, 15])
            if flag == 1.0:
                errors.append(f"Non-ego agent {i} has is_ego=1.0")

    # 3. Check all A2A edges originate from ego
    a2a_edge_index = data['agent', 'to', 'agent'].edge_index
    if a2a_edge_index.shape[1] > 0:
        src_nodes = a2a_edge_index[0].numpy()
        non_ego_src = np.sum(src_nodes != ego_idx)
        if non_ego_src > 0:
            errors.append(f"A2A edges: {non_ego_src} edges don't originate from ego")

    # 4. Check A2L edge count
    n_agents = data['agent'].x.shape[0]
    n_lanes = data['lane'].x.shape[0]
    a2l_edge_index = data['agent', 'to', 'lane'].edge_index
    expected_a2l = n_agents * min(a2l_k, n_lanes)
    actual_a2l = a2l_edge_index.shape[1]
    if actual_a2l != expected_a2l and n_lanes > 0:
        errors.append(f"A2L edge count: expected {expected_a2l}, got {actual_a2l}")

    # 5. Semantic label consistency (via validate_relations)
    relation_mismatches = validate_relations(data.a2a_relations, verbose=False)
    if relation_mismatches:
        errors.append(f"Semantic label mismatches: {len(relation_mismatches)} found")

    # 6. Check for NaN/Inf in all tensors
    tensors_to_check = [
        ("agent.x", data['agent'].x),
        ("lane.x", data['lane'].x),
        ("a2a.edge_attr", data['agent', 'to', 'agent'].edge_attr),
        ("a2l.edge_attr", data['agent', 'to', 'lane'].edge_attr),
        ("roadgraph_ego", data.roadgraph_ego),
    ]
    for name, tensor in tensors_to_check:
        if tensor.numel() > 0:
            if torch.isnan(tensor).any():
                errors.append(f"NaN values found in {name}")
            if torch.isinf(tensor).any():
                errors.append(f"Inf values found in {name}")

    # Print results
    if verbose:
        print("\n=== Graph Validation ===")
        print(f"Ego index: {ego_idx}")
        print(f"Ego position: ({ego_x:.4f}, {ego_y:.4f})")
        print(f"Ego is_ego flag: {is_ego_flag}")
        print(f"A2A edges from ego: {a2a_edge_index.shape[1]}")
        print(f"A2L edges: {actual_a2l} (expected {expected_a2l})")

        if errors:
            print(f"\n[FAIL] {len(errors)} validation errors:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\n[PASS] All validation checks passed!")

    return len(errors) == 0, errors


def compare_with_old_graph(new_relations: List[Dict], old_graph: Dict) -> None:
    """Compare new HeteroData relations with old JSONL graph."""
    old_edges = {e["dst"]: e for e in old_graph.get("edges", [])}
    new_edges = {r["dst"]: r for r in new_relations}

    print("\n=== Comparison with Old Graph ===")
    print(f"Old graph edges: {len(old_edges)}")
    print(f"New graph edges: {len(new_edges)}")

    # Find common edges
    common_dsts = set(old_edges.keys()) & set(new_edges.keys())
    print(f"Common edges: {len(common_dsts)}")

    # Compare relations for common edges
    relation_diffs = []
    for dst in sorted(common_dsts)[:10]:  # Check first 10
        old_rels = set(old_edges[dst].get("relations", []))
        new_rels = set(new_edges[dst].get("relations", []))

        if old_rels != new_rels:
            only_old = old_rels - new_rels
            only_new = new_rels - old_rels
            relation_diffs.append((dst, only_old, only_new))

    if relation_diffs:
        print("\nRelation differences (first 10):")
        for dst, only_old, only_new in relation_diffs:
            print(f"  dst={dst}: old_only={only_old}, new_only={only_new}")
    else:
        print("\nNo relation differences in checked edges!")

    # Compare numeric values
    print("\nNumeric comparison (first 5 common edges):")
    for dst in sorted(common_dsts)[:5]:
        old_e = old_edges[dst]
        new_e = new_edges[dst]
        print(f"  dst={dst}:")
        print(f"    old: dist={old_e.get('distance_m', 0):.2f}, "
              f"x_local={old_e.get('x_local_m', 0):.2f}, "
              f"y_local={old_e.get('y_local_m', 0):.2f}, "
              f"closing={old_e.get('closing_speed', 0):.2f}")
        print(f"    new: dist={new_e['distance_m']:.2f}, "
              f"x_local={new_e['x_local_m']:.2f}, "
              f"y_local={new_e['y_local_m']:.2f}, "
              f"closing={new_e['closing_speed']:.2f}")


def get_edge_color(relations: List[str]) -> str:
    """Get edge color based on semantic relations."""
    if "ttc_imminent" in relations:
        return "red"
    elif "ttc_soon" in relations:
        return "orange"
    elif "approaching" in relations:
        return "yellow"
    elif "leading" in relations or "following" in relations:
        return "cyan"
    elif "moving_away" in relations:
        return "green"
    return "gray"


def plot_hetero_graph(
    data,
    show_ids: bool = False,
    show_lanes: bool = True,
    max_edge_dist: float = 80.0,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the HeteroData graph with semantic edge coloring.
    Uses ONLY data from the HeteroData object - no TFRecord re-reading.
    """
    record_index = data.record_index

    # Get ego state from HeteroData
    ego_x = data.ego_x
    ego_y = data.ego_y
    ego_yaw = data.ego_yaw

    # Get roadgraph points (already in ego frame, stored in HeteroData)
    rg_ego = data.roadgraph_ego.numpy()

    # Get agent positions (already in ego frame)
    agent_x_ego = data['agent'].x[:, 0].numpy()
    agent_y_ego = data['agent'].x[:, 1].numpy()

    # Transform agents back to world (for world plot)
    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)
    agent_x_world = agent_x_ego * cos_yaw - agent_y_ego * sin_yaw + ego_x
    agent_y_world = agent_x_ego * sin_yaw + agent_y_ego * cos_yaw + ego_y

    # Transform roadgraph back to world (for world plot)
    rg_x_world = rg_ego[:, 0] * cos_yaw - rg_ego[:, 1] * sin_yaw + ego_x
    rg_y_world = rg_ego[:, 0] * sin_yaw + rg_ego[:, 1] * cos_yaw + ego_y

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ============ Left plot: World coordinates ============
    ax1 = axes[0]
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(f"Record {record_index} - World Coordinates")

    # Plot roadgraph
    ax1.scatter(rg_x_world, rg_y_world, s=1, alpha=0.3, c="gray", label="Roadgraph")

    # Plot agents
    ax1.scatter(agent_x_world, agent_y_world, s=40, c="blue", zorder=5, label="Agents")

    # Highlight ego
    ego_idx = data.ego_idx
    ax1.scatter([agent_x_world[ego_idx]], [agent_y_world[ego_idx]],
                s=200, marker="*", c="gold", edgecolors="black", zorder=10, label="Ego")

    # Ego heading arrow
    arrow_len = 8.0
    ax1.arrow(agent_x_world[ego_idx], agent_y_world[ego_idx],
              arrow_len * np.cos(ego_yaw), arrow_len * np.sin(ego_yaw),
              head_width=2.0, length_includes_head=True, fc="gold", ec="black", zorder=10)

    # Plot edges with semantic coloring
    for rel in data.a2a_relations:
        if rel["distance_m"] > max_edge_dist:
            continue
        dst = rel["dst"]
        color = get_edge_color(rel["relations"])
        ax1.plot([agent_x_world[ego_idx], agent_x_world[dst]],
                 [agent_y_world[ego_idx], agent_y_world[dst]],
                 linewidth=1.5, alpha=0.7, c=color, zorder=3)

    if show_ids:
        for i in range(len(agent_x_world)):
            ax1.text(agent_x_world[i], agent_y_world[i], str(i), fontsize=8)

    ax1.set_xlabel("X (world)")
    ax1.set_ylabel("Y (world)")
    ax1.legend(loc="upper right", fontsize=8)

    # ============ Right plot: Ego-centric coordinates ============
    ax2 = axes[1]
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title(f"Record {record_index} - Ego-Centric Frame")

    # Roadgraph already in ego frame
    ax2.scatter(rg_ego[:, 0], rg_ego[:, 1], s=1, alpha=0.3, c="gray")

    # Plot agents in ego frame
    ax2.scatter(agent_x_ego, agent_y_ego, s=40, c="blue", zorder=5)

    # Ego at origin
    ax2.scatter([0], [0], s=200, marker="*", c="gold", edgecolors="black", zorder=10)

    # Ego heading arrow (points in +x direction in ego frame)
    ax2.arrow(0, 0, arrow_len, 0,
              head_width=2.0, length_includes_head=True, fc="gold", ec="black", zorder=10)

    # Plot edges with semantic coloring and labels
    for rel in data.a2a_relations:
        if rel["distance_m"] > max_edge_dist:
            continue
        dst = rel["dst"]
        color = get_edge_color(rel["relations"])
        ax2.plot([0, agent_x_ego[dst]], [0, agent_y_ego[dst]],
                 linewidth=1.5, alpha=0.7, c=color, zorder=3)

    # Add lane features if available
    if show_lanes and data['lane'].x.shape[0] > 0:
        lane_x = data['lane'].x[:, 0].numpy()  # centroid x
        lane_y = data['lane'].x[:, 1].numpy()  # centroid y
        ax2.scatter(lane_x, lane_y, s=15, c="purple", alpha=0.5, marker="s", zorder=2, label="Lane centroids")

    if show_ids:
        for i in range(len(agent_x_ego)):
            ax2.text(agent_x_ego[i], agent_y_ego[i], str(i), fontsize=8)

    # Add quadrant labels
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.text(30, 15, "ahead + left", fontsize=9, alpha=0.5)
    ax2.text(30, -15, "ahead + right", fontsize=9, alpha=0.5)
    ax2.text(-30, 15, "behind + left", fontsize=9, alpha=0.5)
    ax2.text(-30, -15, "behind + right", fontsize=9, alpha=0.5)

    ax2.set_xlabel("X_local (forward)")
    ax2.set_ylabel("Y_local (left)")
    ax2.set_xlim(-60, 60)
    ax2.set_ylim(-40, 40)

    # Legend for edge colors
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='TTC imminent'),
        Line2D([0], [0], color='orange', lw=2, label='TTC soon'),
        Line2D([0], [0], color='yellow', lw=2, label='Approaching'),
        Line2D([0], [0], color='cyan', lw=2, label='Leading/Following'),
        Line2D([0], [0], color='green', lw=2, label='Moving away'),
        Line2D([0], [0], color='gray', lw=2, label='Other'),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def print_edge_stats(relations: List[Dict]) -> None:
    """Print statistics about edge semantic labels."""
    print("\n=== Edge Statistics ===")
    print(f"Total A2A edges: {len(relations)}")

    # Count each relation type
    rel_counts: Dict[str, int] = {}
    for rel in relations:
        for label in rel["relations"]:
            rel_counts[label] = rel_counts.get(label, 0) + 1

    print("\nRelation counts:")
    for label in sorted(rel_counts.keys()):
        print(f"  {label:20s}: {rel_counts[label]:4d}")

    # Distance stats
    dists = [rel["distance_m"] for rel in relations]
    if dists:
        print(f"\nDistance stats:")
        print(f"  min: {min(dists):.1f}m, max: {max(dists):.1f}m, mean: {np.mean(dists):.1f}m")


def main():
    parser = argparse.ArgumentParser(description="Visualize HeteroGraph")
    parser.add_argument("--tfrecord", required=True, help="Path to TFRecord")
    parser.add_argument("--record", type=int, default=0, help="Record index")
    parser.add_argument("--old_graphs", type=str, default=None,
                        help="Optional: old semantic_graphs.jsonl for comparison")
    parser.add_argument("--show_ids", action="store_true", help="Show agent IDs")
    parser.add_argument("--no_lanes", action="store_true", help="Hide lane centroids")
    parser.add_argument("--max_dist", type=float, default=80.0,
                        help="Max edge distance to display")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting (validation only)")
    args = parser.parse_args()

    print(f"Building HeteroGraph for record {args.record}...")
    data = build_hetero_graph(args.tfrecord, args.record)

    print("\n=== HeteroData Structure ===")
    print(f"Agent nodes: {data['agent'].x.shape}")
    print(f"Lane nodes:  {data['lane'].x.shape}")
    print(f"A2A edges:   {data['agent', 'to', 'agent'].edge_index.shape[1]} edges, "
          f"{data['agent', 'to', 'agent'].edge_attr.shape[1]} features each")
    print(f"A2L edges:   {data['agent', 'to', 'lane'].edge_index.shape[1]} edges")
    print(f"Ego index:   {data.ego_idx}")

    # Print edge statistics
    print_edge_stats(data.a2a_relations)

    # Comprehensive graph validation
    all_valid, validation_errors = validate_hetero_graph(data)

    # Validate semantic relations
    relation_mismatches = validate_relations(data.a2a_relations)

    # Compare with old graph if provided
    if args.old_graphs:
        old_graph = load_old_graph(args.old_graphs, args.record)
        if old_graph:
            compare_with_old_graph(data.a2a_relations, old_graph)
        else:
            print(f"\nWarning: Could not load old graph from {args.old_graphs}")

    # Plot
    if not args.no_plot:
        plot_hetero_graph(
            data,
            show_ids=args.show_ids,
            show_lanes=not args.no_lanes,
            max_edge_dist=args.max_dist,
            save_path=args.save,
        )

    # Summary
    print("\n=== Final Summary ===")
    if all_valid and len(relation_mismatches) == 0:
        print("All validations passed! Graph is structurally sound.")
    else:
        total_issues = len(validation_errors) + len(relation_mismatches)
        print(f"Found {total_issues} total issues - review output above")


if __name__ == "__main__":
    main()
