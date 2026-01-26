#!/usr/bin/env python3
"""
Spatial Graph BEV Visualization

Visualizes spatial scene graphs in Bird's Eye View (BEV) with:
- Color-coded nodes (ego, map features, traffic lights, route)
- Styled edges (distances, confidence scores)
- Semantic labels (type names, states)
- Multiple visualization modes (simple, detailed, semantic)

Usage:
    python visualize_spatial_graph.py --graph outputs/record0_spatial_graph.json
    python visualize_spatial_graph.py --graph outputs/record0_spatial_graph.json --mode detailed
    python visualize_spatial_graph.py --dir outputs/ --output viz/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from pathlib import Path
import os


# ============================================================================
# Color Schemes
# ============================================================================

NODE_COLORS = {
    'EGO': '#FF0000',                      # Red
    'MAP_FEATURE': '#888888',              # Gray
    'TRAFFIC_LIGHT_CLUSTER': '#00FF00',    # Green (will vary by state)
    'ROUTE': '#0000FF',                    # Blue
}

MAP_TYPE_COLORS = {
    'FREEWAY': '#4A4A4A',
    'SURFACE_STREET': '#666666',
    'BIKE_LANE': '#FFA500',
    'ROAD_EDGE_BOUNDARY': '#8B4513',
    'ROAD_EDGE_MEDIAN': '#A0522D',
    'CROSSWALK': '#FFFF00',
    'STOP_SIGN': '#FF0000',
    'SPEED_BUMP': '#FF69B4',
    'DRIVABLE': '#555555',
    'BOUNDARY': '#8B4513',
    'MARKING': '#FFFFFF',
    'SPECIAL': '#FFD700',
}

TL_STATE_COLORS = {
    0: '#808080',  # UNKNOWN - Gray
    1: '#8B0000',  # ARROW_STOP - Dark Red
    2: '#FFD700',  # ARROW_CAUTION - Gold
    3: '#00FF00',  # ARROW_GO - Green
    4: '#FF0000',  # STOP (RED) - Red
    5: '#FFA500',  # CAUTION (YELLOW) - Orange
    6: '#00FF00',  # GO (GREEN) - Green
    7: '#FF1493',  # FLASHING_STOP - Deep Pink
    8: '#FFD700',  # FLASHING_CAUTION - Gold
}

EDGE_COLORS = {
    'EGO_TO_MAP': '#888888',
    'EGO_TO_TL': '#00FFFF',
    'EGO_TO_ROUTE': '#0000FF',
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_tl_state_name(state):
    """Convert TL state code to human-readable name."""
    names = {
        0: "UNKNOWN",
        1: "ARROW_STOP",
        2: "ARROW_CAUTION",
        3: "ARROW_GO",
        4: "STOP",
        5: "CAUTION",
        6: "GO",
        7: "FLASHING_STOP",
        8: "FLASHING_CAUTION",
    }
    return names.get(state, f"STATE_{state}")


def compute_bev_bounds(graph, margin=10.0):
    """Compute BEV plot bounds from graph nodes."""
    ego = [n for n in graph['nodes'] if n['type'] == 'EGO'][0]
    ego_x = ego['attrs']['x']
    ego_y = ego['attrs']['y']
    
    # Default bounds centered on ego
    x_min, x_max = ego_x - 60, ego_x + 60
    y_min, y_max = ego_y - 60, ego_y + 60
    
    # Expand to include all nodes
    for node in graph['nodes']:
        if node['type'] == 'MAP_FEATURE':
            bbox = node['attrs'].get('bbox_xy_min')
            if bbox:
                x_min = min(x_min, bbox[0] - margin)
                y_min = min(y_min, bbox[1] - margin)
            bbox = node['attrs'].get('bbox_xy_max')
            if bbox:
                x_max = max(x_max, bbox[0] + margin)
                y_max = max(y_max, bbox[1] + margin)
        elif node['type'] == 'TRAFFIC_LIGHT_CLUSTER':
            x, y = node['attrs']['x'], node['attrs']['y']
            x_min = min(x_min, x - margin)
            x_max = max(x_max, x + margin)
            y_min = min(y_min, y - margin)
            y_max = max(y_max, y + margin)
    
    return x_min, x_max, y_min, y_max


def rotate_vector(x, y, yaw):
    """Rotate vector by yaw angle (for ego heading visualization)."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x_rot = x * cos_yaw - y * sin_yaw
    y_rot = x * sin_yaw + y * cos_yaw
    return x_rot, y_rot


# ============================================================================
# Simple Visualization (Clean, Publication-Ready)
# ============================================================================

def visualize_simple(graph, ax=None, show_labels=True):
    """
    Simple, clean BEV visualization.
    
    Shows:
    - Ego vehicle (red arrow)
    - Map features (gray polylines)
    - Traffic lights (colored by state)
    - Route (blue dashed line)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    
    # Get nodes
    ego = [n for n in graph['nodes'] if n['type'] == 'EGO'][0]
    map_nodes = [n for n in graph['nodes'] if n['type'] == 'MAP_FEATURE']
    tl_nodes = [n for n in graph['nodes'] if n['type'] == 'TRAFFIC_LIGHT_CLUSTER']
    route_nodes = [n for n in graph['nodes'] if n['type'] == 'ROUTE']
    
    ego_x = ego['attrs']['x']
    ego_y = ego['attrs']['y']
    ego_yaw = ego['attrs']['yaw']
    
    # 1. Draw map features (polylines)
    for node in map_nodes:
        points = node['attrs'].get('sample_points_xy', [])
        if len(points) >= 2:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], 
                   color='#CCCCCC', linewidth=2, alpha=0.6, zorder=1)
    
    # 2. Draw route (if exists)
    for node in route_nodes:
        # Route doesn't have sample_points in current schema, skip for now
        centroid = node['attrs']['centroid_xy']
        ax.scatter(centroid[0], centroid[1], 
                  color='#0000FF', s=200, alpha=0.3, zorder=2, label='Route')
    
    # 3. Draw traffic lights
    for node in tl_nodes:
        x = node['attrs']['x']
        y = node['attrs']['y']
        state = node['attrs']['representative_state']
        color = TL_STATE_COLORS.get(state, '#808080')
        
        ax.scatter(x, y, color=color, s=300, 
                  edgecolors='black', linewidth=2, zorder=4,
                  marker='o')
        
        if show_labels:
            state_name = get_tl_state_name(state)
            ax.annotate(state_name, (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    # 4. Draw ego vehicle (arrow)
    arrow_length = 5.0
    arrow_width = 2.0
    
    # Compute arrow tip
    dx, dy = rotate_vector(arrow_length, 0, ego_yaw)
    
    ax.arrow(ego_x, ego_y, dx, dy,
            head_width=arrow_width, head_length=2.0,
            fc='#FF0000', ec='black', linewidth=2, zorder=5)
    
    if show_labels:
        speed = ego['attrs']['speed_mps']
        ax.annotate(f'EGO\n{speed:.1f} m/s', (ego_x, ego_y),
                   xytext=(-15, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # 5. Set bounds and styling
    x_min, x_max, y_min, y_max = compute_bev_bounds(graph)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Spatial Graph - Bird\'s Eye View', fontsize=14, fontweight='bold')
    
    return ax


# ============================================================================
# Detailed Visualization (With Edge Connections)
# ============================================================================

def visualize_detailed(graph, ax=None, show_edges=True, show_confidence=True):
    """
    Detailed BEV visualization with edges and metadata.
    
    Shows:
    - All features from simple view
    - Edge connections (ego to map/TL/route)
    - Confidence scores (color-coded)
    - Distance labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 14))
    
    # First draw simple visualization
    visualize_simple(graph, ax=ax, show_labels=True)
    
    if not show_edges:
        return ax
    
    ego = [n for n in graph['nodes'] if n['type'] == 'EGO'][0]
    ego_x = ego['attrs']['x']
    ego_y = ego['attrs']['y']
    
    # Build node lookup
    node_lookup = {n['id']: n for n in graph['nodes']}
    
    # Draw edges
    for edge in graph['edges']:
        src_node = node_lookup.get(edge['src'])
        dst_node = node_lookup.get(edge['dst'])
        
        if not src_node or not dst_node:
            continue
        
        edge_type = edge['type']
        attrs = edge.get('attrs', {})
        
        # Get coordinates
        if src_node['type'] == 'EGO':
            x1, y1 = ego_x, ego_y
        else:
            x1 = src_node['attrs'].get('x', src_node['attrs'].get('centroid_xy', [0, 0])[0])
            y1 = src_node['attrs'].get('y', src_node['attrs'].get('centroid_xy', [0, 0])[1])
        
        if dst_node['type'] == 'MAP_FEATURE':
            closest = attrs.get('closest_world_xy')
            if closest:
                x2, y2 = closest
            else:
                x2, y2 = dst_node['attrs']['centroid_xy']
        elif dst_node['type'] == 'TRAFFIC_LIGHT_CLUSTER':
            x2, y2 = dst_node['attrs']['x'], dst_node['attrs']['y']
        elif dst_node['type'] == 'ROUTE':
            x2, y2 = dst_node['attrs']['centroid_xy']
        else:
            continue
        
        # Color by edge type and confidence
        base_color = EDGE_COLORS.get(edge_type, '#888888')
        alpha = 0.3
        linewidth = 1
        
        if edge_type == 'EGO_TO_TL' and show_confidence:
            confidence = attrs.get('control_confidence', 0.0)
            alpha = 0.2 + 0.6 * confidence  # Higher confidence = more visible
            linewidth = 1 + 2 * confidence
        
        # Draw edge
        ax.plot([x1, x2], [y1, y2], 
               color=base_color, alpha=alpha, linewidth=linewidth,
               linestyle='--', zorder=3)
        
        # Add distance label for close edges
        if edge_type == 'EGO_TO_TL':
            dist = attrs.get('dist_m', 0)
            if dist < 20:  # Only label nearby TLs
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(f'{dist:.1f}m', (mid_x, mid_y),
                           fontsize=7, color='cyan',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='black', alpha=0.7))
    
    ax.set_title('Spatial Graph - Detailed View (with Edges)', 
                fontsize=14, fontweight='bold')
    
    return ax


# ============================================================================
# Semantic Visualization (Color-Coded by Type)
# ============================================================================

def visualize_semantic(graph, ax=None, show_type_names=True):
    """
    Semantic BEV visualization with type-based coloring.
    
    Shows:
    - Map features colored by type (SURFACE_STREET, BIKE_LANE, etc.)
    - Type name labels
    - Speed limit annotations
    - Legend with type categories
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 14))
    
    ego = [n for n in graph['nodes'] if n['type'] == 'EGO'][0]
    map_nodes = [n for n in graph['nodes'] if n['type'] == 'MAP_FEATURE']
    tl_nodes = [n for n in graph['nodes'] if n['type'] == 'TRAFFIC_LIGHT_CLUSTER']
    
    ego_x = ego['attrs']['x']
    ego_y = ego['attrs']['y']
    ego_yaw = ego['attrs']['yaw']
    
    # 1. Draw map features with type-based colors
    type_counts = {}
    for node in map_nodes:
        points = node['attrs'].get('sample_points_xy', [])
        type_name = node['attrs'].get('type_name', 'UNKNOWN')
        type_category = node['attrs'].get('type_category', 'UNKNOWN')
        
        # Get color (try specific type, then category, then default)
        color = MAP_TYPE_COLORS.get(type_name, 
                MAP_TYPE_COLORS.get(type_category, '#888888'))
        
        if len(points) >= 2:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], 
                   color=color, linewidth=3, alpha=0.8, zorder=1)
            
            # Label with type name (only for DRIVABLE and SPECIAL)
            if show_type_names and type_category in ['DRIVABLE', 'SPECIAL']:
                centroid = node['attrs']['centroid_xy']
                ax.annotate(type_name, centroid,
                           fontsize=6, color='white',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor=color, alpha=0.9))
            
            # Track type for legend
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    # 2. Draw traffic lights (same as simple)
    for node in tl_nodes:
        x = node['attrs']['x']
        y = node['attrs']['y']
        state = node['attrs']['representative_state']
        color = TL_STATE_COLORS.get(state, '#808080')
        
        ax.scatter(x, y, color=color, s=400, 
                  edgecolors='black', linewidth=3, zorder=4,
                  marker='s')  # Square for TLs
        
        state_name = get_tl_state_name(state)
        ax.annotate(state_name, (x, y), 
                   xytext=(0, -20), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95))
    
    # 3. Draw ego (same as simple)
    arrow_length = 6.0
    dx, dy = rotate_vector(arrow_length, 0, ego_yaw)
    ax.arrow(ego_x, ego_y, dx, dy,
            head_width=3.0, head_length=2.5,
            fc='#FF0000', ec='black', linewidth=3, zorder=5)
    
    speed = ego['attrs']['speed_mps']
    ax.annotate(f'EGO\n{speed:.1f} m/s', (ego_x, ego_y),
               xytext=(-20, -20), textcoords='offset points',
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95,
                        edgecolor='red', linewidth=2))
    
    # 4. Create legend
    legend_elements = []
    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1])[:8]:
        color = MAP_TYPE_COLORS.get(type_name, '#888888')
        legend_elements.append(
            mpatches.Patch(color=color, label=f'{type_name} ({count})')
        )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                 title='Map Feature Types', framealpha=0.9)
    
    # 5. Set bounds and styling
    x_min, x_max, y_min, y_max = compute_bev_bounds(graph)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title('Spatial Graph - Semantic View (Type-Coded)', 
                fontsize=15, fontweight='bold')
    
    return ax


# ============================================================================
# Combined Multi-Panel Visualization
# ============================================================================

def visualize_combined(graph, save_path=None):
    """
    Create a 2x2 panel visualization showing all modes.
    
    Panels:
    1. Simple view
    2. Detailed view (with edges)
    3. Semantic view (type-coded)
    4. Metadata summary
    """
    fig = plt.figure(figsize=(20, 20))
    
    # Panel 1: Simple
    ax1 = plt.subplot(2, 2, 1)
    visualize_simple(graph, ax=ax1, show_labels=True)
    ax1.set_title('Simple View', fontsize=14, fontweight='bold')
    
    # Panel 2: Detailed
    ax2 = plt.subplot(2, 2, 2)
    visualize_detailed(graph, ax=ax2, show_edges=True, show_confidence=True)
    ax2.set_title('Detailed View (Edges + Confidence)', fontsize=14, fontweight='bold')
    
    # Panel 3: Semantic
    ax3 = plt.subplot(2, 2, 3)
    visualize_semantic(graph, ax=ax3, show_type_names=True)
    ax3.set_title('Semantic View (Type-Coded)', fontsize=14, fontweight='bold')
    
    # Panel 4: Metadata summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    meta = graph.get('meta', {})
    
    # Build summary text
    summary_lines = [
        "SPATIAL GRAPH SUMMARY",
        "=" * 40,
        "",
        f"Nodes: {meta.get('num_nodes', 'N/A')}",
        f"Edges: {meta.get('num_edges', 'N/A')}",
        "",
        "Map Features:",
        f"  Raw: {meta.get('map_features', {}).get('raw_count', 'N/A')}",
        f"  Filtered: {meta.get('map_features', {}).get('filtered_count', 'N/A')}",
        "",
        "Traffic Lights:",
        f"  Total: {meta.get('traffic_lights', {}).get('total_valid', 'N/A')}",
        f"  Clusters: {meta.get('traffic_lights', {}).get('num_clusters', 'N/A')}",
        "",
        "Filter Stats:",
    ]
    
    filter_stats = meta.get('map_features', {}).get('filter_stats', {})
    if filter_stats:
        for key, val in filter_stats.items():
            summary_lines.append(f"  {key}: {val}")
    
    # Add route info
    route_edges = [e for e in graph['edges'] if e['type'] == 'EGO_TO_ROUTE']
    if route_edges:
        route_edge = route_edges[0]
        summary_lines.extend([
            "",
            "Route Status:",
            f"  Status: {route_edge['attrs'].get('status', 'N/A')}",
            f"  Lateral: {route_edge['attrs'].get('signed_lateral_m', 0):.2f} m",
        ])
    
    # Add TL info
    tl_edges = [e for e in graph['edges'] if e['type'] == 'EGO_TO_TL']
    if tl_edges:
        nearby_tls = [e for e in tl_edges if e['attrs'].get('likely_controls_ego', False)]
        summary_lines.extend([
            "",
            f"Nearby TLs: {len(nearby_tls)} / {len(tl_edges)}",
        ])
        
        if nearby_tls:
            top_tl = max(nearby_tls, key=lambda e: e['attrs'].get('control_confidence', 0))
            conf = top_tl['attrs'].get('control_confidence', 0)
            dist = top_tl['attrs'].get('dist_m', 0)
            summary_lines.append(f"  Top confidence: {conf:.2f} at {dist:.1f}m")
    
    summary_text = "\n".join(summary_lines)
    ax4.text(0.1, 0.95, summary_text, 
            transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization -> {save_path}")
    else:
        plt.show()
    
    return fig


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize spatial scene graphs in BEV')
    parser.add_argument('--graph', type=str, help='Path to spatial graph JSON file')
    parser.add_argument('--dir', type=str, help='Directory of JSON files to process')
    parser.add_argument('--output', type=str, default='visualizations/',
                       help='Output directory for saved figures')
    parser.add_argument('--mode', type=str, default='combined',
                       choices=['simple', 'detailed', 'semantic', 'combined'],
                       help='Visualization mode')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument('--show', action='store_true', help='Show plot instead of saving')
    
    args = parser.parse_args()
    
    # Create output directory
    if not args.show:
        os.makedirs(args.output, exist_ok=True)
    
    # Process single file
    if args.graph:
        with open(args.graph, 'r') as f:
            graph = json.load(f)
        
        if args.mode == 'simple':
            fig, ax = plt.subplots(figsize=(12, 12))
            visualize_simple(graph, ax=ax)
        elif args.mode == 'detailed':
            fig, ax = plt.subplots(figsize=(14, 14))
            visualize_detailed(graph, ax=ax)
        elif args.mode == 'semantic':
            fig, ax = plt.subplots(figsize=(14, 14))
            visualize_semantic(graph, ax=ax)
        else:  # combined
            if args.show:
                visualize_combined(graph, save_path=None)
            else:
                base_name = Path(args.graph).stem
                save_path = os.path.join(args.output, f'{base_name}_combined.png')
                visualize_combined(graph, save_path=save_path)
                return
        
        if args.show:
            plt.show()
        else:
            base_name = Path(args.graph).stem
            save_path = os.path.join(args.output, f'{base_name}_{args.mode}.png')
            plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved visualization -> {save_path}")
        
        plt.close()
    
    # Process directory
    elif args.dir:
        json_files = sorted(Path(args.dir).glob('record*_spatial_graph.json'))
        
        if not json_files:
            print(f"No spatial graph JSON files found in {args.dir}")
            return
        
        print(f"Found {len(json_files)} files to visualize...")
        
        for i, json_file in enumerate(json_files):
            print(f"[{i+1}/{len(json_files)}] Processing {json_file.name}...")
            
            with open(json_file, 'r') as f:
                graph = json.load(f)
            
            base_name = json_file.stem
            save_path = os.path.join(args.output, f'{base_name}_{args.mode}.png')
            
            try:
                if args.mode == 'combined':
                    visualize_combined(graph, save_path=save_path)
                else:
                    fig, ax = plt.subplots(figsize=(12, 12))
                    if args.mode == 'simple':
                        visualize_simple(graph, ax=ax)
                    elif args.mode == 'detailed':
                        visualize_detailed(graph, ax=ax)
                    else:  # semantic
                        visualize_semantic(graph, ax=ax)
                    
                    plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved -> {save_path}")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\nDone! Visualizations saved to {args.output}/")
    
    else:
        print("Error: Must provide --graph or --dir")
        parser.print_help()


if __name__ == '__main__':
    main()
