#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import tensorflow as tf

# ==========================================
# 1. SEMANTIC DICTIONARIES (The "Meaning")
# ==========================================
# Based on Waymo Open Motion Dataset V1.2 standard
ROADGRAPH_TYPE_MAP = {
    1: "LANE_CENTER",
    2: "ROAD_EDGE_BOUNDARY",
    3: "ROAD_EDGE_MEDIAN",
    6: "CROSSWALK",
    15: "SPEED_BUMP",
    16: "DRIVEWAY",
    17: "STOP_SIGN",
    18: "SURFACE_STREET",
    19: "BIKE_LANE"
}

TL_STATE_MAP = {
    0: "UNKNOWN",
    1: "ARROW_STOP",
    2: "ARROW_CAUTION",
    3: "ARROW_GO",
    4: "STOP (RED)",
    5: "CAUTION (YELLOW)",
    6: "GO (GREEN)",
    7: "FLASHING_STOP",
    8: "FLASHING_CAUTION"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def has(f: Dict[str, tf.train.Feature], key: str) -> bool:
    return key in f and (len(f[key].float_list.value) > 0 or 
                         len(f[key].int64_list.value) > 0 or 
                         len(f[key].bytes_list.value) > 0)

def F(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    return np.array(f[key].float_list.value, dtype=np.float32)

def I(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    return np.array(f[key].int64_list.value, dtype=np.int64)

def load_record(tfrecord_path: str, record_index: int) -> tf.train.Example:
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    for i, raw_record in enumerate(raw_dataset):
        if i == record_index:
            return tf.train.Example.FromString(raw_record.numpy())
    raise IndexError(f"Record {record_index} not found.")

def ego_frame_proj(ego_pos, ego_yaw, target_pos):
    """Projects target_pos into Ego's Frenet-like frame (Longitudinal, Lateral)."""
    dx = target_pos[:, 0] - ego_pos[0]
    dy = target_pos[:, 1] - ego_pos[1]
    
    # Rotate by -yaw
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    longitudinal = dx * c - dy * s  # Front/Back
    lateral = dx * s + dy * c       # Left/Right
    
    return longitudinal, lateral, np.sqrt(dx**2 + dy**2)

# ==========================================
# 3. MAIN INSPECTION LOGIC
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True, help="Path to .tfrecord file")
    parser.add_argument("--record", type=int, default=0, help="Index of record to inspect")
    parser.add_argument("--radius", type=float, default=50.0, help="Search radius around Ego")
    parser.add_argument("--deep_scan", action="store_true", help="Print ALL keys in the file (to find hidden fields)")
    args = parser.parse_args()

    ex = load_record(args.tfrecord, args.record)
    f = ex.features.feature
    
    # --- 0. DEEP SCAN (Find Missing Data) ---
    if args.deep_scan:
        print("\n🔍 [DEEP SCAN] Listing ALL keys found in this record:")
        keys = sorted(f.keys())
        grouped = defaultdict(list)
        for k in keys:
            root = k.split('/')[0]
            grouped[root].append(k)
        
        for root, k_list in grouped.items():
            print(f"  📂 {root}/ ({len(k_list)} features)")
            for k in k_list[:5]: print(f"     - {k}")
            if len(k_list) > 5: print(f"     ... ({len(k_list)-5} more)")
        print("-" * 60)

    # --- 1. EGO STATE (The "Anchor" Node) ---
    cur_x = F(f, "state/current/x")
    cur_y = F(f, "state/current/y")
    cur_valid = I(f, "state/current/valid")
    is_sdc = I(f, "state/is_sdc")
    ego_idx = np.argmax(is_sdc)
    
    # Fallback for yaw if bbox_yaw is missing
    if has(f, "state/current/bbox_yaw"):
        ego_yaw = F(f, "state/current/bbox_yaw")[ego_idx]
    else:
        ego_yaw = F(f, "state/current/vel_yaw")[ego_idx]

    ego_pos = np.array([cur_x[ego_idx], cur_y[ego_idx]])
    
    print(f"\n🚗 [EGO NODE] Index: {ego_idx}")
    print(f"   Pos: ({ego_pos[0]:.2f}, {ego_pos[1]:.2f})")
    print(f"   Yaw: {ego_yaw:.2f} rad")
    print(f"   Valid Agents in Scene: {np.sum(cur_valid)}")

    # --- 2. ROADGRAPH (The "Spatial" Nodes) ---
    print(f"\n🗺️ [SPATIAL GRAPH NODES] (Radius: {args.radius}m)")
    
    if has(f, "roadgraph_samples/xyz"):
        rg_xyz = F(f, "roadgraph_samples/xyz").reshape(-1, 3)
        rg_type = I(f, "roadgraph_samples/type")
        rg_valid = I(f, "roadgraph_samples/valid").astype(bool)
        
        # Check for speed limits (Crucial for Rule-Based Edges)
        rg_speed = None
        if has(f, "roadgraph_samples/speed_limit"):
             rg_speed = F(f, "roadgraph_samples/speed_limit")
             print("   ✅ FOUND: Speed Limits available!")
        else:
             print("   ⚠️ MISSING: 'roadgraph_samples/speed_limit' not found.")

        # Filter valid points
        valid_indices = np.where(rg_valid)[0]
        pts = rg_xyz[valid_indices]
        types = rg_type[valid_indices]
        
        # Calculate relations (Edges)
        long, lat, dist = ego_frame_proj(ego_pos, ego_yaw, pts)
        
        # Filter by radius
        mask = dist < args.radius
        nearby_idx = valid_indices[mask]
        
        # Group by Type
        print(f"   Found {len(nearby_idx)} points nearby. Grouping by Semantic Type:")
        
        for t_code in sorted(np.unique(types[mask])):
            t_name = ROADGRAPH_TYPE_MAP.get(t_code, f"UNKNOWN_{t_code}")
            t_mask = (types[mask] == t_code)
            
            # Get closest point of this type
            subset_dist = dist[mask][t_mask]
            subset_long = long[mask][t_mask]
            subset_lat = lat[mask][t_mask]
            
            closest_i = np.argmin(subset_dist)
            c_dist = subset_dist[closest_i]
            c_long = subset_long[closest_i]
            c_lat = subset_lat[closest_i]
            
            # Speed Limit Check for Lane Centers
            speed_info = ""
            if t_code == 1 and rg_speed is not None: # Lane Center
                # Map back to original index
                orig_idx = nearby_idx[t_mask][closest_i]
                limit = rg_speed[orig_idx]
                speed_info = f"| 🛑 LIMIT: {limit*2.23:.0f} mph" # Approx convert m/s to mph
            
            # SEMANTIC EDGE LOGIC (Simulated)
            edge_desc = "Unknown Relation"
            if abs(c_lat) < 1.5 and c_long > 0: edge_desc = "ON_PATH / AHEAD"
            elif c_lat < -2.0: edge_desc = "TO_RIGHT"
            elif c_lat > 2.0: edge_desc = "TO_LEFT"
            
            print(f"     • {t_name:<18} : Closest is {c_dist:4.1f}m away {speed_info}")
            print(f"       -> Graph Edge: (Ego) --[{edge_desc} (Lat:{c_lat:.1f}m)]--> ({t_name})")

    # --- 3. TRAFFIC LIGHTS (The "Regulatory" Nodes) ---
    print(f"\n🚦 [TRAFFIC LIGHT NODES]")
    
    tl_keys = [k for k in f.keys() if "traffic_light" in k and "current/x" in k]
    if tl_keys:
        prefix = "traffic_light_state/current"
        tl_x = F(f, f"{prefix}/x")
        tl_y = F(f, f"{prefix}/y")
        tl_state = I(f, f"{prefix}/state")
        tl_valid = I(f, f"{prefix}/valid").astype(bool)
        
        # Check for Lane Association (Crucial for linking Signal -> Lane)
        tl_lane_ids = None
        # V-Max paper doesn't explicitly name this, so we check standard variations
        # Note: Often Waymo stores this connection in the MAP (Lane -> controlled_by), not the LIGHT
        # But we check anyway.
        
        valid_idx = np.where(tl_valid)[0]
        pts = np.stack([tl_x[valid_idx], tl_y[valid_idx]], axis=1)
        long, lat, dist = ego_frame_proj(ego_pos, ego_yaw, pts)
        
        mask = dist < 80.0  # TLs matter even if far away
        relevant_idx = valid_idx[mask]
        
        if len(relevant_idx) == 0:
            print("   No relevant traffic lights nearby.")
        else:
            for i, idx in enumerate(relevant_idx):
                s_code = tl_state[idx]
                s_name = TL_STATE_MAP.get(s_code, "UNKNOWN")
                d = dist[mask][i]
                l = long[mask][i]
                
                # Logic: Is it in front?
                rel = "BEHIND" if l < 0 else "AHEAD"
                
                print(f"     • Light ID {idx}: {s_name} is {d:.1f}m {rel}")
                if s_code in [4, 5, 7] and l > 0 and l < 40:
                    print(f"       -> ⚠️ CRITICAL EDGE: (Ego) --[MUST_STOP]--> (TL_{idx})")
    else:
        print("   No traffic light data found.")

if __name__ == "__main__":
    main()