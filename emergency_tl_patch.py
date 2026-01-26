#!/usr/bin/env python3
"""
EMERGENCY QUICK-FIX for Traffic Light Bug

Apply this patch to immediately see if TL data exists in your TFRecords.
This is a TEMPORARY diagnostic tool, not a permanent fix.

Usage:
    python emergency_tl_patch.py \
      --tfrecord training.tfrecord \
      --record 2 \
      --output fixed_record2_spatial_graph.json
"""

import tensorflow as tf
import numpy as np
import json
import argparse
from pathlib import Path


def extract_and_force_tls(tfrecord_path, record_idx):
    """
    Emergency extraction: Bypass all filtering, force TLs into graph.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for i, raw_bytes in enumerate(dataset):
        if i != record_idx:
            continue
        
        example = tf.train.Example()
        example.ParseFromString(raw_bytes.numpy())
        features = example.features.feature
        
        # Extract ego
        ego_x = features['state/current/x'].float_list.value[0]
        ego_y = features['state/current/y'].float_list.value[0]
        ego_yaw = features['state/current/bbox_yaw'].float_list.value[0]
        ego_vx = features['state/current/velocity_x'].float_list.value[0]
        ego_vy = features['state/current/velocity_y'].float_list.value[0]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        
        # Extract TLs - FORCE EXTRACTION
        tl_state_raw = features['traffic_light_state/current/state'].int64_list.value
        tl_valid_raw = features['traffic_light_state/current/valid'].int64_list.value
        tl_x_raw = features['traffic_light_state/current/x'].float_list.value
        tl_y_raw = features['traffic_light_state/current/y'].float_list.value
        tl_id_raw = features['traffic_light_state/current/id'].int64_list.value
        
        # Convert to arrays
        tl_state = np.array(tl_state_raw, dtype=np.int32)
        tl_valid = np.array([bool(v) for v in tl_valid_raw], dtype=bool)
        tl_x = np.array(tl_x_raw, dtype=np.float32)
        tl_y = np.array(tl_y_raw, dtype=np.float32)
        tl_id = np.array(tl_id_raw, dtype=np.int32)
        
        print(f"\n{'='*70}")
        print(f"EMERGENCY TL EXTRACTION - Record {record_idx}")
        print(f"{'='*70}")
        print(f"Ego: ({ego_x:.2f}, {ego_y:.2f}), yaw={ego_yaw:.2f}, speed={ego_speed:.2f} m/s")
        print(f"\nTraffic Lights:")
        print(f"  Total slots: {len(tl_state)}")
        print(f"  Valid: {np.sum(tl_valid)}")
        
        # Create graph with FORCED TL inclusion
        nodes = []
        edges = []
        
        # Ego node
        nodes.append({
            "id": "ego",
            "type": "EGO",
            "attrs": {
                "x": float(ego_x),
                "y": float(ego_y),
                "yaw": float(ego_yaw),
                "speed_mps": float(ego_speed),
                "speed_source": "computed"
            }
        })
        
        # TL nodes - FORCE ALL VALID TLs (no filtering!)
        tl_count = 0
        for j in range(len(tl_state)):
            if tl_valid[j]:
                # Compute distance
                dx = tl_x[j] - ego_x
                dy = tl_y[j] - ego_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # Create TL node (NO CLUSTERING, just raw TLs)
                node_id = f"tl_{j}"
                nodes.append({
                    "id": node_id,
                    "type": "TRAFFIC_LIGHT_CLUSTER",
                    "attrs": {
                        "cluster_id": int(tl_id[j]),
                        "member_tl_ids": [int(tl_id[j])],
                        "member_states": [int(tl_state[j])],
                        "representative_state": int(tl_state[j]),
                        "x": float(tl_x[j]),
                        "y": float(tl_y[j]),
                        "num_members": 1
                    }
                })
                
                # Create edge
                edges.append({
                    "src": "ego",
                    "dst": node_id,
                    "type": "EGO_TO_TL",
                    "attrs": {
                        "dist_m": float(dist),
                        "x_local_m": float(dx),  # Simplified, not rotated
                        "y_local_m": float(dy),
                        "ahead": bool(dx > 0),
                        "likely_controls_ego": bool(dx > 0 and abs(dy) < 10),
                        "control_confidence": 1.0 if (dx > 0 and abs(dy) < 10) else 0.0
                    }
                })
                
                tl_count += 1
                
                state_name = {
                    0: "UNKNOWN", 1: "ARROW_STOP", 2: "ARROW_CAUTION",
                    3: "ARROW_GO", 4: "STOP", 5: "CAUTION", 6: "GO",
                    7: "FLASHING_STOP", 8: "FLASHING_CAUTION"
                }.get(tl_state[j], f"STATE_{tl_state[j]}")
                
                print(f"  TL {j}: state={state_name:15s}  "
                      f"pos=({tl_x[j]:8.2f}, {tl_y[j]:8.2f})  "
                      f"dist={dist:6.2f}m")
        
        print(f"\nCreated {tl_count} TL nodes (FORCED, no filtering)")
        print(f"{'='*70}\n")
        
        # Create minimal graph
        graph = {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "record_idx": record_idx,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "emergency_extraction": True,
                "traffic_lights": {
                    "total_valid": tl_count,
                    "num_clusters": tl_count,
                    "note": "EMERGENCY PATCH - ALL FILTERING DISABLED"
                }
            }
        }
        
        return graph
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Emergency TL extraction patch')
    parser.add_argument('--tfrecord', required=True, help='Path to TFRecord file')
    parser.add_argument('--record', type=int, required=True, help='Record index')
    parser.add_argument('--output', default='emergency_graph.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    # Extract with forced TLs
    graph = extract_and_force_tls(args.tfrecord, args.record)
    
    if graph is None:
        print(f"Error: Could not extract record {args.record}")
        return
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"Saved emergency graph -> {args.output}")
    print(f"\nNext steps:")
    print(f"1. Visualize: python visualize_spatial_graph.py --graph {args.output} --mode combined")
    print(f"2. If TLs appear: Your TFRecord HAS TL data, it's a filtering bug")
    print(f"3. If TLs still missing: Your TFRecord might be corrupted")


if __name__ == '__main__':
    main()