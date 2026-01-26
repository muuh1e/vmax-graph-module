#!/usr/bin/env python3
"""
Debug traffic light extraction from TFRecords

This script helps diagnose why traffic lights aren't appearing in spatial graphs.
"""

import tensorflow as tf
import numpy as np
import json
import sys

def debug_traffic_lights(tfrecord_path, record_idx):
    """
    Extract and print ALL traffic light data from a specific record.
    """
    print(f"\n{'='*70}")
    print(f"DEBUGGING TRAFFIC LIGHTS - Record {record_idx}")
    print(f"{'='*70}\n")
    
    # Read TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for i, raw_bytes in enumerate(dataset):
        if i != record_idx:
            continue
        
        example = tf.train.Example()
        example.ParseFromString(raw_bytes.numpy())
        features = example.features.feature
        
        # 1. Extract traffic light states
        print("1. TRAFFIC LIGHT STATES (current timestep)")
        print("-" * 70)
        
        tl_state = features['traffic_light_state/current/state'].int64_list.value
        tl_valid = features['traffic_light_state/current/valid'].int64_list.value
        tl_x = features['traffic_light_state/current/x'].float_list.value
        tl_y = features['traffic_light_state/current/y'].float_list.value
        tl_id = features['traffic_light_state/current/id'].int64_list.value
        
        num_tls = len(tl_state)
        print(f"Total TL slots: {num_tls}")
        
        valid_count = 0
        for j in range(num_tls):
            if tl_valid[j]:
                valid_count += 1
                state = tl_state[j]
                state_name = {
                    0: "UNKNOWN", 1: "ARROW_STOP", 2: "ARROW_CAUTION", 
                    3: "ARROW_GO", 4: "STOP", 5: "CAUTION", 6: "GO",
                    7: "FLASHING_STOP", 8: "FLASHING_CAUTION"
                }.get(state, f"STATE_{state}")
                
                print(f"  TL {j:2d}: id={tl_id[j]:4d}  state={state_name:15s}  "
                      f"pos=({tl_x[j]:8.2f}, {tl_y[j]:8.2f})  valid={bool(tl_valid[j])}")
        
        print(f"\nValid TLs: {valid_count} / {num_tls}")
        
        # 2. Check past states
        print("\n2. TRAFFIC LIGHT STATES (past)")
        print("-" * 70)
        
        tl_past_state = features['traffic_light_state/past/state'].int64_list.value
        tl_past_valid = features['traffic_light_state/past/valid'].int64_list.value
        
        # Reshape: (10 timesteps, num_tls)
        tl_past_state = np.array(tl_past_state).reshape(10, num_tls)
        tl_past_valid = np.array(tl_past_valid).reshape(10, num_tls)
        
        # Check how many TLs have valid past data
        tls_with_past = 0
        for j in range(num_tls):
            if np.any(tl_past_valid[:, j]):
                tls_with_past += 1
        
        print(f"TLs with valid past data: {tls_with_past} / {num_tls}")
        
        # 3. Check future states
        print("\n3. TRAFFIC LIGHT STATES (future)")
        print("-" * 70)
        
        tl_future_state = features['traffic_light_state/future/state'].int64_list.value
        tl_future_valid = features['traffic_light_state/future/valid'].int64_list.value
        
        # Reshape: (80 timesteps, num_tls)
        tl_future_state = np.array(tl_future_state).reshape(80, num_tls)
        tl_future_valid = np.array(tl_future_valid).reshape(80, num_tls)
        
        tls_with_future = 0
        for j in range(num_tls):
            if np.any(tl_future_valid[:, j]):
                tls_with_future += 1
        
        print(f"TLs with valid future data: {tls_with_future} / {num_tls}")
        
        # 4. Get ego position for distance calc
        print("\n4. EGO POSITION")
        print("-" * 70)
        
        ego_x = features['state/current/x'].float_list.value[0]
        ego_y = features['state/current/y'].float_list.value[0]
        
        print(f"Ego position: ({ego_x:.2f}, {ego_y:.2f})")
        
        # 5. Calculate distances
        print("\n5. TRAFFIC LIGHT DISTANCES FROM EGO")
        print("-" * 70)
        
        within_80m = 0
        within_50m = 0
        within_20m = 0
        
        for j in range(num_tls):
            if tl_valid[j]:
                dx = tl_x[j] - ego_x
                dy = tl_y[j] - ego_y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 80:
                    within_80m += 1
                if dist < 50:
                    within_50m += 1
                if dist < 20:
                    within_20m += 1
                
                print(f"  TL {j:2d}: distance = {dist:6.2f}m")
        
        print(f"\nTLs within 80m: {within_80m}")
        print(f"TLs within 50m: {within_50m}")
        print(f"TLs within 20m: {within_20m}")
        
        # 6. Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total TL slots:           {num_tls}")
        print(f"Valid current TLs:        {valid_count}")
        print(f"TLs with past data:       {tls_with_past}")
        print(f"TLs with future data:     {tls_with_future}")
        print(f"TLs within 80m radius:    {within_80m}")
        print(f"TLs within 50m radius:    {within_50m}")
        print(f"TLs within 20m radius:    {within_20m}")
        print("="*70)
        
        # 7. Check what your code would extract
        print("\n7. WHAT YOUR CODE SHOULD SEE")
        print("-" * 70)
        
        # Simulate extraction logic from tfexample_io.py
        current_idx = 10  # Current frame is at index 10 in Waymo data
        
        # Current state
        tl_state_current = tl_state
        tl_valid_current = [bool(v) for v in tl_valid]
        
        print(f"current_state shape: {len(tl_state_current)}")
        print(f"current_valid shape: {len(tl_valid_current)}")
        print(f"Valid TLs at current frame: {sum(tl_valid_current)}")
        
        if sum(tl_valid_current) == 0:
            print("\n⚠️  WARNING: NO VALID TRAFFIC LIGHTS IN CURRENT FRAME!")
            print("This is why they don't appear in your spatial graph.")
        
        return

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python debug_traffic_lights.py <tfrecord_path> <record_idx>")
        print("Example: python debug_traffic_lights.py training.tfrecord 2")
        sys.exit(1)
    
    tfrecord_path = sys.argv[1]
    record_idx = int(sys.argv[2])
    
    debug_traffic_lights(tfrecord_path, record_idx)