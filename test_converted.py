#!/usr/bin/env python3
"""
Verify that converted TFRecord data works with Waymax/V-Max.
Checks for the CORRECT Waymax feature format (state/current/*, state/past/*, etc.)
"""

import os
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_inspect_tfrecord(tfrecord_path: str):
    """Load TFRecord and inspect its structure."""
    
    print(f"📂 Loading TFRecord from: {tfrecord_path}")
    print("=" * 60)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        features = example.features.feature
        feature_keys = sorted(features.keys())
        
        print(f"Total features: {len(feature_keys)}")
        print()
        
        # Print ALL features
        print("📋 ALL FEATURES:")
        print("-" * 40)
        for key in feature_keys:
            f = features[key]
            if f.HasField('float_list'):
                vals = list(f.float_list.value)
                print(f"  {key}: float[{len(vals)}]")
            elif f.HasField('int64_list'):
                vals = list(f.int64_list.value)
                print(f"  {key}: int64[{len(vals)}]")
            elif f.HasField('bytes_list'):
                vals = list(f.bytes_list.value)
                print(f"  {key}: bytes[{len(vals)}]")
        
        print()
        print("=" * 60)
        print("🔍 WAYMAX COMPATIBILITY CHECK:")
        print("-" * 40)
        
        # Correct Waymax features (state/current/*, state/past/*, state/future/*)
        critical_waymax_features = [
            # Current state
            'state/current/x',
            'state/current/y',
            'state/current/velocity_x',
            'state/current/velocity_y',
            'state/current/bbox_yaw',      # This is heading
            'state/current/valid',
            'state/current/speed',
            # Past states
            'state/past/x',
            'state/past/y',
            'state/past/valid',
            # Future states (for training targets)
            'state/future/x',
            'state/future/y',
            'state/future/valid',
            # Agent info
            'state/id',
            'state/type',
            'state/is_sdc',
            'state/tracks_to_predict',
            # Roadgraph
            'roadgraph_samples/xyz',
            'roadgraph_samples/type',
            'roadgraph_samples/valid',
            # Traffic lights
            'traffic_light_state/current/state',
            'traffic_light_state/current/valid',
            # SDC paths (V-Max specific)
            'path_samples/xyz',
            'path_samples/valid',
        ]
        
        found = 0
        missing = 0
        for feat in critical_waymax_features:
            present = feat in feature_keys
            status = "✅" if present else "❌"
            print(f"  {status} {feat}")
            if present:
                found += 1
            else:
                missing += 1
        
        print()
        print(f"Found: {found}/{len(critical_waymax_features)} critical features")
        
        if missing == 0:
            print("\n🎉 ALL critical features present! Data is fully V-Max compatible.")
        elif found >= 15:
            print("\n✅ Most critical features present. Data should work with V-Max.")
        else:
            print("\n⚠️ Some features missing. May have issues with V-Max.")
        
        # Sample some actual values
        print()
        print("=" * 60)
        print("📈 SAMPLE DATA VALUES:")
        print("-" * 40)
        
        if 'state/current/x' in features:
            x = list(features['state/current/x'].float_list.value)
            print(f"state/current/x: {len(x)} agents")
            print(f"  First 5 x-positions: {x[:5]}")
            non_zero = [v for v in x if v != 0]
            if non_zero:
                print(f"  Range (non-zero): [{min(non_zero):.2f}, {max(non_zero):.2f}]")
        
        if 'state/current/y' in features:
            y = list(features['state/current/y'].float_list.value)
            print(f"state/current/y: {len(y)} agents")
            print(f"  First 5 y-positions: {y[:5]}")
        
        if 'state/current/valid' in features:
            valid = list(features['state/current/valid'].int64_list.value)
            valid_count = sum(valid)
            print(f"state/current/valid: {valid_count}/{len(valid)} valid agents")
        
        if 'state/type' in features:
            types = list(features['state/type'].int64_list.value)
            from collections import Counter
            type_counts = Counter(types)
            # Type mapping: 1=vehicle, 2=pedestrian, 3=cyclist
            type_names = {0: 'unset', 1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}
            print(f"state/type distribution:")
            for t, count in sorted(type_counts.items()):
                name = type_names.get(t, f'unknown({t})')
                print(f"  {name}: {count}")
        
        if 'state/is_sdc' in features:
            is_sdc = list(features['state/is_sdc'].int64_list.value)
            sdc_count = sum(is_sdc)
            print(f"state/is_sdc: {sdc_count} SDC agent(s)")
        
        if 'scenario/id' in features:
            scenario_id = features['scenario/id'].bytes_list.value[0].decode('utf-8')
            print(f"scenario/id: {scenario_id}")
        
        # Roadgraph info
        if 'roadgraph_samples/xyz' in features:
            xyz = list(features['roadgraph_samples/xyz'].float_list.value)
            print(f"roadgraph_samples/xyz: {len(xyz)//3} points (xyz format)")
        
        if 'roadgraph_samples/valid' in features:
            rg_valid = list(features['roadgraph_samples/valid'].int64_list.value)
            rg_valid_count = sum(rg_valid)
            print(f"roadgraph_samples/valid: {rg_valid_count}/{len(rg_valid)} valid points")
        
        # Path samples (V-Max SDC paths)
        if 'path_samples/xyz' in features:
            path_xyz = list(features['path_samples/xyz'].float_list.value)
            print(f"path_samples/xyz: {len(path_xyz)//3} path points")
        
        if 'path_samples/on_route' in features:
            on_route = list(features['path_samples/on_route'].int64_list.value)
            on_route_count = sum(on_route)
            print(f"path_samples/on_route: {on_route_count}/{len(on_route)} on-route paths")

    # Count total
    count = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    print()
    print("=" * 60)
    print(f"📊 TOTAL SCENARIOS: {count}")
    print("=" * 60)
    
    return count


if __name__ == "__main__":
    import sys
    
    default_path = os.path.expanduser("~/vmax/data/waymo_converted/training.tfrecord")
    tfrecord_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    if not os.path.exists(tfrecord_path):
        print(f"❌ File not found: {tfrecord_path}")
        sys.exit(1)
    
    load_and_inspect_tfrecord(tfrecord_path)