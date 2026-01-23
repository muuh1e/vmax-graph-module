import tensorflow as tf
import numpy as np
import math
import os
import json

# --- CONFIGURATION ---
TFRECORD_PATH = os.path.expanduser("~/vmax/data/waymo_converted/training.tfrecord")

def get_semantic_relations(ego, agent):
    """
    Calculates semantic edges between Ego and another Agent.
    Returns: (list of relation strings, distance)
    """
    relations = []
    
    # 1. Spatial Calculation
    dx = agent['x'] - ego['x']
    dy = agent['y'] - ego['y']
    distance = math.sqrt(dx**2 + dy**2)
    
    # 2. Relative Velocity (Closing speed)
    # Agent velocity relative to Ego
    vx_rel = agent['vx'] - ego['vx']
    vy_rel = agent['vy'] - ego['vy']
    
    # Projection of relative velocity onto position vector (dot product)
    # If negative, they are moving closer.
    # Avoid divide by zero
    if distance > 0.1:
        closing_speed = -(vx_rel * dx + vy_rel * dy) / distance
    else:
        closing_speed = 0

    # Thresholds (Tweak these based on your domain needs)
    if distance < 10.0:
        relations.append("very_close")
    elif distance < 30.0:
        relations.append("nearby")
    else:
        relations.append("distant")
        
    # 3. Orientation (In front vs Behind)
    # Rotate neighbor position into Ego's local frame using Ego's heading (yaw)
    # Formula: x_local = dx * cos(-yaw) - dy * sin(-yaw)
    # (Assuming standard coordinate system where Yaw=0 is East)
    ego_yaw = ego['yaw'] if ego['yaw'] is not None else 0.0
    x_local = dx * math.cos(-ego_yaw) - dy * math.sin(-ego_yaw)
    
    if x_local > 0:
        relations.append("in_front")
    else:
        relations.append("behind")

    # 4. Critical Events
    if closing_speed > 2.0: # Closing at > 2 m/s
         relations.append("approaching_fast")
    elif closing_speed > 0.5:
         relations.append("approaching")

    return relations, distance

def analyze_dynamics(history_vx, history_vy):
    """
    Simple heuristic to interpret history: Is the agent braking or accelerating?
    """
    if len(history_vx) < 2:
        return "stable"
    
    # Speed magnitude at start (t-10) vs end (t-1)
    v_start = math.sqrt(history_vx[0]**2 + history_vy[0]**2)
    v_end = math.sqrt(history_vx[-1]**2 + history_vy[-1]**2)
    
    diff = v_end - v_start
    
    if diff < -1.0:
        return "braking"
    elif diff > 1.0:
        return "accelerating"
    else:
        return "maintaining_speed"

def build_graph(tfrecord_path):
    print(f"Reading from: {tfrecord_path}")
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    # Take the first scenario
    raw = next(iter(dataset.take(1)))
    example = tf.train.Example.FromString(raw.numpy())
    f = example.features.feature

    # --- 1. EXTRACT RAW DATA (Your validated logic) ---
    def F(key): return np.array(f[key].float_list.value, dtype=np.float32)
    def I(key): return np.array(f[key].int64_list.value, dtype=np.int64)

    # Shapes
    # Safe check if data is missing
    if "state/current/x" not in f: return None
    
    cur_x = F("state/current/x")
    N = cur_x.shape[0]
    
    # If past data exists, calculate Kpast
    if "state/past/x" in f:
        Kpast = F("state/past/x").shape[0] // N
    else:
        Kpast = 0

    # Current State
    cur_y = F("state/current/y")
    cur_vx = F("state/current/velocity_x")
    cur_vy = F("state/current/velocity_y")
    cur_yaw = F("state/current/bbox_yaw") if "state/current/bbox_yaw" in f else np.zeros(N)
    cur_valid = I("state/current/valid").astype(bool)
    types = F("state/type") # 1=Vehicle, 2=Ped, 3=Cyc

    # Past State (Reshaped)
    if Kpast > 0:
        past_vx = F("state/past/velocity_x").reshape(N, Kpast)
        past_vy = F("state/past/velocity_y").reshape(N, Kpast)
    
    # Ego Identification
    is_sdc = I("state/is_sdc")
    ego_idx = int(np.argmax(is_sdc))

    # --- 2. BUILD NODES ---
    type_map = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}
    graph_nodes = []
    ego_node_data = {}

    # First pass: Build Node Objects
    for i in range(N):
        if not cur_valid[i]:
            continue
            
        agent_type = type_map.get(int(types[i]), "Unknown")
        
        # Interpret Dynamics from history
        dynamic_status = "unknown"
        if Kpast > 0:
            dynamic_status = analyze_dynamics(past_vx[i], past_vy[i])

        node = {
            "id": int(i),
            "type": agent_type,
            "x": float(cur_x[i]),
            "y": float(cur_y[i]),
            "vx": float(cur_vx[i]),
            "vy": float(cur_vy[i]),
            "yaw": float(cur_yaw[i]),
            "dynamic_state": dynamic_status
        }
        
        if i == ego_idx:
            node["is_ego"] = True
            ego_node_data = node
        else:
            node["is_ego"] = False
            graph_nodes.append(node)

    # --- 3. BUILD EDGES (Semantic Logic) ---
    graph_edges = []
    
    for agent_node in graph_nodes:
        # Calculate relation relative to Ego
        relations, dist = get_semantic_relations(ego_node_data, agent_node)
        
        # Only include edges if relevant (e.g., within 100m)
        if dist < 100.0:
            # Add explicit edges for the graph
            for rel in relations:
                graph_edges.append({
                    "source": "Ego",
                    "target": f"{agent_node['type']}_{agent_node['id']}",
                    "relation": rel
                })
            
            # OPTIONAL: Enrich the node itself with these relations for the LLM
            # (Makes the JSON easier for an LLM to read linearly)
            agent_node["relation_to_ego"] = relations
            agent_node["distance_to_ego"] = round(dist, 2)

    # --- 4. FINAL STRUCTURE ---
    final_graph = {
        "scenario_info": {
            "ego_action": ego_node_data.get("dynamic_state", "unknown"),
            "ego_velocity": round(math.sqrt(ego_node_data['vx']**2 + ego_node_data['vy']**2), 2)
        },
        "context_nodes": graph_nodes, # These are the OTHER agents
        "semantic_edges": graph_edges
    }
    
    return final_graph

if __name__ == "__main__":
    if not os.path.exists(TFRECORD_PATH):
        print(f"❌ Error: File not found at {TFRECORD_PATH}")
    else:
        graph = build_graph(TFRECORD_PATH)
        if graph:
            print(json.dumps(graph, indent=2))
            print(f"\n✅ Graph Built! Found {len(graph['context_nodes'])} relevant agents around the Ego.")