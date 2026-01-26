# Detailed Analysis: build_semantic_graph.py

## Overview

`build_semantic_graph.py` is a script that extracts autonomous vehicle scenario data from Waymo's TFRecord format and constructs a semantic graph representation. The graph captures spatial relationships and dynamic behaviors between the ego vehicle and surrounding agents (vehicles, pedestrians, cyclists).

---

## Table of Contents
1. [Data Extraction Pipeline](#1-data-extraction-pipeline)
2. [Graph Node Construction](#2-graph-node-construction)
3. [Semantic Edge Generation](#3-semantic-edge-generation)
4. [Helper Functions](#4-helper-functions)
5. [Output Structure](#5-output-structure)

---

## 1. Data Extraction Pipeline

### 1.1 Configuration and Data Loading

**Code Snippet (Lines 7-90):**
```python
TFRECORD_PATH = os.path.expanduser("~/vmax/data/waymo_converted/training.tfrecord")

def build_graph(tfrecord_path):
    print(f"Reading from: {tfrecord_path}")
    dataset = tf.data.TFRecordDataset([tfrecord_path])

    # Take the first scenario
    raw = next(iter(dataset.take(1)))
    example = tf.train.Example.FromString(raw.numpy())
    f = example.features.feature
```

**Logic Explanation:**
- **TFRecord Loading**: Uses TensorFlow's `TFRecordDataset` to read binary serialized data
- **Protocol Buffer Parsing**: Deserializes the raw bytes into a `tf.train.Example` protobuf structure
- **Feature Access**: Extracts the `features.feature` dictionary containing all scenario data

### 1.2 Feature Extraction Helper Functions

**Code Snippet (Lines 93-94):**
```python
def F(key): return np.array(f[key].float_list.value, dtype=np.float32)
def I(key): return np.array(f[key].int64_list.value, dtype=np.int64)
```

**Logic Explanation:**
- **F(key)**: Extracts floating-point arrays (positions, velocities, angles)
- **I(key)**: Extracts integer arrays (validation flags, types, IDs)
- **Type Conversion**: Converts TensorFlow list values to NumPy arrays for efficient computation

### 1.3 Current State Data Extraction

**Code Snippet (Lines 96-116):**
```python
# Safe check if data is missing
if "state/current/x" not in f: return None

cur_x = F("state/current/x")
N = cur_x.shape[0]  # Total number of agents in the scenario

# Current State
cur_y = F("state/current/y")
cur_vx = F("state/current/velocity_x")
cur_vy = F("state/current/velocity_y")
cur_yaw = F("state/current/bbox_yaw") if "state/current/bbox_yaw" in f else np.zeros(N)
cur_valid = I("state/current/valid").astype(bool)
types = F("state/type")  # 1=Vehicle, 2=Ped, 3=Cyc
```

**Logic Explanation:**
1. **Safety Check**: Validates the presence of required fields before proceeding
2. **Agent Count**: Determines `N` (total agents) from the length of position arrays
3. **State Variables**:
   - `cur_x`, `cur_y`: Global coordinates in meters
   - `cur_vx`, `cur_vy`: Velocity components in m/s
   - `cur_yaw`: Heading angle in radians (default to 0 if missing)
   - `cur_valid`: Boolean mask to filter out invalid/padded agents
   - `types`: Agent classification (Vehicle=1, Pedestrian=2, Cyclist=3)

### 1.4 Historical Data Extraction

**Code Snippet (Lines 103-121):**
```python
# If past data exists, calculate Kpast
if "state/past/x" in f:
    Kpast = F("state/past/x").shape[0] // N
else:
    Kpast = 0

# Past State (Reshaped)
if Kpast > 0:
    past_vx = F("state/past/velocity_x").reshape(N, Kpast)
    past_vy = F("state/past/velocity_y").reshape(N, Kpast)
```

**Logic Explanation:**
- **Kpast Calculation**: Determines the number of historical timesteps available
  - Formula: `total_past_elements / num_agents` gives timesteps per agent
  - Typical value: Kpast = 10 (1 second of history at 10 Hz)
- **Reshaping**: Converts flat array `[agent0_t0, agent0_t1, ..., agent1_t0, ...]`
  - Into structured matrix: `[N_agents × Kpast_timesteps]`
  - Enables per-agent trajectory analysis

### 1.5 Ego Vehicle Identification

**Code Snippet (Lines 122-124):**
```python
is_sdc = I("state/is_sdc")  # Self-Driving Car flag
ego_idx = int(np.argmax(is_sdc))
```

**Logic Explanation:**
- **is_sdc Array**: Binary flags where 1 indicates the ego vehicle
- **argmax**: Finds the index of the ego vehicle (exactly one per scenario)
- **Purpose**: All relationships will be computed relative to this agent

---

## 2. Graph Node Construction

### 2.1 Node Creation Loop

**Code Snippet (Lines 126-159):**
```python
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
```

**Logic Explanation:**

**Step 1: Validity Filtering**
- Skips agents where `cur_valid[i] == False`
- These are padding slots or agents that left the scene

**Step 2: Type Mapping**
- Converts numeric type codes to human-readable strings
- Handles unknown types gracefully with "Unknown" fallback

**Step 3: Dynamic State Analysis**
- Calls `analyze_dynamics()` to classify agent behavior (see section 4.2)
- Possible states: "braking", "accelerating", "maintaining_speed", "stable"

**Step 4: Node Structure**
Each node contains:
- **id**: Unique agent identifier (0 to N-1)
- **type**: Semantic class (Vehicle/Pedestrian/Cyclist)
- **x, y**: Position in global frame (meters)
- **vx, vy**: Velocity in global frame (m/s)
- **yaw**: Heading angle (radians)
- **dynamic_state**: Inferred motion behavior

**Step 5: Ego Separation**
- Ego vehicle is stored separately in `ego_node_data`
- Other agents are added to `graph_nodes` list
- This separation enables ego-centric relationship computation

---

## 3. Semantic Edge Generation

### 3.1 Relationship Computation

**Code Snippet (Lines 161-181):**
```python
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

        # Enrich the node itself with these relations
        agent_node["relation_to_ego"] = relations
        agent_node["distance_to_ego"] = round(dist, 2)
```

**Logic Explanation:**

**Step 1: Pairwise Analysis**
- Iterates through each non-ego agent
- Computes semantic relationships using `get_semantic_relations()` (see section 4.1)

**Step 2: Distance Filtering**
- Only creates edges for agents within 100 meters
- Reduces graph size and focuses on relevant context

**Step 3: Edge Creation**
- **source**: Always "Ego" (all relationships are ego-centric)
- **target**: Formatted as `{type}_{id}` (e.g., "Vehicle_5", "Pedestrian_12")
- **relation**: Semantic label (e.g., "nearby", "in_front", "approaching_fast")

**Step 4: Node Enrichment**
- Adds `relation_to_ego` list directly to each node
- Adds `distance_to_ego` metric for quick reference
- This dual representation (edges + node attributes) supports both graph queries and LLM consumption

---

## 4. Helper Functions

### 4.1 Semantic Relationship Calculation

**Code Snippet (Lines 10-61):**
```python
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
```

**Logic - Euclidean Distance:**
- **Delta Calculation**: `dx`, `dy` are relative position vectors
- **Distance Formula**: Standard 2D Euclidean norm
- **Purpose**: Primary metric for spatial relationships

**Code Snippet (Lines 22-33):**
```python
# 2. Relative Velocity (Closing speed)
vx_rel = agent['vx'] - ego['vx']
vy_rel = agent['vy'] - ego['vy']

# Projection of relative velocity onto position vector (dot product)
if distance > 0.1:
    closing_speed = -(vx_rel * dx + vy_rel * dy) / distance
else:
    closing_speed = 0
```

**Logic - Closing Speed:**
- **Relative Velocity**: Agent velocity in ego's reference frame
- **Dot Product Projection**:
  - Formula: `(v_rel · pos_vec) / |pos_vec|`
  - Positive: Agents moving apart
  - Negative: Agents approaching (hence the minus sign)
- **Normalization**: Divides by distance to get scalar speed along the line of sight
- **Safety Check**: Avoids division by zero for co-located agents

**Code Snippet (Lines 35-41):**
```python
if distance < 10.0:
    relations.append("very_close")
elif distance < 30.0:
    relations.append("nearby")
else:
    relations.append("distant")
```

**Logic - Distance-Based Relations:**
- **very_close**: < 10m (immediate interaction zone, critical for collision avoidance)
- **nearby**: 10-30m (relevant for planning, typical interaction range)
- **distant**: > 30m (peripheral awareness)

**Code Snippet (Lines 43-53):**
```python
# 3. Orientation (In front vs Behind)
ego_yaw = ego['yaw'] if ego['yaw'] is not None else 0.0
x_local = dx * math.cos(-ego_yaw) - dy * math.sin(-ego_yaw)

if x_local > 0:
    relations.append("in_front")
else:
    relations.append("behind")
```

**Logic - Coordinate Frame Transformation:**
- **Rotation Matrix**: Transforms global coordinates to ego's local frame
  - Formula: `x_local = dx*cos(-θ) - dy*sin(-θ)`
  - This is a 2D rotation by angle `-ego_yaw`
- **Interpretation**:
  - `x_local > 0`: Agent is in ego's forward direction
  - `x_local < 0`: Agent is behind ego
- **Coordinate System**: Assumes standard convention where yaw=0 points East

**Code Snippet (Lines 55-60):**
```python
if closing_speed > 2.0:  # Closing at > 2 m/s
     relations.append("approaching_fast")
elif closing_speed > 0.5:
     relations.append("approaching")

return relations, distance
```

**Logic - Dynamic Relations:**
- **approaching_fast**: > 2 m/s closing rate (7.2 km/h, requires immediate attention)
- **approaching**: 0.5-2 m/s (moderate closing, worth monitoring)
- **No label**: Agents moving apart or parallel (closing_speed ≤ 0.5)

### 4.2 Dynamic State Analysis

**Code Snippet (Lines 63-81):**
```python
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
```

**Logic Explanation:**

**Step 1: Data Availability Check**
- Returns "stable" if insufficient history (< 2 timesteps)

**Step 2: Speed Calculation**
- Computes scalar speed at first timestep: `v_start = ||v(t-10)||`
- Computes scalar speed at last timestep: `v_end = ||v(t-1)||`
- Uses magnitude to capture speed change regardless of direction

**Step 3: Threshold-Based Classification**
- **braking**: Speed decreased by > 1 m/s (significant deceleration)
- **accelerating**: Speed increased by > 1 m/s (significant acceleration)
- **maintaining_speed**: Change within ±1 m/s (relatively constant)

**Insight**: This simple heuristic captures the most safety-critical behaviors (sudden braking/acceleration) without complex trajectory modeling.

---

## 5. Output Structure

### 5.1 Final Graph Assembly

**Code Snippet (Lines 183-191):**
```python
final_graph = {
    "scenario_info": {
        "ego_action": ego_node_data.get("dynamic_state", "unknown"),
        "ego_velocity": round(math.sqrt(ego_node_data['vx']**2 + ego_node_data['vy']**2), 2)
    },
    "context_nodes": graph_nodes,
    "semantic_edges": graph_edges
}
```

**Structure Explanation:**

**1. scenario_info**
- **ego_action**: Ego's current dynamic state (braking/accelerating/maintaining_speed)
- **ego_velocity**: Scalar speed in m/s (rounded to 2 decimals)
- **Purpose**: Provides high-level scenario context

**2. context_nodes**
- List of all non-ego agents with enriched attributes
- Each node contains:
  - Basic state (id, type, position, velocity, yaw)
  - Dynamic classification (dynamic_state)
  - Ego relationships (relation_to_ego, distance_to_ego)

**3. semantic_edges**
- Explicit edge list for graph construction
- Each edge: `{source: "Ego", target: "Type_ID", relation: "label"}`
- Multiple edges can exist between same node pair (different semantic relations)

### 5.2 Example Output

```json
{
  "scenario_info": {
    "ego_action": "maintaining_speed",
    "ego_velocity": 12.34
  },
  "context_nodes": [
    {
      "id": 5,
      "type": "Vehicle",
      "x": 45.2,
      "y": 12.8,
      "vx": 10.5,
      "vy": -2.3,
      "yaw": 0.785,
      "dynamic_state": "braking",
      "relation_to_ego": ["nearby", "in_front", "approaching"],
      "distance_to_ego": 25.3
    }
  ],
  "semantic_edges": [
    {"source": "Ego", "target": "Vehicle_5", "relation": "nearby"},
    {"source": "Ego", "target": "Vehicle_5", "relation": "in_front"},
    {"source": "Ego", "target": "Vehicle_5", "relation": "approaching"}
  ]
}
```

---

## Key Design Decisions

### 1. Ego-Centric Representation
All relationships are computed relative to the ego vehicle, matching autonomous driving's perspective requirements.

### 2. Multi-Label Edges
Each agent pair can have multiple simultaneous relations (e.g., "nearby" + "in_front" + "approaching_fast"), capturing rich semantic context.

### 3. Distance Filtering
100-meter radius limits graph size while preserving all safety-critical context for highway/urban scenarios.

### 4. Temporal Simplification
Uses only start/end velocities from history rather than full trajectory analysis, balancing informativeness with computational efficiency.

### 5. Coordinate Frame Transformation
Local frame orientation detection (in_front/behind) is more interpretable than raw angle calculations for downstream decision-making.

---

## Potential Extensions

1. **Lane Graph Integration**: Add road geometry nodes (lanes, intersections)
2. **Multi-Hop Relations**: Compute agent-to-agent relationships beyond ego
3. **Temporal Edges**: Link nodes across timesteps for trajectory graphs
4. **Probabilistic Relations**: Add confidence scores to relations based on uncertainty
5. **Attention Weights**: Compute importance scores for each relation based on criticality

---

## Usage Example

```bash
python build_semantic_graph.py
```

**Output**: JSON-formatted semantic graph printed to console, showing all agents within 100m of ego with their spatial, dynamic, and semantic relationships.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-21
**Related Files**: `build_semantic_graph_v2.py`, `vmax/agents/networks/encoders/attention_utils.py`
