# Merge Strategy: Unified Heterogeneous Graph

## Executive Summary

This document outlines the strategy for merging two independent graph construction systems into a single unified heterogeneous graph pipeline.

---

## 1. Data Extraction Strategy

### Options Analysis

| Option | Pros | Cons |
|--------|------|------|
| **A: Merge JSON outputs** | No code changes to existing systems; quick to prototype | Reads TFRecord twice; JSON parsing overhead; loses float precision; harder to add new edge types |
| **B: Unified extractor from scratch** | Cleanest architecture; optimal performance | Throws away tested code; high development effort; risk of introducing bugs |
| **C: Shared extraction, branch for nodes** | Single TFRecord read; reuses tfexample_io.py; modular; testable | Moderate refactoring; need to define shared interfaces |

### Recommendation: Option C

**Reasoning:**

1. **Efficiency**: Single TFRecord parse via existing `tfexample_io.py` functions
2. **Reuse**: `get_ego_state()`, `get_roadgraph_samples()`, `get_traffic_lights_current()` already tested
3. **Modularity**: Each node builder is independent - can test/debug in isolation
4. **Extensibility**: Adding new node types (e.g., STOP_SIGN) is straightforward
5. **Incremental migration**: Can validate against existing outputs during development

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED ARCHITECTURE                            │
│                                                                             │
│  TFRecord ──► tfexample_io.py ──► Shared Raw Data ──┬──► AgentNodeBuilder  │
│                                                      │                      │
│                                                      ├──► LaneNodeBuilder   │
│                                                      │                      │
│                                                      ├──► TLNodeBuilder     │
│                                                      │                      │
│                                                      └──► EdgeBuilders      │
│                                                             │               │
│                                                             ▼               │
│                                                      UnifiedGraph           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Coordinate Alignment

### Current State

| System | Coordinates | Transform Location |
|--------|-------------|-------------------|
| `build_semantic_graph_v2.py` | **World frame** (absolute x,y) | Only for `relations` computation, not storage |
| `spatial_graph_builder.py` | **Ego frame** | Uses `ego_frame()` from `geometry.py` |

### Decision: Transform at Extraction Time

**Where**: Immediately after extracting raw positions, before building node features.

**Why**:
- All downstream code operates in one consistent frame
- Avoids scattering transforms throughout the codebase
- Edge computation (agent-to-lane distance) requires both in same frame
- Matches the design doc requirement

### Implementation Strategy

```python
# In unified extractor, right after parsing:

ego_state = get_ego_state(features)
EGO_XY = np.array([ego_state.x, ego_state.y], dtype=np.float32)
EGO_YAW = float(ego_state.yaw)

def to_ego_frame(world_x, world_y):
    """Transform world coordinates to ego-centric frame."""
    dx = world_x - EGO_XY[0]
    dy = world_y - EGO_XY[1]
    return ego_frame(dx, dy, EGO_YAW)  # from geometry.py

# Apply to ALL positions before node building:
# - Agent current/past positions
# - Lane polyline points
# - Traffic light positions
```

### Velocity Transformation

Velocities also need rotation (but not translation):

```python
def velocity_to_ego_frame(world_vx, world_vy):
    """Rotate velocity vector to ego frame."""
    c = np.cos(-EGO_YAW)
    s = np.sin(-EGO_YAW)
    vx_ego = world_vx * c - world_vy * s
    vy_ego = world_vx * s + world_vy * c
    return vx_ego, vy_ego
```

### Yaw Transformation

```python
def yaw_to_ego_frame(world_yaw):
    """Relative heading: 0 means same direction as ego."""
    return normalize_angle(world_yaw - EGO_YAW)
```

---

## 3. Ego Node Resolution

### Current Representations

| System | Ego Representation |
|--------|-------------------|
| `build_semantic_graph_v2.py` | Full agent: id, track_id, type, current {x,y,vx,vy,yaw}, past {x[],y[],vx[],vy[],valid[]} |
| `spatial_graph_builder.py` | Minimal: x, y, yaw, speed, speed_source |

### Decision: Use Semantic Graph's Rich Representation

**Unified Ego = AGENT node with `is_ego=1`**

**Reasoning:**
1. Past trajectory needed for motion prediction tasks
2. Consistent with other agents (same feature vector structure)
3. `is_ego` flag distinguishes it in the graph
4. Can add `speed_source` as metadata if needed

### Handling Duplication

```python
# Ego appears in:
# 1. agent_states (as one of N agents with is_sdc=1)
# 2. get_ego_state() (extracted separately for spatial graph)

# Resolution: Use agent_states as source of truth
# get_ego_state() only used to establish the coordinate transform reference

ego_idx = np.argmax(is_sdc)  # Find ego in agent list

for i, agent in enumerate(agents):
    node = build_agent_node(agent)
    node.is_ego = (i == ego_idx)  # Mark ego
    agent_nodes.append(node)

# Ego's position in ego frame is always (0, 0) by definition
# But we still include full features for consistency
```

### Ego-Frame Position of Ego

After transformation, ego's position is `(0, 0)` and yaw is `0`. This is correct and expected:

```python
# In ego frame:
ego_node.features.x = 0.0      # ego is at origin
ego_node.features.y = 0.0
ego_node.features.yaw = 0.0    # ego heading is reference
ego_node.features.vx = speed   # forward velocity
ego_node.features.vy = 0.0     # (approximately, if going straight)
```

---

## 4. Creating AGENT_TO_LANE Edges

### Data Requirements

| From Agents | From Lanes |
|-------------|------------|
| Position (x, y) in ego frame | Polyline points in ego frame |
| Velocity (vx, vy) | Lane heading |
| Heading (yaw) | Speed limit |

### Algorithm: K-Nearest Lanes with Occupancy Check

```python
def build_agent_to_lane_edges(agent_nodes, lane_nodes, k=3):
    edges = []

    for agent in agent_nodes:
        agent_xy = np.array([agent.x, agent.y])
        agent_yaw = agent.yaw
        agent_speed = agent.speed

        # Compute distance to each lane
        lane_distances = []
        for lane in lane_nodes:
            cp = closest_point_on_polyline(lane.points_ego, agent_xy)
            if cp is not None:
                lane_distances.append((lane, cp))

        # Sort by distance, take k nearest
        lane_distances.sort(key=lambda x: x[1].dist)
        nearest = lane_distances[:k]

        for lane, cp in nearest:
            # Compute edge features
            edge = AgentToLaneEdge(
                src=agent.id,
                dst=lane.id,
                distance_m=cp.dist,
                lateral_offset_m=cp.signed_lateral,
                heading_alignment=cos(agent_yaw - lane.heading),
                s_progress=cp.s / lane.length,
                is_on_lane=(cp.dist < LANE_WIDTH / 2),
                is_approaching=check_approaching(agent, lane, cp),
                is_crossing=(abs(cos(agent_yaw - lane.heading)) < 0.5),
                speed_compliance=agent_speed / lane.speed_limit if lane.speed_limit else 0,
                has_speed_limit=(lane.speed_limit is not None and lane.speed_limit > 0),
            )
            edges.append(edge)

    return edges
```

### Key Function: `closest_point_on_polyline`

Already exists in `geometry.py`:

```python
# Returns:
#   dist: distance to closest point
#   closest_xy: the point itself
#   s: arc-length along polyline
#   signed_lateral: left(+) / right(-) offset
```

### Approaching Check

```python
def check_approaching(agent, lane, cp):
    """Is agent moving toward lane centerline?"""
    # Vector from agent to closest point
    to_lane = cp.closest_xy - np.array([agent.x, agent.y])
    to_lane_norm = to_lane / (np.linalg.norm(to_lane) + 1e-6)

    # Agent velocity
    vel = np.array([agent.vx, agent.vy])

    # Dot product: positive if moving toward lane
    approaching = np.dot(vel, to_lane_norm) > 0.5

    # Only "approaching" if not already on lane
    return approaching and cp.dist > LANE_WIDTH / 2
```

---

## 5. ID Mapping

### Problem

- Agent IDs: integers 0..N-1 (index in TFRecord)
- Lane IDs: arbitrary integers from `roadgraph_samples/id` (can be large, non-sequential)
- TL IDs: cluster IDs (sequential after clustering)
- PyG needs: sequential indices 0..K per node type

### Solution: Type-Specific Index Mapping

```python
@dataclass
class NodeMapping:
    """Bidirectional mapping between original IDs and unified indices."""

    # Forward: original_id -> unified_idx
    agent_to_idx: Dict[int, int]
    lane_to_idx: Dict[int, int]
    tl_to_idx: Dict[int, int]

    # Reverse: unified_idx -> original_id (for debugging)
    idx_to_agent: Dict[int, int]
    idx_to_lane: Dict[int, int]
    idx_to_tl: Dict[int, int]


def build_node_mapping(agent_nodes, lane_nodes, tl_nodes) -> NodeMapping:
    mapping = NodeMapping(
        agent_to_idx={}, lane_to_idx={}, tl_to_idx={},
        idx_to_agent={}, idx_to_lane={}, idx_to_tl={},
    )

    # Agents: keep original order (already 0..N-1)
    for idx, agent in enumerate(agent_nodes):
        mapping.agent_to_idx[agent.original_id] = idx
        mapping.idx_to_agent[idx] = agent.original_id

    # Lanes: map polyline_id to sequential index
    for idx, lane in enumerate(lane_nodes):
        mapping.lane_to_idx[lane.polyline_id] = idx
        mapping.idx_to_lane[idx] = lane.polyline_id

    # TLs: map cluster_id to sequential index
    for idx, tl in enumerate(tl_nodes):
        mapping.tl_to_idx[tl.cluster_id] = idx
        mapping.idx_to_tl[idx] = tl.cluster_id

    return mapping
```

### Edge Index Conversion

```python
def edges_to_pyg_format(edges, mapping, edge_type):
    """Convert edges to PyG COO format."""
    src_type, _, dst_type = edge_type  # e.g., ("agent", "to", "lane")

    src_map = getattr(mapping, f"{src_type}_to_idx")
    dst_map = getattr(mapping, f"{dst_type}_to_idx")

    src_indices = []
    dst_indices = []
    features = []

    for edge in edges:
        src_idx = src_map[edge.src_original_id]
        dst_idx = dst_map[edge.dst_original_id]
        src_indices.append(src_idx)
        dst_indices.append(dst_idx)
        features.append(edge.features)

    return (
        torch.tensor(src_indices, dtype=torch.long),
        torch.tensor(dst_indices, dtype=torch.long),
        torch.stack(features),  # [E, feature_dim]
    )
```

### Storing Original IDs for Debugging

```python
# Option 1: Store in node features (not recommended - wastes space)
# Option 2: Store in graph metadata
graph.metadata = {
    "agent_original_ids": [n.original_id for n in agent_nodes],
    "lane_polyline_ids": [n.polyline_id for n in lane_nodes],
    "tl_cluster_ids": [n.cluster_id for n in tl_nodes],
    "node_mapping": mapping,
}

# Option 3: Store as separate tensors (for batch processing)
graph["agent"].original_id = torch.tensor([...])
```

---

## 6. Handling Missing Data

### Scenarios and Solutions

| Scenario | Frequency | Solution |
|----------|-----------|----------|
| No traffic lights | Common (rural roads) | Empty TL tensors: `[0, 12]`; No LANE_TO_TL edges |
| No drivable lanes | Rare (parking lots?) | Keep BOUNDARY lanes as fallback; warn in metadata |
| Only ego (no other agents) | Uncommon | Valid graph with 1 agent; no A2A edges |
| No valid agents at all | Should never happen | Raise error - ego must exist |
| TFRecord parse failure | Rare | Raise error with record index |

### Implementation

```python
def build_unified_graph(example, config):
    # ... extraction ...

    # Validate ego exists
    if ego_idx is None:
        raise ValueError(f"No ego found in record {record_index}")

    # Build nodes (may be empty lists)
    agent_nodes = build_agent_nodes(...)  # At least ego
    lane_nodes = build_lane_nodes(...)    # May be empty
    tl_nodes = build_tl_nodes(...)        # May be empty

    # Fallback for lanes
    if len(lane_nodes) == 0:
        warnings.warn(f"Record {record_index}: No drivable lanes, using boundaries")
        lane_nodes = build_lane_nodes(..., include_boundaries=True)

    # Build edges (gracefully handle empty node lists)
    a2a_edges = build_a2a_edges(agent_nodes) if len(agent_nodes) > 1 else []
    a2l_edges = build_a2l_edges(agent_nodes, lane_nodes) if len(lane_nodes) > 0 else []
    l2l_edges = build_l2l_edges(lane_nodes) if len(lane_nodes) > 1 else []
    l2tl_edges = build_l2tl_edges(lane_nodes, tl_nodes) if len(tl_nodes) > 0 and len(lane_nodes) > 0 else []

    # Create tensors (handle empty case)
    def safe_stack(nodes, feature_dim):
        if len(nodes) == 0:
            return torch.zeros(0, feature_dim)
        return torch.stack([n.features for n in nodes])

    graph = HeteroData()
    graph["agent"].x = safe_stack(agent_nodes, AGENT_FEATURE_DIM)
    graph["lane"].x = safe_stack(lane_nodes, LANE_FEATURE_DIM)
    graph["tl"].x = safe_stack(tl_nodes, TL_FEATURE_DIM)

    # ... edges ...

    return graph
```

### Empty Edge Handling for PyG

```python
def safe_edge_index(src_list, dst_list):
    """Return valid edge_index even if empty."""
    if len(src_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)
```

---

## 7. Validation

### Invariants (Must Always Hold)

```python
def validate_graph(graph, mapping):
    errors = []

    # 1. Ego must exist
    ego_mask = graph["agent"].x[:, 0] == 1.0  # is_ego feature
    if ego_mask.sum() != 1:
        errors.append(f"Expected exactly 1 ego, found {ego_mask.sum()}")

    # 2. All edge indices in bounds
    for edge_type in graph.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = graph[edge_type].edge_index

        num_src = graph[src_type].x.shape[0]
        num_dst = graph[dst_type].x.shape[0]

        if edge_index.shape[1] > 0:
            if edge_index[0].max() >= num_src:
                errors.append(f"{edge_type}: src index out of bounds")
            if edge_index[1].max() >= num_dst:
                errors.append(f"{edge_type}: dst index out of bounds")

    # 3. No NaN or Inf in features
    for node_type in graph.node_types:
        x = graph[node_type].x
        if torch.isnan(x).any():
            errors.append(f"{node_type}: NaN in features")
        if torch.isinf(x).any():
            errors.append(f"{node_type}: Inf in features")

    # 4. Feature dimensions match spec
    assert graph["agent"].x.shape[1] == 45, "Agent feature dim mismatch"
    assert graph["lane"].x.shape[1] == 55, "Lane feature dim mismatch"
    assert graph["tl"].x.shape[1] == 12, "TL feature dim mismatch"

    # 5. Ego position is (0, 0) in ego frame
    ego_idx = ego_mask.nonzero().item()
    ego_x = graph["agent"].x[ego_idx, 5]  # x position
    ego_y = graph["agent"].x[ego_idx, 6]  # y position
    if abs(ego_x) > 1e-3 or abs(ego_y) > 1e-3:
        errors.append(f"Ego not at origin: ({ego_x}, {ego_y})")

    return errors
```

### Sanity Checks (Compare Against Existing Outputs)

```python
def compare_with_legacy(unified_graph, semantic_json, spatial_json):
    """Cross-validate against existing graph outputs."""

    report = {}

    # 1. Agent count should match
    legacy_agents = len(semantic_json["nodes"])
    unified_agents = unified_graph["agent"].x.shape[0]
    report["agent_count_match"] = (legacy_agents == unified_agents)

    # 2. Lane count should be similar (may differ due to filtering)
    legacy_map_features = len([n for n in spatial_json["nodes"] if n["type"] == "MAP_FEATURE"])
    unified_lanes = unified_graph["lane"].x.shape[0]
    report["lane_count_ratio"] = unified_lanes / max(legacy_map_features, 1)

    # 3. Ego relations should be preserved
    legacy_ego_edges = len(semantic_json["edges"])
    unified_a2a = unified_graph["agent", "to", "agent"].edge_index.shape[1]
    report["a2a_edge_count_match"] = (legacy_ego_edges == unified_a2a)

    # 4. Distance values should be similar
    # ... sample a few agents and compare distances ...

    return report
```

### Automated Test Suite

```python
def test_unified_graph_builder():
    """Integration test with sample TFRecord."""

    # Load test record
    example = load_test_tfrecord("test_data/sample.tfrecord", index=0)

    # Build graph
    graph = build_unified_graph(example)

    # Validate
    errors = validate_graph(graph)
    assert len(errors) == 0, f"Validation failed: {errors}"

    # Check shapes
    assert graph["agent"].x.dim() == 2
    assert graph["lane"].x.dim() == 2
    assert graph["tl"].x.dim() == 2

    # Check edge connectivity
    assert graph["agent", "to", "agent"].edge_index.shape[0] == 2
    assert graph["agent", "to", "lane"].edge_index.shape[0] == 2

    print("All tests passed!")
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              UNIFIED GRAPH PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    TFRecord
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      tfexample_io.py         │
                        │  (shared extraction layer)   │
                        └──────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │  Agent States    │    │   Roadgraph      │    │  Traffic Lights  │
    │  state/*         │    │   Samples        │    │  traffic_light_  │
    │                  │    │   roadgraph_*    │    │  state/*         │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
             │                       │                       │
             │              ┌────────┴─────────┐             │
             │              │                  │             │
             ▼              ▼                  ▼             ▼
    ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ EGO STATE    │ │ get_ego_    │ │ polyline_   │ │ cluster_    │
    │ EXTRACTION   │ │ state()     │ │ grouper     │ │ traffic_    │
    │              │ │             │ │             │ │ lights      │
    └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │                │               │               │
           │                ▼               │               │
           │    ┌───────────────────────┐   │               │
           │    │   COORDINATE FRAME    │   │               │
           │    │   ego_xy, ego_yaw     │   │               │
           │    └───────────┬───────────┘   │               │
           │                │               │               │
           ▼                ▼               ▼               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TO_EGO_FRAME TRANSFORM                       │
    │                                                                 │
    │   All positions, velocities, headings transformed here          │
    └─────────────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │  AgentNode       │    │  LaneNode        │    │  TLNode          │
    │  Builder         │    │  Builder         │    │  Builder         │
    │                  │    │                  │    │                  │
    │  [N_a, 45]       │    │  [N_l, 55]       │    │  [N_t, 12]       │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
             │                       │                       │
             └───────────┬───────────┴───────────┬───────────┘
                         │                       │
                         ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │  NODE MAPPING    │    │  EDGE BUILDERS   │
              │  original → idx  │    │                  │
              └────────┬─────────┘    │  A2A, A2L, L2L,  │
                       │              │  L2TL            │
                       │              └────────┬─────────┘
                       │                       │
                       └───────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │    VALIDATION        │
                        │    - invariants      │
                        │    - bounds check    │
                        │    - NaN/Inf check   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   PyG HeteroData     │
                        │                      │
                        │   nodes: agent,      │
                        │          lane, tl    │
                        │   edges: a2a, a2l,   │
                        │          l2l, l2tl   │
                        └──────────────────────┘
```

---

## Critical Merge Function Pseudocode

```python
def build_unified_graph(
    raw_record: bytes,
    record_index: int,
    config: UnifiedGraphConfig,
) -> Tuple[HeteroData, NodeMapping, Dict[str, Any]]:
    """
    Main entry point: TFRecord bytes → PyG HeteroData graph.

    Returns:
        graph: PyG HeteroData with nodes and edges
        mapping: ID mapping for debugging
        metadata: Statistics and warnings
    """

    # ================================================================
    # PHASE 1: Parse TFExample (shared extraction)
    # ================================================================
    example = tf.train.Example.FromString(raw_record)
    features = example.features.feature

    # Extract all raw data using existing tfexample_io functions
    ego_state = get_ego_state(features)
    roadgraph = get_roadgraph_samples(features)
    tl_current = get_traffic_lights_current(features)
    tl_future = get_traffic_lights_time_major(features, "future", num_lights=tl_current.valid.size)

    # Extract agent states (from build_semantic_graph_v2.py patterns)
    N, Kpast, Kfuture = infer_N_Kpast_Kfuture(features)
    agent_states = extract_agent_states(features, N, Kpast)

    # ================================================================
    # PHASE 2: Establish coordinate frame
    # ================================================================
    EGO_XY = np.array([ego_state.x, ego_state.y], dtype=np.float32)
    EGO_YAW = float(ego_state.yaw)

    def to_ego_pos(x, y):
        return ego_frame(x - EGO_XY[0], y - EGO_XY[1], EGO_YAW)

    def to_ego_vel(vx, vy):
        c, s = np.cos(-EGO_YAW), np.sin(-EGO_YAW)
        return vx * c - vy * s, vx * s + vy * c

    def to_ego_yaw(yaw):
        return normalize_angle(yaw - EGO_YAW)

    # ================================================================
    # PHASE 3: Build AGENT nodes
    # ================================================================
    agent_nodes = []
    ego_agent_idx = None

    for i, agent in enumerate(agent_states):
        if not agent.current_valid:
            continue

        # Transform to ego frame
        x_ego, y_ego = to_ego_pos(agent.x, agent.y)
        vx_ego, vy_ego = to_ego_vel(agent.vx, agent.vy)
        yaw_ego = to_ego_yaw(agent.yaw)

        # Transform past trajectory
        past_x_ego, past_y_ego = [], []
        for t in range(config.past_steps):
            if t < len(agent.past_x) and agent.past_valid[t]:
                px, py = to_ego_pos(agent.past_x[t], agent.past_y[t])
            else:
                px, py = x_ego, y_ego  # fallback to current
            past_x_ego.append(px)
            past_y_ego.append(py)

        # Dynamics analysis
        is_braking, is_accel = analyze_dynamics(agent.past_vx, agent.past_vy, agent.past_valid)

        # Build feature vector [45]
        is_ego = (agent.is_sdc == 1)
        if is_ego:
            ego_agent_idx = len(agent_nodes)

        features = torch.tensor([
            float(is_ego),                           # 0: is_ego
            float(agent.type == 1),                  # 1: type_vehicle
            float(agent.type == 2),                  # 2: type_pedestrian
            float(agent.type == 3),                  # 3: type_cyclist
            float(agent.type not in [1,2,3]),        # 4: type_unknown
            float(x_ego), float(y_ego),              # 5-6: position
            float(vx_ego), float(vy_ego),            # 7-8: velocity
            float(yaw_ego),                          # 9: yaw
            float(np.hypot(vx_ego, vy_ego)),         # 10: speed
            float(agent.length), float(agent.width), # 11-12: bbox
            *past_x_ego,                             # 13-22: past_x
            *past_y_ego,                             # 23-32: past_y
            *[float(v) for v in agent.past_valid[:config.past_steps]],  # 33-42: past_valid
            float(is_braking),                       # 43: is_braking
            float(is_accel),                         # 44: is_accelerating
        ], dtype=torch.float32)

        agent_nodes.append(AgentNode(
            original_id=i,
            features=features,
            x_ego=x_ego, y_ego=y_ego,
            vx_ego=vx_ego, vy_ego=vy_ego,
            yaw_ego=yaw_ego,
            is_ego=is_ego,
        ))

    # Validate ego exists
    if ego_agent_idx is None:
        raise ValueError(f"Record {record_index}: No ego agent found")

    # ================================================================
    # PHASE 4: Build LANE nodes
    # ================================================================
    lane_nodes = []
    DRIVABLE_TYPES = {1, 2, 3, 20}  # FREEWAY, SURFACE_STREET, BIKE_LANE, DRIVEWAY

    # Group roadgraph points by polyline ID
    polylines = group_by_id(roadgraph)

    for poly_id, points in polylines.items():
        type_mode = get_mode(points.types)

        if type_mode not in DRIVABLE_TYPES:
            continue

        # Filter to ego-frame bounding box
        points_xy = points.xyz[:, :2]
        points_ego = np.stack(to_ego_pos(points_xy[:, 0], points_xy[:, 1]), axis=1)

        in_box = (
            (points_ego[:, 0] >= -config.lane_back_m) &
            (points_ego[:, 0] <= config.lane_fwd_m) &
            (np.abs(points_ego[:, 1]) <= config.lane_side_m)
        )

        if in_box.sum() < 2:
            continue

        points_ego = points_ego[in_box]

        # Compute lane features
        centroid = np.mean(points_ego, axis=0)
        heading = compute_polyline_heading(points_ego)

        cp = closest_point_on_polyline(points_ego, np.array([0, 0]))

        samples = sample_polyline_uniform(points_ego, n=10)

        speed_limit = get_median_speed_limit(points.speed_limits[in_box])

        # Build feature vector [55]
        features = torch.tensor([
            float(type_mode == 1),      # 0: type_freeway
            float(type_mode == 2),      # 1: type_surface_street
            float(type_mode == 3),      # 2: type_bike_lane
            float(type_mode == 20),     # 3: type_driveway
            float(centroid[0]), float(centroid[1]),  # 4-5: centroid
            float(np.cos(heading)), float(np.sin(heading)),  # 6-7: heading
            float(speed_limit) if speed_limit else 0.0,      # 8: speed_limit
            float(speed_limit is not None and speed_limit > 0),  # 9: has_speed_limit
            float(compute_polyline_length(points_ego)),      # 10: length
            float(cp.dist),                                  # 11: closest_dist
            float(cp.closest_xy[0]), float(cp.closest_xy[1]),  # 12-13: closest_pt
            float(cp.closest_xy[0] > 0),                     # 14: is_ahead
            *[s[0] for s in samples],                        # 15-24: sample_x
            *[s[1] for s in samples],                        # 25-34: sample_y
            *[1.0] * len(samples) + [0.0] * (10 - len(samples)),  # 35-44: sample_valid
        ], dtype=torch.float32)

        lane_nodes.append(LaneNode(
            polyline_id=poly_id,
            features=features,
            points_ego=points_ego,
            heading=heading,
            speed_limit=speed_limit,
        ))

    # Sort by distance, keep top K
    lane_nodes.sort(key=lambda n: n.features[11])  # closest_dist
    lane_nodes = lane_nodes[:config.max_lanes]

    # ================================================================
    # PHASE 5: Build TRAFFIC_LIGHT nodes
    # ================================================================
    tl_nodes = []

    valid_tl_idx = np.where(tl_current.valid)[0]

    for i in valid_tl_idx:
        x_ego, y_ego = to_ego_pos(tl_current.x[i], tl_current.y[i])
        dist = np.hypot(x_ego, y_ego)

        if dist > config.tl_radius_m:
            continue

        state = int(tl_current.state[i]) if tl_current.state is not None else 0

        is_ahead = x_ego > 0
        control_conf = max(0, 1 - abs(y_ego) / 6.0) * float(is_ahead)

        time_to_change = compute_time_to_change(i, state, tl_future)

        # Build feature vector [12]
        features = torch.tensor([
            float(x_ego), float(y_ego),              # 0-1: position
            float(dist),                             # 2: distance
            float(state == 0),                       # 3: state_unknown
            float(state == 4 or state == 7),         # 4: state_red
            float(state == 5 or state == 8),         # 5: state_yellow
            float(state == 6),                       # 6: state_green
            float(state in [1, 2, 3]),               # 7: state_flashing
            float(is_ahead),                         # 8: is_ahead
            float(control_conf),                     # 9: control_confidence
            float(time_to_change / 80.0) if time_to_change else 0.0,  # 10: time_to_change_norm
            float(time_to_change is not None),       # 11: has_future_info
        ], dtype=torch.float32)

        tl_nodes.append(TLNode(
            cluster_id=int(tl_current.tl_id[i]) if tl_current.tl_id is not None else i,
            features=features,
            x_ego=x_ego, y_ego=y_ego,
        ))

    # Cluster nearby TLs
    tl_nodes = cluster_tl_nodes(tl_nodes, config.tl_cluster_dist_m)
    tl_nodes = tl_nodes[:config.max_traffic_lights]

    # ================================================================
    # PHASE 6: Build node mapping
    # ================================================================
    mapping = build_node_mapping(agent_nodes, lane_nodes, tl_nodes)

    # ================================================================
    # PHASE 7: Build edges
    # ================================================================

    # AGENT_TO_AGENT edges
    a2a_edges = []
    for src in agent_nodes:
        for dst in agent_nodes:
            if src.original_id == dst.original_id:
                continue

            # Only ego→all required; other→other if close
            dist = np.hypot(dst.x_ego - src.x_ego, dst.y_ego - src.y_ego)
            if not src.is_ego and dist > config.agent_interaction_radius_m:
                continue

            features = compute_a2a_features(src, dst, config)
            a2a_edges.append((
                mapping.agent_to_idx[src.original_id],
                mapping.agent_to_idx[dst.original_id],
                features,
            ))

    # AGENT_TO_LANE edges
    a2l_edges = []
    for agent in agent_nodes:
        agent_xy = np.array([agent.x_ego, agent.y_ego])

        # Find k nearest lanes
        lane_dists = [(lane, closest_point_on_polyline(lane.points_ego, agent_xy))
                      for lane in lane_nodes]
        lane_dists = [(l, cp) for l, cp in lane_dists if cp is not None]
        lane_dists.sort(key=lambda x: x[1].dist)

        for lane, cp in lane_dists[:config.k_nearest_lanes]:
            features = compute_a2l_features(agent, lane, cp, config)
            a2l_edges.append((
                mapping.agent_to_idx[agent.original_id],
                mapping.lane_to_idx[lane.polyline_id],
                features,
            ))

    # LANE_TO_LANE edges
    l2l_edges = []
    for lane_a in lane_nodes:
        for lane_b in lane_nodes:
            if lane_a.polyline_id == lane_b.polyline_id:
                continue

            min_dist = min_polyline_distance(lane_a.points_ego, lane_b.points_ego)
            if min_dist > config.lane_adjacency_threshold_m:
                continue

            features = compute_l2l_features(lane_a, lane_b, min_dist)
            l2l_edges.append((
                mapping.lane_to_idx[lane_a.polyline_id],
                mapping.lane_to_idx[lane_b.polyline_id],
                features,
            ))

    # LANE_TO_TL edges
    l2tl_edges = []
    for lane in lane_nodes:
        lane_end = lane.points_ego[-1]

        for tl in tl_nodes:
            tl_xy = np.array([tl.x_ego, tl.y_ego])
            dist = np.linalg.norm(lane_end - tl_xy)

            if dist > config.tl_lane_radius_m:
                continue

            features = compute_l2tl_features(lane, tl, dist)
            l2tl_edges.append((
                mapping.lane_to_idx[lane.polyline_id],
                mapping.tl_to_idx[tl.cluster_id],
                features,
            ))

    # ================================================================
    # PHASE 8: Assemble PyG HeteroData
    # ================================================================
    graph = HeteroData()

    # Node features
    graph["agent"].x = torch.stack([n.features for n in agent_nodes]) if agent_nodes else torch.zeros(0, 45)
    graph["lane"].x = torch.stack([n.features for n in lane_nodes]) if lane_nodes else torch.zeros(0, 55)
    graph["tl"].x = torch.stack([n.features for n in tl_nodes]) if tl_nodes else torch.zeros(0, 12)

    # Edge indices and features
    def pack_edges(edges, feature_dim):
        if len(edges) == 0:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, feature_dim)
        src, dst, feats = zip(*edges)
        return torch.tensor([src, dst], dtype=torch.long), torch.stack(feats)

    graph["agent", "to", "agent"].edge_index, graph["agent", "to", "agent"].edge_attr = pack_edges(a2a_edges, 20)
    graph["agent", "to", "lane"].edge_index, graph["agent", "to", "lane"].edge_attr = pack_edges(a2l_edges, 9)
    graph["lane", "to", "lane"].edge_index, graph["lane", "to", "lane"].edge_attr = pack_edges(l2l_edges, 6)
    graph["lane", "to", "tl"].edge_index, graph["lane", "to", "tl"].edge_attr = pack_edges(l2tl_edges, 3)

    # ================================================================
    # PHASE 9: Validate
    # ================================================================
    errors = validate_graph(graph, mapping)
    if errors:
        raise ValueError(f"Graph validation failed: {errors}")

    # Metadata
    metadata = {
        "record_index": record_index,
        "num_agents": len(agent_nodes),
        "num_lanes": len(lane_nodes),
        "num_tls": len(tl_nodes),
        "num_a2a_edges": len(a2a_edges),
        "num_a2l_edges": len(a2l_edges),
        "num_l2l_edges": len(l2l_edges),
        "num_l2tl_edges": len(l2tl_edges),
    }

    return graph, mapping, metadata
```

---

## Edge Cases Checklist

| # | Edge Case | Handling |
|---|-----------|----------|
| 1 | Ego has no past history | Fill past positions with current position |
| 2 | Agent type is unknown (0 or >4) | Map to `type_unknown=1` |
| 3 | Speed limit is negative (-1) | Treat as missing (`has_speed_limit=0`) |
| 4 | Lane polyline has <2 points | Skip (can't compute heading) |
| 5 | TL state is unknown (0) | Set `state_unknown=1` |
| 6 | No future TL info available | `has_future_info=0`, `time_to_change_norm=0` |
| 7 | Agent at exactly ego position | Valid (it's ego itself at origin) |
| 8 | Zero-length lane segment | Skip or clamp length to epsilon |
| 9 | NaN in TFRecord features | Detect and replace with 0 or skip |
| 10 | Very large coordinates (>1000m) | Clip or warn (likely error) |
| 11 | Duplicate polyline IDs | Use first occurrence or merge |
| 12 | Circular lane references | L2L edges naturally handle (A→B, B→A) |

---

## Complexity Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Shared extraction refactor | 1-2 days | Mostly reusing tfexample_io.py |
| Coordinate transform layer | 0.5 day | Simple, already have ego_frame() |
| Agent node builder | 1 day | Port from build_semantic_graph_v2.py |
| Lane node builder | 1 day | Port from polyline_grouper.py |
| TL node builder | 0.5 day | Port from spatial_graph_builder.py |
| A2A edge builder | 1 day | Port relation logic from semantic graph |
| A2L edge builder | 1-2 days | New, needs closest_point_on_polyline |
| L2L edge builder | 0.5 day | Simple spatial proximity |
| L2TL edge builder | 0.5 day | Simple spatial proximity |
| ID mapping + PyG assembly | 1 day | Boilerplate |
| Validation + testing | 2 days | Important for correctness |
| Integration + debugging | 2-3 days | Unexpected issues |

**Total estimate: 12-15 working days (~2-3 weeks)**

This is a **medium-sized refactoring project**, not a weekend hack but also not a multi-month effort. The key risk is subtle bugs in coordinate transforms and edge cases in TFRecord parsing.

---

## Recommended Development Order

1. **Week 1**: Shared extraction + coordinate transform + agent nodes
   - Get agent nodes working in ego frame
   - Validate against build_semantic_graph_v2.py output

2. **Week 2**: Lane + TL nodes + A2A edges
   - Get full node set working
   - Validate node counts against spatial_graph_builder.py

3. **Week 3**: A2L + L2L + L2TL edges + integration
   - Complete edge builders
   - Full validation suite
   - Performance optimization if needed
