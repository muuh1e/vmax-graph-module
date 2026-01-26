# Spatial Graph Building Pipeline - Detailed Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Data Extraction Layer](#data-extraction-layer)
5. [Module Details](#module-details)
6. [Build Process Flow](#build-process-flow)
7. [Configuration Options](#configuration-options)

---

## Overview

The Spatial Graph Building Pipeline transforms Waymo Open Dataset TFRecord files into LLM-friendly JSON graph representations. The pipeline extracts ego vehicle state, roadgraph features (map), traffic lights, and route information, then builds a structured graph with nodes and edges that captures spatial relationships for autonomous driving decision-making.

**Key Design Principles:**
- **Phase-based processing**: Modular extraction → grouping → graph construction
- **Semantic enrichment**: Raw numeric types converted to human-readable names
- **Intelligent filtering**: Relevance-based selection to keep only decision-critical features
- **No topology assumptions**: Works with point-cloud data without lane connectivity

---

## Directory Structure

```
spatial_graph/
├── __init__.py                    # Package initializer
├── README.md                      # Basic project information
│
├── modules/                       # Core graph building modules
│   ├── __init__.py
│   ├── polyline_grouper.py       # Phase 1: Roadgraph point cloud → polyline features
│   ├── route_monitor.py          # Phase 2: Route extraction and ego-route relationship
│   └── spatial_graph_builder.py  # Phase 3: Final graph assembly with TL clustering
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── tfexample_io.py           # TFRecord parsing and data extraction
│   └── geometry.py               # Geometric transformations and calculations
│
├── scripts/                       # Executable scripts
│   ├── run_spatial_graph_build.py      # CLI for building spatial graphs
│   ├── visualize_spatial_graph.py      # Visualization tools
│   └── plot_spatial_graph.py           # Plotting utilities
│
├── schemas/                       # Data schemas (if any)
├── outputs/                       # Generated graph outputs
├── visualizations/                # Generated visualizations
└── notes/                        # Development notes
```

---

## Pipeline Architecture

The pipeline operates in **three main phases**:

```
TFRecord → [Data Extraction] → [Phase 1: Polyline Grouping]
                              → [Phase 2: Route Analysis]
                              → [Phase 3: Graph Assembly]
                              → JSON Output
```

### Phase Flow Diagram
```
┌─────────────────┐
│  TFRecord File  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  DATA EXTRACTION (tfexample_io.py)          │
│  • Ego state                                │
│  • Roadgraph samples (point cloud)          │
│  • Traffic lights (current/past/future)     │
│  • Path samples (route)                     │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  PHASE 1: Polyline Grouper                  │
│  (polyline_grouper.py)                      │
│  • Group points by roadgraph ID             │
│  • V-Max spatial filtering                  │
│  • Compute closest distances to ego         │
│  • Add semantic type names                  │
│  Output: List[MapFeature]                   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  PHASE 2: Route Monitor                     │
│  (route_monitor.py)                         │
│  • Extract path polyline(s)                 │
│  • Compute Frenet coordinates (s, d)        │
│  • Classify route-following status          │
│  Output: (RouteNode, EgoRouteEdge)          │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  PHASE 3: Graph Builder                     │
│  (spatial_graph_builder.py)                 │
│  • Cluster traffic lights                   │
│  • Semantic map filtering                   │
│  • Build nodes (ego, map, TL, route)        │
│  • Build edges (spatial relationships)      │
│  Output: {meta, nodes, edges}               │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   JSON Graph    │
└─────────────────┘
```

---

## Data Extraction Layer

### File: `utils/tfexample_io.py`

This module handles all TFRecord parsing and provides clean Python interfaces for Waymo dataset features.

#### Core Data Structures

##### 1. **EgoState** (Lines 65-72)
```python
@dataclass(frozen=True)
class EgoState:
    idx: int           # Index in state arrays
    x: float           # World X coordinate (meters)
    y: float           # World Y coordinate (meters)
    yaw: float         # Heading angle (radians)
    speed: float       # Speed (m/s)
    speed_source: str  # "feature" | "computed" | "missing"
```

**Extraction Logic** (`get_ego_state`, lines 75-128):
1. Find ego vehicle: `is_sdc` array where value == 1
2. Extract position from `state/current/x` and `state/current/y`
3. Prefer `bbox_yaw` over `vel_yaw` for heading
4. Speed priority: `state/current/speed` → computed from velocity components → 0

```python
# Critical code snippet (lines 85-98)
ego_idx = int(np.argmax(is_sdc))  # SDC = Self-Driving Car

yaw = 0.0
bbox_yaw = _float_feature(features, "state/current/bbox_yaw")
vel_yaw = _float_feature(features, "state/current/vel_yaw")
if bbox_yaw is not None and bbox_yaw.size > ego_idx:
    yaw = float(bbox_yaw[ego_idx])
elif vel_yaw is not None and vel_yaw.size > ego_idx:
    yaw = float(vel_yaw[ego_idx])
```

##### 2. **RoadgraphSamples** (Lines 134-141)
Point cloud representation of the HD map.

```python
@dataclass(frozen=True)
class RoadgraphSamples:
    xyz: np.ndarray                    # (M, 3) positions
    valid: np.ndarray                  # (M,) validity mask
    rg_type: Optional[np.ndarray]      # (M,) type codes (1=FREEWAY, 2=SURFACE_STREET, etc.)
    rg_id: Optional[np.ndarray]        # (M,) polyline IDs for grouping
    rg_dir: Optional[np.ndarray]       # (M, 3) direction vectors
    speed_limit: Optional[np.ndarray]  # (M,) speed limits in m/s
```

**Extraction Logic** (`get_roadgraph_samples`, lines 144-179):
- Reshapes flat arrays into structured format
- Validates all arrays have consistent length M
- Handles missing optional fields gracefully

##### 3. **TrafficLightSet** (Lines 185-191)
Current state of all traffic lights.

```python
@dataclass(frozen=True)
class TrafficLightSet:
    valid: np.ndarray                # (L,) validity
    x: np.ndarray                    # (L,) world X
    y: np.ndarray                    # (L,) world Y
    state: Optional[np.ndarray]      # (L,) state enum
    tl_id: Optional[np.ndarray]      # (L,) IDs
```

**Time-Major Extraction** (`get_traffic_lights_time_major`, lines 216-245):
For past/future predictions, reshapes `(T*L,)` arrays into `(T, L)` time-major format.

```python
# Reshape logic (lines 241-243)
if arr.size % num_lights != 0:
    continue
T = arr.size // num_lights
out[name] = arr.reshape(T, num_lights)
```

##### 4. **PathSamples** (Lines 251-256)
Route prior information.

```python
@dataclass(frozen=True)
class PathSamples:
    xyz: np.ndarray                      # (P, 3) path points
    valid: np.ndarray                    # (P,) validity
    path_id: Optional[np.ndarray]        # (P,) route IDs (if multiple)
    arc_length: Optional[np.ndarray]     # (P,) cumulative distance
```

---

## Module Details

### Module 1: `modules/polyline_grouper.py`
**Purpose**: Convert roadgraph point cloud into semantic polyline-level features.

#### Key Components

##### Type Mapping (Lines 30-73)
Converts Waymo numeric codes to human-readable names:

```python
WAYMO_ROADGRAPH_TYPES = {
    0: "UNKNOWN",
    1: "FREEWAY",
    2: "SURFACE_STREET",
    3: "BIKE_LANE",
    6: "ROAD_EDGE_BOUNDARY",
    7: "ROAD_EDGE_MEDIAN",
    9: "BROKEN_SINGLE_WHITE",
    10: "SOLID_SINGLE_WHITE",
    # ... more types
    17: "CROSSWALK",
    18: "SPEED_BUMP",
    19: "STOP_SIGN",
}

TYPE_CATEGORIES = {
    "DRIVABLE": {1, 2, 3, 20},      # Can drive on
    "BOUNDARY": {6, 7},              # Road edges
    "MARKING": {9, 10, 11, ...},    # Lane markings
    "SPECIAL": {17, 18, 19},        # Crosswalks, bumps, signs
}
```

##### MapFeature Dataclass (Lines 79-109)
Output structure for each polyline:

```python
@dataclass(frozen=True)
class MapFeature:
    polyline_id: int

    # Semantics (ENHANCED)
    type_mode: Optional[int]    # Raw numeric code
    type_name: str              # "SURFACE_STREET"
    type_category: str          # "DRIVABLE"

    # Geometry
    num_points: int
    centroid_xy: Tuple[float, float]
    bbox_xy_min: Tuple[float, float]
    bbox_xy_max: Tuple[float, float]
    speed_limit_mps: Optional[float]  # Median speed limit (m/s)

    # Ego relationship
    closest_dist_m: float
    closest_world_xy: Tuple[float, float]
    closest_x_local_m: float    # Forward distance
    closest_y_local_m: float    # Lateral distance (left+)
    heading_alignment_cos: Optional[float]  # cos(angle) with ego

    sample_points_xy: List[Tuple[float, float]]  # Compact samples
```

##### Main Processing Function (Lines 184-237)
**`build_map_features_from_roadgraph`**

**Step-by-step logic**:

1. **Filter valid points** (lines 200-203):
```python
points_xy = xyz[:, :2].astype(np.float32)
valid_mask = rg.valid.astype(bool)
```

2. **Apply V-Max spatial filter** (lines 204-215):
   - Ego-centered box: forward (50m), back (5m), lateral (20m)
   - Reduces computational load by ignoring distant points
```python
if cfg.use_vmax_box:
    in_box = select_points_in_vmax_box(
        points_xy=points_xy,
        ego_xy=np.array([ego.x, ego.y], dtype=np.float32),
        ego_yaw=float(ego.yaw),
        fwd=float(cfg.box_fwd_m),
        back=float(cfg.box_back_m),
        side=float(cfg.box_side_m),
    )
    mask = valid_mask & in_box
```

3. **Group by polyline ID** (lines 217-233):
```python
if rg.rg_id is None:
    # Treat all as single polyline
    return [_summarize_one_polyline(polyline_id=0, idx=idx_all, ...)]

ids = rg.rg_id[idx_all]
uniq_ids = np.unique(ids)

for pid in uniq_ids:
    idx = idx_all[ids == pid]
    features.append(_summarize_one_polyline(polyline_id=int(pid), idx=idx, ...))
```

4. **Sort by distance and limit** (lines 235-237):
```python
features.sort(key=lambda mf: mf.closest_dist_m)
return features[: cfg.max_polylines]  # Default: 80
```

##### Polyline Summarization (Lines 240-324)
**`_summarize_one_polyline`** - Core feature extraction:

**Geometry computation** (lines 249-252):
```python
centroid = np.mean(points_xy, axis=0)
bb_min = np.min(points_xy, axis=0)
bb_max = np.max(points_xy, axis=0)
```

**Closest point to ego** (lines 255-268):
```python
dx = points_xy[:, 0] - float(ego.x)
dy = points_xy[:, 1] - float(ego.y)
dist2 = dx * dx + dy * dy
local_min = int(np.argmin(dist2))
closest_dist = float(np.sqrt(dist2[local_min]))

# Transform to ego frame (forward/lateral)
x_local, y_local = ego_frame(
    dx=np.array([dx[local_min]], dtype=np.float32),
    dy=np.array([dy[local_min]], dtype=np.float32),
    ego_yaw=float(ego.yaw),
)
```

**Speed limit extraction** (lines 279-291):
```python
def _median_speed(speed, idx, ignore_negative):
    vals = speed[idx].astype(np.float32)
    if ignore_negative:
        valid = vals[vals >= 0.0]  # Filter -1 (missing) values
        if valid.size > 0:
            vals = valid
    return float(np.median(vals))
```

**Heading alignment** (lines 294-295):
- Computes `cos(angle)` between ego heading and polyline direction
- Values: 1 (aligned), 0 (perpendicular), -1 (opposite)

---

### Module 2: `modules/route_monitor.py`
**Purpose**: Extract route information and compute ego-route relationship using Frenet-like coordinates.

#### Key Data Structures

##### RouteNode (Lines 40-47)
Global route summary:
```python
@dataclass(frozen=True)
class RouteNode:
    route_id: int
    num_points: int
    centroid_xy: Tuple[float, float]
    bbox_xy_min: Tuple[float, float]
    bbox_xy_max: Tuple[float, float]
    has_arc_length: bool  # If original arc_length available
```

##### EgoRouteEdge (Lines 50-62)
Ego's relationship to route:
```python
@dataclass(frozen=True)
class EgoRouteEdge:
    route_id: int
    closest_dist_m: float
    closest_world_xy: Tuple[float, float]

    # Frenet-ish coordinates
    s_m: float                # Along-track position (arc length)
    signed_lateral_m: float   # Lateral error (left positive)

    # Status classification
    status: str  # "ON_TRACK" | "DEVIATING" | "OFF_ROUTE"
```

#### Main Processing Function (Lines 87-178)
**`build_route_node_and_edge`**

**Step-by-step logic**:

1. **Validation** (lines 96-100):
```python
if path is None:
    return None
valid_idx = np.where(path.valid)[0]
if valid_idx.size < 2:  # Need at least 2 points for polyline
    return None
```

2. **Handle multiple route IDs** (lines 104-137):
```python
if path.path_id is not None:
    ids = path.path_id[valid_idx]
    uniq = np.unique(ids)

    # Build candidate routes
    candidates = []
    for rid in uniq:
        idx = valid_idx[ids == rid]
        if idx.size < 2:
            continue
        route_xy = path.xyz[idx, :2].astype(np.float32)
        candidates.append((int(rid), route_xy))

    # Select closest route
    if cfg.route_selection == "closest":
        best = None
        best_dist = float("inf")
        for rid, route_xy in candidates:
            res = closest_point_on_polyline(route_xy, ego_xy)
            if res.dist < best_dist:
                best_dist = res.dist
                best = (rid, route_xy, res)
```

3. **Status classification** (lines 157-167):
Uses hierarchical thresholds:
```python
if cp.dist >= cfg.off_route_dist_m:  # Default: 10m
    status = "OFF_ROUTE"
else:
    lat = abs(cp.signed_lateral)
    if lat <= cfg.on_track_lat_m:     # Default: 1.5m
        status = "ON_TRACK"
    elif lat <= cfg.deviating_lat_m:  # Default: 4.0m
        status = "DEVIATING"
    else:
        status = "OFF_ROUTE"
```

**Frenet coordinates** are computed in `utils/geometry.py` by projecting ego onto route segments.

---

### Module 3: `modules/spatial_graph_builder.py`
**Purpose**: Orchestrate all phases and build final graph JSON with traffic light clustering and semantic filtering.

#### Configuration (Lines 54-80)

```python
@dataclass(frozen=True)
class SpatialGraphConfig:
    poly_cfg: PolylineGrouperConfig       # Phase 1 config
    route_cfg: RouteMonitorConfig         # Phase 2 config

    # Traffic light selection
    tl_radius_m: float = 80.0             # Search radius
    tl_topk: int = 20                     # Max TLs to keep
    tl_cluster_dist_m: float = 1.0        # Cluster within 1m

    # TL relevance heuristics
    tl_relevance_y_thresh_m: float = 6.0  # "Near centerline"
    tl_relevance_min_x_m: float = 0.0     # Only ahead

    # Map filtering
    map_max_features: int = 20            # Final cap (reduced from 80)
    map_safety_dist_m: float = 8.0        # Always keep if < 8m
    map_lane_proxy_align_thresh: float = 0.7  # cos(45°)
    map_lane_proxy_lateral_m: float = 5.0

    # Edge bucketing
    dist_very_close_m: float = 5.0
    dist_close_m: float = 15.0
    dist_medium_m: float = 30.0
```

#### Traffic Light Clustering (Lines 113-192)
**`_cluster_traffic_lights`** - **NEW ENHANCEMENT**

Reduces redundancy by merging nearby traffic lights:

**Algorithm** (agglomerative clustering):
```python
# Compute pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(positions))

clusters = []
used = set()

for i in range(len(positions)):
    if i in used:
        continue

    # Find all TLs within cluster_dist (default 1.0m)
    close = np.where(distances[i] <= cluster_dist)[0]
    cluster_indices = [j for j in close if j not in used]
    used.update(cluster_indices)

    # Representative: closest to cluster centroid
    cluster_pos = positions[cluster_indices]
    centroid = np.mean(cluster_pos, axis=0)
    dists_to_centroid = np.linalg.norm(cluster_pos - centroid, axis=1)
    rep_index = cluster_indices[int(np.argmin(dists_to_centroid))]

    # Representative state: median (robust to outliers)
    rep_state = int(np.median(member_states))

    clusters.append({
        "representative_index": rep_index,
        "position": (x, y),
        "member_ids": [...],
        "member_states": [...],
        "representative_state": rep_state,
    })
```

#### TL Control Confidence (Lines 198-227)
**`_compute_tl_control_confidence`** - **NEW ENHANCEMENT**

Heuristic to estimate if a traffic light controls ego's lane:

```python
def _compute_tl_control_confidence(x_local, y_local, heading_alignment):
    # Ahead factor (binary)
    ahead_conf = 1.0 if x_local > 0 else 0.0

    # Lateral centering (drops linearly to 0 at 6m)
    lat_conf = max(0.0, 1.0 - abs(y_local) / 6.0)

    # Heading alignment (optional, if TL direction available)
    align_conf = (heading_alignment + 1.0) / 2.0  # Map [-1,1] → [0,1]

    # Combined confidence (product)
    return ahead_conf * lat_conf * align_conf
```

Example outputs:
- TL 10m ahead, 0.5m left: `confidence ≈ 0.92`
- TL 15m ahead, 5m left: `confidence ≈ 0.17`
- TL behind ego: `confidence = 0.0`

#### Semantic Map Filtering (Lines 232-331)
**`_filter_map_features_semantic`** - **NEW ENHANCEMENT**

Reduces map features from 80 → 20 using priority-based selection:

**Priority categories** (lines 256-290):
1. **Safety-critical**: `dist < 8m` (always keep)
2. **Lane candidates**: Aligned + ahead + close lateral offset
3. **TL-relevant lanes**: Polylines near traffic lights
4. **Forward context**: Other ahead features
5. **Others**: Remaining features

```python
# Category 1: Safety-critical
if mf.closest_dist_m < cfg.map_safety_dist_m:
    safety.append(mf)
    continue

# Category 2: Lane-proxy candidates
if (mf.heading_alignment_cos > cfg.map_lane_proxy_align_thresh and  # cos > 0.7 (< 45°)
    mf.closest_x_local_m > 0 and                                     # Ahead
    abs(mf.closest_y_local_m) < cfg.map_lane_proxy_lateral_m):      # < 5m lateral
    lane_candidates.append(mf)
    continue

# Category 3: TL-relevant lanes
if mf.polyline_id in tl_polyline_ids:
    tl_lanes.append(mf)
    continue
```

**Budget allocation** (lines 293-318):
```python
result = []
result.extend(sorted(safety, key=lambda m: m.closest_dist_m))
remaining = cfg.map_max_features - len(result)

# At least half remaining budget for lane candidates
if remaining > 0:
    budget = max(1, remaining // 2)
    result.extend(sorted(lane_candidates, key=lambda m: m.closest_dist_m)[:budget])
    remaining = cfg.map_max_features - len(result)

# Half budget for TL lanes
if remaining > 0:
    budget = max(1, remaining // 2)
    result.extend(sorted(tl_lanes, key=lambda m: m.closest_dist_m)[:budget])
    remaining = cfg.map_max_features - len(result)

# Fill with forward context
if remaining > 0:
    result.extend(sorted(forward_context, key=lambda m: m.closest_dist_m)[:remaining])
```

#### Main Graph Builder (Lines 393-670)
**`build_spatial_graph_from_example`**

**Complete flow**:

1. **Extract raw data** (lines 407-415):
```python
ego: EgoState = get_ego_state(features)
rg: RoadgraphSamples = get_roadgraph_samples(features)
tl_cur: TrafficLightSet = get_traffic_lights_current(features)
path: Optional[PathSamples] = get_path_samples(features)

tl_future = get_traffic_lights_time_major(features, "future", num_lights)
tl_past = get_traffic_lights_time_major(features, "past", num_lights)
```

2. **Phase 1: Map features** (line 418):
```python
map_features_raw: List[MapFeature] = build_map_features_from_roadgraph(
    rg=rg, ego=ego, cfg=cfg.poly_cfg
)
```

3. **Phase 2: Route** (lines 421-425):
```python
route_pack = build_route_node_and_edge(ego=ego, path=path, cfg=cfg.route_cfg)
if route_pack is not None:
    route_node, route_edge = route_pack
```

4. **Phase 3a: TL clustering** (lines 428-463):
```python
tl_valid_idx = np.where(tl_cur.valid.astype(bool))[0]

# Filter by radius
dx = tl_cur.x[tl_valid_idx] - ego.x
dy = tl_cur.y[tl_valid_idx] - ego.y
dist = np.sqrt(dx * dx + dy * dy)
in_rad = dist <= cfg.tl_radius_m

# Top-k closest
order = np.argsort(dist)[: cfg.tl_topk]
idx_pick = tl_valid_idx[in_rad][order]

# Cluster
tl_clusters = _cluster_traffic_lights(
    tl_x=tl_cur.x[idx_pick],
    tl_y=tl_cur.y[idx_pick],
    tl_idx=idx_pick,
    tl_cur=tl_cur,
    cluster_dist=cfg.tl_cluster_dist_m,
)
```

5. **Phase 3b: Semantic filtering** (lines 471-477):
```python
map_features, filter_stats = _filter_map_features_semantic(
    map_features=map_features_raw,
    ego=ego,
    route_edge=route_edge,
    tl_polyline_ids=tl_polyline_ids,  # Currently empty (no TL-lane association)
    cfg=cfg,
)
```

6. **Build graph nodes** (lines 479-624):

**Ego node** (lines 484-496):
```python
nodes.append({
    "id": "ego",
    "type": "EGO",
    "attrs": {
        "x": ego.x,
        "y": ego.y,
        "yaw": ego.yaw,
        "speed_mps": ego.speed,
        "speed_source": ego.speed_source,
    }
})
```

**Map feature nodes** (lines 499-520):
```python
for mf in map_features:
    nid = f"map_{mf.polyline_id}"
    nodes.append({
        "id": nid,
        "type": "MAP_FEATURE",
        "attrs": {
            "polyline_id": mf.polyline_id,
            "type_mode": mf.type_mode,
            "type_name": mf.type_name,           # "SURFACE_STREET"
            "type_category": mf.type_category,   # "DRIVABLE"
            "num_points": mf.num_points,
            "centroid_xy": mf.centroid_xy,
            "speed_limit_mps": mf.speed_limit_mps,
            "heading_alignment_cos": mf.heading_alignment_cos,
            # ...
        }
    })
```

**Traffic light cluster nodes** (lines 544-618):
```python
for cluster in tl_clusters:
    tid = cluster["member_ids"][0]
    nid = f"tl_cluster_{tid}"

    # Compute relevance
    ahead = bool(x_local >= 0.0)
    centered = bool(abs(y_local) <= 6.0)
    likely_controls_ego = ahead and centered

    control_confidence = _compute_tl_control_confidence(
        x_local=x_local,
        y_local=y_local,
    )

    nodes.append({
        "id": nid,
        "type": "TRAFFIC_LIGHT_CLUSTER",
        "attrs": {
            "cluster_id": tid,
            "member_tl_ids": cluster["member_ids"],
            "member_states": cluster["member_states"],
            "representative_state": cluster["representative_state"],
            "num_members": len(cluster["member_ids"]),
            # ...
        }
    })
```

**Route node** (lines 621-639):
```python
if route_node is not None:
    nodes.append({
        "id": f"route_{route_node.route_id}",
        "type": "ROUTE",
        "attrs": {
            "route_id": route_node.route_id,
            "num_points": route_node.num_points,
            "centroid_xy": route_node.centroid_xy,
            # ...
        }
    })
```

7. **Build graph edges** (lines 521-639):

**Ego → Map edges** (lines 526-542):
```python
edges.append({
    "src": "ego",
    "dst": f"map_{mf.polyline_id}",
    "type": "EGO_TO_MAP",
    "attrs": {
        "closest_dist_m": mf.closest_dist_m,
        "x_local_m": mf.closest_x_local_m,
        "y_local_m": mf.closest_y_local_m,
        "distance_bucket": "very_close" | "close" | "medium" | "far",
        "ahead": bool(x_local > 0),
        "left": bool(y_local > 0),
        "overspeeding": bool(ego.speed > speed_limit + margin),
    }
})
```

**Ego → TL edges** (lines 602-618):
```python
edges.append({
    "src": "ego",
    "dst": nid,
    "type": "EGO_TO_TL",
    "attrs": {
        "dist_m": dist_tl,
        "x_local_m": x_local,
        "y_local_m": y_local,
        "ahead": ahead,
        "likely_controls_ego": likely_controls_ego,
        "control_confidence": control_confidence,  # NEW
        "distance_bucket": bucket,
        "future_unique_states_valid": [...],       # Future prediction
        "time_to_change_steps": 5,                 # Steps until state change
        # ...
    }
})
```

**Ego → Route edge** (lines 632-639):
```python
edges.append({
    "src": "ego",
    "dst": f"route_{route_id}",
    "type": "EGO_TO_ROUTE",
    "attrs": {
        "closest_dist_m": route_edge.closest_dist_m,
        "s_m": route_edge.s_m,                    # Arc length
        "signed_lateral_m": route_edge.signed_lateral_m,
        "status": "ON_TRACK" | "DEVIATING" | "OFF_ROUTE",
        "is_following_route": bool(status == "ON_TRACK"),
        "is_drifting_left": bool(lateral > 0.5),
        "is_drifting_right": bool(lateral < -0.5),
    }
})
```

8. **Build metadata** (lines 641-669):
```python
meta = {
    "record_index": record_index,
    "num_nodes": len(nodes),
    "num_edges": len(edges),
    "map_features": {
        "raw_count": len(map_features_raw),
        "filtered_count": len(map_features),
        "filter_stats": {
            "safety": 3,
            "lane_candidates": 5,
            "tl_lanes": 0,
            "forward_context": 12,
            # ...
        }
    },
    "traffic_lights": {
        "total_valid": 45,
        "within_radius": 25,
        "num_clusters": 8,
    },
    "limits": {
        "max_polylines_raw": 80,
        "max_map_features_filtered": 20,
        "tl_topk": 20,
        "tl_cluster_dist_m": 1.0,
    },
    "notes": {
        "no_lane_topology_in_tfexample": True,
        "tl_relevance_is_heuristic": True,
        "tl_clustering_enabled": True,
        "semantic_map_filtering_enabled": True,
    }
}
```

---

## Build Process Flow

### Entry Point: `scripts/run_spatial_graph_build.py`

**CLI Usage**:
```bash
python scripts/run_spatial_graph_build.py \
  --tfrecord /path/to/training.tfrecord \
  --record 0 \
  --out outputs/graph.json \
  --max_polylines 80 \
  --tl_topk 20 \
  --tl_radius 80.0
```

**Main execution** (lines 56-92):

1. **Read TFRecord** (line 57):
```python
ex = read_example_from_tfrecord(args.tfrecord, args.record)
```

2. **Configure pipeline** (lines 59-82):
```python
poly_cfg = PolylineGrouperConfig(
    use_vmax_box=not args.no_vmax_box,
    box_fwd_m=args.box_fwd,      # Default: 50m
    box_back_m=args.box_back,    # Default: 5m
    box_side_m=args.box_side,    # Default: 20m
    max_polylines=args.max_polylines,  # Default: 80
    max_sample_points_per_polyline=args.max_sample_points,  # Default: 40
)

route_cfg = RouteMonitorConfig(
    on_track_lat_m=args.on_track_lat,      # Default: 1.5m
    deviating_lat_m=args.deviating_lat,    # Default: 4.0m
    off_route_dist_m=args.off_route_dist,  # Default: 10.0m
)

cfg = SpatialGraphConfig(
    poly_cfg=poly_cfg,
    route_cfg=route_cfg,
    tl_radius_m=args.tl_radius,            # Default: 80m
    tl_topk=args.tl_topk,                  # Default: 20
    tl_relevance_y_thresh_m=args.tl_y_thresh,  # Default: 6m
)
```

3. **Build graph** (line 84):
```python
graph = build_spatial_graph_from_example(
    example=ex,
    record_index=args.record,
    cfg=cfg
)
```

4. **Save output** (lines 86-92):
```python
with open(args.out, "w") as fp:
    json.dump(graph, fp, indent=2)
```

**Output JSON structure**:
```json
{
  "meta": {
    "record_index": 0,
    "num_nodes": 32,
    "num_edges": 31,
    "map_features": {...},
    "traffic_lights": {...},
    "limits": {...},
    "notes": {...}
  },
  "nodes": [
    {
      "id": "ego",
      "type": "EGO",
      "attrs": {"x": 123.4, "y": 567.8, "yaw": 1.57, "speed_mps": 8.3}
    },
    {
      "id": "map_42",
      "type": "MAP_FEATURE",
      "attrs": {
        "type_name": "SURFACE_STREET",
        "type_category": "DRIVABLE",
        "speed_limit_mps": 13.89,
        ...
      }
    },
    {
      "id": "tl_cluster_123",
      "type": "TRAFFIC_LIGHT_CLUSTER",
      "attrs": {
        "representative_state": 4,
        "member_tl_ids": [123, 124],
        "num_members": 2,
        ...
      }
    },
    {
      "id": "route_0",
      "type": "ROUTE",
      "attrs": {...}
    }
  ],
  "edges": [
    {
      "src": "ego",
      "dst": "map_42",
      "type": "EGO_TO_MAP",
      "attrs": {
        "closest_dist_m": 2.3,
        "x_local_m": 12.5,
        "y_local_m": -1.2,
        "distance_bucket": "very_close",
        "ahead": true,
        "overspeeding": false
      }
    },
    {
      "src": "ego",
      "dst": "tl_cluster_123",
      "type": "EGO_TO_TL",
      "attrs": {
        "dist_m": 45.2,
        "likely_controls_ego": true,
        "control_confidence": 0.85,
        "time_to_change_steps": 7,
        ...
      }
    },
    {
      "src": "ego",
      "dst": "route_0",
      "type": "EGO_TO_ROUTE",
      "attrs": {
        "s_m": 123.4,
        "signed_lateral_m": 0.3,
        "status": "ON_TRACK"
      }
    }
  ]
}
```

---

## Configuration Options

### PolylineGrouperConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vmax_box` | True | Enable ego-centered spatial filtering |
| `box_fwd_m` | 50.0 | Forward distance (meters) |
| `box_back_m` | 5.0 | Backward distance (meters) |
| `box_side_m` | 20.0 | Lateral distance (meters) |
| `max_polylines` | 80 | Maximum polylines to keep |
| `max_sample_points_per_polyline` | 40 | Points per polyline for sampling |
| `speed_limit_ignore_negative` | True | Ignore -1 (missing) speed limits |

### RouteMonitorConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `on_track_lat_m` | 1.5 | Lateral threshold for "ON_TRACK" status |
| `deviating_lat_m` | 4.0 | Lateral threshold for "DEVIATING" status |
| `off_route_dist_m` | 10.0 | Distance threshold for "OFF_ROUTE" status |
| `route_selection` | "closest" | How to choose from multiple routes |

### SpatialGraphConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tl_radius_m` | 80.0 | Search radius for traffic lights |
| `tl_topk` | 20 | Maximum traffic lights to keep |
| `tl_cluster_dist_m` | 1.0 | Clustering distance threshold |
| `tl_relevance_y_thresh_m` | 6.0 | Lateral threshold for TL relevance |
| `tl_relevance_min_x_m` | 0.0 | Minimum forward distance for TL relevance |
| `map_max_features` | 20 | Final map feature count (after filtering) |
| `map_safety_dist_m` | 8.0 | Safety-critical distance threshold |
| `map_lane_proxy_align_thresh` | 0.7 | Heading alignment threshold (cos(45°)) |
| `map_lane_proxy_lateral_m` | 5.0 | Lateral distance for lane candidates |
| `dist_very_close_m` | 5.0 | "Very close" bucket threshold |
| `dist_close_m` | 15.0 | "Close" bucket threshold |
| `dist_medium_m` | 30.0 | "Medium" bucket threshold |
| `overspeed_margin_mps` | 0.5 | Speed tolerance for overspeeding detection |

---

## Key Enhancements in Current Version

### 1. Traffic Light Clustering
- **Problem**: Redundant TLs at same intersection
- **Solution**: Cluster TLs within 1m, use representative state
- **Benefit**: Reduces graph size, cleaner for LLM

### 2. TL Control Confidence
- **Problem**: No explicit TL-lane association in dataset
- **Solution**: Heuristic confidence score (ahead × lateral × alignment)
- **Benefit**: LLM can prioritize relevant TLs

### 3. Semantic Map Filtering
- **Problem**: 80 map features too verbose for LLM
- **Solution**: Priority-based filtering → 20 features
- **Categories**: Safety → Lane candidates → TL lanes → Forward context
- **Benefit**: Focused, decision-relevant features

### 4. Human-Readable Type Names
- **Problem**: Numeric type codes (1, 2, 6, etc.)
- **Solution**: Added type dictionary + categories
- **Benefit**: "SURFACE_STREET" more interpretable than "2"

### 5. Speed Limit Handling
- **Units**: Documented as m/s (not km/h)
- **Validation**: Warns if > 60 m/s (216 km/h)
- **Missing values**: Ignores -1 (sentinel)

---

## Geometry Utilities (`utils/geometry.py`)

### Ego Frame Transform
**Purpose**: Convert world coordinates to ego-centered (forward/left) frame.

```python
def ego_frame(dx, dy, ego_yaw):
    """
    Rotate world-frame deltas into ego frame:
      x_local = forward (positive = ahead)
      y_local = left (positive = left side)
    """
    c = np.cos(-ego_yaw)
    s = np.sin(-ego_yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    return x_local, y_local
```

### V-Max Box Selection
**Purpose**: Filter points within ego-centered bounding box.

```python
def select_points_in_vmax_box(points_xy, ego_xy, ego_yaw, fwd=50, back=5, side=20):
    """
    Returns boolean mask for points in box:
      -back <= x_local <= fwd
      |y_local| <= side
    """
    pts_ego = world_to_ego_points_xy(points_xy, ego_xy, ego_yaw)
    x, y = pts_ego[:, 0], pts_ego[:, 1]
    return (x <= fwd) & (x >= -back) & (np.abs(y) <= side)
```

### Closest Point on Polyline
**Purpose**: Project point onto polyline to compute Frenet coordinates.

```python
def closest_point_on_polyline(poly_xy, query_xy):
    """
    Projects query onto each segment, returns:
      - dist: Euclidean distance
      - closest_xy: Closest point on polyline
      - seg_idx: Segment index [i, i+1]
      - t: Parameter in [0, 1] along segment
      - s: Arc length (meters)
      - signed_lateral: Left positive
    """
    # Project onto all segments
    p, q = poly_xy[:-1], poly_xy[1:]
    v = q - p
    w = query_xy - p

    vv = np.sum(v * v, axis=1)
    t = np.clip(np.sum(w * v, axis=1) / vv, 0.0, 1.0)

    proj = p + v * t.reshape(-1, 1)
    dist2 = np.sum((proj - query_xy) ** 2, axis=1)
    seg_idx = int(np.argmin(dist2))

    # Arc length
    seg_lens = np.sqrt(np.sum(v * v, axis=1))
    s = np.sum(seg_lens[:seg_idx]) + t[seg_idx] * seg_lens[seg_idx]

    # Signed lateral (2D cross product)
    heading = v[seg_idx] / np.linalg.norm(v[seg_idx])
    e = query_xy - proj[seg_idx]
    signed_lateral = heading[0] * e[1] - heading[1] * e[0]

    return ClosestPointOnPolyline(dist, proj[seg_idx], seg_idx, t[seg_idx], s, signed_lateral)
```

---

## Summary

This spatial graph pipeline provides a complete, production-ready solution for converting Waymo TFRecords into LLM-friendly graph representations. Key strengths:

1. **Modular design**: Clear separation of concerns (extraction → grouping → assembly)
2. **Semantic enrichment**: Human-readable type names, confidence scores, status labels
3. **Intelligent filtering**: Priority-based selection keeps only decision-relevant features
4. **Robust handling**: Graceful degradation for missing fields (path, TL direction, etc.)
5. **Configurable**: Extensive parameters for tuning spatial extents, thresholds, limits
6. **Well-documented**: Inline comments, dataclass annotations, docstrings

**Use cases**:
- Training spatial reasoning LLMs for autonomous driving
- Debugging V-Max model predictions
- Visualizing scene understanding
- Analyzing ego-route relationships

**Future enhancements**:
- Lane topology if Waymo adds connectivity data
- Explicit TL-lane associations (currently heuristic)
- Dynamic object tracking (current focus is static scene)
- Multi-agent graph edges (ego-object, object-object)
