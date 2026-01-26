# Unified Heterogeneous Graph Design

## Overview

A single ego-centric heterogeneous graph combining dynamic agents, static lane geometry, and traffic signals for autonomous driving scene representation.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  UNIFIED SCENE GRAPH                        │
                    │                                                             │
                    │    ┌─────────┐        AGENT_TO_AGENT         ┌─────────┐   │
                    │    │  AGENT  │◄─────────────────────────────►│  AGENT  │   │
                    │    │  (ego)  │                               │ (other) │   │
                    │    └────┬────┘                               └────┬────┘   │
                    │         │                                         │        │
                    │         │ AGENT_TO_LANE                           │        │
                    │         │                                         │        │
                    │         ▼                                         ▼        │
                    │    ┌─────────┐        LANE_TO_LANE           ┌─────────┐   │
                    │    │  LANE   │◄─────────────────────────────►│  LANE   │   │
                    │    └────┬────┘                               └─────────┘   │
                    │         │                                                  │
                    │         │ LANE_TO_TL                                       │
                    │         │                                                  │
                    │         ▼                                                  │
                    │    ┌─────────┐                                             │
                    │    │   TL    │                                             │
                    │    └─────────┘                                             │
                    │                                                             │
                    └─────────────────────────────────────────────────────────────┘
```

---

## Coordinate System

**All features are in EGO-CENTRIC frame:**
- Origin: Ego vehicle position at current timestep
- X-axis: Ego heading direction (forward positive)
- Y-axis: Perpendicular to heading (left positive)
- Yaw: Relative to ego heading (0 = same direction as ego)

---

## Node Types

### 1. AGENT Node

Represents dynamic actors: ego vehicle, other vehicles, pedestrians, cyclists.

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `is_ego` | float | {0, 1} | 1 if this is ego vehicle |
| 1 | `type_vehicle` | float | {0, 1} | One-hot: vehicle |
| 2 | `type_pedestrian` | float | {0, 1} | One-hot: pedestrian |
| 3 | `type_cyclist` | float | {0, 1} | One-hot: cyclist |
| 4 | `type_unknown` | float | {0, 1} | One-hot: unknown type |
| 5 | `x` | float | [-100, 100] m | Position X in ego frame |
| 6 | `y` | float | [-50, 50] m | Position Y in ego frame |
| 7 | `vx` | float | [-30, 30] m/s | Velocity X in ego frame |
| 8 | `vy` | float | [-10, 10] m/s | Velocity Y in ego frame |
| 9 | `yaw` | float | [-π, π] rad | Heading relative to ego |
| 10 | `speed` | float | [0, 40] m/s | Scalar speed |
| 11 | `length` | float | [0, 20] m | Bounding box length |
| 12 | `width` | float | [0, 5] m | Bounding box width |
| 13-22 | `past_x[0:10]` | float | m | Past 10 X positions (ego frame) |
| 23-32 | `past_y[0:10]` | float | m | Past 10 Y positions (ego frame) |
| 33-42 | `past_valid[0:10]` | float | {0, 1} | Past position validity mask |
| 43 | `is_braking` | float | {0, 1} | Deceleration detected |
| 44 | `is_accelerating` | float | {0, 1} | Acceleration detected |

**AGENT feature vector size: 45**

---

### 2. LANE Node

Represents drivable surface polylines (FREEWAY, SURFACE_STREET, BIKE_LANE, DRIVEWAY).

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `type_freeway` | float | {0, 1} | One-hot: freeway |
| 1 | `type_surface_street` | float | {0, 1} | One-hot: surface street |
| 2 | `type_bike_lane` | float | {0, 1} | One-hot: bike lane |
| 3 | `type_driveway` | float | {0, 1} | One-hot: driveway |
| 4 | `centroid_x` | float | m | Centroid X in ego frame |
| 5 | `centroid_y` | float | m | Centroid Y in ego frame |
| 6 | `heading_cos` | float | [-1, 1] | cos(lane_heading - ego_heading) |
| 7 | `heading_sin` | float | [-1, 1] | sin(lane_heading - ego_heading) |
| 8 | `speed_limit_mps` | float | [0, 40] m/s | Speed limit (0 if unknown) |
| 9 | `has_speed_limit` | float | {0, 1} | Speed limit available |
| 10 | `length_m` | float | [0, 200] m | Polyline length |
| 11 | `closest_dist_m` | float | [0, 100] m | Closest point distance to ego |
| 12 | `closest_x` | float | m | Closest point X in ego frame |
| 13 | `closest_y` | float | m | Closest point Y in ego frame |
| 14 | `is_ahead` | float | {0, 1} | Closest point ahead of ego |
| 15-34 | `sample_x[0:10]` | float | m | 10 sampled X positions (ego frame) |
| 35-44 | `sample_y[0:10]` | float | m | 10 sampled Y positions (ego frame) |
| 45-54 | `sample_valid[0:10]` | float | {0, 1} | Sample point validity |

**LANE feature vector size: 55**

---

### 3. TRAFFIC_LIGHT Node

Represents traffic signal state.

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `x` | float | m | Position X in ego frame |
| 1 | `y` | float | m | Position Y in ego frame |
| 2 | `dist_m` | float | [0, 100] m | Distance to ego |
| 3 | `state_unknown` | float | {0, 1} | One-hot: unknown state |
| 4 | `state_red` | float | {0, 1} | One-hot: red |
| 5 | `state_yellow` | float | {0, 1} | One-hot: yellow |
| 6 | `state_green` | float | {0, 1} | One-hot: green |
| 7 | `state_flashing` | float | {0, 1} | One-hot: flashing |
| 8 | `is_ahead` | float | {0, 1} | TL is ahead of ego |
| 9 | `control_confidence` | float | [0, 1] | Likelihood controls ego lane |
| 10 | `time_to_change_norm` | float | [0, 1] | Normalized time to state change |
| 11 | `has_future_info` | float | {0, 1} | Future prediction available |

**TRAFFIC_LIGHT feature vector size: 12**

---

## Edge Types

### 1. AGENT_TO_AGENT Edge

Directed edge from one agent to another (including ego→other and other→other).

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `rel_x` | float | m | Target X relative to source (source frame) |
| 1 | `rel_y` | float | m | Target Y relative to source (source frame) |
| 2 | `rel_vx` | float | m/s | Relative velocity X |
| 3 | `rel_vy` | float | m/s | Relative velocity Y |
| 4 | `distance_m` | float | [0, 100] m | Euclidean distance |
| 5 | `closing_speed` | float | m/s | Positive = approaching |
| 6 | `ttc` | float | [0, 10] s | Time-to-collision (clamped) |
| 7 | `has_ttc` | float | {0, 1} | TTC is valid (approaching) |
| 8 | `is_ahead` | float | {0, 1} | Target ahead of source |
| 9 | `is_behind` | float | {0, 1} | Target behind source |
| 10 | `is_left` | float | {0, 1} | Target left of source |
| 11 | `is_right` | float | {0, 1} | Target right of source |
| 12 | `is_approaching` | float | {0, 1} | Closing speed > 0.5 m/s |
| 13 | `is_moving_away` | float | {0, 1} | Closing speed < -0.5 m/s |
| 14 | `is_leading` | float | {0, 1} | Same lane proxy, ahead |
| 15 | `is_following` | float | {0, 1} | Same lane proxy, behind |
| 16 | `dist_bucket_very_close` | float | {0, 1} | < 5m |
| 17 | `dist_bucket_close` | float | {0, 1} | 5-15m |
| 18 | `dist_bucket_medium` | float | {0, 1} | 15-30m |
| 19 | `dist_bucket_far` | float | {0, 1} | > 30m |

**AGENT_TO_AGENT feature vector size: 20**

---

### 2. AGENT_TO_LANE Edge

Connects an agent to nearby lane(s) it may be occupying or approaching.

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `distance_m` | float | [0, 20] m | Distance to lane centerline |
| 1 | `lateral_offset_m` | float | [-10, 10] m | Signed lateral offset (left +) |
| 2 | `heading_alignment` | float | [-1, 1] | cos(agent_heading - lane_heading) |
| 3 | `s_progress` | float | [0, 1] | Normalized arc-length position |
| 4 | `is_on_lane` | float | {0, 1} | Agent within lane bounds |
| 5 | `is_approaching_lane` | float | {0, 1} | Moving toward lane |
| 6 | `is_crossing_lane` | float | {0, 1} | Heading perpendicular to lane |
| 7 | `speed_compliance` | float | [0, 2] | agent_speed / lane_speed_limit |
| 8 | `has_speed_limit` | float | {0, 1} | Lane has speed limit info |

**AGENT_TO_LANE feature vector size: 9**

---

### 3. LANE_TO_LANE Edge (Optional)

Represents lane adjacency/connectivity. Only created if lanes are spatially close.

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `min_distance_m` | float | [0, 10] m | Minimum distance between lanes |
| 1 | `is_adjacent_left` | float | {0, 1} | Target lane is left neighbor |
| 2 | `is_adjacent_right` | float | {0, 1} | Target lane is right neighbor |
| 3 | `is_successor` | float | {0, 1} | Target is downstream continuation |
| 4 | `is_predecessor` | float | {0, 1} | Target is upstream continuation |
| 5 | `heading_diff_cos` | float | [-1, 1] | Heading similarity |

**LANE_TO_LANE feature vector size: 6**

---

### 4. LANE_TO_TL Edge

Connects lanes to traffic lights that may control them.

| Index | Feature | Type | Range/Values | Description |
|-------|---------|------|--------------|-------------|
| 0 | `distance_m` | float | [0, 50] m | TL distance to lane endpoint |
| 1 | `control_confidence` | float | [0, 1] | Heuristic: TL controls this lane |
| 2 | `is_ahead_of_lane_end` | float | {0, 1} | TL is ahead of lane terminus |

**LANE_TO_TL feature vector size: 3**

---

## Graph Structure Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           NODE COUNTS (Typical)                            │
├────────────────────────────────────────────────────────────────────────────┤
│  AGENT:         1 ego + up to N_agents (max ~32)                          │
│  LANE:          up to N_lanes (max ~20 after filtering)                   │
│  TRAFFIC_LIGHT: up to N_tl (max ~10 after clustering)                     │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                           EDGE CONNECTIVITY                                │
├────────────────────────────────────────────────────────────────────────────┤
│  AGENT_TO_AGENT:  ego → all others (required)                             │
│                   other → other within distance threshold (optional)       │
│                                                                            │
│  AGENT_TO_LANE:   each agent → K nearest lanes (K=3 typical)              │
│                                                                            │
│  LANE_TO_LANE:    adjacent lanes within distance threshold                │
│                                                                            │
│  LANE_TO_TL:      lanes → nearby TLs within radius + ahead                │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Construction Pipeline Pseudocode

```
function build_unified_graph(tfrecord_example):

    # ========================================
    # PHASE 1: Parse TFExample
    # ========================================

    ego_state = get_ego_state(example)  # from tfexample_io
    agents_raw = parse_agent_states(example)  # state/current/*, state/past/*
    roadgraph = get_roadgraph_samples(example)  # from tfexample_io
    traffic_lights = get_traffic_lights_current(example)
    tl_future = get_traffic_lights_time_major(example, "future")

    # ========================================
    # PHASE 2: Build AGENT Nodes
    # ========================================

    agent_nodes = []
    ego_xy = (ego_state.x, ego_state.y)
    ego_yaw = ego_state.yaw

    for agent in agents_raw:
        if not agent.current_valid:
            continue

        # Transform to ego frame
        x_ego, y_ego = world_to_ego(agent.x, agent.y, ego_xy, ego_yaw)
        vx_ego, vy_ego = rotate_velocity(agent.vx, agent.vy, ego_yaw)
        yaw_rel = normalize_angle(agent.yaw - ego_yaw)

        # Transform past trajectory
        past_x_ego, past_y_ego = [], []
        for t in range(10):
            if agent.past_valid[t]:
                px, py = world_to_ego(agent.past_x[t], agent.past_y[t], ego_xy, ego_yaw)
            else:
                px, py = 0.0, 0.0  # or last valid position
            past_x_ego.append(px)
            past_y_ego.append(py)

        # Compute dynamics flags
        is_braking, is_accel = analyze_dynamics(agent.past_vx, agent.past_vy, agent.past_valid)

        # Build feature vector [45]
        features = build_agent_features(
            is_ego=(agent.id == ego_state.idx),
            agent_type=agent.type,
            x=x_ego, y=y_ego,
            vx=vx_ego, vy=vy_ego,
            yaw=yaw_rel,
            speed=hypot(vx_ego, vy_ego),
            length=agent.length, width=agent.width,
            past_x=past_x_ego, past_y=past_y_ego,
            past_valid=agent.past_valid,
            is_braking=is_braking, is_accel=is_accel
        )

        agent_nodes.append(AgentNode(id=agent.id, features=features))

    # ========================================
    # PHASE 3: Build LANE Nodes
    # ========================================

    # Filter to drivable surfaces only
    drivable_types = {1, 2, 3, 20}  # FREEWAY, SURFACE_STREET, BIKE_LANE, DRIVEWAY

    lane_nodes = []
    polylines = group_roadgraph_by_id(roadgraph)

    for poly_id, points in polylines.items():
        if get_type_mode(points) not in drivable_types:
            continue

        # Transform to ego frame
        points_ego = world_to_ego_batch(points.xy, ego_xy, ego_yaw)

        # Compute lane features
        centroid = mean(points_ego)
        heading = compute_polyline_heading(points_ego)
        heading_rel = heading - 0  # already in ego frame

        closest_idx, closest_dist = find_closest_point(points_ego, origin=(0,0))
        closest_pt = points_ego[closest_idx]

        # Sample 10 evenly spaced points
        samples = sample_polyline(points_ego, n=10)

        # Build feature vector [55]
        features = build_lane_features(
            lane_type=get_type_mode(points),
            centroid=centroid,
            heading_cos=cos(heading_rel),
            heading_sin=sin(heading_rel),
            speed_limit=get_speed_limit(points),
            length=compute_polyline_length(points_ego),
            closest_dist=closest_dist,
            closest_pt=closest_pt,
            samples=samples
        )

        lane_nodes.append(LaneNode(id=poly_id, features=features, points_ego=points_ego))

    # Sort by distance, keep top K
    lane_nodes.sort(key=lambda n: n.features.closest_dist)
    lane_nodes = lane_nodes[:MAX_LANES]

    # ========================================
    # PHASE 4: Build TRAFFIC_LIGHT Nodes
    # ========================================

    tl_nodes = []

    for tl_idx in range(len(traffic_lights)):
        if not traffic_lights.valid[tl_idx]:
            continue

        # Transform to ego frame
        x_ego, y_ego = world_to_ego(
            traffic_lights.x[tl_idx],
            traffic_lights.y[tl_idx],
            ego_xy, ego_yaw
        )

        dist = hypot(x_ego, y_ego)
        if dist > TL_RADIUS_MAX:
            continue

        state = traffic_lights.state[tl_idx]

        # Compute control confidence
        is_ahead = x_ego > 0
        control_conf = compute_tl_control_confidence(x_ego, y_ego)

        # Future state features
        time_to_change = compute_time_to_change(tl_idx, state, tl_future)

        # Build feature vector [12]
        features = build_tl_features(
            x=x_ego, y=y_ego,
            dist=dist,
            state=state,
            is_ahead=is_ahead,
            control_conf=control_conf,
            time_to_change=time_to_change
        )

        tl_nodes.append(TLNode(id=tl_idx, features=features))

    # Cluster nearby TLs, keep representatives
    tl_nodes = cluster_traffic_lights(tl_nodes, cluster_dist=1.0)
    tl_nodes = tl_nodes[:MAX_TL]

    # ========================================
    # PHASE 5: Build AGENT_TO_AGENT Edges
    # ========================================

    a2a_edges = []
    ego_node = find_ego_node(agent_nodes)

    for src in agent_nodes:
        for dst in agent_nodes:
            if src.id == dst.id:
                continue

            # Only ego→all required; optionally other→other if close
            if not src.is_ego and distance(src, dst) > AGENT_INTERACTION_RADIUS:
                continue

            # Compute relative features (in src agent's frame)
            rel_x = dst.x - src.x  # already in ego frame, but relative
            rel_y = dst.y - src.y

            # Rotate to source agent's local frame
            src_yaw = src.features.yaw  # relative to ego
            rel_x_local, rel_y_local = rotate_to_frame(rel_x, rel_y, -src_yaw)

            rel_vx = dst.vx - src.vx
            rel_vy = dst.vy - src.vy

            dist = hypot(rel_x, rel_y)
            closing_speed = compute_closing_speed(rel_x, rel_y, rel_vx, rel_vy)

            ttc = dist / closing_speed if closing_speed > 0 else INF
            ttc = min(ttc, 10.0)  # clamp

            # Semantic relations
            is_ahead = rel_x_local > 0
            is_behind = rel_x_local < 0
            is_left = rel_y_local > 0
            is_right = rel_y_local < 0

            is_approaching = closing_speed > 0.5
            is_moving_away = closing_speed < -0.5

            # Lane proxy heuristic (same as semantic graph)
            is_leading = is_ahead and abs(rel_y_local) < LANE_WIDTH
            is_following = is_behind and abs(rel_y_local) < LANE_WIDTH

            # Distance buckets
            dist_buckets = compute_distance_buckets(dist)

            # Build feature vector [20]
            features = build_a2a_features(
                rel_x=rel_x_local, rel_y=rel_y_local,
                rel_vx=rel_vx, rel_vy=rel_vy,
                distance=dist,
                closing_speed=closing_speed,
                ttc=ttc, has_ttc=is_approaching,
                is_ahead=is_ahead, is_behind=is_behind,
                is_left=is_left, is_right=is_right,
                is_approaching=is_approaching,
                is_moving_away=is_moving_away,
                is_leading=is_leading, is_following=is_following,
                dist_buckets=dist_buckets
            )

            a2a_edges.append(Edge(src=src.id, dst=dst.id, type="AGENT_TO_AGENT", features=features))

    # ========================================
    # PHASE 6: Build AGENT_TO_LANE Edges
    # ========================================

    a2l_edges = []

    for agent in agent_nodes:
        agent_xy = (agent.features.x, agent.features.y)
        agent_yaw = agent.features.yaw
        agent_speed = agent.features.speed

        # Find K nearest lanes
        lanes_by_dist = sorted(lane_nodes, key=lambda L: distance_to_polyline(agent_xy, L.points_ego))
        nearest_lanes = lanes_by_dist[:K_NEAREST_LANES]

        for lane in nearest_lanes:
            # Compute closest point on lane polyline
            cp = closest_point_on_polyline(lane.points_ego, agent_xy)

            distance_m = cp.dist
            lateral_offset = cp.signed_lateral
            heading_align = cos(agent_yaw - lane.heading)
            s_progress = cp.s / lane.length  # normalized

            is_on_lane = distance_m < LANE_WIDTH / 2

            # Check if approaching (velocity toward lane)
            vel_toward_lane = ... # dot product of velocity and lane normal
            is_approaching_lane = vel_toward_lane > 0.5 and distance_m > LANE_WIDTH / 2

            # Check if crossing (perpendicular heading)
            is_crossing = abs(heading_align) < 0.5  # cos(60°) ≈ 0.5

            # Speed compliance
            if lane.speed_limit > 0:
                speed_compliance = agent_speed / lane.speed_limit
                has_speed_limit = 1.0
            else:
                speed_compliance = 0.0
                has_speed_limit = 0.0

            # Build feature vector [9]
            features = build_a2l_features(
                distance_m=distance_m,
                lateral_offset=lateral_offset,
                heading_align=heading_align,
                s_progress=s_progress,
                is_on_lane=is_on_lane,
                is_approaching=is_approaching_lane,
                is_crossing=is_crossing,
                speed_compliance=speed_compliance,
                has_speed_limit=has_speed_limit
            )

            a2l_edges.append(Edge(src=agent.id, dst=lane.id, type="AGENT_TO_LANE", features=features))

    # ========================================
    # PHASE 7: Build LANE_TO_LANE Edges (Optional)
    # ========================================

    l2l_edges = []

    for lane_a in lane_nodes:
        for lane_b in lane_nodes:
            if lane_a.id == lane_b.id:
                continue

            min_dist = min_distance_between_polylines(lane_a.points_ego, lane_b.points_ego)

            if min_dist > LANE_ADJACENCY_THRESHOLD:
                continue

            # Determine relationship type
            is_adj_left, is_adj_right = check_lateral_adjacency(lane_a, lane_b)
            is_successor = check_longitudinal_connection(lane_a, lane_b, direction="forward")
            is_predecessor = check_longitudinal_connection(lane_a, lane_b, direction="backward")

            heading_diff = cos(lane_a.heading - lane_b.heading)

            # Build feature vector [6]
            features = build_l2l_features(
                min_distance=min_dist,
                is_adj_left=is_adj_left,
                is_adj_right=is_adj_right,
                is_successor=is_successor,
                is_predecessor=is_predecessor,
                heading_diff_cos=heading_diff
            )

            l2l_edges.append(Edge(src=lane_a.id, dst=lane_b.id, type="LANE_TO_LANE", features=features))

    # ========================================
    # PHASE 8: Build LANE_TO_TL Edges
    # ========================================

    l2tl_edges = []

    for lane in lane_nodes:
        lane_end = lane.points_ego[-1]  # downstream end

        for tl in tl_nodes:
            tl_xy = (tl.features.x, tl.features.y)

            dist_to_end = distance(lane_end, tl_xy)

            if dist_to_end > TL_LANE_RADIUS:
                continue

            # Check if TL is ahead of lane end
            is_ahead = tl_xy[0] > lane_end[0]  # in ego frame

            # Heuristic control confidence
            control_conf = compute_lane_tl_control_confidence(lane, tl)

            # Build feature vector [3]
            features = build_l2tl_features(
                distance=dist_to_end,
                control_conf=control_conf,
                is_ahead=is_ahead
            )

            l2tl_edges.append(Edge(src=lane.id, dst=tl.id, type="LANE_TO_TL", features=features))

    # ========================================
    # PHASE 9: Assemble Graph
    # ========================================

    graph = HeteroGraph(
        nodes={
            "AGENT": stack_features([n.features for n in agent_nodes]),      # [N_a, 45]
            "LANE": stack_features([n.features for n in lane_nodes]),        # [N_l, 55]
            "TRAFFIC_LIGHT": stack_features([n.features for n in tl_nodes]), # [N_t, 12]
        },
        edges={
            "AGENT_TO_AGENT": (src_ids, dst_ids, stack_features(a2a_edges)),  # [E_aa, 20]
            "AGENT_TO_LANE": (src_ids, dst_ids, stack_features(a2l_edges)),   # [E_al, 9]
            "LANE_TO_LANE": (src_ids, dst_ids, stack_features(l2l_edges)),    # [E_ll, 6]
            "LANE_TO_TL": (src_ids, dst_ids, stack_features(l2tl_edges)),     # [E_lt, 3]
        },
        node_counts={
            "AGENT": len(agent_nodes),
            "LANE": len(lane_nodes),
            "TRAFFIC_LIGHT": len(tl_nodes),
        }
    )

    return graph
```

---

## Configuration Parameters

```python
@dataclass
class UnifiedGraphConfig:
    # Node limits
    max_agents: int = 32
    max_lanes: int = 20
    max_traffic_lights: int = 10

    # History
    past_steps: int = 10

    # Spatial filtering
    agent_radius_m: float = 50.0
    lane_fwd_m: float = 50.0
    lane_back_m: float = 10.0
    lane_side_m: float = 20.0
    tl_radius_m: float = 80.0

    # Edge creation thresholds
    agent_interaction_radius_m: float = 30.0
    k_nearest_lanes: int = 3
    lane_adjacency_threshold_m: float = 5.0
    tl_lane_radius_m: float = 30.0

    # Semantic thresholds
    lane_width_m: float = 3.5
    closing_speed_threshold_mps: float = 0.5

    # Distance buckets
    dist_very_close_m: float = 5.0
    dist_close_m: float = 15.0
    dist_medium_m: float = 30.0

    # TL clustering
    tl_cluster_dist_m: float = 1.0
```

---

## Tensor Shapes Summary

| Component | Shape | Description |
|-----------|-------|-------------|
| `nodes["AGENT"]` | `[N_a, 45]` | Agent node features |
| `nodes["LANE"]` | `[N_l, 55]` | Lane node features |
| `nodes["TRAFFIC_LIGHT"]` | `[N_t, 12]` | TL node features |
| `edges["AGENT_TO_AGENT"]` | `[E_aa, 20]` | Agent interaction edges |
| `edges["AGENT_TO_LANE"]` | `[E_al, 9]` | Agent-lane occupancy edges |
| `edges["LANE_TO_LANE"]` | `[E_ll, 6]` | Lane adjacency edges |
| `edges["LANE_TO_TL"]` | `[E_lt, 3]` | Lane-TL control edges |

---

## Notes & Open Questions

1. **Lane topology**: TFExample lacks explicit lane connectivity. LANE_TO_LANE edges are inferred from spatial proximity + heading alignment.

2. **TL-Lane association**: No ground truth for which TL controls which lane. Using spatial heuristics (ahead + centered).

3. **Padding strategy**: For batching, nodes/edges need padding to fixed max counts. Suggest:
   - Mask tensors for valid nodes/edges
   - Zero-padding for unused slots

4. **Coordinate normalization**: Consider normalizing positions by `max_radius` for network stability.

5. **Temporal extension**: Current design is single-timestep. For sequence models, stack graphs across time or add temporal edges.
