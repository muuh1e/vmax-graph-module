#!/usr/bin/env python3
"""
hetero_graph.py

Build a PyTorch Geometric HeteroData object combining:
- Semantic graph (agents) from TFExample
- Spatial graph (lanes) from roadgraph_samples

Output structure:
- data['agent'].x: [N_agents, ~45] agent features
- data['lane'].x: [N_lanes, ~40] lane features
- data['agent', 'to', 'agent'].edge_index: [2, E_a2a]
- data['agent', 'to', 'agent'].edge_attr: [E_a2a, F_a2a]
- data['agent', 'to', 'lane'].edge_index: [2, E_a2l]
- data['agent', 'to', 'lane'].edge_attr: [E_a2l, F_a2l]

All coordinates are transformed to ego-centric frame (ego at origin, heading=0).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Reuse existing utilities from spatial_graph at repo root
from spatial_graph.utils.tfexample_io import (
    read_example_from_tfrecord,
    get_ego_state,
    get_roadgraph_samples,
    get_traffic_lights_current,
    EgoState,
)
from spatial_graph.utils.geometry import ego_frame, world_to_ego_points_xy
from spatial_graph.modules.polyline_grouper import (
    build_map_features_from_roadgraph,
    MapFeature,
    PolylineGrouperConfig,
)

if TYPE_CHECKING:
    from ..configs import GraphConfig


# -----------------------------
# Agent type encoding
# -----------------------------
TYPE_TO_IDX = {
    "unknown": 0,
    "vehicle": 1,
    "pedestrian": 2,
    "cyclist": 3,
    "other": 4,
}

TYPE_MAP = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}


def type_label(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    return TYPE_MAP.get(int(code), "unknown")


# -----------------------------
# TFExample reading helpers (adapted from build_semantic_graph_v2.py)
# -----------------------------
def _float_feature(features, key: str) -> Optional[np.ndarray]:
    feat = features.get(key, None)
    if feat is None:
        return None
    vals = feat.float_list.value
    if len(vals) == 0:
        return None
    return np.asarray(vals, dtype=np.float32)


def _int_feature(features, key: str) -> Optional[np.ndarray]:
    feat = features.get(key, None)
    if feat is None:
        return None
    vals = feat.int64_list.value
    if len(vals) == 0:
        return None
    return np.asarray(vals, dtype=np.int64)


def _reshape_or_none(arr: Optional[np.ndarray], shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    expected = int(np.prod(shape))
    if arr.size != expected:
        return None
    return arr.reshape(shape)


def read_agent_types(features, N: int) -> Optional[np.ndarray]:
    """Read agent types, handling both float and int storage."""
    if "state/type" not in features:
        return None
    feat = features["state/type"]
    if len(feat.float_list.value) > 0:
        arr = np.array(feat.float_list.value, dtype=np.float32)
        if arr.size != N:
            return None
        return np.rint(arr).astype(np.int64)
    if len(feat.int64_list.value) > 0:
        arr = np.array(feat.int64_list.value, dtype=np.int64)
        if arr.size != N:
            return None
        return arr
    return None


# -----------------------------
# Coordinate transformation
# -----------------------------
def transform_to_ego_frame(
    x: np.ndarray,
    y: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform world coordinates to ego-centric frame."""
    dx = x - ego_x
    dy = y - ego_y
    x_local, y_local = ego_frame(dx, dy, ego_yaw)
    return x_local, y_local


def transform_velocity_to_ego_frame(
    vx: np.ndarray,
    vy: np.ndarray,
    ego_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform velocities to ego-centric frame (rotation only)."""
    vx_local, vy_local = ego_frame(vx, vy, ego_yaw)
    return vx_local, vy_local


def transform_yaw_to_ego_frame(yaw: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Transform yaw angles to ego-centric frame."""
    return yaw - ego_yaw


# -----------------------------
# Agent feature extraction
# -----------------------------
def extract_agent_features(
    features,
    ego: EgoState,
    K_past: int = 10,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract agent features from TFExample.

    Returns:
        agent_features: [N_valid, F] numpy array
        valid_indices: indices of valid agents in original array
        ego_local_idx: index of ego in valid_indices (for edge construction)
    """
    # Infer dimensions
    cur_x = _float_feature(features, "state/current/x")
    if cur_x is None:
        raise KeyError("Missing state/current/x")
    N = cur_x.shape[0]

    past_x_flat = _float_feature(features, "state/past/x")
    if past_x_flat is None:
        raise KeyError("Missing state/past/x")
    Kpast_total = past_x_flat.size // N
    K = min(K_past, Kpast_total)

    # Read current state
    cur_y = _float_feature(features, "state/current/y")
    cur_vx = _float_feature(features, "state/current/velocity_x")
    cur_vy = _float_feature(features, "state/current/velocity_y")
    cur_yaw = _float_feature(features, "state/current/bbox_yaw")
    if cur_yaw is None:
        cur_yaw = _float_feature(features, "state/current/vel_yaw")
    if cur_yaw is None:
        cur_yaw = np.zeros(N, dtype=np.float32)
    cur_valid = _int_feature(features, "state/current/valid")
    if cur_valid is None:
        cur_valid = np.ones(N, dtype=np.int64)
    cur_valid = cur_valid.astype(bool)

    # Read past state
    past_x = _reshape_or_none(_float_feature(features, "state/past/x"), (N, Kpast_total))
    past_y = _reshape_or_none(_float_feature(features, "state/past/y"), (N, Kpast_total))
    past_vx = _reshape_or_none(_float_feature(features, "state/past/velocity_x"), (N, Kpast_total))
    past_vy = _reshape_or_none(_float_feature(features, "state/past/velocity_y"), (N, Kpast_total))
    past_valid = _reshape_or_none(_int_feature(features, "state/past/valid"), (N, Kpast_total))

    # Slice to last K steps
    if past_x is not None:
        past_x = past_x[:, -K:]
    if past_y is not None:
        past_y = past_y[:, -K:]
    if past_vx is not None:
        past_vx = past_vx[:, -K:]
    if past_vy is not None:
        past_vy = past_vy[:, -K:]
    if past_valid is not None:
        past_valid = past_valid[:, -K:].astype(bool)

    # Agent types
    types = read_agent_types(features, N)
    if types is None:
        types = np.zeros(N, dtype=np.int64)

    # Bounding box dimensions (if available)
    length = _float_feature(features, "state/current/length")
    width = _float_feature(features, "state/current/width")
    if length is None:
        length = np.ones(N, dtype=np.float32) * 4.5  # default vehicle length
    if width is None:
        width = np.ones(N, dtype=np.float32) * 2.0  # default vehicle width

    # Filter to valid agents
    valid_indices = np.where(cur_valid)[0]
    N_valid = len(valid_indices)

    # Find ego in valid indices
    is_sdc = _int_feature(features, "state/is_sdc")
    ego_orig_idx = int(np.argmax(is_sdc))
    ego_local_idx = int(np.where(valid_indices == ego_orig_idx)[0][0])

    # Transform coordinates to ego frame
    ego_x, ego_y, ego_yaw_val = ego.x, ego.y, ego.yaw

    # Current position in ego frame
    x_ego, y_ego = transform_to_ego_frame(
        cur_x[valid_indices], cur_y[valid_indices], ego_x, ego_y, ego_yaw_val
    )

    # Current velocity in ego frame
    vx_ego, vy_ego = transform_velocity_to_ego_frame(
        cur_vx[valid_indices], cur_vy[valid_indices], ego_yaw_val
    )

    # Yaw in ego frame
    yaw_ego = transform_yaw_to_ego_frame(cur_yaw[valid_indices], ego_yaw_val)

    # Speed magnitude
    speed = np.sqrt(cur_vx[valid_indices]**2 + cur_vy[valid_indices]**2)

    # Build feature vector for each agent
    # Features: [x, y, vx, vy, yaw, speed, cos(yaw), sin(yaw), length, width,
    #            type_onehot(5), is_ego(1), past_x(K), past_y(K), past_vx(K), past_vy(K), past_valid(K)]
    # Total: 10 + 5 + 1 + 5*K = 16 + 5*K

    n_base_features = 16
    n_past_features = 5 * K  # x, y, vx, vy, valid
    n_features = n_base_features + n_past_features

    agent_features = np.zeros((N_valid, n_features), dtype=np.float32)

    # Base features
    agent_features[:, 0] = x_ego
    agent_features[:, 1] = y_ego
    agent_features[:, 2] = vx_ego
    agent_features[:, 3] = vy_ego
    agent_features[:, 4] = yaw_ego
    agent_features[:, 5] = speed
    agent_features[:, 6] = np.cos(yaw_ego)
    agent_features[:, 7] = np.sin(yaw_ego)
    agent_features[:, 8] = length[valid_indices]
    agent_features[:, 9] = width[valid_indices]

    # Type one-hot encoding
    for i, idx in enumerate(valid_indices):
        t = int(types[idx])
        if 0 <= t < 5:
            agent_features[i, 10 + t] = 1.0

    # Is ego flag
    agent_features[ego_local_idx, 15] = 1.0

    # Past trajectory features (in ego frame)
    if past_x is not None and past_y is not None:
        for i, idx in enumerate(valid_indices):
            px, py = transform_to_ego_frame(
                past_x[idx], past_y[idx], ego_x, ego_y, ego_yaw_val
            )
            agent_features[i, n_base_features:n_base_features+K] = px
            agent_features[i, n_base_features+K:n_base_features+2*K] = py

    # Past velocity features (in ego frame)
    if past_vx is not None and past_vy is not None:
        for i, idx in enumerate(valid_indices):
            pvx, pvy = transform_velocity_to_ego_frame(
                past_vx[idx], past_vy[idx], ego_yaw_val
            )
            agent_features[i, n_base_features+2*K:n_base_features+3*K] = pvx
            agent_features[i, n_base_features+3*K:n_base_features+4*K] = pvy

    # Past validity mask
    if past_valid is not None:
        for i, idx in enumerate(valid_indices):
            agent_features[i, n_base_features+4*K:n_base_features+5*K] = past_valid[idx].astype(np.float32)

    return agent_features, valid_indices, ego_local_idx


# -----------------------------
# Future trajectory extraction
# -----------------------------
def extract_future_trajectory(
    features,
    ego: EgoState,
    valid_indices: np.ndarray,
    T_future: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract future trajectory labels for motion prediction.

    Args:
        features: TFExample features dict
        ego: EgoState with x, y, yaw for coordinate transform
        valid_indices: indices of valid agents (from extract_agent_features)
        T_future: number of future timesteps (default 80 = 8 seconds at 10Hz)

    Returns:
        future_xy: [N_valid, T_future, 2] numpy array in ego frame
        future_valid: [N_valid, T_future] numpy array (1.0 if valid, 0.0 if not)
    """
    # Get number of agents
    cur_x = _float_feature(features, "state/current/x")
    if cur_x is None:
        raise KeyError("Missing state/current/x")
    N = cur_x.shape[0]

    # Read future state
    fut_x_flat = _float_feature(features, "state/future/x")
    fut_y_flat = _float_feature(features, "state/future/y")
    fut_valid_flat = _int_feature(features, "state/future/valid")

    N_valid = len(valid_indices)

    # Handle missing future data
    if fut_x_flat is None or fut_y_flat is None:
        # Return zeros with all invalid
        future_xy = np.zeros((N_valid, T_future, 2), dtype=np.float32)
        future_valid = np.zeros((N_valid, T_future), dtype=np.float32)
        return future_xy, future_valid

    # Infer actual future timesteps from data
    T_actual = fut_x_flat.size // N
    T = min(T_future, T_actual)

    # Reshape to [N, T_actual]
    fut_x = _reshape_or_none(fut_x_flat, (N, T_actual))
    fut_y = _reshape_or_none(fut_y_flat, (N, T_actual))
    fut_valid = _reshape_or_none(fut_valid_flat, (N, T_actual))

    if fut_x is None or fut_y is None:
        future_xy = np.zeros((N_valid, T_future, 2), dtype=np.float32)
        future_valid = np.zeros((N_valid, T_future), dtype=np.float32)
        return future_xy, future_valid

    # Slice to T_future timesteps (take first T)
    fut_x = fut_x[:, :T]
    fut_y = fut_y[:, :T]
    if fut_valid is not None:
        fut_valid = fut_valid[:, :T].astype(np.float32)
    else:
        fut_valid = np.ones((N, T), dtype=np.float32)

    # Transform to ego frame
    ego_x, ego_y, ego_yaw = ego.x, ego.y, ego.yaw

    # Output arrays
    future_xy = np.zeros((N_valid, T_future, 2), dtype=np.float32)
    future_valid_out = np.zeros((N_valid, T_future), dtype=np.float32)

    for i, idx in enumerate(valid_indices):
        # Transform future positions to ego frame
        fx, fy = transform_to_ego_frame(
            fut_x[idx], fut_y[idx], ego_x, ego_y, ego_yaw
        )
        future_xy[i, :T, 0] = fx
        future_xy[i, :T, 1] = fy
        future_valid_out[i, :T] = fut_valid[idx]

    return future_xy, future_valid_out


# -----------------------------
# Lane feature extraction
# -----------------------------
def extract_lane_features(
    map_features: List[MapFeature],
    ego: EgoState,
    max_sample_points: int = 10,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Extract lane features from MapFeature list.

    Returns:
        lane_features: [N_lanes, F] numpy array
        lane_polylines: list of [P, 2] arrays with polyline points (ego frame)
    """
    if not map_features:
        return np.zeros((0, 40), dtype=np.float32), []

    N_lanes = len(map_features)

    # Features per lane:
    # [centroid_x, centroid_y, closest_x, closest_y, closest_dist,
    #  bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
    #  speed_limit, heading_cos, num_points,
    #  type_category_onehot(5): UNKNOWN, DRIVABLE, BOUNDARY, MARKING, SPECIAL,
    #  sample_points_x(10), sample_points_y(10)]
    # Total: 12 + 5 + 20 = 37, pad to ~40

    n_type_cats = 5
    n_base = 12
    n_samples = max_sample_points
    n_features = n_base + n_type_cats + 2 * n_samples + 3  # pad to 40

    lane_features = np.zeros((N_lanes, n_features), dtype=np.float32)
    lane_polylines = []

    ego_x, ego_y, ego_yaw = ego.x, ego.y, ego.yaw

    type_cat_map = {"UNKNOWN": 0, "DRIVABLE": 1, "BOUNDARY": 2, "MARKING": 3, "SPECIAL": 4}

    for i, mf in enumerate(map_features):
        # Transform centroid to ego frame
        cx, cy = transform_to_ego_frame(
            np.array([mf.centroid_xy[0]]),
            np.array([mf.centroid_xy[1]]),
            ego_x, ego_y, ego_yaw
        )
        lane_features[i, 0] = cx[0]
        lane_features[i, 1] = cy[0]

        # Closest point already in ego frame from MapFeature
        lane_features[i, 2] = mf.closest_x_local_m
        lane_features[i, 3] = mf.closest_y_local_m
        lane_features[i, 4] = mf.closest_dist_m

        # Transform bbox to ego frame
        bmin_x, bmin_y = transform_to_ego_frame(
            np.array([mf.bbox_xy_min[0]]),
            np.array([mf.bbox_xy_min[1]]),
            ego_x, ego_y, ego_yaw
        )
        bmax_x, bmax_y = transform_to_ego_frame(
            np.array([mf.bbox_xy_max[0]]),
            np.array([mf.bbox_xy_max[1]]),
            ego_x, ego_y, ego_yaw
        )
        lane_features[i, 5] = bmin_x[0]
        lane_features[i, 6] = bmin_y[0]
        lane_features[i, 7] = bmax_x[0]
        lane_features[i, 8] = bmax_y[0]

        # Speed limit (normalized, assume max ~40 m/s)
        speed_limit = mf.speed_limit_mps if mf.speed_limit_mps is not None else 0.0
        lane_features[i, 9] = speed_limit / 40.0

        # Heading alignment
        heading_cos = mf.heading_alignment_cos if mf.heading_alignment_cos is not None else 0.0
        lane_features[i, 10] = heading_cos

        # Number of points (normalized)
        lane_features[i, 11] = min(mf.num_points / 100.0, 1.0)

        # Type category one-hot
        cat_idx = type_cat_map.get(mf.type_category, 0)
        lane_features[i, n_base + cat_idx] = 1.0

        # Sample points (transform to ego frame)
        sample_pts = mf.sample_points_xy[:n_samples]
        if sample_pts:
            pts_world = np.array(sample_pts, dtype=np.float32)
            pts_ego = world_to_ego_points_xy(pts_world, np.array([ego_x, ego_y]), ego_yaw)
            n_pts = pts_ego.shape[0]
            offset = n_base + n_type_cats
            lane_features[i, offset:offset+n_pts] = pts_ego[:, 0]
            lane_features[i, offset+n_samples:offset+n_samples+n_pts] = pts_ego[:, 1]
            lane_polylines.append(pts_ego)
        else:
            lane_polylines.append(np.zeros((0, 2), dtype=np.float32))

    return lane_features, lane_polylines


# -----------------------------
# L2L Configuration
# -----------------------------
L2L_MAX_DIST = 15.0          # Max distance to consider any L2L edge
L2L_SUCCESSOR_DIST = 5.0     # End of lane_i to start of lane_j
L2L_NEIGHBOR_DIST = 8.0      # Parallel lanes threshold
L2L_HEADING_THRESH = 0.7     # cos(heading_diff) for "same direction"


# -----------------------------
# Lane heading helper
# -----------------------------
def compute_lane_heading(polyline: np.ndarray) -> float:
    """Compute overall heading of a polyline (start to end)."""
    if len(polyline) < 2:
        return 0.0
    direction = polyline[-1] - polyline[0]
    return float(np.arctan2(direction[1], direction[0]))


def min_polyline_distance(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute minimum distance between two polylines."""
    if poly1.shape[0] == 0 or poly2.shape[0] == 0:
        return float('inf')
    # Brute force pairwise distance
    min_dist = float('inf')
    for p1 in poly1:
        for p2 in poly2:
            d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if d < min_dist:
                min_dist = d
    return min_dist


# -----------------------------
# L2L Edge construction
# -----------------------------
def build_l2l_edges(
    lane_features: np.ndarray,
    lane_polylines: List[np.ndarray],
    max_dist: float = L2L_MAX_DIST,
    successor_dist: float = L2L_SUCCESSOR_DIST,
    neighbor_dist: float = L2L_NEIGHBOR_DIST,
    heading_thresh: float = L2L_HEADING_THRESH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lane-to-lane edges based on spatial relationships.

    Edge features (7):
        [0]: min_distance_m
        [1]: is_successor (end_i near start_j, same direction)
        [2]: is_predecessor (start_i near end_j, same direction)
        [3]: is_left_neighbor (parallel, j left of i)
        [4]: is_right_neighbor (parallel, j right of i)
        [5]: heading_diff_cos
        [6]: same_direction

    Returns:
        edge_index: [2, E_l2l]
        edge_attr: [E_l2l, 7]
    """
    N_EDGE_FEATURES = 7
    N_lanes = len(lane_polylines)

    if N_lanes < 2:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    src_list = []
    dst_list = []
    attr_list = []

    for i in range(N_lanes):
        poly_i = lane_polylines[i]
        if poly_i.shape[0] < 2:
            continue

        heading_i = compute_lane_heading(poly_i)
        start_i = poly_i[0]
        end_i = poly_i[-1]
        mid_i = poly_i[len(poly_i) // 2]

        for j in range(N_lanes):
            if i == j:
                continue

            poly_j = lane_polylines[j]
            if poly_j.shape[0] < 2:
                continue

            # Compute minimum distance between polylines
            min_dist = min_polyline_distance(poly_i, poly_j)

            if min_dist > max_dist:
                continue

            # Compute headings
            heading_j = compute_lane_heading(poly_j)
            heading_diff = heading_j - heading_i
            heading_diff_cos = float(np.cos(heading_diff))
            same_direction = 1.0 if heading_diff_cos > heading_thresh else 0.0

            start_j = poly_j[0]
            end_j = poly_j[-1]
            mid_j = poly_j[len(poly_j) // 2]

            # Check successor: end_i near start_j, same direction
            dist_end_i_start_j = np.sqrt((end_i[0] - start_j[0])**2 + (end_i[1] - start_j[1])**2)
            is_successor = 1.0 if (dist_end_i_start_j < successor_dist and same_direction > 0.5) else 0.0

            # Check predecessor: start_i near end_j, same direction
            dist_start_i_end_j = np.sqrt((start_i[0] - end_j[0])**2 + (start_i[1] - end_j[1])**2)
            is_predecessor = 1.0 if (dist_start_i_end_j < successor_dist and same_direction > 0.5) else 0.0

            # Check left/right neighbor
            is_left_neighbor = 0.0
            is_right_neighbor = 0.0

            if same_direction > 0.5 and min_dist < neighbor_dist:
                # Compute lateral offset of lane_j's midpoint relative to lane_i's frame
                # Lane i direction vector
                lane_dir = np.array([np.cos(heading_i), np.sin(heading_i)])

                # Vector from mid_i to mid_j
                to_mid_j = mid_j - mid_i

                # Cross product to get signed lateral distance
                # positive = j is to the left of i
                lateral = lane_dir[0] * to_mid_j[1] - lane_dir[1] * to_mid_j[0]

                if lateral > 1.0:
                    is_left_neighbor = 1.0
                elif lateral < -1.0:
                    is_right_neighbor = 1.0

            # Only add edge if at least one relationship exists OR min_dist < 5.0 (proximal)
            has_relationship = (is_successor > 0.5 or is_predecessor > 0.5 or
                               is_left_neighbor > 0.5 or is_right_neighbor > 0.5 or
                               min_dist < 5.0)

            if not has_relationship:
                continue

            edge_attr = [
                min_dist,            # 0: min_distance_m
                is_successor,        # 1: is_successor
                is_predecessor,      # 2: is_predecessor
                is_left_neighbor,    # 3: is_left_neighbor
                is_right_neighbor,   # 4: is_right_neighbor
                heading_diff_cos,    # 5: heading_diff_cos
                same_direction,      # 6: same_direction
            ]

            src_list.append(i)
            dst_list.append(j)
            attr_list.append(edge_attr)

    if not src_list:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)

    return edge_index, edge_attr


# -----------------------------
# Traffic Light Node Extraction
# -----------------------------
def extract_tl_features(
    features,
    ego: EgoState,
    max_radius: float = 80.0,
    max_tls: int = 16,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract traffic light node features.

    Node features (12):
        [0-1]: x_ego, y_ego (position in ego frame)
        [2]: dist_m
        [3]: state_unknown (one-hot)
        [4]: state_red
        [5]: state_yellow
        [6]: state_green
        [7]: state_flashing
        [8]: is_ahead (x_ego > 0)
        [9]: is_relevant (ahead AND close AND laterally aligned)
        [10]: normalized_dist (dist / max_radius)
        [11]: heading_to_tl (atan2 normalized to [-1, 1])

    Returns:
        tl_features: [N_tl, 12]
        tl_ids: original TL indices for reference
    """
    N_FEATURES = 12

    try:
        tl_set = get_traffic_lights_current(features)
    except KeyError:
        # No traffic lights in this record
        return np.zeros((0, N_FEATURES), dtype=np.float32), []

    if tl_set is None or tl_set.valid is None:
        return np.zeros((0, N_FEATURES), dtype=np.float32), []

    # Filter by valid mask
    valid_mask = tl_set.valid.astype(bool)
    if not np.any(valid_mask):
        return np.zeros((0, N_FEATURES), dtype=np.float32), []

    valid_indices = np.where(valid_mask)[0]
    tl_x = tl_set.x[valid_mask]
    tl_y = tl_set.y[valid_mask]
    tl_state = tl_set.state[valid_mask] if tl_set.state is not None else np.zeros(len(valid_indices), dtype=np.int64)

    # Transform to ego frame
    x_ego, y_ego = transform_to_ego_frame(tl_x, tl_y, ego.x, ego.y, ego.yaw)

    # Compute distances
    dist = np.sqrt(x_ego**2 + y_ego**2)

    # Filter by max_radius
    radius_mask = dist <= max_radius
    if not np.any(radius_mask):
        return np.zeros((0, N_FEATURES), dtype=np.float32), []

    x_ego = x_ego[radius_mask]
    y_ego = y_ego[radius_mask]
    dist = dist[radius_mask]
    tl_state = tl_state[radius_mask]
    valid_indices = valid_indices[radius_mask]

    # Sort by distance, take top max_tls
    sort_idx = np.argsort(dist)[:max_tls]
    x_ego = x_ego[sort_idx]
    y_ego = y_ego[sort_idx]
    dist = dist[sort_idx]
    tl_state = tl_state[sort_idx]
    valid_indices = valid_indices[sort_idx]

    N_tl = len(x_ego)
    tl_features = np.zeros((N_tl, N_FEATURES), dtype=np.float32)

    for i in range(N_tl):
        # Position
        tl_features[i, 0] = x_ego[i]
        tl_features[i, 1] = y_ego[i]
        tl_features[i, 2] = dist[i]

        # State one-hot encoding
        # Waymo TL states: 0=unknown, 1=arrow_stop, 2=arrow_caution, 3=arrow_go,
        #                  4=stop, 5=caution, 6=go, 7=flashing_stop, 8=flashing_caution
        state = int(tl_state[i])
        state_unknown = 0.0
        state_red = 0.0
        state_yellow = 0.0
        state_green = 0.0
        state_flashing = 0.0

        if state == 0:
            state_unknown = 1.0
        elif state in [1, 4]:  # arrow_stop, stop
            state_red = 1.0
        elif state in [2, 5]:  # arrow_caution, caution
            state_yellow = 1.0
        elif state in [3, 6]:  # arrow_go, go
            state_green = 1.0
        elif state in [7, 8]:  # flashing_stop, flashing_caution
            state_flashing = 1.0
        else:
            state_unknown = 1.0

        tl_features[i, 3] = state_unknown
        tl_features[i, 4] = state_red
        tl_features[i, 5] = state_yellow
        tl_features[i, 6] = state_green
        tl_features[i, 7] = state_flashing

        # Is ahead
        is_ahead = 1.0 if x_ego[i] > 0 else 0.0
        tl_features[i, 8] = is_ahead

        # Is relevant (ahead AND close AND laterally aligned)
        is_relevant = 1.0 if (is_ahead > 0.5 and abs(y_ego[i]) < 10 and dist[i] < 50) else 0.0
        tl_features[i, 9] = is_relevant

        # Normalized distance
        tl_features[i, 10] = dist[i] / max_radius

        # Heading to TL (normalized to [-1, 1])
        heading_to_tl = np.arctan2(y_ego[i], x_ego[i])
        tl_features[i, 11] = heading_to_tl / np.pi

    return tl_features, valid_indices.tolist()


# -----------------------------
# A2TL Edge construction
# -----------------------------
def build_a2tl_edges(
    agent_features: np.ndarray,
    tl_features: np.ndarray,
    max_dist: float = 60.0,
    k_nearest: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build agent-to-traffic-light edges.

    Each agent connects to up to k nearest TLs within max_dist.

    Edge features (6):
        [0-1]: dx, dy (relative position)
        [2]: dist
        [3]: is_ahead (TL ahead of agent)
        [4]: is_relevant (ahead AND laterally close)
        [5]: angle_to_tl (normalized)

    Returns:
        edge_index: [2, E_a2tl]
        edge_attr: [E_a2tl, 6]
    """
    N_EDGE_FEATURES = 6
    N_agents = agent_features.shape[0]
    N_tls = tl_features.shape[0]

    if N_agents == 0 or N_tls == 0:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    # Agent positions
    agent_x = agent_features[:, 0]
    agent_y = agent_features[:, 1]

    # TL positions (already in ego frame)
    tl_x = tl_features[:, 0]
    tl_y = tl_features[:, 1]

    src_list = []
    dst_list = []
    attr_list = []

    for i in range(N_agents):
        ax, ay = agent_x[i], agent_y[i]

        # Compute distance to each TL
        tl_info = []
        for j in range(N_tls):
            dx = tl_x[j] - ax
            dy = tl_y[j] - ay
            dist = np.sqrt(dx*dx + dy*dy)

            if dist <= max_dist:
                tl_info.append((j, dx, dy, dist))

        # Sort by distance and take k nearest
        tl_info.sort(key=lambda x: x[3])

        for tl_idx, dx, dy, dist in tl_info[:k_nearest]:
            # Is ahead (TL ahead of agent)
            is_ahead = 1.0 if dx > 0 else 0.0

            # Is relevant (ahead AND laterally close)
            is_relevant = 1.0 if (is_ahead > 0.5 and abs(dy) < 6) else 0.0

            # Angle to TL (normalized to [-1, 1])
            angle_to_tl = np.arctan2(dy, dx) / np.pi

            edge_attr = [
                dx,            # 0
                dy,            # 1
                dist,          # 2
                is_ahead,      # 3
                is_relevant,   # 4
                angle_to_tl,   # 5
            ]

            src_list.append(i)
            dst_list.append(tl_idx)
            attr_list.append(edge_attr)

    if not src_list:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)

    return edge_index, edge_attr


# -----------------------------
# L2TL Edge construction
# -----------------------------
def build_l2tl_edges(
    lane_features: np.ndarray,
    lane_polylines: List[np.ndarray],
    tl_features: np.ndarray,
    max_dist: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lane-to-traffic-light edges.

    Connect lanes to TLs that are near the lane's end and ahead.

    Edge features (5):
        [0]: dist (from lane end to TL)
        [1]: is_ahead (TL ahead along lane direction)
        [2]: lateral_offset (signed)
        [3]: heading_alignment (cos of angle between lane heading and direction to TL)
        [4]: is_controlling (heuristic: ahead AND close AND aligned)

    Returns:
        edge_index: [2, E_l2tl]
        edge_attr: [E_l2tl, 5]
    """
    N_EDGE_FEATURES = 5
    N_lanes = len(lane_polylines)
    N_tls = tl_features.shape[0]

    if N_lanes == 0 or N_tls == 0:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    # TL positions (already in ego frame)
    tl_x = tl_features[:, 0]
    tl_y = tl_features[:, 1]

    src_list = []
    dst_list = []
    attr_list = []

    for i in range(N_lanes):
        poly = lane_polylines[i]
        if poly.shape[0] < 2:
            continue

        # Lane end point and heading
        lane_end = poly[-1]
        lane_heading = compute_lane_heading(poly)
        lane_dir = np.array([np.cos(lane_heading), np.sin(lane_heading)])

        for j in range(N_tls):
            tl_pos = np.array([tl_x[j], tl_y[j]])

            # Vector from lane end to TL
            to_tl = tl_pos - lane_end
            dist = np.sqrt(to_tl[0]**2 + to_tl[1]**2)

            if dist > max_dist:
                continue

            # Is ahead (TL is in front of lane end along lane heading)
            forward_dist = np.dot(to_tl, lane_dir)
            is_ahead = 1.0 if forward_dist > 0 else 0.0

            # Lateral offset (signed perpendicular distance)
            lateral_offset = lane_dir[0] * to_tl[1] - lane_dir[1] * to_tl[0]

            # Heading alignment (cos of angle between lane direction and vector to TL)
            if dist > 1e-6:
                to_tl_unit = to_tl / dist
                heading_alignment = float(np.dot(lane_dir, to_tl_unit))
            else:
                heading_alignment = 1.0

            # Is controlling (heuristic: ahead AND close AND aligned)
            is_controlling = 1.0 if (is_ahead > 0.5 and dist < 20 and abs(lateral_offset) < 5) else 0.0

            edge_attr = [
                dist,               # 0
                is_ahead,           # 1
                lateral_offset,     # 2
                heading_alignment,  # 3
                is_controlling,     # 4
            ]

            src_list.append(i)
            dst_list.append(j)
            attr_list.append(edge_attr)

    if not src_list:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)

    return edge_index, edge_attr


# -----------------------------
# Enhanced Edge Feature Helpers (Phase 1)
# -----------------------------

def compute_lateral_velocity(
    agent_features: np.ndarray,
    agent_idx: int,
) -> float:
    """
    Compute velocity perpendicular to agent's heading direction.
    
    Args:
        agent_features: [N, F] agent node features
        agent_idx: Index of the agent
        
    Returns:
        Lateral velocity in m/s (positive = moving left, negative = moving right)
    """
    vx = agent_features[agent_idx, 2]
    vy = agent_features[agent_idx, 3]
    yaw = agent_features[agent_idx, 4]
    
    # Perpendicular direction (left of heading)
    perp_x = -np.sin(yaw)
    perp_y = np.cos(yaw)
    
    return float(vx * perp_x + vy * perp_y)


def compute_yaw_rate(
    agent_features: np.ndarray,
    agent_idx: int,
    k_past: int = 10,
    dt: float = 0.1,
) -> float:
    """
    Compute angular velocity (yaw rate) from past trajectory.
    
    Uses past positions to estimate heading change over time.
    Agent features layout: past_x at [16:16+k], past_y at [16+k:16+2k]
    
    Args:
        agent_features: [N, F] agent node features
        agent_idx: Index of the agent
        k_past: Number of past timesteps in features
        dt: Time step between samples (seconds)
        
    Returns:
        Yaw rate in rad/s
    """
    # Extract past positions (last 2 valid points)
    past_x_start = 16
    past_y_start = 16 + k_past
    past_valid_start = 16 + 4 * k_past
    
    past_x = agent_features[agent_idx, past_x_start:past_x_start + k_past]
    past_y = agent_features[agent_idx, past_y_start:past_y_start + k_past]
    past_valid = agent_features[agent_idx, past_valid_start:past_valid_start + k_past]
    
    # Find last two valid timesteps
    valid_indices = np.where(past_valid > 0.5)[0]
    if len(valid_indices) < 2:
        return 0.0
    
    # Use positions to estimate heading at two timesteps
    idx1 = valid_indices[-2]
    idx2 = valid_indices[-1]
    
    # Current heading (from most recent movement)
    dx = past_x[idx2] - past_x[idx1]
    dy = past_y[idx2] - past_y[idx1]
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    
    # If we have more history, compute heading change
    if len(valid_indices) >= 3:
        idx0 = valid_indices[-3]
        dx_prev = past_x[idx1] - past_x[idx0]
        dy_prev = past_y[idx1] - past_y[idx0]
        
        if abs(dx_prev) > 1e-6 or abs(dy_prev) > 1e-6:
            yaw_curr = np.arctan2(dy, dx)
            yaw_prev = np.arctan2(dy_prev, dx_prev)
            
            # Normalize angle difference to [-pi, pi]
            dyaw = yaw_curr - yaw_prev
            while dyaw > np.pi:
                dyaw -= 2 * np.pi
            while dyaw < -np.pi:
                dyaw += 2 * np.pi
            
            time_diff = (idx2 - idx0) * dt
            if time_diff > 1e-6:
                return float(dyaw / time_diff)
    
    return 0.0


def compute_predicted_distance(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    pos_j: np.ndarray,
    vel_j: np.ndarray,
    dt: float,
) -> float:
    """
    Compute predicted distance between two agents at future time dt.
    
    Uses constant velocity linear prediction.
    
    Args:
        pos_i: [2] position of agent i
        vel_i: [2] velocity of agent i
        pos_j: [2] position of agent j
        vel_j: [2] velocity of agent j
        dt: Time in seconds for prediction
        
    Returns:
        Predicted distance in meters
    """
    pred_pos_i = pos_i + vel_i * dt
    pred_pos_j = pos_j + vel_j * dt
    return float(np.linalg.norm(pred_pos_j - pred_pos_i))


def compute_interaction_scores(
    dx: float,
    dy: float,
    closing_speed: float,
    lateral_vel_j_toward_i: float,
) -> tuple:
    """
    Compute soft interaction type scores based on geometry and dynamics.
    
    Args:
        dx: Relative x position (j - i) in ego frame
        dy: Relative y position (j - i) in ego frame
        closing_speed: Rate of approach (positive = approaching)
        lateral_vel_j_toward_i: Agent j's lateral velocity toward i's position
        
    Returns:
        (merging_score, yielding_score, cutting_in_score) all in [0, 1]
    """
    # Merging: different lanes, converging trajectories, similar longitudinal position
    lateral_separation = abs(dy)
    converging = closing_speed > 1.0 and lateral_separation > 2.0 and lateral_separation < 8.0
    similar_longitudinal = abs(dx) < 10.0
    merging_score = 1.0 if (converging and similar_longitudinal) else 0.0
    
    # Cutting-in: j has high lateral velocity toward i's lane
    # sigmoid approximation: 1 / (1 + exp(-x))
    x = lateral_vel_j_toward_i - 0.5  # threshold at 0.5 m/s
    cutting_in_score = 1.0 / (1.0 + np.exp(-x * 4))  # scaled sigmoid
    
    # Yielding: based on relative position and simple right-of-way rules
    # Vehicle on right has priority, or vehicle ahead has priority
    j_on_right = 1.0 if dy < 0 else 0.0
    j_ahead = 1.0 if dx > 0 else 0.0
    yielding_score = 0.5 * j_on_right + 0.3 * j_ahead
    
    return float(merging_score), float(yielding_score), float(cutting_in_score)


# -----------------------------
# Edge construction
# -----------------------------
def _compute_a2a_edge_attrs(
    agent_features: np.ndarray,
    edge_index: np.ndarray,
    lane_width: float = 2.0,
    use_enhanced: bool = True,
    collision_threshold: float = 3.0,
    prediction_horizon: float = 2.0,
    k_past: int = 10,
) -> np.ndarray:
    """
    Compute A2A edge attributes for a given edge index.
    
    This is a helper function for new A2A modes that compute edge indices
    differently but need the same edge attribute computation.
    
    Args:
        agent_features: [N, F] agent node features
        edge_index: [2, E] source and destination indices
        lane_width: Width for same-lane detection
        use_enhanced: If True, add Phase 1 enhanced features (20→32 dims)
        collision_threshold: Distance threshold for collision prediction (meters)
        prediction_horizon: Time horizon for trajectory prediction (seconds)
        k_past: Number of past timesteps in agent features
        
    Returns:
        edge_attr: [E, 20] or [E, 32] edge attribute array
    """
    N_BASE_FEATURES = 20
    N_ENHANCED_FEATURES = 12
    N_EDGE_FEATURES = N_BASE_FEATURES + (N_ENHANCED_FEATURES if use_enhanced else 0)
    
    if edge_index.shape[1] == 0:
        return np.zeros((0, N_EDGE_FEATURES), dtype=np.float32)
    
    # Extract positions and velocities
    x = agent_features[:, 0]
    y = agent_features[:, 1]
    vx = agent_features[:, 2]
    vy = agent_features[:, 3]
    
    E = edge_index.shape[1]
    attr_list = []
    
    for e in range(E):
        i = edge_index[0, e]
        j = edge_index[1, e]
        
        src_x, src_y = x[i], y[i]
        src_vx, src_vy = vx[i], vy[i]
        
        dx = x[j] - src_x
        dy = y[j] - src_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Relative velocity
        rel_vx = vx[j] - src_vx
        rel_vy = vy[j] - src_vy
        
        # Closing speed
        if dist > 1e-6:
            rel_pos_unit = np.array([dx, dy]) / dist
            rel_vel = np.array([rel_vx, rel_vy])
            closing_speed = -float(np.dot(rel_pos_unit, rel_vel))
        else:
            closing_speed = 0.0
        
        # TTC
        has_ttc = 1.0 if closing_speed > 0.5 else 0.0
        ttc = min(dist / closing_speed, 10.0) if closing_speed > 0.5 else 10.0
        
        # Positional flags
        is_ahead = 1.0 if dx > 0 else 0.0
        is_behind = 1.0 if dx < 0 else 0.0
        is_left = 1.0 if dy > 0 else 0.0
        is_right = 1.0 if dy < 0 else 0.0
        
        # Distance buckets
        dist_very_close = 1.0 if dist < 5.0 else 0.0
        dist_close = 1.0 if (5.0 <= dist < 15.0) else 0.0
        dist_medium = 1.0 if (15.0 <= dist < 30.0) else 0.0
        dist_far = 1.0 if dist >= 30.0 else 0.0
        
        # Approach flags
        is_approaching = 1.0 if closing_speed > 0.5 else 0.0
        is_moving_away = 1.0 if closing_speed < -0.5 else 0.0
        
        # Lane-proxy flags
        in_same_lane = abs(dy) < lane_width
        is_leading = 1.0 if (in_same_lane and dx > 0) else 0.0
        is_following = 1.0 if (in_same_lane and dx < 0) else 0.0
        
        # Base features (20 dims)
        edge_attr = [
            dx, dy, dist,
            rel_vx, rel_vy, closing_speed,
            ttc, has_ttc,
            is_ahead, is_behind, is_left, is_right,
            dist_very_close, dist_close, dist_medium, dist_far,
            is_approaching, is_moving_away,
            is_leading, is_following,
        ]
        
        # Enhanced features (12 additional dims)
        if use_enhanced:
            # Trajectory-based features (6 dims)
            lateral_vel_i = compute_lateral_velocity(agent_features, i)
            lateral_vel_j = compute_lateral_velocity(agent_features, j)
            lateral_movement_i = np.sign(lateral_vel_i) if abs(lateral_vel_i) > 0.3 else 0.0
            lateral_movement_j = np.sign(lateral_vel_j) if abs(lateral_vel_j) > 0.3 else 0.0
            yaw_rate_i = compute_yaw_rate(agent_features, i, k_past=k_past)
            yaw_rate_j = compute_yaw_rate(agent_features, j, k_past=k_past)
            
            # Predicted collision features (3 dims)
            pos_i = np.array([src_x, src_y])
            vel_i = np.array([src_vx, src_vy])
            pos_j = np.array([x[j], y[j]])
            vel_j = np.array([vx[j], vy[j]])
            
            pred_dist_1s = compute_predicted_distance(pos_i, vel_i, pos_j, vel_j, 1.0)
            pred_dist_2s = compute_predicted_distance(pos_i, vel_i, pos_j, vel_j, prediction_horizon)
            min_pred_dist = min(pred_dist_1s, pred_dist_2s)
            will_collide = 1.0 if min_pred_dist < collision_threshold else 0.0
            
            # Semantic interaction type (3 dims)
            # Compute j's lateral velocity toward i's position
            if dist > 1e-6:
                to_i_unit = np.array([-dx, -dy]) / dist
                # Perpendicular to to_i direction
                perp_to_i = np.array([-to_i_unit[1], to_i_unit[0]])
                lateral_vel_j_toward_i = abs(np.dot(vel_j, perp_to_i))
            else:
                lateral_vel_j_toward_i = 0.0
            
            merging_score, yielding_score, cutting_in_score = compute_interaction_scores(
                dx, dy, closing_speed, lateral_vel_j_toward_i
            )
            
            # Append enhanced features
            edge_attr.extend([
                lateral_vel_i, lateral_vel_j,           # 20-21
                lateral_movement_i, lateral_movement_j, # 22-23
                yaw_rate_i, yaw_rate_j,                 # 24-25
                pred_dist_1s, pred_dist_2s,             # 26-27
                will_collide,                           # 28
                merging_score, yielding_score, cutting_in_score,  # 29-31
            ])
        
        attr_list.append(edge_attr)
    
    return np.array(attr_list, dtype=np.float32)



def build_a2a_edges(
    agent_features: np.ndarray,
    ego_local_idx: int,
    max_dist: float = 50.0,
    lane_width: float = 2.0,
    ego_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Build agent-to-agent edges with semantic features.

    When ego_only=True: only ego -> other agents (original behavior)
    When ego_only=False: all agent pairs within max_dist (full connectivity)

    Edge attributes (20 features):
    - [0-1]: dx, dy (relative position)
    - [2]: dist (Euclidean distance)
    - [3-4]: rel_vx, rel_vy (relative velocity)
    - [5]: closing_speed (positive = approaching)
    - [6]: ttc (time-to-collision, clamped to 10s)
    - [7]: has_ttc (1.0 if closing_speed > 0.5, else 0.0)
    - [8]: is_ahead (dx > 0)
    - [9]: is_behind (dx < 0)
    - [10]: is_left (dy > 0)
    - [11]: is_right (dy < 0)
    - [12]: dist_very_close (dist < 5)
    - [13]: dist_close (5 <= dist < 15)
    - [14]: dist_medium (15 <= dist < 30)
    - [15]: dist_far (dist >= 30)
    - [16]: is_approaching (closing_speed > 0.5)
    - [17]: is_moving_away (closing_speed < -0.5)
    - [18]: is_leading (|dy| < 2.0 AND dx > 0)
    - [19]: is_following (|dy| < 2.0 AND dx < 0)

    Returns:
        edge_index: [2, E] array
        edge_attr: [E, 20] array with edge features
        edge_relations: List of dicts with human-readable relation labels (ego->others only)
    """
    N = agent_features.shape[0]
    N_EDGE_FEATURES = 20

    if N <= 1:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32),
                [])

    # Extract positions from features
    x = agent_features[:, 0]
    y = agent_features[:, 1]
    vx = agent_features[:, 2]
    vy = agent_features[:, 3]

    src_list = []
    dst_list = []
    attr_list = []
    relations_list = []  # Only populated for ego -> others edges

    for i in range(N):
        # If ego_only mode, skip non-ego sources
        if ego_only and i != ego_local_idx:
            continue

        src_x, src_y = x[i], y[i]
        src_vx, src_vy = vx[i], vy[i]

        for j in range(N):
            if i == j:
                continue

            # Position relative to source agent
            dx = x[j] - src_x
            dy = y[j] - src_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > max_dist:
                continue

            # Relative velocity
            rel_vx = vx[j] - src_vx
            rel_vy = vy[j] - src_vy

            # Closing speed (positive = approaching)
            if dist > 1e-6:
                rel_pos_unit = np.array([dx, dy]) / dist
                rel_vel = np.array([rel_vx, rel_vy])
                closing_speed = -float(np.dot(rel_pos_unit, rel_vel))
            else:
                closing_speed = 0.0

            # TTC computation
            has_ttc = 1.0 if closing_speed > 0.5 else 0.0
            if closing_speed > 0.5:
                ttc = min(dist / closing_speed, 10.0)  # clamp to 10s
            else:
                ttc = 10.0

            # Positional flags
            is_ahead = 1.0 if dx > 0 else 0.0
            is_behind = 1.0 if dx < 0 else 0.0
            is_left = 1.0 if dy > 0 else 0.0
            is_right = 1.0 if dy < 0 else 0.0

            # Distance buckets (one-hot)
            dist_very_close = 1.0 if dist < 5.0 else 0.0
            dist_close = 1.0 if (5.0 <= dist < 15.0) else 0.0
            dist_medium = 1.0 if (15.0 <= dist < 30.0) else 0.0
            dist_far = 1.0 if dist >= 30.0 else 0.0

            # Approach flags
            is_approaching = 1.0 if closing_speed > 0.5 else 0.0
            is_moving_away = 1.0 if closing_speed < -0.5 else 0.0

            # Lane-proxy flags
            in_same_lane = abs(dy) < lane_width
            is_leading = 1.0 if (in_same_lane and dx > 0) else 0.0
            is_following = 1.0 if (in_same_lane and dx < 0) else 0.0

            # Build edge attribute vector (20 features)
            edge_attr = [
                dx, dy,                                  # 0-1
                dist,                                    # 2
                rel_vx, rel_vy,                          # 3-4
                closing_speed,                           # 5
                ttc, has_ttc,                            # 6-7
                is_ahead, is_behind,                     # 8-9
                is_left, is_right,                       # 10-11
                dist_very_close, dist_close,             # 12-13
                dist_medium, dist_far,                   # 14-15
                is_approaching, is_moving_away,          # 16-17
                is_leading, is_following,                # 18-19
            ]

            src_list.append(i)
            dst_list.append(j)
            attr_list.append(edge_attr)

            # Only build human-readable relations for ego->others to avoid bloat
            if i == ego_local_idx:
                relations = []
                if dist_very_close: relations.append("very_close")
                elif dist_close: relations.append("close")
                elif dist_medium: relations.append("medium")
                else: relations.append("far")

                relations.append("ahead_of" if is_ahead else "behind_of")
                relations.append("left_of" if is_left else "right_of")

                if is_approaching:
                    relations.append("approaching")
                    if ttc < 1.0: relations.append("ttc_imminent")
                    elif ttc < 3.0: relations.append("ttc_soon")
                    elif ttc < 6.0: relations.append("ttc_later")
                elif is_moving_away:
                    relations.append("moving_away")

                if is_leading: relations.append("leading")
                if is_following: relations.append("following")

                relations_list.append({
                    "src": i,
                    "dst": j,
                    "distance_m": dist,
                    "x_local_m": dx,
                    "y_local_m": dy,
                    "closing_speed": closing_speed,
                    "ttc": ttc,
                    "relations": relations,
                })

    if not src_list:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, N_EDGE_FEATURES), dtype=np.float32),
                [])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)

    return edge_index, edge_attr, relations_list


def compute_closest_point_on_polyline(
    poly: np.ndarray,
    query: np.ndarray,
) -> Tuple[float, np.ndarray, float, float]:
    """
    Compute closest point on polyline to query point.

    Returns:
        dist: distance to closest point
        closest_pt: [2] closest point coordinates
        progress: normalized progress along polyline (0 to 1)
        lane_heading: heading at closest segment (radians)
    """
    if poly.shape[0] < 2:
        return float('inf'), query, 0.0, 0.0

    # Project onto each segment
    p = poly[:-1]  # (S, 2)
    q = poly[1:]   # (S, 2)
    v = q - p      # segment vectors
    w = query - p  # (S, 2)

    vv = np.sum(v * v, axis=1)  # (S,)
    vv = np.maximum(vv, 1e-9)   # avoid div by zero

    t = np.sum(w * v, axis=1) / vv
    t_clamped = np.clip(t, 0.0, 1.0)

    proj = p + v * t_clamped.reshape(-1, 1)  # (S, 2)
    d = proj - query
    dist2 = np.sum(d * d, axis=1)
    seg_idx = int(np.argmin(dist2))

    closest_pt = proj[seg_idx]
    dist = float(np.sqrt(dist2[seg_idx]))
    t_best = float(t_clamped[seg_idx])

    # Compute progress along polyline
    seg_lens = np.sqrt(np.sum(v * v, axis=1))
    total_len = float(np.sum(seg_lens))
    if total_len < 1e-6:
        progress = 0.0
    else:
        s_prefix = float(np.sum(seg_lens[:seg_idx]))
        s = s_prefix + t_best * float(seg_lens[seg_idx])
        progress = s / total_len

    # Lane heading at closest segment
    heading_vec = v[seg_idx]
    lane_heading = float(np.arctan2(heading_vec[1], heading_vec[0]))

    return dist, closest_pt, progress, lane_heading


def build_a2l_edges(
    agent_features: np.ndarray,
    lane_features: np.ndarray,
    lane_polylines: List[np.ndarray],
    k: int = 3,
    use_frenet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build agent-to-lane edges (each agent to k nearest lanes).

    Edge attributes (9 base features, 15 if use_frenet=True):
    - [0-1]: dx, dy (relative position to closest point on lane)
    - [2]: dist (distance to closest point)
    - [3]: lateral_offset (signed: positive = lane is left of agent)
    - [4]: heading_alignment (cos(agent_yaw - lane_heading))
    - [5]: is_on_lane (1.0 if lateral distance < 2m)
    - [6]: is_approaching_lane (1.0 if velocity points toward lane)
    - [7]: progress_along_lane (0 to 1)
    - [8]: angle_to_lane (atan2(dy, dx))
    
    Enhanced features (if use_frenet=True):
    - [9]: s (longitudinal Frenet coordinate - meters from lane start)
    - [10]: d (lateral Frenet coordinate - same as lateral_offset, for clarity)
    - [11]: d_dot (lateral velocity in lane frame)
    - [12]: v_longitudinal (velocity along lane direction)
    - [13]: v_lateral (velocity perpendicular to lane)
    - [14]: v_longitudinal_normalized (v_longitudinal / speed_limit)

    Returns:
        edge_index: [2, E] array
        edge_attr: [E, 9] or [E, 15] array with edge features
    """
    N_BASE_FEATURES = 9
    N_FRENET_FEATURES = 6
    N_EDGE_FEATURES = N_BASE_FEATURES + (N_FRENET_FEATURES if use_frenet else 0)
    N_agents = agent_features.shape[0]
    N_lanes = lane_features.shape[0]

    if N_agents == 0 or N_lanes == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, N_EDGE_FEATURES), dtype=np.float32)

    # Agent state (ego frame)
    agent_x = agent_features[:, 0]
    agent_y = agent_features[:, 1]
    agent_vx = agent_features[:, 2]
    agent_vy = agent_features[:, 3]
    agent_yaw = agent_features[:, 4]

    src_list = []
    dst_list = []
    attr_list = []

    for i in range(N_agents):
        ax, ay = agent_x[i], agent_y[i]
        avx, avy = agent_vx[i], agent_vy[i]
        a_yaw = agent_yaw[i]
        query = np.array([ax, ay], dtype=np.float32)

        # Compute distance to each lane
        lane_info = []
        for j in range(N_lanes):
            poly = lane_polylines[j]
            if poly.shape[0] >= 2:
                dist, closest_pt, progress, lane_heading = compute_closest_point_on_polyline(poly, query)
                # Compute lane length for Frenet s coordinate
                lane_length = float(np.sum(np.sqrt(np.sum(np.diff(poly, axis=0)**2, axis=1))))
            else:
                # Fallback to lane centroid
                cx, cy = lane_features[j, 0], lane_features[j, 1]
                closest_pt = np.array([cx, cy])
                dist = float(np.sqrt((cx - ax)**2 + (cy - ay)**2))
                progress = 0.5
                lane_heading = 0.0
                lane_length = 10.0  # default

            lane_info.append((j, dist, closest_pt, progress, lane_heading, lane_length))

        # Sort by distance and take k nearest
        lane_info.sort(key=lambda x: x[1])

        for lane_idx, dist, closest_pt, progress, lane_heading, lane_length in lane_info[:k]:
            # Relative position
            dx = closest_pt[0] - ax
            dy = closest_pt[1] - ay

            # Lateral offset (signed distance: positive = lane is left)
            # Using cross product: lane_dir x agent_to_lane
            lane_dir = np.array([np.cos(lane_heading), np.sin(lane_heading)])
            lane_perp = np.array([-np.sin(lane_heading), np.cos(lane_heading)])  # Left is positive
            to_lane = np.array([dx, dy])
            lateral_offset = float(lane_dir[0] * to_lane[1] - lane_dir[1] * to_lane[0])

            # Heading alignment
            heading_alignment = float(np.cos(a_yaw - lane_heading))

            # Is on lane (lateral distance < 2m)
            is_on_lane = 1.0 if abs(lateral_offset) < 2.0 else 0.0

            # Is approaching lane (velocity points toward lane)
            if dist > 1e-6:
                to_lane_unit = to_lane / dist
                vel = np.array([avx, avy])
                vel_toward_lane = float(np.dot(vel, to_lane_unit))
                is_approaching_lane = 1.0 if vel_toward_lane > 0.5 else 0.0
            else:
                is_approaching_lane = 0.0

            # Angle to lane
            angle_to_lane = float(np.arctan2(dy, dx))

            # Build base edge attribute vector (9 features)
            edge_attr = [
                dx, dy,                  # 0-1
                dist,                    # 2
                lateral_offset,          # 3
                heading_alignment,       # 4
                is_on_lane,              # 5
                is_approaching_lane,     # 6
                progress,                # 7
                angle_to_lane,           # 8
            ]
            
            # Enhanced Frenet features (6 additional dims)
            if use_frenet:
                # s = longitudinal position along lane (meters)
                s = progress * lane_length
                
                # d = lateral offset (already computed, but include for clarity in Frenet coords)
                d = lateral_offset
                
                # d_dot = lateral velocity (rate of change of lateral offset)
                vel = np.array([avx, avy])
                d_dot = float(np.dot(vel, lane_perp))
                
                # Lane-relative velocity decomposition
                v_longitudinal = float(np.dot(vel, lane_dir))
                v_lateral = float(np.dot(vel, lane_perp))  # same as d_dot
                
                # Normalized longitudinal velocity (by speed limit if available)
                # lane_features[lane_idx, 9] is speed_limit (normalized by 40.0)
                speed_limit_norm = lane_features[lane_idx, 9] if lane_idx < lane_features.shape[0] else 0.0
                speed_limit = speed_limit_norm * 40.0  # denormalize
                if speed_limit > 1e-6:
                    v_longitudinal_normalized = v_longitudinal / speed_limit
                else:
                    v_longitudinal_normalized = 0.0
                
                edge_attr.extend([
                    s,                        # 9
                    d,                        # 10
                    d_dot,                    # 11
                    v_longitudinal,           # 12
                    v_lateral,                # 13
                    v_longitudinal_normalized,  # 14
                ])

            src_list.append(i)
            dst_list.append(lane_idx)
            attr_list.append(edge_attr)

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, N_EDGE_FEATURES), dtype=np.float32)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)

    return edge_index, edge_attr


# -----------------------------
# Main function
# -----------------------------
def build_hetero_graph(
    tfrecord_path: str,
    record_index: int,
    config: Optional["GraphConfig"] = None,
    lane_config: Optional[PolylineGrouperConfig] = None,
) -> HeteroData:
    """
    Build a HeteroData graph from a TFRecord.

    Args:
        tfrecord_path: Path to TFRecord file
        record_index: Index of record to load
        config: GraphConfig instance controlling graph structure.
                If None, uses default (baseline) settings.
        lane_config: Configuration for lane extraction

    Returns:
        HeteroData with:
            - data['agent'].x: [N_agents, F_agent]
            - data['lane'].x: [N_lanes, F_lane]
            - data['tl'].x: [N_tl, 12] (traffic light nodes)
            - data['agent', 'to', 'agent'].edge_index, .edge_attr
            - data['agent', 'to', 'lane'].edge_index, .edge_attr
            - data['lane', 'to', 'lane'].edge_index, .edge_attr (L2L edges)
            - data['agent', 'to', 'tl'].edge_index, .edge_attr (A2TL edges)
            - data['lane', 'to', 'tl'].edge_index, .edge_attr (L2TL edges)
            - data.y: [N_agents, 80, 2] future trajectory labels
            - data.future_valid: [N_agents, 80] validity mask
            - data.ego_future: [80, 2] ego future trajectory
    """
    # Import here to avoid circular import
    from ..configs import GraphConfig
    
    # Use default config if not provided
    if config is None:
        config = GraphConfig.baseline()
    
    # Extract config values
    K_past = config.k_past
    a2l_k = config.a2l_k
    a2a_max_dist = config.a2a_max_dist
    full_a2a = config.full_a2a
    include_tl = config.include_tl
    include_l2l = config.include_l2l
    include_a2tl = config.include_a2tl
    include_l2tl = config.include_l2tl
    
    if lane_config is None:
        lane_config = PolylineGrouperConfig()

    # Load TFExample
    example = read_example_from_tfrecord(tfrecord_path, record_index)
    features = example.features.feature

    # Get ego state
    ego = get_ego_state(features)

    # Extract agent features
    agent_features, valid_indices, ego_local_idx = extract_agent_features(
        features, ego, K_past=K_past
    )

    # Extract future trajectory labels
    future_xy, future_valid = extract_future_trajectory(
        features, ego, valid_indices
    )

    # Extract lane features
    rg = get_roadgraph_samples(features)
    map_features = build_map_features_from_roadgraph(rg, ego, cfg=lane_config)
    lane_features, lane_polylines = extract_lane_features(map_features, ego)
    
    # Apply lane filtering if mode is not "all"
    lane_filter_mode = config.lane_filter_mode if hasattr(config, 'lane_filter_mode') else "all"
    
    if lane_filter_mode != "all" and lane_features.shape[0] > 1:
        from .edge_builders import filter_lanes
        
        lane_positions = torch.from_numpy(lane_features[:, :2]).float()  # First 2 features are x, y
        ego_position = torch.tensor([0.0, 0.0])  # Ego is at origin in ego-centric frame
        ego_heading = 0.0  # Ego heading is 0 in ego-centric frame
        
        lane_mask, _ = filter_lanes(
            lane_positions, ego_position, ego_heading,
            max_count=config.lane_max_count,
            max_distance=config.lane_max_distance,
            filter_mode=lane_filter_mode,
            drivable_only=config.lane_drivable_only,
        )
        
        # Apply mask to lanes
        if lane_mask.sum() < lane_features.shape[0]:
            lane_features = lane_features[lane_mask.numpy()]
            lane_polylines = [lane_polylines[i] for i, m in enumerate(lane_mask) if m]

    # Build L2L edges based on l2l_mode
    l2l_mode = config.l2l_mode if hasattr(config, 'l2l_mode') else "all"
    typed_l2l_edges = {}  # Will hold typed edges if l2l_mode == "typed"
    
    if include_l2l and len(lane_polylines) > 1:
        if l2l_mode == "typed":
            # Use typed L2L edges (successor/predecessor/left/right)
            from .edge_builders import build_typed_l2l_edges
            
            # Compute lane headings from polylines
            lane_headings = np.array([
                np.arctan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0]) if len(p) > 1 else 0.0
                for p in lane_polylines
            ])
            
            typed_l2l_edges = build_typed_l2l_edges(
                lane_polylines, lane_headings,
                max_successor_dist=config.l2l_max_successor_dist,
                max_neighbor_dist=config.l2l_max_neighbor_dist,
            )
            
            # Still create the standard l2l edges (as fallback/for backwards compatibility)
            l2l_edge_index, l2l_edge_attr = build_l2l_edges(
                lane_features, lane_polylines
            )
        else:
            # Default: single L2L edge type
            l2l_edge_index, l2l_edge_attr = build_l2l_edges(
                lane_features, lane_polylines
            )
    else:
        l2l_edge_index = np.zeros((2, 0), dtype=np.int64)
        l2l_edge_attr = np.zeros((0, 7), dtype=np.float32)

    # Build TL nodes and edges
    # TL nodes are needed if either A2TL or L2TL edges are enabled
    needs_tl = include_tl or include_a2tl or include_l2tl
    tl_filter_mode = config.tl_filter_mode if hasattr(config, 'tl_filter_mode') else "all"
    
    if needs_tl:
        tl_features, tl_ids = extract_tl_features(features, ego)

        if tl_features.shape[0] > 0:
            # Apply TL filtering if mode is not "all"
            if tl_filter_mode != "all" and tl_features.shape[0] > 1:
                from .edge_builders import filter_relevant_traffic_lights
                
                tl_positions = torch.from_numpy(tl_features[:, :2]).float()
                ego_position = torch.tensor([0.0, 0.0])  # Ego is at origin in ego-centric frame
                ego_heading = 0.0  # Ego heading is 0 in ego-centric frame
                
                tl_mask, _ = filter_relevant_traffic_lights(
                    tl_positions, ego_position, ego_heading,
                    max_distance=config.tl_max_distance,
                    max_relevant=config.tl_max_relevant,
                    ahead_only=config.tl_ahead_only,
                    filter_mode=tl_filter_mode,
                )
                
                # Apply mask
                if tl_mask.sum() < tl_features.shape[0]:
                    tl_features = tl_features[tl_mask.numpy()]
                    tl_ids = [tl_ids[i] for i, m in enumerate(tl_mask) if m]
            
            # Build A2TL edges if enabled
            if include_a2tl or (include_tl and config.tl_connect_to in ["agents", "both"]):
                a2tl_edge_index, a2tl_edge_attr = build_a2tl_edges(
                    agent_features, tl_features
                )
            else:
                a2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
                a2tl_edge_attr = np.zeros((0, 6), dtype=np.float32)
            
            # Build L2TL edges if enabled  
            if include_l2tl or (include_tl and config.tl_connect_to in ["lanes", "both"]):
                l2tl_edge_index, l2tl_edge_attr = build_l2tl_edges(
                    lane_features, lane_polylines, tl_features
                )
            else:
                l2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
                l2tl_edge_attr = np.zeros((0, 5), dtype=np.float32)
        else:
            # Empty TL graph components
            a2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
            a2tl_edge_attr = np.zeros((0, 6), dtype=np.float32)
            l2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
            l2tl_edge_attr = np.zeros((0, 5), dtype=np.float32)
    else:
        tl_features = np.zeros((0, 12), dtype=np.float32)
        tl_ids = []
        a2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
        a2tl_edge_attr = np.zeros((0, 6), dtype=np.float32)
        l2tl_edge_index = np.zeros((2, 0), dtype=np.int64)
        l2tl_edge_attr = np.zeros((0, 5), dtype=np.float32)

    # Build A2A edges based on a2a_mode
    a2a_mode = config.a2a_mode if hasattr(config, 'a2a_mode') else "ego_only"
    
    if a2a_mode == "k_nearest":
        # Use new k-nearest mode from edge_builders
        from .edge_builders import build_a2a_edges_k_nearest
        positions = torch.from_numpy(agent_features[:, :2]).float()
        a2a_src, a2a_dst = build_a2a_edges_k_nearest(
            positions, ego_local_idx,
            k=config.a2a_k_nearest,
            max_distance=a2a_max_dist
        )
        # Build edge attributes using existing function
        a2a_edge_index_new = np.stack([a2a_src.numpy(), a2a_dst.numpy()], axis=0)
        use_enhanced = getattr(config, 'use_enhanced_a2a_features', True)
        collision_threshold = getattr(config, 'a2a_collision_threshold', 3.0)
        prediction_horizon = getattr(config, 'a2a_prediction_horizon', 2.0)
        a2a_edge_attr = _compute_a2a_edge_attrs(
            agent_features, a2a_edge_index_new,
            use_enhanced=use_enhanced,
            collision_threshold=collision_threshold,
            prediction_horizon=prediction_horizon,
            k_past=k_past
        )
        a2a_edge_index = a2a_edge_index_new
        a2a_relations = []  # Not computed for new modes
        
    elif a2a_mode == "directional":
        # Use directional mode (only agents ahead in FOV)
        from .edge_builders import build_a2a_edges_directional
        positions = torch.from_numpy(agent_features[:, :2]).float()
        headings = torch.from_numpy(agent_features[:, 4]).float()  # heading is feature 4
        a2a_src, a2a_dst = build_a2a_edges_directional(
            positions, headings, ego_local_idx,
            k=config.a2a_k_nearest,
            max_distance=a2a_max_dist,
            fov_angle=config.a2a_fov_angle
        )
        a2a_edge_index_new = np.stack([a2a_src.numpy(), a2a_dst.numpy()], axis=0)
        use_enhanced = getattr(config, 'use_enhanced_a2a_features', True)
        collision_threshold = getattr(config, 'a2a_collision_threshold', 3.0)
        prediction_horizon = getattr(config, 'a2a_prediction_horizon', 2.0)
        a2a_edge_attr = _compute_a2a_edge_attrs(
            agent_features, a2a_edge_index_new,
            use_enhanced=use_enhanced,
            collision_threshold=collision_threshold,
            prediction_horizon=prediction_horizon,
            k_past=k_past
        )
        a2a_edge_index = a2a_edge_index_new
        a2a_relations = []
        
    else:
        # Default: use existing build_a2a_edges (ego_only or full_a2a)
        a2a_edge_index, a2a_edge_attr, a2a_relations = build_a2a_edges(
            agent_features, ego_local_idx,
            max_dist=a2a_max_dist,
            ego_only=not full_a2a
        )

    # Build A2L edges
    use_frenet = getattr(config, 'use_frenet_a2l_features', True)
    a2l_edge_index, a2l_edge_attr = build_a2l_edges(
        agent_features, lane_features, lane_polylines, k=a2l_k, use_frenet=use_frenet
    )

    # Create HeteroData
    data = HeteroData()

    # Node features
    data['agent'].x = torch.from_numpy(agent_features)
    data['lane'].x = torch.from_numpy(lane_features)
    data['tl'].x = torch.from_numpy(tl_features)

    # Agent-to-agent edges
    data['agent', 'to', 'agent'].edge_index = torch.from_numpy(a2a_edge_index)
    data['agent', 'to', 'agent'].edge_attr = torch.from_numpy(a2a_edge_attr)

    # Agent-to-lane edges
    data['agent', 'to', 'lane'].edge_index = torch.from_numpy(a2l_edge_index)
    data['agent', 'to', 'lane'].edge_attr = torch.from_numpy(a2l_edge_attr)

    # Lane-to-lane edges
    if l2l_mode == "typed" and typed_l2l_edges:
        # Store typed L2L edges separately
        for edge_type_name, (src, dst) in typed_l2l_edges.items():
            if len(src) > 0:
                data['lane', edge_type_name, 'lane'].edge_index = torch.stack([src, dst])
        
        # Also store generic L2L for backwards compatibility
        data['lane', 'to', 'lane'].edge_index = torch.from_numpy(l2l_edge_index)
        data['lane', 'to', 'lane'].edge_attr = torch.from_numpy(l2l_edge_attr)
    else:
        data['lane', 'to', 'lane'].edge_index = torch.from_numpy(l2l_edge_index)
        data['lane', 'to', 'lane'].edge_attr = torch.from_numpy(l2l_edge_attr)

    # Agent-to-traffic-light edges
    data['agent', 'to', 'tl'].edge_index = torch.from_numpy(a2tl_edge_index)
    data['agent', 'to', 'tl'].edge_attr = torch.from_numpy(a2tl_edge_attr)

    # Lane-to-traffic-light edges
    data['lane', 'to', 'tl'].edge_index = torch.from_numpy(l2tl_edge_index)
    data['lane', 'to', 'tl'].edge_attr = torch.from_numpy(l2tl_edge_attr)

    # Store metadata
    data.ego_idx = ego_local_idx
    data.record_index = record_index
    data.a2a_relations = a2a_relations  # Human-readable relations for visualization
    data.tl_ids = tl_ids  # Original TL indices for reference

    # Store future trajectory labels for motion prediction
    data.y = torch.from_numpy(future_xy)                # [N_agents, 80, 2]
    data.future_valid = torch.from_numpy(future_valid)  # [N_agents, 80]
    data.ego_future = data.y[ego_local_idx]             # [80, 2] convenience accessor

    # Store ego state for coordinate transforms in visualization
    data.ego_x = ego.x
    data.ego_y = ego.y
    data.ego_yaw = ego.yaw

    # Store roadgraph points (ego frame) for visualization
    # Subsample to max ~2000 points for efficiency
    rg_valid_mask = rg.valid.astype(bool)
    rg_xyz_valid = rg.xyz[rg_valid_mask]
    max_rg_points = 2000
    if rg_xyz_valid.shape[0] > max_rg_points:
        indices = np.linspace(0, rg_xyz_valid.shape[0] - 1, max_rg_points, dtype=int)
        rg_xyz_valid = rg_xyz_valid[indices]
    rg_ego = world_to_ego_points_xy(
        rg_xyz_valid[:, :2],
        np.array([ego.x, ego.y], dtype=np.float32),
        ego.yaw
    )
    data.roadgraph_ego = torch.from_numpy(rg_ego)

    return data


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HeteroGraph from TFRecord")
    parser.add_argument("--tfrecord", type=str, required=True, help="Path to TFRecord")
    parser.add_argument("--index", type=int, default=0, help="Record index")
    parser.add_argument("--no-tl", action="store_true", help="Disable traffic light nodes")
    parser.add_argument("--no-l2l", action="store_true", help="Disable L2L edges")
    parser.add_argument("--ego-only", action="store_true", help="Only ego->others A2A edges")
    args = parser.parse_args()

    print(f"Loading record {args.index} from {args.tfrecord}...")

    data = build_hetero_graph(
        args.tfrecord,
        args.index,
        include_tl=not args.no_tl,
        include_l2l=not args.no_l2l,
        full_a2a=not args.ego_only,
    )

    print("\n=== Enhanced Graph Structure ===")
    print(f"Agent nodes: {data['agent'].x.shape}")
    print(f"Lane nodes:  {data['lane'].x.shape}")
    print(f"TL nodes:    {data['tl'].x.shape}")
    print(f"A2A edges:   {data['agent', 'to', 'agent'].edge_index.shape[1]}")
    print(f"A2L edges:   {data['agent', 'to', 'lane'].edge_index.shape[1]}")
    print(f"L2L edges:   {data['lane', 'to', 'lane'].edge_index.shape[1]}")
    print(f"A2TL edges:  {data['agent', 'to', 'tl'].edge_index.shape[1]}")
    print(f"L2TL edges:  {data['lane', 'to', 'tl'].edge_index.shape[1]}")
    print(f"Ego index:   {data.ego_idx}")

    print("\n=== Future Trajectory Labels ===")
    print(f"Future trajectory (y): {data.y.shape}")
    print(f"Future valid mask:     {data.future_valid.shape}")
    print(f"Ego future:            {data.ego_future.shape}")
    # Count valid future steps for ego
    ego_valid_steps = int(data.future_valid[data.ego_idx].sum().item())
    print(f"Ego valid future steps: {ego_valid_steps}/80 ({ego_valid_steps * 0.1:.1f}s)")
    # Show first 5 future positions for ego
    print(f"Ego future (first 5 steps):")
    for t in range(min(5, data.ego_future.shape[0])):
        x, y = data.ego_future[t].numpy()
        valid = data.future_valid[data.ego_idx, t].item()
        print(f"  t={t}: ({x:7.2f}, {y:7.2f}) valid={valid:.0f}")

    print("\n=== Sample Features ===")
    print(f"Agent[0] features (first 10): {data['agent'].x[0, :10].numpy()}")
    if data['lane'].x.shape[0] > 0:
        print(f"Lane[0] features (first 10):  {data['lane'].x[0, :10].numpy()}")
    if data['tl'].x.shape[0] > 0:
        print(f"TL[0] features:               {data['tl'].x[0].numpy()}")

    print("\n=== L2L Edge Attribute Layout (7 features) ===")
    print("  [0]: min_distance_m")
    print("  [1]: is_successor")
    print("  [2]: is_predecessor")
    print("  [3]: is_left_neighbor")
    print("  [4]: is_right_neighbor")
    print("  [5]: heading_diff_cos")
    print("  [6]: same_direction")

    print("\n=== TL Node Feature Layout (12 features) ===")
    print("  [0-1]: x_ego, y_ego")
    print("  [2]: dist_m")
    print("  [3-7]: state one-hot (unknown, red, yellow, green, flashing)")
    print("  [8]: is_ahead")
    print("  [9]: is_relevant")
    print("  [10]: normalized_dist")
    print("  [11]: heading_to_tl")

    print("\n=== A2TL Edge Attribute Layout (6 features) ===")
    print("  [0-1]: dx, dy (relative position)")
    print("  [2]: dist")
    print("  [3]: is_ahead")
    print("  [4]: is_relevant")
    print("  [5]: angle_to_tl (normalized)")

    print("\n=== L2TL Edge Attribute Layout (5 features) ===")
    print("  [0]: dist (from lane end to TL)")
    print("  [1]: is_ahead")
    print("  [2]: lateral_offset (signed)")
    print("  [3]: heading_alignment")
    print("  [4]: is_controlling")

    print("\n=== A2A Edge Attribute Layout (20 features) ===")
    print("  [0-1]: dx, dy (relative position)")
    print("  [2]: dist")
    print("  [3-4]: rel_vx, rel_vy")
    print("  [5]: closing_speed")
    print("  [6]: ttc (clamped to 10s)")
    print("  [7]: has_ttc")
    print("  [8-9]: is_ahead, is_behind")
    print("  [10-11]: is_left, is_right")
    print("  [12-15]: dist_very_close, dist_close, dist_medium, dist_far")
    print("  [16-17]: is_approaching, is_moving_away")
    print("  [18-19]: is_leading, is_following")

    print("\n=== A2L Edge Attribute Layout (9 features) ===")
    print("  [0-1]: dx, dy (relative position to closest point)")
    print("  [2]: dist")
    print("  [3]: lateral_offset (signed, positive = lane left of agent)")
    print("  [4]: heading_alignment (cos(agent_yaw - lane_heading))")
    print("  [5]: is_on_lane (lateral < 2m)")
    print("  [6]: is_approaching_lane (velocity toward lane)")
    print("  [7]: progress_along_lane (0 to 1)")
    print("  [8]: angle_to_lane")

    print("\n=== Agent Feature Layout (16 + 5*K) ===")
    print("  [0-1]: x, y (ego frame)")
    print("  [2-3]: vx, vy")
    print("  [4]: yaw")
    print("  [5]: speed")
    print("  [6-7]: cos(yaw), sin(yaw)")
    print("  [8-9]: length, width")
    print("  [10-14]: type_onehot (unknown, vehicle, pedestrian, cyclist, other)")
    print("  [15]: is_ego")
    print("  [16:16+K]: past_x")
    print("  [16+K:16+2K]: past_y")
    print("  [16+2K:16+3K]: past_vx")
    print("  [16+3K:16+4K]: past_vy")
    print("  [16+4K:16+5K]: past_valid")

    print("\n=== Sample Semantic Relations (ego -> others) ===")
    for rel in data.a2a_relations[:5]:
        print(f"  ego -> {rel['dst']}: dist={rel['distance_m']:.1f}m, "
              f"x_local={rel['x_local_m']:.1f}m, y_local={rel['y_local_m']:.1f}m")
        print(f"    relations: {rel['relations']}")

    # Show L2L edge samples if available
    if data['lane', 'to', 'lane'].edge_index.shape[1] > 0:
        print("\n=== Sample L2L Edges ===")
        l2l_ei = data['lane', 'to', 'lane'].edge_index
        l2l_ea = data['lane', 'to', 'lane'].edge_attr
        for k in range(min(5, l2l_ei.shape[1])):
            src, dst = l2l_ei[0, k].item(), l2l_ei[1, k].item()
            attrs = l2l_ea[k].numpy()
            relations = []
            if attrs[1] > 0.5: relations.append("successor")
            if attrs[2] > 0.5: relations.append("predecessor")
            if attrs[3] > 0.5: relations.append("left_neighbor")
            if attrs[4] > 0.5: relations.append("right_neighbor")
            if attrs[6] > 0.5: relations.append("same_direction")
            print(f"  lane[{src}] -> lane[{dst}]: dist={attrs[0]:.1f}m, {relations}")

    # Show TL info if available
    if data['tl'].x.shape[0] > 0:
        print("\n=== Traffic Light Summary ===")
        tl_x = data['tl'].x
        for k in range(min(5, tl_x.shape[0])):
            pos = tl_x[k, 0:2].numpy()
            dist = tl_x[k, 2].item()
            states = tl_x[k, 3:8].numpy()
            state_names = ["unknown", "red", "yellow", "green", "flashing"]
            state = state_names[int(np.argmax(states))]
            is_ahead = tl_x[k, 8].item() > 0.5
            is_relevant = tl_x[k, 9].item() > 0.5
            print(f"  TL[{k}]: pos=({pos[0]:.1f}, {pos[1]:.1f}), dist={dist:.1f}m, "
                  f"state={state}, ahead={is_ahead}, relevant={is_relevant}")

    print("\nDone!")
