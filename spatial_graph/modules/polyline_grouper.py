"""
polyline_grouper.py

Phase 1: Convert roadgraph_samples (point cloud) into polyline-level MapFeature entities
by grouping points using roadgraph_samples/id.

ENHANCEMENTS:
- Added Waymo roadgraph type dictionary with semantic names
- Added type categorization (DRIVABLE, BOUNDARY, MARKING)
- Speed limit validation and unit documentation
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from spatial_graph.utils.tfexample_io import EgoState, RoadgraphSamples
from spatial_graph.utils.geometry import ego_frame, select_points_in_vmax_box


# -------------------------
# Waymo Roadgraph Type Mapping
# -------------------------
# Reference: Waymo Open Dataset v1.4.2 documentation
# URL: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/map.proto

WAYMO_ROADGRAPH_TYPES = {
    0: "UNKNOWN",
    1: "FREEWAY",
    2: "SURFACE_STREET",
    3: "BIKE_LANE",
    6: "ROAD_EDGE_BOUNDARY",
    7: "ROAD_EDGE_MEDIAN",
    9: "BROKEN_SINGLE_WHITE",
    10: "SOLID_SINGLE_WHITE",
    11: "SOLID_DOUBLE_WHITE",
    12: "BROKEN_SINGLE_YELLOW",
    13: "BROKEN_DOUBLE_YELLOW",
    14: "SOLID_SINGLE_YELLOW",
    15: "SOLID_DOUBLE_YELLOW",
    16: "PASSING_DOUBLE_YELLOW",
    17: "CROSSWALK",
    18: "SPEED_BUMP",
    19: "STOP_SIGN",
    20: "DRIVEWAY",
}

# Semantic categorization for filtering/prioritization
TYPE_CATEGORIES = {
    "DRIVABLE": {1, 2, 3, 20},  # FREEWAY, SURFACE_STREET, BIKE_LANE, DRIVEWAY
    "BOUNDARY": {6, 7},          # ROAD_EDGE_BOUNDARY, ROAD_EDGE_MEDIAN
    "MARKING": {9, 10, 11, 12, 13, 14, 15, 16},  # Lane markings
    "SPECIAL": {17, 18, 19},     # CROSSWALK, SPEED_BUMP, STOP_SIGN
    "UNKNOWN": {0},
}

def get_type_name(type_code: Optional[int]) -> str:
    """Convert numeric type code to human-readable name."""
    if type_code is None:
        return "UNKNOWN"
    return WAYMO_ROADGRAPH_TYPES.get(type_code, f"UNKNOWN_{type_code}")

def get_type_category(type_code: Optional[int]) -> str:
    """Get semantic category for a type code."""
    if type_code is None:
        return "UNKNOWN"
    for category, codes in TYPE_CATEGORIES.items():
        if type_code in codes:
            return category
    return "UNKNOWN"


# -------------------------
# Public datatypes
# -------------------------
@dataclass(frozen=True)
class MapFeature:
    polyline_id: int

    # Map semantics - ENHANCED with human-readable names
    type_mode: Optional[int]
    type_name: str          # NEW: Human-readable type name
    type_category: str      # NEW: Semantic category

    # Geometry summary
    num_points: int
    centroid_xy: Tuple[float, float]
    bbox_xy_min: Tuple[float, float]
    bbox_xy_max: Tuple[float, float]

    # Speed limit (from roadgraph_samples/speed_limit) if available
    # NOTE: Units are m/s (meters per second) based on Waymo documentation
    # Common values: ~11 m/s (40 km/h urban), ~22 m/s (80 km/h), ~36 m/s (130 km/h highway)
    speed_limit_mps: Optional[float]

    # Closest point to ego (world + ego frame)
    closest_dist_m: float
    closest_world_xy: Tuple[float, float]
    closest_x_local_m: float
    closest_y_local_m: float

    # Direction alignment at closest point if roadgraph_samples/dir exists
    heading_alignment_cos: Optional[float]

    # Optional compact sample points for later heuristics/debug (world frame)
    sample_points_xy: List[Tuple[float, float]]


@dataclass(frozen=True)
class PolylineGrouperConfig:
    # V-Max style spatial selection (ego-frame box)
    use_vmax_box: bool = True
    box_fwd_m: float = 50.0
    box_back_m: float = 5.0
    box_side_m: float = 20.0

    # Output size control
    max_polylines: int = 80
    max_sample_points_per_polyline: int = 40

    # Speed limit handling
    speed_limit_ignore_negative: bool = True  # Ignore -1 (missing) values
    speed_limit_warn_threshold_mps: float = 60.0  # Warn if > 216 km/h (suspicious)


# -------------------------
# Core logic
# -------------------------
def _mode_int(values: np.ndarray) -> Optional[int]:
    if values is None or values.size == 0:
        return None
    # values are int64
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _median_speed(speed: Optional[np.ndarray], idx: np.ndarray, ignore_negative: bool) -> Optional[float]:
    """
    Compute median speed limit for a polyline.
    
    Args:
        speed: Full speed_limit array from roadgraph_samples
        idx: Indices for this polyline
        ignore_negative: If True, filter out -1 (missing) values
    
    Returns:
        Median speed in m/s, or None if no valid values
    """
    if speed is None:
        return None
    vals = speed[idx].astype(np.float32)
    if vals.size == 0:
        return None
    if ignore_negative:
        valid = vals[vals >= 0.0]
        if valid.size > 0:
            vals = valid
    return float(np.median(vals))


def _heading_alignment_cos(
    ego_yaw: float,
    rg_dir: Optional[np.ndarray],
    closest_index: int,
) -> Optional[float]:
    if rg_dir is None:
        return None
    if closest_index < 0 or closest_index >= rg_dir.shape[0]:
        return None

    d = rg_dir[closest_index, :2].astype(np.float32)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return None
    d = d / n

    ego_fwd = np.array([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    return float(np.clip(np.dot(d, ego_fwd), -1.0, 1.0))


def build_map_features_from_roadgraph(
    rg: RoadgraphSamples,
    ego: EgoState,
    cfg: PolylineGrouperConfig = PolylineGrouperConfig(),
) -> List[MapFeature]:
    """
    Convert roadgraph_samples into a list of polyline-level MapFeature objects.

    Steps:
    1) Filter valid points
    2) Optionally apply V-Max ego-frame box filter
    3) Group by roadgraph_samples/id
    4) Summarize each polyline (closest point, speed limit, etc.)
    5) Sort by closest distance to ego and keep top cfg.max_polylines
    """

    xyz = rg.xyz
    points_xy = xyz[:, :2].astype(np.float32)
    valid_mask = rg.valid.astype(bool)

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
    else:
        mask = valid_mask

    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return []

    # If rg_id is missing, we cannot group properly -> treat everything as one polyline
    if rg.rg_id is None:
        return [_summarize_one_polyline(polyline_id=0, idx=idx_all, rg=rg, ego=ego, cfg=cfg)]

    ids = rg.rg_id[idx_all]
    uniq_ids = np.unique(ids)

    features: List[MapFeature] = []
    for pid in uniq_ids:
        idx = idx_all[ids == pid]
        if idx.size == 0:
            continue
        features.append(_summarize_one_polyline(polyline_id=int(pid), idx=idx, rg=rg, ego=ego, cfg=cfg))

    # Sort by closest distance and keep top-K
    features.sort(key=lambda mf: mf.closest_dist_m)
    return features[: cfg.max_polylines]


def _summarize_one_polyline(
    polyline_id: int,
    idx: np.ndarray,
    rg: RoadgraphSamples,
    ego: EgoState,
    cfg: PolylineGrouperConfig,
) -> MapFeature:
    points_xy = rg.xyz[idx, :2].astype(np.float32)

    # Geometry stats
    centroid = np.mean(points_xy, axis=0)
    bb_min = np.min(points_xy, axis=0)
    bb_max = np.max(points_xy, axis=0)

    # Closest point to ego (within this polyline subset)
    dx = points_xy[:, 0] - float(ego.x)
    dy = points_xy[:, 1] - float(ego.y)
    dist2 = dx * dx + dy * dy
    local_min = int(np.argmin(dist2))
    closest_dist = float(np.sqrt(dist2[local_min]))

    closest_world = points_xy[local_min]
    x_local, y_local = ego_frame(
        dx=np.array([dx[local_min]], dtype=np.float32),
        dy=np.array([dy[local_min]], dtype=np.float32),
        ego_yaw=float(ego.yaw),
    )
    closest_x_local = float(x_local[0])
    closest_y_local = float(y_local[0])

    # Type mode (raw int code) + semantic names
    type_mode = None
    if rg.rg_type is not None:
        type_mode = _mode_int(rg.rg_type[idx])
    
    type_name = get_type_name(type_mode)
    type_category = get_type_category(type_mode)

    # Speed limit (median) with validation
    speed_med = _median_speed(
        speed=rg.speed_limit,
        idx=idx,
        ignore_negative=bool(cfg.speed_limit_ignore_negative),
    )
    
    # Warn about suspicious speed limits (optional, can be removed if noisy)
    if speed_med is not None and speed_med > cfg.speed_limit_warn_threshold_mps:
        import warnings
        warnings.warn(
            f"Polyline {polyline_id} has suspicious speed limit {speed_med:.1f} m/s "
            f"({speed_med * 3.6:.1f} km/h). This may indicate incorrect units."
        )

    # Heading alignment at closest point (need original global index)
    closest_global_index = int(idx[local_min])
    align = _heading_alignment_cos(float(ego.yaw), rg.rg_dir, closest_global_index)

    # Sample points: take K closest points in this polyline to ego
    K = min(int(cfg.max_sample_points_per_polyline), int(idx.size))
    if K > 0:
        # partial sort for efficiency
        sel = np.argpartition(dist2, K - 1)[:K]
        sel = sel[np.argsort(dist2[sel])]
        sample = points_xy[sel]
        sample_points = [(float(p[0]), float(p[1])) for p in sample]
    else:
        sample_points = []

    return MapFeature(
        polyline_id=int(polyline_id),
        type_mode=type_mode,
        type_name=type_name,
        type_category=type_category,
        num_points=int(idx.size),
        centroid_xy=(float(centroid[0]), float(centroid[1])),
        bbox_xy_min=(float(bb_min[0]), float(bb_min[1])),
        bbox_xy_max=(float(bb_max[0]), float(bb_max[1])),
        speed_limit_mps=speed_med,
        closest_dist_m=float(closest_dist),
        closest_world_xy=(float(closest_world[0]), float(closest_world[1])),
        closest_x_local_m=float(closest_x_local),
        closest_y_local_m=float(closest_y_local),
        heading_alignment_cos=align,
        sample_points_xy=sample_points,
    )


# -------------------------
# Convenience (debug)
# -------------------------
def map_features_to_dict(features: List[MapFeature]) -> List[Dict]:
    """Handy for JSON serialization later."""
    return [asdict(f) for f in features]