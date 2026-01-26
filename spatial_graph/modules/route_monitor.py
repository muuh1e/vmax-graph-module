"""
route_monitor.py

Phase 2: Use path_samples/* as "route prior" and compute ego↔route relation.

What we have (from TFExample inspection):
- path_samples/xyz (P,3)
- path_samples/valid (P,)
- path_samples/id (P,) optional
- path_samples/arc_length (P,) optional

What we output:
- A RouteNode summary (global route / path polyline)
- An Ego→Route edge with:
  * closest distance to route
  * signed lateral error (left positive) based on local route segment heading
  * along-track position s (meters, approximate)
  * status label: ON_TRACK / DEVIATING / OFF_ROUTE (simple thresholding)

Notes:
- We do NOT assume lane topology exists (it doesn't in TFExample).
- If multiple path_ids exist, we either pick the one that is closest to ego (default),
  or treat them as separate route candidates later (v2).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from spatial_graph.utils.tfexample_io import EgoState, PathSamples
from spatial_graph.utils.geometry import closest_point_on_polyline


# -------------------------
# Public datatypes
# -------------------------
@dataclass(frozen=True)
class RouteNode:
    route_id: int
    num_points: int
    centroid_xy: Tuple[float, float]
    bbox_xy_min: Tuple[float, float]
    bbox_xy_max: Tuple[float, float]
    has_arc_length: bool


@dataclass(frozen=True)
class EgoRouteEdge:
    route_id: int

    closest_dist_m: float
    closest_world_xy: Tuple[float, float]

    # "Frenet-ish" features relative to route polyline
    s_m: float
    signed_lateral_m: float

    # coarse status label for the LLM
    status: str  # ON_TRACK | DEVIATING | OFF_ROUTE


@dataclass(frozen=True)
class RouteMonitorConfig:
    # thresholds for status label
    on_track_lat_m: float = 1.5
    deviating_lat_m: float = 4.0
    off_route_dist_m: float = 10.0

    # If multiple path_ids exist, choose which route:
    # 'closest' = pick route with minimum distance to ego
    route_selection: str = "closest"


# -------------------------
# Core logic
# -------------------------
def _route_geometry_summary(route_xy: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    centroid = np.mean(route_xy, axis=0)
    bb_min = np.min(route_xy, axis=0)
    bb_max = np.max(route_xy, axis=0)
    return (float(centroid[0]), float(centroid[1])), (float(bb_min[0]), float(bb_min[1])), (float(bb_max[0]), float(bb_max[1]))


def build_route_node_and_edge(
    ego: EgoState,
    path: Optional[PathSamples],
    cfg: RouteMonitorConfig = RouteMonitorConfig(),
) -> Optional[Tuple[RouteNode, EgoRouteEdge]]:
    """
    Returns (RouteNode, EgoRouteEdge) if path_samples exist and have >=2 valid points,
    else returns None.
    """
    if path is None:
        return None

    valid_idx = np.where(path.valid)[0]
    if valid_idx.size < 2:
        return None

    # If multiple path ids exist, group them and select one
    if path.path_id is not None:
        ids = path.path_id[valid_idx]
        uniq = np.unique(ids)

        candidates: List[Tuple[int, np.ndarray]] = []
        for rid in uniq:
            idx = valid_idx[ids == rid]
            if idx.size < 2:
                continue
            route_xy = path.xyz[idx, :2].astype(np.float32)
            candidates.append((int(rid), route_xy))

        if len(candidates) == 0:
            return None

        if cfg.route_selection == "closest":
            best = None
            best_dist = float("inf")
            for rid, route_xy in candidates:
                res = closest_point_on_polyline(route_xy, np.array([ego.x, ego.y], dtype=np.float32))
                if res is None:
                    continue
                if res.dist < best_dist:
                    best_dist = res.dist
                    best = (rid, route_xy, res)
            if best is None:
                return None
            route_id, route_xy, cp = best
        else:
            # fallback: pick the first id
            route_id, route_xy = candidates[0]
            cp = closest_point_on_polyline(route_xy, np.array([ego.x, ego.y], dtype=np.float32))
            if cp is None:
                return None
    else:
        # Single unlabelled route
        route_id = 0
        route_xy = path.xyz[valid_idx, :2].astype(np.float32)
        cp = closest_point_on_polyline(route_xy, np.array([ego.x, ego.y], dtype=np.float32))
        if cp is None:
            return None

    centroid, bb_min, bb_max = _route_geometry_summary(route_xy)

    node = RouteNode(
        route_id=int(route_id),
        num_points=int(route_xy.shape[0]),
        centroid_xy=centroid,
        bbox_xy_min=bb_min,
        bbox_xy_max=bb_max,
        has_arc_length=bool(path.arc_length is not None),
    )

    # status labeling: first check absolute distance to route
    if cp.dist >= cfg.off_route_dist_m:
        status = "OFF_ROUTE"
    else:
        lat = abs(cp.signed_lateral)
        if lat <= cfg.on_track_lat_m:
            status = "ON_TRACK"
        elif lat <= cfg.deviating_lat_m:
            status = "DEVIATING"
        else:
            status = "OFF_ROUTE"

    edge = EgoRouteEdge(
        route_id=int(route_id),
        closest_dist_m=float(cp.dist),
        closest_world_xy=(float(cp.closest_xy[0]), float(cp.closest_xy[1])),
        s_m=float(cp.s),
        signed_lateral_m=float(cp.signed_lateral),
        status=status,
    )

    return node, edge


# -------------------------
# Convenience (debug)
# -------------------------
def route_node_to_dict(node: RouteNode) -> Dict:
    return asdict(node)


def ego_route_edge_to_dict(edge: EgoRouteEdge) -> Dict:
    return asdict(edge)
