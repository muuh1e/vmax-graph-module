"""
geometry.py

Small geometry utilities:
- ego-frame transform
- nearest point to point-cloud
- closest point on a polyline (segment projection) -> needed for route monitor / frenet-ish features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def ego_frame(dx: np.ndarray, dy: np.ndarray, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate world-frame delta into ego frame:
      x_local = forward
      y_local = left
    """
    c = np.cos(-ego_yaw)
    s = np.sin(-ego_yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    return x_local, y_local


def world_to_ego_points_xy(points_xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """points_xy: (N,2) -> (N,2) in ego frame"""
    d = points_xy - ego_xy.reshape(1, 2)
    x_local, y_local = ego_frame(d[:, 0], d[:, 1], ego_yaw)
    return np.stack([x_local, y_local], axis=-1)


def select_points_in_vmax_box(
    points_xy: np.ndarray,
    ego_xy: np.ndarray,
    ego_yaw: float,
    fwd: float = 50.0,
    back: float = 5.0,
    side: float = 20.0,
) -> np.ndarray:
    """
    Returns boolean mask for points inside the V-Max style ego-frame bounding box:
      -back <= x_local <= fwd
      |y_local| <= side
    """
    pts_ego = world_to_ego_points_xy(points_xy, ego_xy, ego_yaw)
    x = pts_ego[:, 0]
    y = pts_ego[:, 1]
    return (x <= fwd) & (x >= -back) & (np.abs(y) <= side)


@dataclass(frozen=True)
class ClosestPointResult:
    idx: int
    dist: float
    closest_xy: np.ndarray  # (2,)
    dx: float
    dy: float


def closest_point_in_cloud(points_xy: np.ndarray, query_xy: np.ndarray) -> ClosestPointResult:
    """
    points_xy: (N,2), query_xy: (2,)
    Returns closest point in the cloud.
    """
    d = points_xy - query_xy.reshape(1, 2)
    dist2 = np.sum(d * d, axis=1)
    idx = int(np.argmin(dist2))
    dist = float(np.sqrt(dist2[idx]))
    return ClosestPointResult(
        idx=idx,
        dist=dist,
        closest_xy=points_xy[idx].copy(),
        dx=float(d[idx, 0]),
        dy=float(d[idx, 1]),
    )


@dataclass(frozen=True)
class ClosestPointOnPolyline:
    dist: float
    closest_xy: np.ndarray     # (2,)
    seg_idx: int               # segment index i refers to segment [i, i+1]
    t: float                   # projection parameter on segment [0,1]
    s: float                   # arclength coordinate along polyline (meters)
    signed_lateral: float      # sign by local segment heading (left positive)


def closest_point_on_polyline(poly_xy: np.ndarray, query_xy: np.ndarray) -> Optional[ClosestPointOnPolyline]:
    """
    Compute the closest point on a polyline to query_xy by projecting onto each segment.
    poly_xy: (N,2) N>=2 expected. If N<2 returns None.

    Returns:
      closest point, distance, segment index, t in [0,1], s (approx arclength), signed lateral error.
    """
    if poly_xy.shape[0] < 2:
        return None

    p = poly_xy[:-1]           # (S,2)
    q = poly_xy[1:]            # (S,2)
    v = q - p                  # segment vectors (S,2)
    w = query_xy.reshape(1, 2) - p  # (S,2)

    vv = np.sum(v * v, axis=1)  # (S,)
    # avoid divide-by-zero for degenerate segments
    vv = np.maximum(vv, 1e-9)

    t = np.sum(w * v, axis=1) / vv
    t_clamped = np.clip(t, 0.0, 1.0)

    proj = p + v * t_clamped.reshape(-1, 1)  # (S,2)
    d = proj - query_xy.reshape(1, 2)
    dist2 = np.sum(d * d, axis=1)
    seg_idx = int(np.argmin(dist2))

    closest_xy = proj[seg_idx]
    dist = float(np.sqrt(dist2[seg_idx]))
    t_best = float(t_clamped[seg_idx])

    # arclength s: sum of prior segment lengths + t*current_length
    seg_lens = np.sqrt(np.sum(v * v, axis=1))
    s_prefix = float(np.sum(seg_lens[:seg_idx]))
    s = s_prefix + t_best * float(seg_lens[seg_idx])

    # signed lateral error using segment heading (left positive)
    heading = v[seg_idx]
    h_norm = np.linalg.norm(heading)
    if h_norm < 1e-6:
        signed_lat = 0.0
    else:
        h = heading / h_norm
        # vector from closest to query
        e = (query_xy - closest_xy)
        # 2D cross product z = h_x*e_y - h_y*e_x
        cross_z = float(h[0] * e[1] - h[1] * e[0])
        signed_lat = cross_z  # sign only; magnitude ~ lateral meters if h is unit

    return ClosestPointOnPolyline(
        dist=dist,
        closest_xy=closest_xy.copy(),
        seg_idx=seg_idx,
        t=t_best,
        s=float(s),
        signed_lateral=float(signed_lat),
    )
