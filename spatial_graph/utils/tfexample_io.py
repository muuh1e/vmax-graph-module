"""
tfexample_io.py

Small utilities to load one TFExample record and extract the spatial fields we confirmed exist:
- roadgraph_samples/{xyz,dir,type,id,valid,speed_limit}
- traffic_light_state/{current,past,future}/*
- path_samples/*  (handled defensively if missing in some records)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


# -------------------------
# Basic TFExample access
# -------------------------
def read_example_from_tfrecord(tfrecord_path: str, record_index: int) -> tf.train.Example:
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for i, raw in enumerate(ds):
        if i == record_index:
            return tf.train.Example.FromString(raw.numpy())
    raise IndexError(f"record_index={record_index} not found in {tfrecord_path}")


def _float_feature(features: Dict[str, tf.train.Feature], key: str) -> Optional[np.ndarray]:
    feat = features.get(key, None)
    if feat is None:
        return None
    vals = feat.float_list.value
    if len(vals) == 0:
        return None
    return np.asarray(vals, dtype=np.float32)


def _int_feature(features: Dict[str, tf.train.Feature], key: str) -> Optional[np.ndarray]:
    feat = features.get(key, None)
    if feat is None:
        return None
    vals = feat.int64_list.value
    if len(vals) == 0:
        return None
    return np.asarray(vals, dtype=np.int64)


def has_key(features: Dict[str, tf.train.Feature], key: str) -> bool:
    """True if key exists and has non-empty list."""
    feat = features.get(key, None)
    if feat is None:
        return False
    return (
        len(feat.float_list.value) > 0
        or len(feat.int64_list.value) > 0
        or len(feat.bytes_list.value) > 0
    )


# -------------------------
# Ego helpers
# -------------------------
@dataclass(frozen=True)
class EgoState:
    idx: int
    x: float
    y: float
    yaw: float
    speed: float
    speed_source: str   # "feature" | "computed" | "missing"


def get_ego_state(features: Dict[str, tf.train.Feature]) -> EgoState:
    """
    Ego is the object where state/is_sdc == 1.
    yaw: prefer state/current/bbox_yaw, else state/current/vel_yaw, else 0
    speed: prefer state/current/speed, else sqrt(vx^2 + vy^2) if available, else 0
    """
    is_sdc = _int_feature(features, "state/is_sdc")
    if is_sdc is None:
        raise KeyError("Missing state/is_sdc")

    ego_idx = int(np.argmax(is_sdc))

    cur_x = _float_feature(features, "state/current/x")
    cur_y = _float_feature(features, "state/current/y")
    if cur_x is None or cur_y is None:
        raise KeyError("Missing state/current/x or state/current/y")

    yaw = 0.0
    bbox_yaw = _float_feature(features, "state/current/bbox_yaw")
    vel_yaw = _float_feature(features, "state/current/vel_yaw")
    if bbox_yaw is not None and bbox_yaw.size > ego_idx:
        yaw = float(bbox_yaw[ego_idx])
    elif vel_yaw is not None and vel_yaw.size > ego_idx:
        yaw = float(vel_yaw[ego_idx])

    speed = None
    cur_speed = _float_feature(features, "state/current/speed")
    if cur_speed is not None and cur_speed.size > ego_idx:
        s = float(cur_speed[ego_idx])
        if s >= 0.0:
            speed = s

    
    
    speed_source = "feature"
    
    if speed is None:
        vx = _float_feature(features, "state/current/velocity_x")
        vy = _float_feature(features, "state/current/velocity_y")
        if vx is not None and vy is not None and vx.size > ego_idx and vy.size > ego_idx:
            speed = float(np.sqrt(vx[ego_idx] ** 2 + vy[ego_idx] ** 2))
            speed_source = "computed"
        else:
            speed = 0.0
            speed_source = "missing"
        
    return EgoState(
    idx=ego_idx,
    x=float(cur_x[ego_idx]),
    y=float(cur_y[ego_idx]),
    yaw=float(yaw),
    speed=float(speed),
    speed_source=speed_source,
)


# -------------------------
# Roadgraph extraction
# -------------------------
@dataclass(frozen=True)
class RoadgraphSamples:
    xyz: np.ndarray          # (M,3)
    valid: np.ndarray        # (M,) bool
    rg_type: Optional[np.ndarray]   # (M,) int64
    rg_id: Optional[np.ndarray]     # (M,) int64
    rg_dir: Optional[np.ndarray]    # (M,3) float32
    speed_limit: Optional[np.ndarray]  # (M,) float32


def get_roadgraph_samples(features: Dict[str, tf.train.Feature]) -> RoadgraphSamples:
    xyz = _float_feature(features, "roadgraph_samples/xyz")
    valid = _int_feature(features, "roadgraph_samples/valid")
    if xyz is None or valid is None:
        raise KeyError("Missing roadgraph_samples/xyz or roadgraph_samples/valid")

    xyz = xyz.reshape(-1, 3)
    valid = valid.astype(np.bool_)

    rg_type = _int_feature(features, "roadgraph_samples/type")
    rg_id = _int_feature(features, "roadgraph_samples/id")
    rg_dir = _float_feature(features, "roadgraph_samples/dir")
    if rg_dir is not None:
        rg_dir = rg_dir.reshape(-1, 3)

    speed_limit = _float_feature(features, "roadgraph_samples/speed_limit")

    # Optional sanity alignment (only if present)
    M = xyz.shape[0]
    if rg_type is not None and rg_type.shape[0] != M:
        raise ValueError(f"roadgraph_samples/type length mismatch: {rg_type.shape[0]} vs M={M}")
    if rg_id is not None and rg_id.shape[0] != M:
        raise ValueError(f"roadgraph_samples/id length mismatch: {rg_id.shape[0]} vs M={M}")
    if rg_dir is not None and rg_dir.shape[0] != M:
        raise ValueError(f"roadgraph_samples/dir length mismatch: {rg_dir.shape[0]} vs M={M}")
    if speed_limit is not None and speed_limit.shape[0] != M:
        raise ValueError(f"roadgraph_samples/speed_limit length mismatch: {speed_limit.shape[0]} vs M={M}")

    return RoadgraphSamples(
        xyz=xyz,
        valid=valid,
        rg_type=rg_type,
        rg_id=rg_id,
        rg_dir=rg_dir,
        speed_limit=speed_limit,
    )


# -------------------------
# Traffic light extraction
# -------------------------
@dataclass(frozen=True)
class TrafficLightSet:
    valid: np.ndarray      # (L,) bool
    x: np.ndarray          # (L,)
    y: np.ndarray          # (L,)
    state: Optional[np.ndarray]  # (L,) int64
    tl_id: Optional[np.ndarray]  # (L,) int64


def get_traffic_lights_current(features: Dict[str, tf.train.Feature]) -> TrafficLightSet:
    base = "traffic_light_state/current"
    valid = _int_feature(features, f"{base}/valid")
    x = _float_feature(features, f"{base}/x")
    y = _float_feature(features, f"{base}/y")
    if valid is None or x is None or y is None:
        raise KeyError("Missing traffic_light_state/current/{valid,x,y}")

    state = _int_feature(features, f"{base}/state")
    tl_id = _int_feature(features, f"{base}/id")

    return TrafficLightSet(
        valid=valid.astype(np.bool_),
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        state=state,
        tl_id=tl_id,
    )




def get_traffic_lights_time_major(
    features: Dict[str, tf.train.Feature],
    prefix: str,
    num_lights: int,
) -> Dict[str, np.ndarray]:
    """
    Returns time-major arrays for prefix in {'past','future'} when possible.
    Example keys:
      traffic_light_state/past/state  -> reshape (T, L)
    """
    assert prefix in ("past", "future")
    base = f"traffic_light_state/{prefix}"

    out: Dict[str, np.ndarray] = {}
    for name, getter in [
        ("valid", _int_feature),
        ("x", _float_feature),
        ("y", _float_feature),
        ("state", _int_feature),
        ("id", _int_feature),
    ]:
        arr = getter(features, f"{base}/{name}")
        if arr is None:
            continue
        if arr.size % num_lights != 0:
            continue
        T = arr.size // num_lights
        out[name] = arr.reshape(T, num_lights)

    return out


# -------------------------
# Path samples extraction (route prior)
# -------------------------
@dataclass(frozen=True)
class PathSamples:
    xyz: np.ndarray             # (P,3)
    valid: np.ndarray           # (P,) bool
    path_id: Optional[np.ndarray]      # (P,) int64
    arc_length: Optional[np.ndarray]   # (P,) float32


def get_path_samples(features: Dict[str, tf.train.Feature]) -> Optional[PathSamples]:
    """
    Extract path_samples/* if present.
    Some records may not contain these; return None in that case.
    """
    if not (has_key(features, "path_samples/xyz") and has_key(features, "path_samples/valid")):
        return None

    xyz = _float_feature(features, "path_samples/xyz")
    valid = _int_feature(features, "path_samples/valid")
    if xyz is None or valid is None:
        return None

    xyz = xyz.reshape(-1, 3)
    valid = valid.astype(np.bool_)

    path_id = _int_feature(features, "path_samples/id")
    arc = _float_feature(features, "path_samples/arc_length")

    P = xyz.shape[0]
    if path_id is not None and path_id.shape[0] != P:
        path_id = None
    if arc is not None and arc.shape[0] != P:
        arc = None

    return PathSamples(
        xyz=xyz,
        valid=valid,
        path_id=path_id,
        arc_length=arc,
    )