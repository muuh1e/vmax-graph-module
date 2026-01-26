"""
spatial_graph_builder.py

Phase 3: Build the final LLM-friendly Spatial Graph JSON from a TFExample record.

ENHANCEMENTS:
- Traffic light clustering (reduce redundancy)
- Semantic relevance filtering (keep only decision-relevant map features)
- Confidence scores for TL control heuristics
- Improved meta/debug statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from spatial_graph.utils.tfexample_io import (
    EgoState,
    RoadgraphSamples,
    TrafficLightSet,
    PathSamples,
    get_ego_state,
    get_roadgraph_samples,
    get_traffic_lights_current,
    get_traffic_lights_time_major,
    get_path_samples,
)

from spatial_graph.utils.geometry import ego_frame

from spatial_graph.modules.polyline_grouper import (
    PolylineGrouperConfig,
    MapFeature,
    build_map_features_from_roadgraph,
)

from spatial_graph.modules.route_monitor import (
    RouteMonitorConfig,
    RouteNode,
    EgoRouteEdge,
    build_route_node_and_edge,
    route_node_to_dict,
    ego_route_edge_to_dict,
)


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class SpatialGraphConfig:
    poly_cfg: PolylineGrouperConfig = PolylineGrouperConfig()
    route_cfg: RouteMonitorConfig = RouteMonitorConfig()

    # Traffic-light selection
    tl_radius_m: float = 80.0
    tl_topk: int = 20
    tl_cluster_dist_m: float = 1.0  # NEW: Cluster TLs within this distance

    # TL relevance heuristic (since we don't have TL↔lane association)
    tl_relevance_y_thresh_m: float = 6.0  # "near centerline" heuristic
    tl_relevance_min_x_m: float = 0.0     # only consider ahead by default

    # Map feature semantic filtering (NEW)
    map_max_features: int = 20            # Final cap (down from 80)
    map_safety_dist_m: float = 8.0        # Very close = always keep
    map_lane_proxy_align_thresh: float = 0.7  # cos(theta) > 0.7 (theta < 45°)
    map_lane_proxy_lateral_m: float = 5.0     # lateral offset for lane candidates

    # Edge bucketing
    dist_very_close_m: float = 5.0
    dist_close_m: float = 15.0
    dist_medium_m: float = 30.0

    # Speed limit overspeed margin (m/s)
    overspeed_margin_mps: float = 0.5


# -------------------------
# Helpers
# -------------------------
def _dist_bucket(d: float, cfg: SpatialGraphConfig) -> str:
    if d < cfg.dist_very_close_m:
        return "very_close"
    if d < cfg.dist_close_m:
        return "close"
    if d < cfg.dist_medium_m:
        return "medium"
    return "far"


def _node(node_id: str, node_type: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": node_id, "type": node_type, "attrs": attrs}


def _edge(src: str, dst: str, edge_type: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {"src": src, "dst": dst, "type": edge_type, "attrs": attrs}


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# -------------------------
# Traffic light clustering (NEW)
# -------------------------
def _cluster_traffic_lights(
    tl_x: np.ndarray,
    tl_y: np.ndarray,
    tl_idx: np.ndarray,
    tl_cur: TrafficLightSet,
    cluster_dist: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Cluster traffic lights that are very close together (< cluster_dist meters).
    
    Returns:
        List of TL cluster dicts, each containing:
        - representative_index: int (index in tl_cur arrays)
        - position: (x, y)
        - member_ids: list of tl_ids
        - member_states: list of states
        - representative_state: most common state
    """
    if len(tl_x) == 0:
        return []
    
    # Simple agglomerative clustering
    positions = np.stack([tl_x, tl_y], axis=1)  # (N, 2)
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(positions))
    
    # Find clusters (greedy approach)
    clusters = []
    used = set()
    
    for i in range(len(positions)):
        if i in used:
            continue
        
        # Find all TLs within cluster_dist
        close = np.where(distances[i] <= cluster_dist)[0]
        cluster_indices = [j for j in close if j not in used]
        
        if len(cluster_indices) == 0:
            continue
        
        # Mark as used
        used.update(cluster_indices)
        
        # Representative: use the one closest to cluster centroid
        cluster_pos = positions[cluster_indices]
        centroid = np.mean(cluster_pos, axis=0)
        dists_to_centroid = np.linalg.norm(cluster_pos - centroid, axis=1)
        rep_local = int(np.argmin(dists_to_centroid))
        rep_index = cluster_indices[rep_local]
        
        # Gather cluster info
        member_ids = []
        member_states = []
        for idx in cluster_indices:
            global_idx = tl_idx[idx]
            if tl_cur.tl_id is not None:
                member_ids.append(int(tl_cur.tl_id[global_idx]))
            else:
                member_ids.append(int(global_idx))
            if tl_cur.state is not None:
                member_states.append(int(tl_cur.state[global_idx]))
        
        # Representative state: most common
        if len(member_states) > 0:
            rep_state = int(np.median(member_states))  # Median for robustness
        else:
            rep_state = None
        
        clusters.append({
            "representative_index": rep_index,
            "position": (float(tl_x[rep_index]), float(tl_y[rep_index])),
            "member_ids": member_ids,
            "member_states": member_states,
            "representative_state": rep_state,
        })
    
    return clusters


# -------------------------
# TL control confidence (NEW)
# -------------------------
def _compute_tl_control_confidence(
    x_local: float,
    y_local: float,
    heading_alignment: Optional[float] = None,
) -> float:
    """
    Compute confidence that this TL controls ego's lane.
    
    Args:
        x_local: Forward distance in ego frame
        y_local: Lateral distance in ego frame
        heading_alignment: cos(theta) between ego and TL direction (if available)
    
    Returns:
        Confidence in [0, 1]
    """
    # Ahead factor (binary: 1 if ahead, 0 otherwise)
    ahead_conf = 1.0 if x_local > 0 else 0.0
    
    # Lateral centering factor (drops linearly to 0 at 6m)
    lat_conf = max(0.0, 1.0 - abs(y_local) / 6.0)
    
    # Heading alignment factor (optional, if we have TL direction)
    align_conf = 1.0
    if heading_alignment is not None:
        # Map [-1, 1] to [0, 1]: aligned=1, opposite=0
        align_conf = (heading_alignment + 1.0) / 2.0
    
    # Combined confidence (product)
    return ahead_conf * lat_conf * align_conf


# -------------------------
# Map feature relevance filtering (NEW)
# -------------------------
def _filter_map_features_semantic(
    map_features: List[MapFeature],
    ego: EgoState,
    route_edge: Optional[EgoRouteEdge],
    tl_polyline_ids: Set[int],
    cfg: SpatialGraphConfig,
) -> Tuple[List[MapFeature], Dict[str, int]]:
    """
    Intelligent filtering to keep only decision-relevant map features.
    
    Priority:
    1. Safety-critical: very close (< 8m)
    2. Lane-proxy: aligned + ahead + small lateral offset
    3. TL-relevant: polylines associated with traffic lights
    4. Forward context: ahead and not too far
    
    Returns:
        (filtered_features, debug_stats)
    """
    if len(map_features) <= cfg.map_max_features:
        return map_features, {"total": len(map_features), "kept_all": True}
    
    # Categorize
    safety = []
    lane_candidates = []
    tl_lanes = []
    forward_context = []
    others = []
    
    for mf in map_features:
        # Category 1: Safety-critical (very close)
        if mf.closest_dist_m < cfg.map_safety_dist_m:
            safety.append(mf)
            continue
        
        # Category 2: Lane-proxy candidates
        # - Aligned with ego heading (cos > 0.7 means angle < 45°)
        # - Ahead of ego
        # - Small lateral offset (likely in same lane)
        if (mf.heading_alignment_cos is not None and
            mf.heading_alignment_cos > cfg.map_lane_proxy_align_thresh and
            mf.closest_x_local_m > 0 and
            abs(mf.closest_y_local_m) < cfg.map_lane_proxy_lateral_m):
            lane_candidates.append(mf)
            continue
        
        # Category 3: TL-relevant lanes
        if mf.polyline_id in tl_polyline_ids:
            tl_lanes.append(mf)
            continue
        
        # Category 4: Forward context (ahead but not categorized above)
        if mf.closest_x_local_m > 0:
            forward_context.append(mf)
            continue
        
        # Category 5: Everything else
        others.append(mf)
    
    # Assemble with priority
    result = []
    
    # Always include safety-critical
    result.extend(sorted(safety, key=lambda m: m.closest_dist_m))
    remaining = cfg.map_max_features - len(result)
    
    # Include lane candidates (sorted by distance)
    if remaining > 0:
        budget = max(1, remaining // 2)  # At least half the remaining budget
        result.extend(sorted(lane_candidates, key=lambda m: m.closest_dist_m)[:budget])
        remaining = cfg.map_max_features - len(result)
    
    # Include TL lanes
    if remaining > 0:
        budget = max(1, remaining // 2)
        result.extend(sorted(tl_lanes, key=lambda m: m.closest_dist_m)[:budget])
        remaining = cfg.map_max_features - len(result)
    
    # Include forward context
    if remaining > 0:
        result.extend(sorted(forward_context, key=lambda m: m.closest_dist_m)[:remaining])
        remaining = cfg.map_max_features - len(result)
    
    # Fill with others if we still have room
    if remaining > 0:
        result.extend(sorted(others, key=lambda m: m.closest_dist_m)[:remaining])
    
    # Debug statistics
    debug_stats = {
        "total_input": len(map_features),
        "safety": len(safety),
        "lane_candidates": len(lane_candidates),
        "tl_lanes": len(tl_lanes),
        "forward_context": len(forward_context),
        "others": len(others),
        "kept": len(result),
    }
    
    return result[:cfg.map_max_features], debug_stats


# -------------------------
# Traffic light future features
# -------------------------
def _compute_tl_future_features(
    tl_index: int,
    cur_state: Optional[int],
    tl_future: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Compute future-change features WITHOUT assuming enum meaning.
    We report:
      - future_unique_states_valid
      - time_to_change_steps (first step where state != cur_state)
      - time_to_first_seen_state_steps (for each unique state)
    """
    if "state" not in tl_future or "valid" not in tl_future:
        return {}

    state = tl_future["state"]  # (T, L)
    valid = tl_future["valid"]  # (T, L) int/float -> we'll treat >0 as valid

    if tl_index < 0 or tl_index >= state.shape[1]:
        return {}

    s = state[:, tl_index].astype(np.int64)
    v = valid[:, tl_index].astype(np.int64) > 0

    s_valid = s[v]
    if s_valid.size == 0:
        return {}

    uniq = np.unique(s_valid).tolist()

    # Time to change from current
    time_to_change = None
    if cur_state is not None:
        diff = (s != int(cur_state)) & v
        where = np.where(diff)[0]
        if where.size > 0:
            time_to_change = int(where[0])

    # Time to first occurrence of each state (valid only)
    first_seen: Dict[int, int] = {}
    for st in uniq:
        where = np.where((s == st) & v)[0]
        if where.size > 0:
            first_seen[int(st)] = int(where[0])

    return {
        "future_unique_states_valid": uniq,
        "time_to_change_steps": time_to_change,
        "time_to_first_seen_state_steps": first_seen,
        "future_horizon_steps": int(state.shape[0]),
    }


# -------------------------
# Main builder
# -------------------------
def build_spatial_graph_from_example(
    example: Any,  # tf.train.Example
    record_index: Optional[int] = None,
    cfg: SpatialGraphConfig = SpatialGraphConfig(),
) -> Dict[str, Any]:
    """
    Build the spatial graph JSON from one TFExample record.

    Returns:
      dict with keys: meta, nodes, edges
    """
    features = example.features.feature

    # ---- Extract core data ----
    ego: EgoState = get_ego_state(features)
    rg: RoadgraphSamples = get_roadgraph_samples(features)
    tl_cur: TrafficLightSet = get_traffic_lights_current(features)
    path: Optional[PathSamples] = get_path_samples(features)

    # Time-major TL sequences for future/past (used for change features)
    num_lights = int(tl_cur.valid.size)
    tl_future = get_traffic_lights_time_major(features, "future", num_lights=num_lights)
    tl_past = get_traffic_lights_time_major(features, "past", num_lights=num_lights)

    # ---- Phase 1: map features (polylines) ----
    map_features_raw: List[MapFeature] = build_map_features_from_roadgraph(rg=rg, ego=ego, cfg=cfg.poly_cfg)

    # ---- Phase 2: route ----
    route_pack = build_route_node_and_edge(ego=ego, path=path, cfg=cfg.route_cfg)
    route_node: Optional[RouteNode] = None
    route_edge: Optional[EgoRouteEdge] = None
    if route_pack is not None:
        route_node, route_edge = route_pack

    # ---- Phase 3: Traffic light clustering ----
    tl_valid_idx = np.where(tl_cur.valid.astype(bool))[0]
    tl_clusters = []
    
    if tl_valid_idx.size > 0:
        tl_x = tl_cur.x[tl_valid_idx]
        tl_y = tl_cur.y[tl_valid_idx]

        dx = tl_x - float(ego.x)
        dy = tl_y - float(ego.y)
        dist = np.sqrt(dx * dx + dy * dy)

        # Radius filter
        in_rad = dist <= float(cfg.tl_radius_m)
        idx2 = tl_valid_idx[in_rad]
        dist2 = dist[in_rad]
        dx2 = dx[in_rad]
        dy2 = dy[in_rad]

        # Ego frame for relevance
        x_local, y_local = ego_frame(dx2.astype(np.float32), dy2.astype(np.float32), float(ego.yaw))

        # Top-k closest
        order = np.argsort(dist2)[: min(int(cfg.tl_topk), int(idx2.size))]
        idx_pick = idx2[order]
        
        # Cluster the selected TLs
        tl_x_pick = tl_cur.x[idx_pick]
        tl_y_pick = tl_cur.y[idx_pick]
        
        tl_clusters = _cluster_traffic_lights(
            tl_x=tl_x_pick,
            tl_y=tl_y_pick,
            tl_idx=idx_pick,
            tl_cur=tl_cur,
            cluster_dist=cfg.tl_cluster_dist_m,
        )

    # Collect TL polyline IDs (for map filtering)
    tl_polyline_ids: Set[int] = set()
    # Note: We don't have explicit TL→lane association, so this set remains empty
    # In a future version, you could infer this from spatial proximity

    # ---- Phase 4: Semantic map feature filtering ----
    map_features, filter_stats = _filter_map_features_semantic(
        map_features=map_features_raw,
        ego=ego,
        route_edge=route_edge,
        tl_polyline_ids=tl_polyline_ids,
        cfg=cfg,
    )

    # ---- Build nodes ----
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    ego_id = "ego"
    nodes.append(
        _node(
            node_id=ego_id,
            node_type="EGO",
            attrs={
                "x": float(ego.x),
                "y": float(ego.y),
                "yaw": float(ego.yaw),
                "speed_mps": float(ego.speed),
                "speed_source": ego.speed_source,
            },
        )
    )

    # Map feature nodes + ego->map edges
    for mf in map_features:
        nid = f"map_{mf.polyline_id}"
        nodes.append(
            _node(
                node_id=nid,
                node_type="MAP_FEATURE",
                attrs={
                    "polyline_id": int(mf.polyline_id),
                    "type_mode": mf.type_mode,
                    "type_name": mf.type_name,  # NEW: Human-readable
                    "type_category": mf.type_category,  # NEW: Semantic category
                    "num_points": int(mf.num_points),
                    "centroid_xy": list(mf.centroid_xy),
                    "bbox_xy_min": list(mf.bbox_xy_min),
                    "bbox_xy_max": list(mf.bbox_xy_max),
                    "speed_limit_mps": mf.speed_limit_mps,
                    "heading_alignment_cos": mf.heading_alignment_cos,
                    # Keep samples small (debug/heuristics)
                    "sample_points_xy": mf.sample_points_xy,
                },
            )
        )

        overspeeding = None
        if mf.speed_limit_mps is not None:
            overspeeding = bool(ego.speed > (mf.speed_limit_mps + cfg.overspeed_margin_mps))

        edges.append(
            _edge(
                src=ego_id,
                dst=nid,
                edge_type="EGO_TO_MAP",
                attrs={
                    "closest_dist_m": float(mf.closest_dist_m),
                    "closest_world_xy": list(mf.closest_world_xy),
                    "x_local_m": float(mf.closest_x_local_m),
                    "y_local_m": float(mf.closest_y_local_m),
                    "distance_bucket": _dist_bucket(float(mf.closest_dist_m), cfg),
                    "ahead": bool(mf.closest_x_local_m > 0.0),
                    "left": bool(mf.closest_y_local_m > 0.0),
                    "overspeeding": overspeeding,
                },
            )
        )

    # Traffic light cluster nodes + ego->TL edges
    for cluster in tl_clusters:
        rep_idx = cluster["representative_index"]
        # The representative_index is in the "picked" array, need to map back
        tl_idx = idx_pick[rep_idx]
        
        tid_list = cluster["member_ids"]
        tid = tid_list[0] if len(tid_list) > 0 else tl_idx  # Use first member ID
        nid = f"tl_cluster_{tid}"

        # Compute ego-local position for this cluster
        tl_x_world, tl_y_world = cluster["position"]
        dx_tl = tl_x_world - ego.x
        dy_tl = tl_y_world - ego.y
        dist_tl = np.sqrt(dx_tl**2 + dy_tl**2)
        x_loc_tl, y_loc_tl = ego_frame(
            np.array([dx_tl], dtype=np.float32),
            np.array([dy_tl], dtype=np.float32),
            ego.yaw
        )
        
        # Relevance heuristic
        ahead = bool(x_loc_tl[0] >= cfg.tl_relevance_min_x_m)
        centered = bool(abs(y_loc_tl[0]) <= cfg.tl_relevance_y_thresh_m)
        likely_controls_ego = bool(ahead and centered)
        
        # Confidence score
        control_confidence = _compute_tl_control_confidence(
            x_local=float(x_loc_tl[0]),
            y_local=float(y_loc_tl[0]),
            heading_alignment=None,  # Could add TL direction if available
        )

        # Node
        nodes.append(
            _node(
                node_id=nid,
                node_type="TRAFFIC_LIGHT_CLUSTER",
                attrs={
                    "cluster_id": int(tid),
                    "member_tl_ids": tid_list,
                    "member_states": cluster["member_states"],
                    "representative_state": cluster["representative_state"],
                    "x": tl_x_world,
                    "y": tl_y_world,
                    "num_members": len(tid_list),
                },
            )
        )

        # Future features for representative TL
        fut_feats = _compute_tl_future_features(
            tl_index=tl_idx,
            cur_state=cluster["representative_state"],
            tl_future=tl_future,
        )

        # Edge
        edges.append(
            _edge(
                src=ego_id,
                dst=nid,
                edge_type="EGO_TO_TL",
                attrs={
                    "dist_m": float(dist_tl),
                    "x_local_m": float(x_loc_tl[0]),
                    "y_local_m": float(y_loc_tl[0]),
                    "ahead": ahead,
                    "likely_controls_ego": likely_controls_ego,
                    "control_confidence": float(control_confidence),  # NEW: Confidence score
                    "distance_bucket": _dist_bucket(float(dist_tl), cfg),
                    **fut_feats,
                },
            )
        )

    # Route node + ego->route edge
    if route_node is not None and route_edge is not None:
        rid = int(route_node.route_id)
        route_id = f"route_{rid}"
        nodes.append(_node(node_id=route_id, node_type="ROUTE", attrs=route_node_to_dict(route_node)))
        
        # Add semantic tags to route edge
        route_attrs = ego_route_edge_to_dict(route_edge)
        route_attrs["is_following_route"] = route_edge.status == "ON_TRACK"
        route_attrs["is_drifting_left"] = route_edge.signed_lateral_m > 0.5
        route_attrs["is_drifting_right"] = route_edge.signed_lateral_m < -0.5
        
        edges.append(
            _edge(
                src=ego_id,
                dst=route_id,
                edge_type="EGO_TO_ROUTE",
                attrs=route_attrs,
            )
        )

    # ---- Meta ----
    meta = {
        "record_index": int(record_index) if record_index is not None else None,
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "map_features": {
            "raw_count": len(map_features_raw),
            "filtered_count": len(map_features),
            "filter_stats": filter_stats,
        },
        "traffic_lights": {
            "total_valid": int(tl_valid_idx.size),
            "within_radius": int(idx2.size) if tl_valid_idx.size > 0 else 0,
            "num_clusters": len(tl_clusters),
        },
        "limits": {
            "max_polylines_raw": int(cfg.poly_cfg.max_polylines),
            "max_map_features_filtered": int(cfg.map_max_features),
            "tl_topk": int(cfg.tl_topk),
            "tl_cluster_dist_m": float(cfg.tl_cluster_dist_m),
        },
        "notes": {
            "no_lane_topology_in_tfexample": True,
            "no_tl_lane_control_in_tfexample": True,
            "tl_relevance_is_heuristic": True,
            "tl_clustering_enabled": True,
            "semantic_map_filtering_enabled": True,
        },
    }

    return {"meta": meta, "nodes": nodes, "edges": edges}