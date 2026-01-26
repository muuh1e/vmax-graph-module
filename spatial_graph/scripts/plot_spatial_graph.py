#!/usr/bin/env python3
"""
Pretty plot for spatial_graph JSON (ego-local view).

Usage:
  cd ~/vmax/V-Max
  python -m spatial_graph.scripts.plot_spatial_graph \
    --graph spatial_graph/outputs/record100_spatial_graph.json \
    --out spatial_graph/outputs/record100_plot.png
"""

from __future__ import annotations
import argparse
import json
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def world_to_ego_xy(xs: np.ndarray, ys: np.ndarray, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """World -> ego frame. Ego frame: x forward, y left."""
    dx = xs - ego_x
    dy = ys - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


def is_world_coords(points_xy: List[List[float]]) -> bool:
    """Heuristic: world coords are usually large magnitude (thousands)."""
    if not points_xy:
        return False
    arr = np.asarray(points_xy, dtype=np.float32)
    return bool(np.max(np.abs(arr)) > 200.0)


def get_node(nodes: List[Dict[str, Any]], node_type: str) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if n.get("type") == node_type:
            return n
    return None


def cluster_points_xy(points: List[Tuple[float, float, Dict[str, Any]]], grid_m: float) -> Dict[Tuple[int, int], List[Tuple[float, float, Dict[str, Any]]]]:
    def key(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / grid_m)), int(round(y / grid_m)))
    clusters: Dict[Tuple[int, int], List[Tuple[float, float, Dict[str, Any]]]] = {}
    for x, y, a in points:
        k = key(x, y)
        clusters.setdefault(k, []).append((x, y, a))
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="Path to spatial_graph JSON.")
    ap.add_argument("--out", default="", help="Optional output image path. If empty, show interactively.")
    ap.add_argument("--title", default="", help="Optional title override.")
    ap.add_argument("--pad", type=float, default=6.0, help="Plot padding in meters.")
    ap.add_argument("--max_map_polylines", type=int, default=9999)
    ap.add_argument("--max_tl_labels", type=int, default=8)
    ap.add_argument("--tl_grid", type=float, default=0.75, help="TL clustering grid (ego-local meters).")
    ap.add_argument("--draw_all_edges", action="store_true", help="Draw all edges (can be cluttered).")
    ap.add_argument("--no_labels", action="store_true")
    args = ap.parse_args()

    with open(args.graph, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes = g["nodes"]
    edges = g["edges"]
    meta = g.get("meta", {})

    ego_node = get_node(nodes, "EGO")
    if ego_node is None:
        raise RuntimeError("No EGO node found")
    ego = ego_node["attrs"]
    ego_x = float(ego["x"])
    ego_y = float(ego["y"])
    ego_yaw = float(ego.get("yaw", 0.0))
    ego_speed = float(ego.get("speed_mps", 0.0))
    ego_speed_src = ego.get("speed_source", None)

    # Build edge lookup by dst for MAP features (so we can style polylines)
    map_edge_by_dst: Dict[str, Dict[str, Any]] = {}
    for e in edges:
        if e.get("type") == "EGO_TO_MAP":
            map_edge_by_dst[e["dst"]] = e.get("attrs", {})

    # ---- MAP polylines ----
    map_nodes = [n for n in nodes if n.get("type") == "MAP_FEATURE"]
    map_nodes = map_nodes[: min(len(map_nodes), int(args.max_map_polylines))]

    plotted_x: List[np.ndarray] = []
    plotted_y: List[np.ndarray] = []

    for mn in map_nodes:
        nid = mn.get("id")
        attrs = mn.get("attrs", {})
        pts = attrs.get("sample_points_xy", [])
        if not pts:
            continue

        pts_arr = np.asarray(pts, dtype=np.float32)
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]

        if is_world_coords(pts):
            xl, yl = world_to_ego_xy(xs, ys, ego_x, ego_y, ego_yaw)
        else:
            xl, yl = xs, ys

        # styling by edge semantics
        eattrs = map_edge_by_dst.get(nid, {})
        hazard = bool(eattrs.get("near_lateral_hazard", False))
        align = attrs.get("heading_alignment_cos", None)
        is_lane_proxy = bool(align is not None and float(align) >= 0.7)

        lw = 1.0
        alpha = 0.75
        if is_lane_proxy:
            lw = 1.8
            alpha = 0.85
        if hazard:
            lw = 2.8
            alpha = 0.95

        plt.plot(xl, yl, linewidth=lw, alpha=alpha)
        plotted_x.append(np.asarray(xl))
        plotted_y.append(np.asarray(yl))

    # ---- ROUTE: use route edge closest point ----
    route_edge = next((e for e in edges if e.get("type") == "EGO_TO_ROUTE"), None)
    if route_edge is not None:
        ra = route_edge.get("attrs", {})
        cxy = ra.get("closest_world_xy", None)
        if isinstance(cxy, list) and len(cxy) == 2:
            xw, yw = float(cxy[0]), float(cxy[1])
            xl, yl = world_to_ego_xy(np.array([xw]), np.array([yw]), ego_x, ego_y, ego_yaw)
            rx, ry = float(xl[0]), float(yl[0])

            plt.scatter([rx], [ry], marker="x", s=140)
            plt.plot([0.0, rx], [0.0, ry], linewidth=1.2, alpha=0.6)

            plotted_x.append(np.array([rx]))
            plotted_y.append(np.array([ry]))

            if not args.no_labels:
                status = ra.get("status", "")
                lat = ra.get("signed_lateral_m", None)
                lab = f"ROUTE {status}"
                if lat is not None:
                    lab += f"\nlat={float(lat):+.2f}m"
                # move label away from center
                plt.annotate(lab, (rx, ry), textcoords="offset points", xytext=(12, -18))

    # ---- Traffic lights: cluster from EGO_TO_TL edges (already in ego-local coords) ----
    tl_points: List[Tuple[float, float, Dict[str, Any]]] = []
    for e in edges:
        if e.get("type") != "EGO_TO_TL":
            continue
        a = e.get("attrs", {})
        if "x_local_m" not in a or "y_local_m" not in a:
            continue
        tl_points.append((float(a["x_local_m"]), float(a["y_local_m"]), a))

    tl_clusters = cluster_points_xy(tl_points, grid_m=float(args.tl_grid))

    # Plot cluster markers
    if tl_clusters:
        cx = []
        cy = []
        sizes = []
        clist = []
        for _, items in tl_clusters.items():
            # representative = nearest (min dist)
            rep = min(items, key=lambda it: float(it[2].get("dist_m", 1e9)))
            x, y, _ = rep
            cx.append(x)
            cy.append(y)
            sizes.append(60 + 25 * (len(items) - 1))
            clist.append(items)

        plt.scatter(np.array(cx), np.array(cy), marker="o", s=np.array(sizes))
        plotted_x.append(np.asarray(cx))
        plotted_y.append(np.asarray(cy))

        # labels (top priority clusters)
        if not args.no_labels:
            def cl_priority(items):
                # any must_stop > any stop_nearby > any ctrl > any known state
                must = any(bool(it[2].get("must_stop", False)) for it in items)
                near = any(bool(it[2].get("stop_signal_nearby", False)) for it in items)
                ctrl = any(bool(it[2].get("likely_controls_ego", False)) for it in items)
                known = any(str(it[2].get("state_label", "UNKNOWN")) != "UNKNOWN" for it in items)
                return (must, near, ctrl, known, len(items))

            sorted_clusters = sorted(clist, key=cl_priority, reverse=True)[: int(args.max_tl_labels)]
            for idx, items in enumerate(sorted_clusters):
                rep = min(items, key=lambda it: float(it[2].get("dist_m", 1e9)))
                x, y, a = rep

                must = any(bool(it[2].get("must_stop", False)) for it in items)
                near = any(bool(it[2].get("stop_signal_nearby", False)) for it in items)
                ctrl = any(bool(it[2].get("likely_controls_ego", False)) for it in items)

                # pick one label (prefer STOP/GO if present)
                labels = [str(it[2].get("state_label", "UNKNOWN")) for it in items]
                label = "UNKNOWN"
                if "STOP" in labels:
                    label = "STOP"
                elif "GO" in labels:
                    label = "GO"

                tags = []
                if must:
                    tags.append("MUST_STOP")
                elif near:
                    tags.append("STOP_NEARBY")
                if ctrl:
                    tags.append("CTRL")
                txt = f"TL {label}"
                if tags:
                    txt += " [" + ",".join(tags) + "]"
                if len(items) > 1:
                    txt += f"\n(n={len(items)})"

                # stagger label offsets to reduce overlap
                plt.annotate(txt, (x, y), textcoords="offset points", xytext=(8, 8 + 14 * idx))

    # ---- Ego marker ----
    ego_tri = np.array([[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0], [2.0, 0.0]], dtype=np.float32)
    plt.plot(ego_tri[:, 0], ego_tri[:, 1], linewidth=2.2)
    plt.scatter([0.0], [0.0], s=45)

    # ---- Edges (reduced clutter by default) ----
    if args.draw_all_edges:
        for e in edges:
            t = e.get("type")
            a = e.get("attrs", {})
            if t in ("EGO_TO_MAP", "EGO_TO_TL") and ("x_local_m" in a and "y_local_m" in a):
                plt.plot([0.0, float(a["x_local_m"])], [0.0, float(a["y_local_m"])], linewidth=0.8, alpha=0.25)
    else:
        # Only draw "important" edges:
        # - route edge already drawn
        for e in edges:
            t = e.get("type")
            a = e.get("attrs", {})
            if t == "EGO_TO_MAP":
                if bool(a.get("near_lateral_hazard", False)):
                    plt.plot([0.0, float(a["x_local_m"])], [0.0, float(a["y_local_m"])], linewidth=1.0, alpha=0.35)
            elif t == "EGO_TO_TL":
                if bool(a.get("must_stop", False)) or bool(a.get("stop_signal_nearby", False)):
                    plt.plot([0.0, float(a["x_local_m"])], [0.0, float(a["y_local_m"])], linewidth=1.0, alpha=0.35)

    # ---- View limits ----
    allx = np.concatenate([np.ravel(x) for x in plotted_x]) if plotted_x else np.array([0.0])
    ally = np.concatenate([np.ravel(y) for y in plotted_y]) if plotted_y else np.array([0.0])

    pad = float(args.pad)
    xmin, xmax = float(allx.min()) - pad, float(allx.max()) + pad
    ymin, ymax = float(ally.min()) - pad, float(ally.max()) + pad

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.25)

    rid = meta.get("record_index", None)
    title = args.title.strip() or (f"Spatial Graph (record={rid})" if rid is not None else "Spatial Graph")
    plt.title(title)
    plt.xlabel("ego-local x (m, forward)")
    plt.ylabel("ego-local y (m, left)")

    footer = f"ego speed={ego_speed:.2f} m/s"
    if ego_speed_src is not None:
        footer += f" ({ego_speed_src})"
    plt.suptitle(footer, y=0.98, fontsize=10)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved plot -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
