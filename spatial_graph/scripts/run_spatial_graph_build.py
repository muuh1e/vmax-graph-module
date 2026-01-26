#!/usr/bin/env python3
"""
run_spatial_graph_build.py

CLI entrypoint to build a spatial graph JSON from a single TFExample record.

Examples:
python scripts/run_spatial_graph_build.py \
  --tfrecord /home/med1e/vmax/data/waymo_converted/training.tfrecord \
  --record 0 \
  --out /home/med1e/vmax/V-Max/spatial_graph/outputs/record0_spatial_graph.json

Print to stdout:
python scripts/run_spatial_graph_build.py --tfrecord ... --record 0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from spatial_graph.utils.tfexample_io import read_example_from_tfrecord
from spatial_graph.modules.spatial_graph_builder import SpatialGraphConfig, build_spatial_graph_from_example
from spatial_graph.modules.polyline_grouper import PolylineGrouperConfig
from spatial_graph.modules.route_monitor import RouteMonitorConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True, help="Path to TFRecord (ScenarioMax-converted TFExample file).")
    ap.add_argument("--record", type=int, default=0, help="Record index inside the TFRecord.")
    ap.add_argument("--out", default="", help="Optional output JSON path (if empty, prints to stdout).")

    # polyline grouper
    ap.add_argument("--max_polylines", type=int, default=80)
    ap.add_argument("--max_sample_points", type=int, default=40)
    ap.add_argument("--no_vmax_box", action="store_true", help="Disable V-Max style roadgraph box filter.")
    ap.add_argument("--box_fwd", type=float, default=50.0)
    ap.add_argument("--box_back", type=float, default=5.0)
    ap.add_argument("--box_side", type=float, default=20.0)

    # traffic lights
    ap.add_argument("--tl_radius", type=float, default=80.0)
    ap.add_argument("--tl_topk", type=int, default=20)
    ap.add_argument("--tl_y_thresh", type=float, default=6.0)
    ap.add_argument("--tl_min_x", type=float, default=0.0)

    # route status thresholds
    ap.add_argument("--on_track_lat", type=float, default=1.5)
    ap.add_argument("--deviating_lat", type=float, default=4.0)
    ap.add_argument("--off_route_dist", type=float, default=10.0)

    args = ap.parse_args()

    ex = read_example_from_tfrecord(args.tfrecord, args.record)

    poly_cfg = PolylineGrouperConfig(
        use_vmax_box=not args.no_vmax_box,
        box_fwd_m=float(args.box_fwd),
        box_back_m=float(args.box_back),
        box_side_m=float(args.box_side),
        max_polylines=int(args.max_polylines),
        max_sample_points_per_polyline=int(args.max_sample_points),
    )

    route_cfg = RouteMonitorConfig(
        on_track_lat_m=float(args.on_track_lat),
        deviating_lat_m=float(args.deviating_lat),
        off_route_dist_m=float(args.off_route_dist),
        route_selection="closest",
    )

    cfg = SpatialGraphConfig(
        poly_cfg=poly_cfg,
        route_cfg=route_cfg,
        tl_radius_m=float(args.tl_radius),
        tl_topk=int(args.tl_topk),
        tl_relevance_y_thresh_m=float(args.tl_y_thresh),
        tl_relevance_min_x_m=float(args.tl_min_x),
    )

    graph = build_spatial_graph_from_example(example=ex, record_index=int(args.record), cfg=cfg)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(graph, fp, indent=2)
        print(f"Saved spatial graph JSON -> {args.out}")
    else:
        print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()
