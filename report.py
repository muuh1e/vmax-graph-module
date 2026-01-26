#!/usr/bin/env python3
import os
import glob
import json
import math
import statistics
from collections import Counter, defaultdict

OUT_DIR = "spatial_graph/outputs"
PATTERN = os.path.join(OUT_DIR, "record*_spatial_graph.json")

def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def _safe_median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None

def _pct(a, b):
    return (100.0 * a / b) if b else None

def _is_finite(x):
    return x is None or (isinstance(x, (int, float)) and math.isfinite(float(x)))

def check_graph(data, fname):
    """Return list of issues found in a graph JSON."""
    issues = []

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    meta = data.get("meta", {})

    if not isinstance(nodes, list) or not isinstance(edges, list):
        issues.append("nodes/edges not lists")
        return issues

    # Required node types
    node_types = [n.get("type") for n in nodes]
    if "EGO" not in node_types:
        issues.append("missing EGO node")
    if "ROUTE" not in node_types:
        issues.append("missing ROUTE node")

    # Basic meta sanity
    if "num_nodes" in meta and meta["num_nodes"] != len(nodes):
        issues.append(f"meta.num_nodes mismatch ({meta['num_nodes']} vs {len(nodes)})")
    if "num_edges" in meta and meta["num_edges"] != len(edges):
        issues.append(f"meta.num_edges mismatch ({meta['num_edges']} vs {len(edges)})")

    # Check ego attrs
    ego = next((n for n in nodes if n.get("type") == "EGO"), None)
    if ego:
        ea = ego.get("attrs", {})
        for k in ["x", "y", "yaw", "speed_mps"]:
            if k in ea and not _is_finite(ea[k]):
                issues.append(f"ego.{k} not finite")
        if "speed_mps" in ea and float(ea["speed_mps"]) < 0:
            issues.append("ego.speed_mps negative")

    # Route edge
    route_edge = next((e for e in edges if e.get("type") == "EGO_TO_ROUTE"), None)
    if route_edge is None:
        issues.append("missing EGO_TO_ROUTE edge")
    else:
        ra = route_edge.get("attrs", {})
        if "signed_lateral_m" in ra and not _is_finite(ra["signed_lateral_m"]):
            issues.append("route.signed_lateral_m not finite")
        if "status" not in ra:
            issues.append("route.status missing")

    # TL edges sanity
    for e in edges:
        if e.get("type") == "EGO_TO_TL":
            a = e.get("attrs", {})
            for k in ["dist_m", "x_local_m", "y_local_m"]:
                if k in a and not _is_finite(a[k]):
                    issues.append(f"TL edge attr {k} not finite")
            if "control_confidence" in a:
                cc = a["control_confidence"]
                if not _is_finite(cc):
                    issues.append("TL control_confidence not finite")
                else:
                    if float(cc) < 0 or float(cc) > 1.0:
                        issues.append("TL control_confidence out of [0,1]")

    # MAP edges sanity (optional)
    for e in edges:
        if e.get("type") == "EGO_TO_MAP":
            a = e.get("attrs", {})
            for k in ["dist_m", "x_local_m", "y_local_m"]:
                if k in a and not _is_finite(a[k]):
                    issues.append(f"MAP edge attr {k} not finite")

    return issues

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(PATTERN))

    if not files:
        print(f"❌ No files found: {PATTERN}")
        return

    # Aggregates
    num_nodes = []
    num_edges = []
    map_raw = []
    map_filt = []
    map_reduction_pct = []
    map_safety = []
    map_lane = []
    map_forward = []

    tl_total = []
    tl_clusters = []
    tl_cluster_reduction_pct = []
    tl_conf_all = []
    tl_must_stop_counts = []
    tl_stop_nearby_counts = []

    node_type_counter = Counter()
    map_type_name_counter = Counter()

    # Readiness checks
    bad = []  # (file, issues)

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        meta = data.get("meta", {})
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Collect basic counts
        num_nodes.append(meta.get("num_nodes", len(nodes)))
        num_edges.append(meta.get("num_edges", len(edges)))
        for n in nodes:
            nt = n.get("type")
            if nt:
                node_type_counter[nt] += 1
            if nt == "MAP_FEATURE":
                tn = n.get("attrs", {}).get("type_name")
                if tn:
                    map_type_name_counter[tn] += 1

        # Map stats (newer meta format)
        mf = meta.get("map_features") or meta.get("map_filter") or meta.get("debug", {}).get("map_filter")
        # You used meta.map_features.{raw_count, filtered_count}
        if isinstance(meta.get("map_features"), dict):
            mf = meta["map_features"]
            raw = mf.get("raw_count")
            filt = mf.get("filtered_count")
            fs = mf.get("filter_stats", {})
        else:
            # fallback older debug shape
            raw = meta.get("num_map_candidates") or (mf.get("candidates_in") if isinstance(mf, dict) else None)
            filt = meta.get("num_map_features") or (mf.get("final_kept") if isinstance(mf, dict) else None)
            fs = mf if isinstance(mf, dict) else {}

        if raw is not None and filt is not None:
            map_raw.append(int(raw))
            map_filt.append(int(filt))
            if int(raw) > 0:
                map_reduction_pct.append(100.0 * (1.0 - (int(filt) / int(raw))))

        # filter stats if present
        if isinstance(fs, dict):
            map_safety.append(fs.get("safety"))
            map_lane.append(fs.get("lane_candidates"))
            map_forward.append(fs.get("forward_context"))

        # TL stats
        tlm = meta.get("traffic_lights") or meta.get("debug", {}).get("traffic_lights")
        # TL sanity (per record, avoids list misalignment)
        if isinstance(tlm, dict):
            total = tlm.get("total_valid") or 0
            within = tlm.get("within_radius") or 0
            clusters = tlm.get("num_clusters") or 0

            if clusters > within or within > total:
                bad.append((os.path.basename(f), [f"TL meta inconsistent total={total} within={within} clusters={clusters}"]))


        # Count TL stop flags + gather confidence
        must_stop = 0
        stop_nearby = 0
        for e in edges:
            if e.get("type") == "EGO_TO_TL":
                a = e.get("attrs", {})
                if "control_confidence" in a and a["control_confidence"] is not None:
                    try:
                        tl_conf_all.append(float(a["control_confidence"]))
                    except Exception:
                        pass
                if a.get("must_stop") is True:
                    must_stop += 1
                if a.get("stop_signal_nearby") is True:
                    stop_nearby += 1

        tl_must_stop_counts.append(must_stop)
        tl_stop_nearby_counts.append(stop_nearby)

        # Validate
        issues = check_graph(data, os.path.basename(f))
        if issues:
            bad.append((os.path.basename(f), issues))

    # Summary statistics
    summary = {
        "num_files": len(files),
        "nodes": {
            "min": min(num_nodes),
            "max": max(num_nodes),
            "mean": _safe_mean(num_nodes),
            "median": _safe_median(num_nodes),
        },
        "edges": {
            "min": min(num_edges),
            "max": max(num_edges),
            "mean": _safe_mean(num_edges),
            "median": _safe_median(num_edges),
        },
        "map": {
            "raw_mean": _safe_mean(map_raw),
            "filtered_mean": _safe_mean(map_filt),
            "reduction_pct_mean": _safe_mean(map_reduction_pct),
            "reduction_pct_median": _safe_median(map_reduction_pct),
            "safety_mean": _safe_mean([x for x in map_safety if x is not None]),
            "lane_mean": _safe_mean([x for x in map_lane if x is not None]),
            "forward_mean": _safe_mean([x for x in map_forward if x is not None]),
        },
        "traffic_lights": {
            "total_valid_mean": _safe_mean(tl_total) if tl_total else None,
            "clusters_mean": _safe_mean(tl_clusters) if tl_clusters else None,
            "cluster_reduction_pct_mean": _safe_mean(tl_cluster_reduction_pct) if tl_cluster_reduction_pct else None,
            "control_confidence": {
                "count": len(tl_conf_all),
                "min": min(tl_conf_all) if tl_conf_all else None,
                "max": max(tl_conf_all) if tl_conf_all else None,
                "mean": _safe_mean(tl_conf_all),
                "median": _safe_median(tl_conf_all),
            },
            "must_stop_edges": {
                "total": sum(tl_must_stop_counts),
                "mean_per_record": _safe_mean(tl_must_stop_counts),
                "records_with_any": sum(1 for x in tl_must_stop_counts if x > 0),
            },
            "stop_nearby_edges": {
                "total": sum(tl_stop_nearby_counts),
                "mean_per_record": _safe_mean(tl_stop_nearby_counts),
                "records_with_any": sum(1 for x in tl_stop_nearby_counts if x > 0),
            },
        },
        "node_type_counts": dict(node_type_counter),
        "top_map_type_names": map_type_name_counter.most_common(15),
        "bad_files": bad,
    }

    # Print a readable report
    print("\n" + "=" * 70)
    print(f"📦 Spatial Graph Report (files={summary['num_files']})")
    print("=" * 70)

    print(f"\nNodes:  min={summary['nodes']['min']} max={summary['nodes']['max']} "
          f"mean={summary['nodes']['mean']:.2f} median={summary['nodes']['median']:.2f}")
    print(f"Edges:  min={summary['edges']['min']} max={summary['edges']['max']} "
          f"mean={summary['edges']['mean']:.2f} median={summary['edges']['median']:.2f}")

    m = summary["map"]
    if m["raw_mean"] is not None:
        print(f"\nMap: raw_mean={m['raw_mean']:.2f} filtered_mean={m['filtered_mean']:.2f} "
              f"reduction_mean={m['reduction_pct_mean']:.1f}% (median {m['reduction_pct_median']:.1f}%)")
        if m["safety_mean"] is not None:
            print(f"     filter_stats mean: safety={m['safety_mean']:.2f} lane={m['lane_mean']:.2f} forward={m['forward_mean']:.2f}")

    tl = summary["traffic_lights"]
    if tl["total_valid_mean"] is not None:
        print(f"\nTraffic lights: total_valid_mean={tl['total_valid_mean']:.2f} clusters_mean={tl['clusters_mean']:.2f} "
              f"cluster_reduction_mean={tl['cluster_reduction_pct_mean']:.1f}%")
    cc = tl["control_confidence"]
    if cc["count"] > 0:
        print(f"TL control_confidence: n={cc['count']} min={cc['min']:.2f} max={cc['max']:.2f} "
              f"mean={cc['mean']:.2f} median={cc['median']:.2f}")

    print(f"\nSTOP semantics:")
    print(f"  must_stop edges: total={tl['must_stop_edges']['total']} "
          f"records_with_any={tl['must_stop_edges']['records_with_any']}/{summary['num_files']} "
          f"mean_per_record={tl['must_stop_edges']['mean_per_record']:.2f}")
    print(f"  stop_nearby edges: total={tl['stop_nearby_edges']['total']} "
          f"records_with_any={tl['stop_nearby_edges']['records_with_any']}/{summary['num_files']} "
          f"mean_per_record={tl['stop_nearby_edges']['mean_per_record']:.2f}")

    print("\nTop MAP_FEATURE type_name (first 15):")
    for name, cnt in summary["top_map_type_names"]:
        print(f"  {name:24s}  {cnt}")

    # Readiness criteria
    print("\n" + "-" * 70)
    ready = True

    if bad:
        ready = False
        print(f"❌ Found {len(bad)} files with issues.")
        print("   First 10 issues:")
        for bf, issues in bad[:10]:
            print(f"   - {bf}: {', '.join(issues)}")
    else:
        print("✅ No structural issues detected (ego/route/finite checks passed).")

    # Expectation checks (soft)
    # - should have ROUTE in all records (your pipeline intends it)
    # - filtered map should be <= 20 typically (if you cap)
    if map_filt and max(map_filt) > 25:
        ready = False
        print(f"❌ Map filtered count too high in some record (max={max(map_filt)}). Expected ~<=20.")
    else:
        print("✅ Map size is within expected budget (<= ~20).")



    print("\nREADY STATUS:", "✅ READY TO PROCEED" if ready else "⚠️ NOT READY (see issues above)")
    print("-" * 70)

    # Save JSON summary
    out_json = os.path.join(OUT_DIR, "report_summary.json")
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved summary -> {out_json}\n")

if __name__ == "__main__":
    main()
