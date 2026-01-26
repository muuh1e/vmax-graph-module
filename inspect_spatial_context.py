#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------
# TFExample helpers
# ---------------------------
def _feat_kind_and_len(feat: tf.train.Feature) -> Tuple[str, int]:
    if len(feat.float_list.value) > 0:
        return "float", len(feat.float_list.value)
    if len(feat.int64_list.value) > 0:
        return "int", len(feat.int64_list.value)
    if len(feat.bytes_list.value) > 0:
        return "bytes", len(feat.bytes_list.value)
    return "empty", 0


def has(f: Dict[str, tf.train.Feature], key: str) -> bool:
    return key in f and _feat_kind_and_len(f[key])[1] > 0


def F(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    return np.asarray(f[key].float_list.value, dtype=np.float32)


def I(f: Dict[str, tf.train.Feature], key: str) -> np.ndarray:
    return np.asarray(f[key].int64_list.value, dtype=np.int64)


def load_record(tfrecord_path: str, record_index: int) -> tf.train.Example:
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for i, raw in enumerate(ds):
        if i == record_index:
            return tf.train.Example.FromString(raw.numpy())
    raise IndexError(f"record_index={record_index} not found in {tfrecord_path}")


# ---------------------------
# Geometry helpers
# ---------------------------
def ego_frame(dx: np.ndarray, dy: np.ndarray, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    # rotate by -yaw
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    x_local = dx * c - dy * s
    y_local = dx * s + dy * c
    return x_local, y_local


def summarize_counter(title: str, counter: Counter, topn: int = 20):
    print(f"\n{title}")
    for k, v in counter.most_common(topn):
        print(f"  {k}: {v}")
    if len(counter) > topn:
        print(f"  ... ({len(counter) - topn} more)")


def stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}


@dataclass
class KeyInfo:
    key: str
    dtype: str
    length: int


def list_nonempty_keys(f: Dict[str, tf.train.Feature]) -> List[KeyInfo]:
    out: List[KeyInfo] = []
    for k in sorted(f.keys()):
        dt, ln = _feat_kind_and_len(f[k])
        if ln == 0:
            continue
        out.append(KeyInfo(k, dt, ln))
    return out


def group_by_prefix(keys: List[KeyInfo]) -> Dict[str, List[KeyInfo]]:
    g: Dict[str, List[KeyInfo]] = defaultdict(list)
    for ki in keys:
        prefix = ki.key.split("/")[0] if "/" in ki.key else ki.key
        g[prefix].append(ki)
    return dict(g)


def grep_keys(keys: List[KeyInfo], pattern: str) -> List[KeyInfo]:
    pat = pattern.lower()
    return [k for k in keys if pat in k.key.lower()]


def reshape_time_major(vec: np.ndarray, per_step: int) -> Optional[np.ndarray]:
    if per_step <= 0:
        return None
    if vec.size % per_step != 0:
        return None
    T = vec.size // per_step
    return vec.reshape(T, per_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--record", type=int, default=0)

    # full key inspection
    ap.add_argument("--list_keys", action="store_true")
    ap.add_argument("--grep", action="append", default=[], help="substring filter (repeatable)")
    ap.add_argument("--scan_missing", action="store_true", help="scan for speed_limit/topology/lane-control style fields")

    # roadgraph / TL printing
    ap.add_argument("--topk_rg", type=int, default=40)
    ap.add_argument("--rg_radius", type=float, default=60.0)
    ap.add_argument("--topk_tl", type=int, default=20)
    ap.add_argument("--tl_radius", type=float, default=80.0)

    # V-Max style selection (Appendix D style)
    ap.add_argument("--vmax_obs", action="store_true", help="also print V-Max-style selections")
    ap.add_argument("--vmax_rg_topn", type=int, default=200)
    ap.add_argument("--vmax_rg_fwd", type=float, default=50.0)
    ap.add_argument("--vmax_rg_back", type=float, default=5.0)
    ap.add_argument("--vmax_rg_side", type=float, default=20.0)
    ap.add_argument("--vmax_tl_k", type=int, default=5)

    ap.add_argument("--dump_json", default="", help="optional JSON summary path")
    args = ap.parse_args()

    ex = load_record(args.tfrecord, args.record)
    f = ex.features.feature
    keys = list_nonempty_keys(f)

    # ---- Ego summary ----
    cur_x = F(f, "state/current/x")
    cur_y = F(f, "state/current/y")
    cur_valid = I(f, "state/current/valid").astype(bool)
    is_sdc = I(f, "state/is_sdc")
    ego_idx = int(np.argmax(is_sdc))

    ego_yaw = 0.0
    if has(f, "state/current/bbox_yaw"):
        ego_yaw = float(F(f, "state/current/bbox_yaw")[ego_idx])
    elif has(f, "state/current/vel_yaw"):
        ego_yaw = float(F(f, "state/current/vel_yaw")[ego_idx])

    ego_x = float(cur_x[ego_idx])
    ego_y = float(cur_y[ego_idx])

    print("\n========== RECORD SUMMARY ==========")
    print("tfrecord:", args.tfrecord)
    print("record_index:", args.record)
    print("N objects:", cur_x.shape[0])
    print("valid objects @ current:", int(cur_valid.sum()))
    print(f"ego_idx: {ego_idx}  ego(x,y)=({ego_x:.3f},{ego_y:.3f})  yaw={ego_yaw:.3f}")

    summary = {
        "tfrecord": args.tfrecord,
        "record_index": int(args.record),
        "ego_idx": int(ego_idx),
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_yaw": float(ego_yaw),
        "num_keys": int(len(keys)),
    }

    # ---- Full key list + grouping ----
    if args.list_keys:
        print("\n========== ALL FEATURE KEYS (non-empty) ==========")
        print(f"Total non-empty keys: {len(keys)}")
        grouped = group_by_prefix(keys)
        print("\nKey counts by top-level prefix:")
        for pfx in sorted(grouped.keys()):
            print(f"  {pfx:28s} {len(grouped[pfx])}")
        print("\nFull key list (dtype, length):")
        for ki in keys:
            print(f"  [{ki.dtype:5s}] {ki.length:7d}  {ki.key}")

    if args.grep:
        print("\n========== GREP RESULTS ==========")
        for pat in args.grep:
            hits = grep_keys(keys, pat)
            print(f"\n-- pattern: '{pat}'  hits={len(hits)}")
            for ki in hits[:250]:
                print(f"  [{ki.dtype:5s}] {ki.length:7d}  {ki.key}")
            if len(hits) > 250:
                print("  ... (truncated)")

    if args.scan_missing:
        print("\n========== MISSING-DATA SCAN ==========")
        want = [
            "speed_limit", "speed", "successor", "predecessor", "lane", "control",
            "controlled", "connector", "boundary", "edge", "stopline", "stop_point", "route", "path",
        ]
        found = {}
        for pat in want:
            hits = grep_keys(keys, pat)
            found[pat] = [h.key for h in hits]
            if hits:
                print(f"[FOUND] '{pat}': {len(hits)} keys, e.g. {found[pat][:5]}")
            else:
                print(f"[MISS ] '{pat}': no keys matched")
        summary["scan"] = {k: v[:50] for k, v in found.items()}

    # ---- Roadgraph detailed ----
    print("\n========== ROADGRAPH_SAMPLES ==========")
    rg_info = {"present": False}
    if has(f, "roadgraph_samples/xyz") and has(f, "roadgraph_samples/valid"):
        rg_xyz = F(f, "roadgraph_samples/xyz").reshape(-1, 3)
        rg_valid = I(f, "roadgraph_samples/valid").astype(bool)
        M = rg_xyz.shape[0]
        rg_info.update({"present": True, "M": int(M), "valid": int(rg_valid.sum())})

        rg_fields = {}
        for k in sorted(f.keys()):
            if k.startswith("roadgraph_samples/") and has(f, k):
                dt, ln = _feat_kind_and_len(f[k])
                rg_fields[k.split("roadgraph_samples/")[1]] = {"dtype": dt, "len": ln}
        rg_info["fields"] = rg_fields

        print(f"M roadgraph points: {int(M)}  valid: {int(rg_valid.sum())}")
        print("Available roadgraph_samples/* fields:", ", ".join(sorted(rg_fields.keys())))

        rg_type = I(f, "roadgraph_samples/type") if has(f, "roadgraph_samples/type") else None
        rg_id = I(f, "roadgraph_samples/id") if has(f, "roadgraph_samples/id") else None
        rg_dir = F(f, "roadgraph_samples/dir").reshape(-1, 3) if has(f, "roadgraph_samples/dir") else None

        # speed-limit probe (common candidate names)
        speed_candidates = [k for k in rg_fields.keys() if "speed" in k.lower()]
        if speed_candidates:
            print("roadgraph speed-related keys present:", speed_candidates)
        else:
            print("roadgraph speed-related keys present: NONE")

        if rg_type is not None:
            summarize_counter("roadgraph type counts (valid):", Counter(rg_type[rg_valid].tolist()), topn=30)

        # nearest points
        pts = rg_xyz[rg_valid]
        dx = pts[:, 0] - ego_x
        dy = pts[:, 1] - ego_y
        dist = np.sqrt(dx * dx + dy * dy)
        mask = dist <= args.rg_radius
        pts2 = pts[mask]
        dist2 = dist[mask]
        dx2 = dx[mask]
        dy2 = dy[mask]
        x_local, y_local = ego_frame(dx2, dy2, ego_yaw)

        if pts2.shape[0] == 0:
            print(f"No roadgraph points within rg_radius={args.rg_radius}m")
        else:
            k = min(args.topk_rg, pts2.shape[0])
            idx = np.argpartition(dist2, k - 1)[:k]
            idx = idx[np.argsort(dist2[idx])]
            valid_idx = np.where(rg_valid)[0]
            sel_valid_idx = valid_idx[mask]

            print(f"\nTop-{k} nearest roadgraph points within {args.rg_radius}m:")
            for t in idx:
                orig = int(sel_valid_idx[t])
                rid = int(rg_id[orig]) if rg_id is not None else None
                rtype = int(rg_type[orig]) if rg_type is not None else None
                print(
                    f"  dist={dist2[t]:6.2f}m  ego_frame=(x={x_local[t]:6.2f}, y={y_local[t]:6.2f})  "
                    f"xyz=({pts2[t,0]:.2f},{pts2[t,1]:.2f})  id={rid} type={rtype}"
                )

        # V-Max style roadgraph selection
        if args.vmax_obs:
            print("\n---- V-Max style roadgraph selection (bounding box) ----")
            valid_idx = np.where(rg_valid)[0]
            ptsv = rg_xyz[valid_idx]
            dxv = ptsv[:, 0] - ego_x
            dyv = ptsv[:, 1] - ego_y
            xloc, yloc = ego_frame(dxv, dyv, ego_yaw)
            in_box = (xloc <= args.vmax_rg_fwd) & (xloc >= -args.vmax_rg_back) & (np.abs(yloc) <= args.vmax_rg_side)
            idx_in = valid_idx[in_box]
            print(
                f"box fwd<= {args.vmax_rg_fwd}m back<= {args.vmax_rg_back}m side<= {args.vmax_rg_side}m -> "
                f"{idx_in.size} points"
            )
            if idx_in.size > 0:
                dx_in = rg_xyz[idx_in, 0] - ego_x
                dy_in = rg_xyz[idx_in, 1] - ego_y
                d_in = np.sqrt(dx_in * dx_in + dy_in * dy_in)
                order = np.argsort(d_in)[: min(args.vmax_rg_topn, idx_in.size)]
                picked = idx_in[order]
                if rg_type is not None:
                    summarize_counter("picked roadgraph type counts:", Counter(rg_type[picked].tolist()), topn=20)
                if rg_id is not None:
                    cid = Counter(rg_id[picked].tolist())
                    print(f"picked unique polyline ids: {len(cid)} (top 10 by point-count)")
                    for pid, cnt in cid.most_common(10):
                        print(f"  id={int(pid)} points={int(cnt)}")
    else:
        print("roadgraph_samples missing (no xyz/valid keys).")

    summary["roadgraph"] = rg_info

    # ---- Traffic lights detailed (with time reshape) ----
    def inspect_tl(prefix: str, num_lights_current: Optional[int]):
        base = f"traffic_light_state/{prefix}"
        print(f"\n========== TRAFFIC LIGHTS ({prefix}) ==========")
        if not has(f, f"{base}/valid"):
            print("No traffic light keys for this prefix.")
            return {"present": False}

        tl_valid = I(f, f"{base}/valid").astype(bool)
        tl_x = F(f, f"{base}/x")
        tl_y = F(f, f"{base}/y")
        tl_state = I(f, f"{base}/state") if has(f, f"{base}/state") else None
        tl_id = I(f, f"{base}/id") if has(f, f"{base}/id") else None

        keys_here = [k for k in sorted(f.keys()) if k.startswith(base + "/") and has(f, k)]
        print(f"num entries: {tl_valid.size}  valid: {int(tl_valid.sum())}")
        print("Available keys:", ", ".join([k.split(base + '/')[1] for k in keys_here]))

        # try reshape for past/future using num lights from current
        if num_lights_current is not None and prefix in ("past", "future"):
            v2 = reshape_time_major(tl_valid.astype(np.int64), num_lights_current)
            id2 = reshape_time_major(tl_id, num_lights_current) if tl_id is not None else None
            st2 = reshape_time_major(tl_state, num_lights_current) if tl_state is not None else None
            if v2 is not None:
                T = v2.shape[0]
                print(f"Reshaped with num_lights_current={num_lights_current} -> T={T}, L={num_lights_current}")
                if id2 is not None:
                    uid = np.unique(id2)
                    print("Unique TL ids (across time):", uid[: min(25, uid.size)], ("..." if uid.size > 25 else ""))
                if st2 is not None:
                    va = v2.reshape(-1).astype(bool)
                    summarize_counter("TL state counts over time (valid):", Counter(st2.reshape(-1)[va].tolist()), topn=20)

        # distance-based print (unique by id if possible)
        if tl_id is not None:
            valid_idx = np.where(tl_valid)[0]
            seen = set()
            uniq = []
            for i in valid_idx:
                tid = int(tl_id[i])
                if tid in seen:
                    continue
                seen.add(tid)
                uniq.append(i)
            idx_use = np.asarray(uniq, dtype=np.int64)
        else:
            idx_use = np.where(tl_valid)[0]

        if idx_use.size == 0:
            print("No valid traffic lights to inspect.")
            return {"present": True}

        dx = tl_x[idx_use] - ego_x
        dy = tl_y[idx_use] - ego_y
        dist = np.sqrt(dx * dx + dy * dy)
        mask = dist <= args.tl_radius
        idx2 = idx_use[mask]
        dist2 = dist[mask]
        dx2 = dx[mask]
        dy2 = dy[mask]
        x_local, y_local = ego_frame(dx2, dy2, ego_yaw)

        if idx2.size == 0:
            print(f"No valid traffic lights within tl_radius={args.tl_radius}m")
        else:
            order = np.argsort(dist2)[: min(args.topk_tl, idx2.size)]
            print(f"Top-{order.size} nearest unique valid traffic lights within {args.tl_radius}m:")
            for j in order:
                i = int(idx2[j])
                st = int(tl_state[i]) if tl_state is not None else None
                tid = int(tl_id[i]) if tl_id is not None else None
                print(
                    f"  dist={dist2[j]:6.2f}m  ego_frame=(x={x_local[j]:6.2f}, y={y_local[j]:6.2f})  "
                    f"xy=({float(tl_x[i]):.2f},{float(tl_y[i]):.2f})  id={tid} state={st}"
                )

            if tl_state is not None:
                summarize_counter(
                    "traffic light state counts (unique valid within radius):",
                    Counter(tl_state[idx2].tolist()),
                    topn=20,
                )

        if args.vmax_obs:
            order_all = np.argsort(dist)[: min(args.vmax_tl_k, idx_use.size)]
            picked = idx_use[order_all]
            print(f"\n---- V-Max style TL selection: {picked.size} closest unique valid ----")
            for i in picked:
                i = int(i)
                st = int(tl_state[i]) if tl_state is not None else None
                tid = int(tl_id[i]) if tl_id is not None else None
                ddx = float(tl_x[i] - ego_x)
                ddy = float(tl_y[i] - ego_y)
                xl, yl = ego_frame(np.array([ddx]), np.array([ddy]), ego_yaw)
                print(
                    f"  dist={float(np.sqrt(ddx*ddx+ddy*ddy)):6.2f}m "
                    f"ego_frame=(x={float(xl[0]):6.2f}, y={float(yl[0]):6.2f}) "
                    f"id={tid} state={st}"
                )

        return {"present": True}

    num_lights_current = int(I(f, "traffic_light_state/current/valid").size) if has(f, "traffic_light_state/current/valid") else None
    tl_info = {"present": num_lights_current is not None, "num_lights_current": num_lights_current}
    tl_info["current"] = inspect_tl("current", num_lights_current)
    tl_info["past"] = inspect_tl("past", num_lights_current)
    tl_info["future"] = inspect_tl("future", num_lights_current)
    summary["traffic_lights"] = tl_info

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print("\nSaved JSON summary:", args.dump_json)


if __name__ == "__main__":
    main()
