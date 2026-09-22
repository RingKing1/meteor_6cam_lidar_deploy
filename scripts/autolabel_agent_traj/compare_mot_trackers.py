#!/usr/bin/env python3
"""A/B comparison: legacy MOTTracker3D vs SimpleTrack on identical bev_box input.

Metrics (all on the SAME scene, SAME detections):
  1. Track persistence : unique tracks, mean/median/p90 length, fragment rate
  2. tvalid coverage   : fraction of (box, horizon) pairs with a future waypoint
  3. ID switch heur.   : new track born within 1.5 m / 2.0 s of another track's
                         death (same class) -> likely re-identification
  4. Trajectory smooth : median / p95 per-horizon |accel| of future waypoints
  5. Wall time
"""
import json
import os
import sys
import time

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "..", "tools", "SimpleTrack-main"))

from build_agent_traj_gt import MOTTracker3D           # legacy engine
from simpletrack_mot import SimpleTrackMOTEngine, build_config  # SimpleTrack

KMAX = 64
HORIZON = 6
STEP_FRAMES = 5
DT = 0.1


def load_scene_dets(scene, root):
    """Same Pass-1 as the production scripts: ego -> world dets."""
    manifest = json.load(open(os.path.join(root, scene, "manifest.json")))
    poses = np.load(os.path.join(root, scene, "ego_motion.npz"))["pose"]
    frames = manifest["frames"]
    frame_dets = []
    for fi in range(len(frames)):
        p = os.path.join(root, scene, "bev_box", f"{fi:04d}.npz")
        dets = []
        if os.path.exists(p) and fi < len(poses):
            z = np.load(p)
            boxes = z["boxes"] if "boxes" in z else []
            b3d = (z["boxes_3d"] if "boxes_3d" in z
                   and len(z["boxes_3d"]) == len(boxes) else None)
            xe, ye, yaw = poses[fi]
            ce, se = np.cos(yaw), np.sin(yaw)
            for bi, b in enumerate(boxes):
                cls_id, cx, cy, l, w, yaw_b = b
                zc, h = 0.0, 1.8
                if b3d is not None:
                    zc, h = b3d[bi][3], b3d[bi][6]
                xw = xe + cx * ce - cy * se
                yw = ye + cx * se + cy * ce
                dets.append({
                    "cls": int(round(cls_id)),
                    "pos_w": [xw, yw, zc],
                    "size": [l, w, h],
                    "yaw_w": yaw_b + yaw,
                    "ego_box": (cls_id, cx, cy, l, w, yaw_b),
                })
        frame_dets.append(dets)
    return frame_dets, poses, len(frames)


def run_legacy(frame_dets):
    tracker = MOTTracker3D(max_lost_frames=10, max_assoc_dist=3.0)
    per_frame = []
    t0 = time.time()
    for fi, dets in enumerate(frame_dets):
        matched = tracker.update_frame(dets, fi)
        per_frame.append([(id(d), matched[id(d)]) for d in dets
                          if id(d) in matched])
    return per_frame, time.time() - t0


def run_simpletrack(frame_dets):
    engine = SimpleTrackMOTEngine(build_config())
    per_frame = []
    t0 = time.time()
    for fi, dets in enumerate(frame_dets):
        matched = engine.update_frame(dets, fi)
        per_frame.append([(id(d), matched[id(d)]) for d in dets
                          if id(d) in matched])
    return per_frame, time.time() - t0


def track_stats(per_frame):
    """per_frame: list of [(det_id, track_id)]"""
    first = {}
    last = {}
    length = {}
    cls_by_tid = {}
    for fi, items in enumerate(per_frame):
        for det_id, tid in items:
            first.setdefault(tid, fi)
            last[tid] = fi
            length[tid] = length.get(tid, 0) + 1
    lengths = np.array(list(length.values()), dtype=np.float64)
    n = len(lengths)
    frag = float((lengths == 1).sum()) / n if n else 0.0
    # persistence: fraction of (frame, box) instances on tracks >= 10 frames
    total_inst = sum(len(x) for x in per_frame)
    persistent = sum(1 for x in per_frame for _, tid in x
                     if length[tid] >= 10)
    return {
        "n_tracks": n,
        "mean_len": float(lengths.mean()) if n else 0.0,
        "median_len": float(np.median(lengths)) if n else 0.0,
        "p90_len": float(np.percentile(lengths, 90)) if n else 0.0,
        "max_len": float(lengths.max()) if n else 0.0,
        "fragment_rate_1f": frag,
        "persistent_ratio": (persistent / total_inst) if total_inst else 0.0,
        "first": first, "last": last, "length": length,
    }


def count_switches(per_frame, stats, frame_dets, window=20, dist=1.5):
    """A new track born within window frames and dist meters (same class) of
    another track's death is a likely re-identification (switch/fragment)."""
    first, last = stats["first"], stats["last"]
    # tid -> (last_frame, pos, cls)
    ends = {}
    for tid, lf in last.items():
        ff = first[tid]
        # position at last frame
        for fi in range(lf, max(ff - 1, -1), -1):
            if fi < 0 or fi >= len(per_frame):
                continue
            hit = [(det_id, tt) for det_id, tt in per_frame[fi] if tt == tid]
            if hit:
                det_id = hit[0][0]
                pos = None
                for d in frame_dets[fi]:
                    if id(d) == det_id:
                        pos = d["pos_w"][:2]
                        cls = d["cls"]
                        break
                if pos is not None:
                    ends[tid] = (lf, np.array(pos), cls)
                break
    switches = 0
    for tid, (lf, pos, cls) in ends.items():
        ff = first[tid]
        # candidate new tracks starting after this one ended
        for tid2, ff2 in first.items():
            if tid2 == tid or ff2 <= lf or ff2 > lf + window:
                continue
            if ends.get(tid2):
                continue  # only consider tracks still alive (not re-ended)
            # find tid2's birth position
            for fi in range(ff2, min(ff2 + 5, len(per_frame))):
                hit = [(det_id, tt) for det_id, tt in per_frame[fi] if tt == tid2]
                if hit:
                    det_id = hit[0][0]
                    for d in frame_dets[fi]:
                        if id(d) == det_id and d["cls"] == cls:
                            if np.linalg.norm(np.array(d["pos_w"][:2]) - pos) <= dist:
                                switches += 1
                            break
                    break
    return switches


def tvalid_and_smoothness(per_frame, frame_dets, poses, n_frames):
    """Recompute Pass-3 future waypoints and derive coverage + smoothness."""
    track_lookup = {}
    for fi, items in enumerate(per_frame):
        for det_id, tid in items:
            for d in frame_dets[fi]:
                if id(d) == det_id:
                    track_lookup.setdefault(tid, {})[fi] = np.array(d["pos_w"][:2])
                    break

    total_horiz = 0
    covered_horiz = 0
    accels = []
    for fi in range(n_frames):
        dets = frame_dets[fi]
        if fi >= len(poses) or not dets:
            continue
        xe, ye, yaw = poses[fi]
        c_inv, s_inv = np.cos(-yaw), np.sin(-yaw)

        def to_curr(xw, yw):
            dx, dy = xw - xe, yw - ye
            return (c_inv * dx - s_inv * dy, s_inv * dx + c_inv * dy)

        tid_of = {det_id: tid for det_id, tid in per_frame[fi]}
        for d in dets:
            if id(d) not in tid_of:
                continue
            tid = tid_of[id(d)]
            cx, cy = d["ego_box"][1], d["ego_box"][2]
            hist = track_lookup.get(tid, {})
            wp = []
            for h in range(HORIZON):
                ffi = fi + STEP_FRAMES * (h + 1)
                if ffi in hist:
                    fx, fy = to_curr(hist[ffi][0], hist[ffi][1])
                    wp.append([fx - cx, fy - cy])
                    covered_horiz += 1
                total_horiz += 1
            # per-horizon velocity (0.5 s) -> |accel| between consecutive steps
            if len(wp) >= 3:
                vels = [np.linalg.norm(np.array(wp[i + 1]) - np.array(wp[i])) / 0.5
                        for i in range(len(wp) - 1)]
                accels.extend(np.abs(np.diff(vels)))
    cov = (covered_horiz / total_horiz) if total_horiz else 0.0
    acc = np.array(accels)
    return cov, (float(np.median(acc)) if len(acc) else 0.0,
                 float(np.percentile(acc, 95)) if len(acc) else 0.0,
                 int(len(acc)))


def main():
    root = "scenes"
    scene = sys.argv[1] if len(sys.argv) > 1 else "data_20260910_061820"
    print(f"=== A/B MOT comparison on {scene} ===")
    frame_dets, poses, n = load_scene_dets(scene, root)
    print(f"frames={n}  dets/frame mean={np.mean([len(x) for x in frame_dets]):.1f}")

    pf_legacy, t_legacy = run_legacy(frame_dets)
    st_legacy = track_stats(pf_legacy)
    sw_legacy = count_switches(pf_legacy, st_legacy, frame_dets)
    cov_legacy, acc_legacy = tvalid_and_smoothness(pf_legacy, frame_dets, poses, n)

    pf_st, t_st = run_simpletrack(frame_dets)
    st_st = track_stats(pf_st)
    sw_st = count_switches(pf_st, st_st, frame_dets)
    cov_st, acc_st = tvalid_and_smoothness(pf_st, frame_dets, poses, n)

    def row(name, legacy, st_new):
        print(f"  {name:<34s} legacy={legacy:<12} simpletrack={st_new}")

    print("\n--- Track persistence ---")
    row("unique tracks", st_legacy["n_tracks"], st_st["n_tracks"])
    row("mean track len (fr)", f"{st_legacy['mean_len']:.1f}", f"{st_st['mean_len']:.1f}")
    row("median track len (fr)", f"{st_legacy['median_len']:.1f}", f"{st_st['median_len']:.1f}")
    row("p90 track len (fr)", f"{st_legacy['p90_len']:.1f}", f"{st_st['p90_len']:.1f}")
    row("max track len (fr)", st_legacy["max_len"], st_st["max_len"])
    row("1-frame fragment rate", f"{st_legacy['fragment_rate_1f']:.3f}",
        f"{st_st['fragment_rate_1f']:.3f}")
    row("persistent (>=10fr) ratio", f"{st_legacy['persistent_ratio']:.3f}",
        f"{st_st['persistent_ratio']:.3f}")

    print("\n--- ID stability / GT completeness ---")
    row("re-id switch count (heur.)", sw_legacy, sw_st)
    row("tvalid coverage", f"{cov_legacy:.3f}", f"{cov_st:.3f}")
    row("|accel| median (m/s^2)", f"{acc_legacy[0]:.2f}", f"{acc_st[0]:.2f}")
    row("|accel| p95 (m/s^2)", f"{acc_legacy[1]:.2f}", f"{acc_st[1]:.2f}")
    row("accel samples", acc_legacy[2], acc_st[2])

    print("\n--- Runtime (CPU) ---")
    row("wall time", f"{t_legacy:.2f}s", f"{t_st:.2f}s")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
