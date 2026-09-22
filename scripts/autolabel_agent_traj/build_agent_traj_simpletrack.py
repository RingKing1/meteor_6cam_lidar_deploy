#!/usr/bin/env python3
"""3D MOT & Future 3.0s Trajectory Waypoint Generation (SimpleTrack engine).

Drop-in replacement for build_agent_traj_gt.py with the MOT stage upgraded to
SimpleTrack (ICRA 2022, mot_3d library): Kalman filter motion model + GIoU
association + track lifecycle + two-stage occlusion redundancy.

Output contract is IDENTICAL to the original (agent_traj/<fi>.npz):
  boxes  float32 [K,6]   (cls, xe, ye, l, w, yaw)   ego frame, closest-first
  count  int64
  traj   float32 [K,6,2] future (dx, dy) relative offsets from box centre
  tvalid float32 [K,6]   1.0 where the instance exists at that horizon
plus a NEW backward-compatible key:
  track_ids int64 [K]    persistent SimpleTrack track id per box
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simpletrack_mot import SimpleTrackMOTEngine, build_config  # noqa: E402

KMAX = 64
HORIZON = 6
STEP_FRAMES = 5  # 5 frames @ 10 Hz = 0.5 s


def process_scene_trajectories(scene, args):
    scene_dir = os.path.join(args.root, scene)
    manifest_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found for scene {scene}, skipping.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    ego_motion_p = os.path.join(scene_dir, "ego_motion.npz")
    if not os.path.exists(ego_motion_p):
        print(f"[!] ego_motion.npz not found for scene {scene}, skipping.")
        return

    ego_data = np.load(ego_motion_p)
    poses = ego_data["pose"]  # [N, 3]: (x_world, y_world, yaw_world)

    bev_box_dir = os.path.join(scene_dir, "bev_box")
    out_traj_dir = os.path.join(scene_dir, args.out_subdir)
    os.makedirs(out_traj_dir, exist_ok=True)

    frames = manifest["frames"]
    n_frames = len(frames)
    print(f"\n[*] Processing 3D MOT (SimpleTrack) & Agent Trajectories for "
          f"{scene} ({n_frames} frames) -> {args.out_subdir} ...")

    configs = build_config(asso=args.asso, max_age=args.max_age,
                           min_hits=args.min_hits, redundancy_mode=args.redundancy)

    # Pass 1: Load all 3D boxes and transform to World Coordinates
    tracker = SimpleTrackMOTEngine(configs)
    frame_dets_world = []

    for fi in range(n_frames):
        b_npz = os.path.join(bev_box_dir, f"{fi:04d}.npz")
        dets = []
        if os.path.exists(b_npz) and fi < len(poses):
            z = np.load(b_npz)
            boxes = z["boxes"] if "boxes" in z else []
            b3d_list = (z["boxes_3d"] if "boxes_3d" in z
                        and len(z["boxes_3d"]) == len(boxes) else None)

            xe_ego, ye_ego, yaw_ego = poses[fi]
            ce, se = np.cos(yaw_ego), np.sin(yaw_ego)

            for bi, b in enumerate(boxes):
                cls_id, cx, cy, l, w, yaw_b = b
                zc, h = 0.0, 1.8
                if b3d_list is not None:
                    zc, h = b3d_list[bi][3], b3d_list[bi][6]

                # Ego -> World
                xw = xe_ego + cx * ce - cy * se
                yw = ye_ego + cx * se + cy * ce
                zw = zc
                yaw_w = yaw_b + yaw_ego

                dets.append({
                    "cls": int(round(cls_id)),
                    "pos_w": [xw, yw, zw],
                    "size": [l, w, h],
                    "yaw_w": yaw_w,
                    "ego_box": (cls_id, cx, cy, l, w, yaw_b),
                })
        frame_dets_world.append(dets)

    # Pass 2: Run SimpleTrack 3D MOT across the scene
    frame_matched_tracks = []
    for fi in range(n_frames):
        matched = tracker.update_frame(frame_dets_world[fi], fi)
        frame_matched_tracks.append(matched)

    # Build persistent track trajectory lookup table: track_id -> dict(frame -> pos_w)
    track_lookup = {}
    for fi, dets in enumerate(frame_dets_world):
        matched = frame_matched_tracks[fi]
        for det in dets:
            tid = matched.get(id(det))
            if tid is None:
                continue
            track_lookup.setdefault(tid, {})[fi] = np.array(det["pos_w"][:2],
                                                            dtype=np.float64)

    # Pass 3: Calculate future 3.0s waypoints (6 horizons) relative to ego at frame fi
    total_agents = 0
    t_start = time.time()

    for fi in range(n_frames):
        out_npz = os.path.join(out_traj_dir, f"{fi:04d}.npz")
        if not args.force and os.path.exists(out_npz):
            if not args.no_manifest:
                manifest["frames"][fi]["agent_traj"] = (
                    f"{args.out_subdir}/{fi:04d}.npz")
            continue

        dets = frame_dets_world[fi]
        matched = frame_matched_tracks[fi]

        B = np.zeros((KMAX, 6), dtype=np.float32)
        T = np.zeros((KMAX, HORIZON, 2), dtype=np.float32)
        V = np.zeros((KMAX, HORIZON), dtype=np.float32)
        IDS = np.zeros((KMAX,), dtype=np.int64)

        if fi < len(poses) and len(dets) > 0:
            xe_curr, ye_curr, yaw_curr = poses[fi]
            c_inv, s_inv = np.cos(-yaw_curr), np.sin(-yaw_curr)

            def to_curr_ego(xw, yw):
                dx = xw - xe_curr
                dy = yw - ye_curr
                return (c_inv * dx - s_inv * dy, s_inv * dx + c_inv * dy)

            cand_agents = []
            for det in dets:
                tid = matched.get(id(det))
                if tid is None:
                    continue  # safety: det without track id (should not happen)
                cx, cy = det["ego_box"][1], det["ego_box"][2]
                dist_sq = cx * cx + cy * cy
                cand_agents.append((dist_sq, tid, det))
            cand_agents.sort()

            k = 0
            for _, tid, det in cand_agents:
                if k >= KMAX:
                    break
                cls_id, cx, cy, l, w, yaw_b = det["ego_box"]
                B[k] = (cls_id, cx, cy, l, w, yaw_b)
                IDS[k] = tid

                hist = track_lookup.get(tid, {})
                for h in range(HORIZON):
                    future_fi = fi + STEP_FRAMES * (h + 1)
                    if future_fi in hist:
                        fw_pos = hist[future_fi]
                        f_xe, f_ye = to_curr_ego(fw_pos[0], fw_pos[1])
                        T[k, h] = (f_xe - cx, f_ye - cy)
                        V[k, h] = 1.0

                k += 1
            total_agents += k
            count_val = np.int64(k)
        else:
            count_val = np.int64(0)

        np.savez_compressed(out_npz, boxes=B, count=count_val, traj=T,
                            tvalid=V, track_ids=IDS)
        if not args.no_manifest:
            manifest["frames"][fi]["agent_traj"] = (
                f"{args.out_subdir}/{fi:04d}.npz")

        if (fi + 1) % 500 == 0 or (fi + 1) == n_frames:
            fps = (fi + 1) / max(0.001, time.time() - t_start)
            print(f"  [{fi+1}/{n_frames}] {fps:.1f} fps | Tracked agents: "
                  f"{total_agents} | SimpleTrack tracks: {len(tracker.trackers)}")

    if not args.no_manifest:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print(f"[+] Scene {scene} agent trajectories completed in {total_time:.1f}s "
          f"({len(tracker.trackers)} class engines).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/meteor_6cam_lidar_deploy/scenes")
    parser.add_argument("--scenes", default="all",
                        help="all or comma-separated scene list")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-subdir", default="agent_traj")
    parser.add_argument("--no-manifest", action="store_true",
                        help="do not touch manifest.json (A/B comparison mode)")
    parser.add_argument("--asso", default="giou", choices=["giou", "iou", "euler"])
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--redundancy", default="mm", choices=["mm", "default"])
    args = parser.parse_args()

    all_scenes = [
        "data_20260910_061820", "data_20260910_062659", "data_20260910_063822",
        "data_20260910_064331", "data_20260910_073823", "data_20260910_074912",
    ]
    if args.scenes == "all":
        target_scenes = all_scenes
    else:
        target_scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print("================================================================================")
    print("  3D MOT (SimpleTrack) & Future 3.0s Agent Trajectory Generation")
    print("================================================================================")
    print(f"Scenes: {target_scenes}")
    print(f"Engine: SimpleTrack asso={args.asso} max_age={args.max_age} "
          f"min_hits={args.min_hits} redundancy={args.redundancy}")
    print(f"Out:    {args.out_subdir}  (manifest update: {not args.no_manifest})")
    print("================================================================================")

    for scene in target_scenes:
        process_scene_trajectories(scene, args)

    print("\n================================================================================")
    print("  All agent trajectories generated successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
