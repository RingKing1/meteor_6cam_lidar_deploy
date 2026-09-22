#!/usr/bin/env python3
"""3D Multi-Object Tracking & 3.0s Future Trajectory Waypoint Generation.

Solves 3D agent trajectories for METEOR:
  - Coordinate alignment: Ego frame -> World ENU coordinate frame via ego_motion.npz
  - 3D MOT: Kalman/Linear motion model + Hungarian distance association
  - Future 3.0s waypoints: 6 steps at +0.5s, +1.0s, +1.5s, +2.0s, +2.5s, +3.0s (+5, +10, ..., +30 frames @ 10Hz)
  - Relative displacement delta: (dx, dy) relative to agent position at frame t in frame t's ego coordinate system

Outputs:
  - scenes/<scene>/agent_traj/<fi:04d>.npz:
      boxes:  float32 [64, 6]   (cls, xe, ye, l, w, yaw)
      count:  int64
      traj:   float32 [64, 6, 2] future (dx, dy) relative offsets from box centre
      tvalid: float32 [64, 6]   validity mask (1.0 = present, 0.0 = absent/occluded)
  - Updates manifest.json with "agent_traj" key
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import numpy as np

KMAX = 64
HORIZON = 6
STEP_FRAMES = 5  # 5 frames @ 10 Hz = 0.5 s


class Track3D:
    def __init__(self, track_id, cls_id, pos_w, size, yaw_w, frame_idx):
        self.track_id = track_id
        self.cls_id = cls_id
        self.pos_w = np.array(pos_w, dtype=np.float64)  # [x, y, z]
        self.size = np.array(size, dtype=np.float64)    # [l, w, h]
        self.yaw_w = float(yaw_w)
        self.last_frame = frame_idx
        self.history = {frame_idx: self.pos_w.copy()}   # frame_idx -> pos_w
        self.vel_w = np.zeros(2, dtype=np.float64)     # [vx, vy] m/s
        self.time_lost = 0

    def predict(self, dt=0.1):
        """Predict position at next frame using constant velocity."""
        return self.pos_w[:2] + self.vel_w * dt

    def update(self, pos_w, size, yaw_w, frame_idx, dt=0.1):
        pos_arr = np.array(pos_w, dtype=np.float64)
        if dt > 0:
            inst_vel = (pos_arr[:2] - self.pos_w[:2]) / dt
            # Exponential smoothing for velocity
            self.vel_w = 0.6 * self.vel_w + 0.4 * inst_vel
        self.pos_w = pos_arr
        self.size = 0.7 * self.size + 0.3 * np.array(size, dtype=np.float64)
        self.yaw_w = float(yaw_w)
        self.last_frame = frame_idx
        self.history[frame_idx] = self.pos_w.copy()
        self.time_lost = 0


class MOTTracker3D:
    def __init__(self, max_lost_frames=10, max_assoc_dist=3.0):
        self.next_id = 1
        self.tracks = []
        self.max_lost_frames = max_lost_frames
        self.max_assoc_dist = max_assoc_dist

    def update_frame(self, detections_w, frame_idx):
        """detections_w: list of dicts: {'cls': c, 'pos_w': [x,y,z], 'size': [l,w,h], 'yaw_w': yaw, 'raw_box': b}"""
        matched_track_ids = {}

        if len(self.tracks) == 0:
            for det in detections_w:
                tr = Track3D(self.next_id, det['cls'], det['pos_w'], det['size'], det['yaw_w'], frame_idx)
                self.tracks.append(tr)
                matched_track_ids[id(det)] = self.next_id
                self.next_id += 1
            return matched_track_ids

        # Predict active track positions
        active_tracks = [tr for tr in self.tracks if (frame_idx - tr.last_frame) <= self.max_lost_frames]
        if len(active_tracks) == 0 or len(detections_w) == 0:
            for det in detections_w:
                tr = Track3D(self.next_id, det['cls'], det['pos_w'], det['size'], det['yaw_w'], frame_idx)
                self.tracks.append(tr)
                matched_track_ids[id(det)] = self.next_id
                self.next_id += 1
            return matched_track_ids

        # Build cost matrix
        cost_matrix = np.full((len(active_tracks), len(detections_w)), 1e6, dtype=np.float64)
        for ti, tr in enumerate(active_tracks):
            pred_xy = tr.predict(dt=0.1 * (frame_idx - tr.last_frame))
            for di, det in enumerate(detections_w):
                # Class compatibility check
                if tr.cls_id != det['cls']:
                    continue
                det_xy = det['pos_w'][:2]
                dist = np.linalg.norm(pred_xy - det_xy)
                if dist < self.max_assoc_dist:
                    cost_matrix[ti, di] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_assoc_dist:
                tr = active_tracks[r]
                det = detections_w[c]
                tr.update(det['pos_w'], det['size'], det['yaw_w'], frame_idx, dt=0.1 * (frame_idx - tr.last_frame))
                matched_track_ids[id(det)] = tr.track_id
                matched_dets.add(c)

        # Unmatched detections start new tracks
        for di, det in enumerate(detections_w):
            if di not in matched_dets:
                tr = Track3D(self.next_id, det['cls'], det['pos_w'], det['size'], det['yaw_w'], frame_idx)
                self.tracks.append(tr)
                matched_track_ids[id(det)] = self.next_id
                self.next_id += 1

        return matched_track_ids


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
    out_traj_dir = os.path.join(scene_dir, "agent_traj")
    os.makedirs(out_traj_dir, exist_ok=True)

    frames = manifest["frames"]
    n_frames = len(frames)
    print(f"\n[*] Processing 3D MOT & Agent Trajectories for {scene} ({n_frames} frames)...")

    # Pass 1: Load all 3D boxes and transform to World Coordinates
    tracker = MOTTracker3D(max_lost_frames=10, max_assoc_dist=3.0)
    frame_dets_world = []

    for fi in range(n_frames):
        b_npz = os.path.join(bev_box_dir, f"{fi:04d}.npz")
        dets = []
        if os.path.exists(b_npz) and fi < len(poses):
            z = np.load(b_npz)
            boxes = z["boxes"] if "boxes" in z else []
            # boxes_3d: (cls, cx, cy, zc, l, w, h, yaw)
            b3d_list = z["boxes_3d"] if "boxes_3d" in z and len(z["boxes_3d"]) == len(boxes) else None

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

    # Pass 2: Run 3D MOT tracking across the scene
    frame_matched_tracks = []
    for fi in range(n_frames):
        matched = tracker.update_frame(frame_dets_world[fi], fi)
        frame_matched_tracks.append(matched)

    # Build persistent track trajectory lookup table: track_id -> dict(frame_idx -> pos_w)
    track_lookup = {tr.track_id: tr.history for tr in tracker.tracks}

    # Pass 3: Calculate future 3.0s waypoints (6 horizons) relative to ego at frame fi
    total_agents = 0
    t_start = time.time()

    for fi in range(n_frames):
        out_npz = os.path.join(out_traj_dir, f"{fi:04d}.npz")
        if not args.force and os.path.exists(out_npz):
            manifest["frames"][fi]["agent_traj"] = f"agent_traj/{fi:04d}.npz"
            continue

        dets = frame_dets_world[fi]
        matched = frame_matched_tracks[fi]

        B = np.zeros((KMAX, 6), dtype=np.float32)
        T = np.zeros((KMAX, HORIZON, 2), dtype=np.float32)
        V = np.zeros((KMAX, HORIZON), dtype=np.float32)

        if fi < len(poses) and len(dets) > 0:
            xe_curr, ye_curr, yaw_curr = poses[fi]
            # Transformation from World -> Ego at current frame fi
            c_inv, s_inv = np.cos(-yaw_curr), np.sin(-yaw_curr)

            def to_curr_ego(xw, yw):
                dx = xw - xe_curr
                dy = yw - ye_curr
                return (c_inv * dx - s_inv * dy, s_inv * dx + c_inv * dy)

            # Sort agents by distance to ego vehicle (closest agents first)
            cand_agents = []
            for det in dets:
                tid = matched[id(det)]
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

                # Future horizons: +0.5s, +1.0s, +1.5s, +2.0s, +2.5s, +3.0s
                hist = track_lookup.get(tid, {})
                for h in range(HORIZON):
                    future_fi = fi + STEP_FRAMES * (h + 1)
                    if future_fi in hist:
                        fw_pos = hist[future_fi]
                        f_xe, f_ye = to_curr_ego(fw_pos[0], fw_pos[1])
                        # Trajectory is relative displacement (dx, dy) from current agent position
                        T[k, h] = (f_xe - cx, f_ye - cy)
                        V[k, h] = 1.0

                k += 1
            total_agents += k
            count_val = np.int64(k)
        else:
            count_val = np.int64(0)

        np.savez_compressed(out_npz, boxes=B, count=count_val, traj=T, tvalid=V)
        manifest["frames"][fi]["agent_traj"] = f"agent_traj/{fi:04d}.npz"

        if (fi + 1) % 500 == 0 or (fi + 1) == n_frames:
            fps = (fi + 1) / max(0.001, time.time() - t_start)
            print(f"  [{fi+1}/{n_frames}] {fps:.1f} fps | Tracked agents: {total_agents} | Active tracks: {len(tracker.tracks)}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print(f"[+] Scene {scene} agent trajectories completed in {total_time:.1f}s ({len(tracker.tracks)} persistent tracks).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/meteor_6cam_lidar_deploy/scenes", help="Scenes root directory")
    parser.add_argument("--scenes", default="all", help="all or comma-separated scene list")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_scenes = [
        "data_20260910_061820",
        "data_20260910_062659",
        "data_20260910_063822",
        "data_20260910_064331",
        "data_20260910_073823",
        "data_20260910_074912",
    ]
    if args.scenes == "all":
        target_scenes = all_scenes
    else:
        target_scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print("================================================================================")
    print("  3D MOT & Future 3.0s Agent Trajectory Generation Pipeline")
    print("================================================================================")
    print(f"Scenes: {target_scenes}")
    print(f"Force:  {args.force}")
    print("================================================================================")

    for scene in target_scenes:
        process_scene_trajectories(scene, args)

    print("\n================================================================================")
    print("  All agent trajectories generated successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
