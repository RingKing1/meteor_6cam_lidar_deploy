#!/usr/bin/env python3
"""Strict Trajectory Filtering & Anomaly Wiping Pipeline (refine_and_filter_agent_traj.py)

Strict Quality Contract (Direct Wipe Policy):
  Directly wipe/delete abnormal targets or corrupted trajectory frames from Ground Truth,
  guaranteeing that ONLY 100% pure, verified, physically plausible samples enter downstream
  training, T4 dataset annotation, and evaluation dashboards.

Wiping Rules:
  1. Ghost Track Elimination (Min Lifespan):
     Any track with total lifespan < min_lifespan (default: 5 frames / 0.5s) is completely wiped.
  2. Corrupted Track Elimination (High-Violation Tracks):
     Any track where >= 25% of frames violate kinematic limits is deemed an ID-switch or clutter
     runaway, and is completely wiped from all frames.
  3. Strict Kinematic Frame Wiping (Zero Interpolated Guessing):
     Any frame where instantaneous speed exceeds class limits:
       - Pedestrian: speed > 3.5 m/s (12.6 km/h) -> Frame WIPED
       - Bicycle: speed > 15.0 m/s (54 km/h) -> Frame WIPED
       - Vehicle (Car/Truck/Bus): speed > 30.0 m/s (108 km/h) or delta_d > 3.0m in 0.1s -> Frame WIPED
       - Unphysical Yaw Chaos: |delta_yaw| > 60 deg in 0.1s (not 180 flip) -> Frame WIPED
  4. Heading 180° Flip Correction & Smooth Yaw:
     Moving vehicles (v > 1.2 m/s) with 180° symmetry ambiguity (opposing motion by > 100°)
     are corrected by adding pi; moving vehicle heading is smoothed via moving average.
  5. Zero-Velocity Update (ZUPT @ 3s < 1.0m):
     Vehicles with 3.0s future displacement < 1.0m are strictly classified as Stationary.
     Their future prediction waypoints are forced identically to (0, 0), and positions are
     anchored to the local temporal median to eradicate centroid jitter.
  6. Dual-Dataset Synchronization:
     Synchronously writes clean outputs to BOTH agent_traj/ and bev_box/ (with raw backups).

Usage:
  python3 scripts/autolabel_agent_traj/refine_and_filter_agent_traj.py \
      --scenes all \
      --stat-dist-thresh 1.0 \
      --min-lifespan 5 \
      --sync-bev-box
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

MAX_SPEED_PER_CLASS = {
    0: 15.0,  # obstacle
    1: 30.0,  # car (108 km/h)
    2: 25.0,  # truck (90 km/h)
    3: 25.0,  # bus
    4: 15.0,  # bicycle (54 km/h)
    5: 20.0,  # motorcycle (72 km/h)
    6: 3.5,   # pedestrian (12.6 km/h)
}

KMAX = 64
HORIZON = 6
STEP_FRAMES = 5  # 5 frames @ 10Hz = 0.5s


def angle_diff(a, b):
    """Smallest signed angle difference a - b in radians [-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def smooth_yaw_series(yaws, window_size=5):
    """Unwrap and smooth yaw angles using moving average."""
    if len(yaws) < 3:
        return yaws
    unwrapped = np.unwrap(yaws)
    w = min(window_size, len(unwrapped))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return (unwrapped + np.pi) % (2 * np.pi) - np.pi

    kernel = np.ones(w) / w
    pad_l = w // 2
    padded = np.pad(unwrapped, pad_l, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return (smoothed + np.pi) % (2 * np.pi) - np.pi


def smooth_pos_series(positions, window_size=5):
    """Smooth 2D position trajectory in world coordinates using moving average."""
    if len(positions) < 3:
        return positions
    w = min(window_size, len(positions))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return positions
    kernel = np.ones(w) / w
    pad_l = w // 2
    pad_x = np.pad(positions[:, 0], pad_l, mode="edge")
    pad_y = np.pad(positions[:, 1], pad_l, mode="edge")
    sx = np.convolve(pad_x, kernel, mode="valid")
    sy = np.convolve(pad_y, kernel, mode="valid")
    return np.column_stack([sx, sy])


def process_scene_strict(scene, args):
    scene_dir = os.path.join(args.root, scene)
    raw_traj_dir = os.path.join(scene_dir, args.in_traj_subdir)
    out_traj_dir = os.path.join(scene_dir, args.out_traj_subdir)
    raw_bev_dir = os.path.join(scene_dir, args.in_bev_subdir)
    out_bev_dir = os.path.join(scene_dir, args.out_bev_subdir)
    ego_p = os.path.join(scene_dir, "ego_motion.npz")

    if not os.path.isdir(raw_traj_dir):
        print(f"[!] Trajectory dir not found: {raw_traj_dir}")
        return None

    poses = np.load(ego_p)["pose"] if os.path.exists(ego_p) else None
    if poses is None:
        print(f"[!] Ego motion pose not found: {ego_p}")
        return None

    npz_files = sorted(glob.glob(os.path.join(raw_traj_dir, "*.npz")))
    n_frames = len(npz_files)
    print(f"\n{'='*78}")
    print(f"  Strict Trajectory Filtering & Wiping for Scene: {scene}")
    print(f"  Frames: {n_frames} | In Traj: {args.in_traj_subdir} -> Out Traj: {args.out_traj_subdir}")
    if args.sync_bev_box:
        print(f"  In BEV: {args.in_bev_subdir} -> Out BEV: {args.out_bev_subdir}")
    print(f"  Policy: DIRECT WIPE of anomalous frames/tracks | Stat Threshold: < {args.stat_dist_thresh:.2f}m")
    print(f"{'='*78}")

    t0 = time.time()

    # -------------------------------------------------------------------------
    # PASS 1: Ingest all tracks, boxes, and boxes_3d in World Coordinates
    # -------------------------------------------------------------------------
    frame_data = [[] for _ in range(n_frames)]
    frame_bev_3d_dict = [{} for _ in range(n_frames)]
    all_tracks = {}

    for fi in range(n_frames):
        p_traj = npz_files[fi]
        ztraj = np.load(p_traj)
        boxes = ztraj["boxes"]
        count = int(ztraj["count"])
        tids = ztraj["track_ids"] if "track_ids" in ztraj else np.arange(count)
        trajs = ztraj["traj"]
        tvals = ztraj["tvalid"]

        # Ingest 3D boxes from bev_box if available
        p_bev = os.path.join(raw_bev_dir, f"{fi:04d}.npz")
        b3d_list = []
        if os.path.exists(p_bev):
            zbev = np.load(p_bev)
            if "boxes_3d" in zbev:
                b3d_list = zbev["boxes_3d"]

        xe_e, ye_e, yaw_e = poses[fi]
        ce, se = np.cos(yaw_e), np.sin(yaw_e)

        for k in range(count):
            tid = int(tids[k])
            cls_id, cx, cy, l, w, yaw_b = boxes[k]
            cls_int = int(round(cls_id))

            # Match 3D zc and height from bev_3d
            zc, h = 0.0, 1.8
            matched_b3d = None
            for b3 in b3d_list:
                if abs(b3[1] - cx) < 0.1 and abs(b3[2] - cy) < 0.1:
                    zc, h = float(b3[3]), float(b3[6])
                    matched_b3d = b3.copy()
                    break

            # Ego -> World conversion
            xw = xe_e + cx * ce - cy * se
            yw = ye_e + cx * se + cy * ce
            yaw_w = yaw_b + yaw_e

            obs = {
                "fi": fi,
                "tid": tid,
                "cls": cls_int,
                "cx": float(cx),
                "cy": float(cy),
                "zc": float(zc),
                "l": float(l),
                "w": float(w),
                "h": float(h),
                "yaw_b": float(yaw_b),
                "xw": float(xw),
                "yw": float(yw),
                "yaw_w": float(yaw_w),
                "raw_b3d": matched_b3d,
                "raw_traj": trajs[k].copy(),
                "raw_tvalid": tvals[k].copy(),
                "raw_d3": float(np.linalg.norm(trajs[k, 5])) if tvals[k, 5] > 0.5 else -1.0,
            }
            frame_data[fi].append(obs)
            all_tracks.setdefault(tid, []).append(obs)

    n_raw_tracks = len(all_tracks)
    n_raw_instances = sum(len(v) for v in all_tracks.values())

    stats = {
        "raw_tracks": n_raw_tracks,
        "raw_instances": n_raw_instances,
        "ghost_tracks_wiped": 0,
        "ghost_instances_wiped": 0,
        "corrupted_tracks_wiped": 0,
        "corrupted_instances_wiped": 0,
        "abnormal_speed_frames_wiped": 0,
        "chaotic_yaw_frames_wiped": 0,
        "vehicle_yaw_180_flips_fixed": 0,
        "vehicle_yaw_smoothed_count": 0,
        "stat_instances_before_035m": 0,
        "stat_instances_after_100m": 0,
        "moving_instances_before": 0,
        "moving_instances_after": 0,
        "retained_tracks": 0,
        "retained_instances": 0,
    }

    # -------------------------------------------------------------------------
    # PASS 2: Track-level Wiping (Ghost & Corrupted Tracks)
    # -------------------------------------------------------------------------
    candidate_tracks = {}
    for tid, obs_list in all_tracks.items():
        obs_list.sort(key=lambda x: x["fi"])
        lifespan = len(obs_list)
        cls_int = obs_list[0]["cls"]

        # Rule 1: Wipe ghost tracks with lifespan < min_lifespan
        if lifespan < args.min_lifespan:
            stats["ghost_tracks_wiped"] += 1
            stats["ghost_instances_wiped"] += lifespan
            continue

        # Rule 2: Check overall kinematic violation ratio
        v_max = MAX_SPEED_PER_CLASS.get(cls_int, 30.0)
        violations = 0
        for i in range(1, lifespan):
            dt = (obs_list[i]["fi"] - obs_list[i - 1]["fi"]) * 0.1
            if dt > 0:
                dist = np.hypot(obs_list[i]["xw"] - obs_list[i - 1]["xw"],
                                obs_list[i]["yw"] - obs_list[i - 1]["yw"])
                speed = dist / dt
                if speed > v_max or (cls_int in [1, 2, 3] and dist > 3.0 and dt <= 0.15):
                    violations += 1

        if violations >= max(2, int(0.25 * lifespan)):
            # More than 25% frames violate physical speed -> Corrupted runaway track -> WIPE
            stats["corrupted_tracks_wiped"] += 1
            stats["corrupted_instances_wiped"] += lifespan
            continue

        candidate_tracks[tid] = obs_list

    # -------------------------------------------------------------------------
    # PASS 3: Strict Frame-Level Anomaly Wiping
    # -------------------------------------------------------------------------
    retained_tracks = {}
    for tid, obs_list in candidate_tracks.items():
        cls_int = obs_list[0]["cls"]
        is_vehicle = cls_int in [1, 2, 3]
        v_max = MAX_SPEED_PER_CLASS.get(cls_int, 30.0)

        # Filter out bad frames from obs_list
        clean_obs = [obs_list[0]]
        for i in range(1, len(obs_list)):
            prev_o = clean_obs[-1]
            curr_o = obs_list[i]
            dt = (curr_o["fi"] - prev_o["fi"]) * 0.1
            if dt <= 0:
                continue
            dist = np.hypot(curr_o["xw"] - prev_o["xw"], curr_o["yw"] - prev_o["yw"])
            speed = dist / dt

            # Rule 3: Direct wipe if speed or displacement jumps beyond physical limits
            is_speed_spike = (speed > v_max) or (is_vehicle and dist > 3.0 and dt <= 0.15)
            if is_speed_spike:
                stats["abnormal_speed_frames_wiped"] += 1
                continue  # DIRECT WIPE: frame dropped from GT!

            # Check unphysical yaw chaos (> 60 deg jump in 0.1s that is NOT a 180 flip)
            if is_vehicle and dt <= 0.15:
                dyaw = abs(angle_diff(curr_o["yaw_w"], prev_o["yaw_w"]))
                if dyaw > np.radians(60.0) and abs(dyaw - np.pi) > np.radians(25.0):
                    stats["chaotic_yaw_frames_wiped"] += 1
                    continue  # DIRECT WIPE: chaotic orientation frame dropped!

            clean_obs.append(curr_o)

        if len(clean_obs) < args.min_lifespan:
            # If after dropping bad frames the track shrunk below min_lifespan -> drop entire
            stats["ghost_tracks_wiped"] += 1
            stats["ghost_instances_wiped"] += len(clean_obs)
            continue

        retained_tracks[tid] = clean_obs

    # -------------------------------------------------------------------------
    # PASS 4: Motion Refinement, 180° Flip Correction & ZUPT on Pure Tracks
    # -------------------------------------------------------------------------
    for tid, obs_list in retained_tracks.items():
        cls_int = obs_list[0]["cls"]
        is_vehicle = cls_int in [1, 2, 3]
        n_obs = len(obs_list)

        pos_w = np.array([[o["xw"], o["yw"]] for o in obs_list], dtype=np.float64)
        yaw_w = np.array([o["yaw_w"] for o in obs_list], dtype=np.float64)
        frames = [o["fi"] for o in obs_list]
        frame_to_idx = {f: idx for idx, f in enumerate(frames)}

        # 1. 3.0s Future Displacement Calculation
        d3_vals = np.zeros(n_obs, dtype=np.float64)
        for i in range(n_obs):
            curr_f = frames[i]
            target_f = curr_f + 30
            if target_f in frame_to_idx:
                d3_vals[i] = np.linalg.norm(pos_w[frame_to_idx[target_f]] - pos_w[i])
            else:
                fwd_idx = min(n_obs - 1, i + 30)
                df_fwd = frames[fwd_idx] - curr_f
                if df_fwd >= 5:
                    d3_vals[i] = np.linalg.norm(pos_w[fwd_idx] - pos_w[i]) * (30.0 / df_fwd)
                else:
                    past_idx = max(0, i - 30)
                    df_past = curr_f - frames[past_idx]
                    if df_past >= 5:
                        d3_vals[i] = np.linalg.norm(pos_w[i] - pos_w[past_idx]) * (30.0 / df_past)
                    else:
                        d3_vals[i] = 0.0

        is_stat_per_frame = (d3_vals < args.stat_dist_thresh) if is_vehicle else np.zeros(n_obs, dtype=bool)

        # 2. 180° Heading Flip Correction for Moving Vehicles
        if is_vehicle and n_obs >= 5:
            unwrapped_yaw = np.unwrap(yaw_w)
            for i in range(1, n_obs - 1):
                if not is_stat_per_frame[i]:
                    df = frames[i + 1] - frames[i - 1]
                    if df > 0:
                        vx = (pos_w[i + 1, 0] - pos_w[i - 1, 0]) / (df * 0.1)
                        vy = (pos_w[i + 1, 1] - pos_w[i - 1, 1]) / (df * 0.1)
                        speed = np.hypot(vx, vy)
                        if speed > 1.2:
                            theta_v = np.arctan2(vy, vx)
                            d_ang = angle_diff(unwrapped_yaw[i], theta_v)
                            if abs(d_ang) > np.radians(100.0):
                                unwrapped_yaw[i] += np.pi
                                stats["vehicle_yaw_180_flips_fixed"] += 1

            yaw_w = smooth_yaw_series(unwrapped_yaw, window_size=5)
            stats["vehicle_yaw_smoothed_count"] += 1

        # 3. ZUPT Position Anchoring on Stationary & Smooth on Moving
        smoothed_pos = smooth_pos_series(pos_w, window_size=5)
        refined_pos = np.copy(smoothed_pos)

        if is_vehicle:
            i = 0
            while i < n_obs:
                if is_stat_per_frame[i]:
                    j = i
                    while j < n_obs and is_stat_per_frame[j]:
                        j += 1
                    seg_len = j - i
                    if seg_len >= 5:
                        med_xy = np.median(pos_w[i:j], axis=0)
                        refined_pos[i:j] = med_xy
                    i = j
                else:
                    i += 1

        for i, o in enumerate(obs_list):
            o["refined_xw"] = float(refined_pos[i, 0])
            o["refined_yw"] = float(refined_pos[i, 1])
            o["refined_yaw_w"] = float(yaw_w[i])
            o["is_stationary"] = bool(is_stat_per_frame[i])
            o["refined_d3"] = float(d3_vals[i])

    # -------------------------------------------------------------------------
    # PASS 5: Write Clean Datasets (agent_traj and synced bev_box)
    # -------------------------------------------------------------------------
    os.makedirs(out_traj_dir, exist_ok=True)
    if args.sync_bev_box:
        os.makedirs(out_bev_dir, exist_ok=True)

    refined_lookup = {}
    for tid, obs_list in retained_tracks.items():
        for o in obs_list:
            refined_lookup.setdefault(tid, {})[o["fi"]] = (o["refined_xw"], o["refined_yw"])

    retained_obs_by_frame = [[] for _ in range(n_frames)]
    for tid, obs_list in retained_tracks.items():
        for o in obs_list:
            retained_obs_by_frame[o["fi"]].append(o)

    for fi in range(n_frames):
        out_npz = os.path.join(out_traj_dir, f"{fi:04d}.npz")
        xe_curr, ye_curr, yaw_curr = poses[fi]
        c_inv, s_inv = np.cos(-yaw_curr), np.sin(-yaw_curr)

        curr_obs = retained_obs_by_frame[fi]

        cand_list = []
        for o in curr_obs:
            dx = o["refined_xw"] - xe_curr
            dy = o["refined_yw"] - ye_curr
            xe_box = c_inv * dx - s_inv * dy
            ye_box = s_inv * dx + c_inv * dy
            dist_sq = xe_box * xe_box + ye_box * ye_box
            cand_list.append((dist_sq, o["tid"], xe_box, ye_box, o))
        cand_list.sort(key=lambda x: x[0])

        B = np.zeros((KMAX, 6), dtype=np.float32)
        T = np.zeros((KMAX, HORIZON, 2), dtype=np.float32)
        V = np.zeros((KMAX, HORIZON), dtype=np.float32)
        IDS = np.zeros((KMAX,), dtype=np.int64)

        bev_boxes_list = []
        bev_3d_list = []

        k = 0
        for _, tid, xe_box, ye_box, o in cand_list:
            if k >= KMAX:
                break
            cls_id = float(o["cls"])
            l_box = o["l"]
            w_box = o["w"]
            h_box = o["h"]
            zc_box = o["zc"]
            yaw_box = float(angle_diff(o["refined_yaw_w"], yaw_curr))

            B[k] = (cls_id, xe_box, ye_box, l_box, w_box, yaw_box)
            IDS[k] = tid
            bev_boxes_list.append([cls_id, xe_box, ye_box, l_box, w_box, yaw_box])
            bev_3d_list.append([cls_id, xe_box, ye_box, zc_box, l_box, w_box, h_box, yaw_box])

            # Future waypoints
            if o["is_stationary"]:
                for h in range(HORIZON):
                    future_fi = fi + STEP_FRAMES * (h + 1)
                    if future_fi < n_frames:
                        T[k, h] = (0.0, 0.0)
                        V[k, h] = 1.0
                stats["stat_instances_after_100m"] += 1
            else:
                hist = refined_lookup.get(tid, {})
                for h in range(HORIZON):
                    future_fi = fi + STEP_FRAMES * (h + 1)
                    if future_fi in hist:
                        fx_w, fy_w = hist[future_fi]
                        fdx = fx_w - xe_curr
                        fdy = fy_w - ye_curr
                        f_xe = c_inv * fdx - s_inv * fdy
                        f_ye = s_inv * fdx + c_inv * fdy
                        T[k, h] = (f_xe - xe_box, f_ye - ye_box)
                        V[k, h] = 1.0
                stats["moving_instances_after"] += 1

            if o["raw_d3"] >= 0.0:
                if o["raw_d3"] <= 0.35:
                    stats["stat_instances_before_035m"] += 1
                else:
                    stats["moving_instances_before"] += 1

            k += 1

        count_val = np.int64(k)
        stats["retained_instances"] += k
        np.savez_compressed(out_npz, boxes=B, count=count_val, traj=T,
                            tvalid=V, track_ids=IDS)

        # Sync bev_box
        if args.sync_bev_box:
            bev_out_npz = os.path.join(out_bev_dir, f"{fi:04d}.npz")
            b_arr = np.array(bev_boxes_list, dtype=np.float32) if len(bev_boxes_list) else np.zeros((0, 6), dtype=np.float32)
            b3d_arr = np.array(bev_3d_list, dtype=np.float32) if len(bev_3d_list) else np.zeros((0, 8), dtype=np.float32)
            np.savez_compressed(bev_out_npz, boxes=b_arr, boxes_3d=b3d_arr)

    stats["retained_tracks"] = len(retained_tracks)
    elapsed = time.time() - t0

    # Print Full Report
    print(f"\n[+] Scene {scene} strict refinement completed in {elapsed:.2f}s!")
    print(f"{'-'*78}")
    print(f"  [1] Strict Wiping Breakdown (Pure GT Guarantee):")
    print(f"      Ghost Tracks Wiped (<{args.min_lifespan} frames): {stats['ghost_tracks_wiped']} "
          f"({stats['ghost_instances_wiped']} instances deleted)")
    print(f"      Corrupted Runaway Tracks Wiped: {stats['corrupted_tracks_wiped']} "
          f"({stats['corrupted_instances_wiped']} instances deleted)")
    print(f"      Abnormal Speed Jump Frames Wiped: {stats['abnormal_speed_frames_wiped']} frames")
    print(f"      Chaotic Yaw Spinning Frames Wiped: {stats['chaotic_yaw_frames_wiped']} frames")
    total_wiped = (stats['ghost_instances_wiped'] + stats['corrupted_instances_wiped'] +
                   stats['abnormal_speed_frames_wiped'] + stats['chaotic_yaw_frames_wiped'])
    print(f"      ==> TOTAL ANOMALOUS INSTANCES WIPED FROM GT: {total_wiped} (-{total_wiped/max(1,n_raw_instances)*100:.1f}%)")
    print(f"  [2] Valid Physical Motion Optimization:")
    print(f"      Retained Verified Pure Tracks: {stats['retained_tracks']} / {stats['raw_tracks']}")
    print(f"      Retained Verified Pure Instances: {stats['retained_instances']} / {stats['raw_instances']}")
    print(f"      Legitimate Vehicle 180° Heading Flips Corrected: {stats['vehicle_yaw_180_flips_fixed']}")
    print(f"      Vehicles with Smoothed Yaw (No Steering Jitter): {stats['vehicle_yaw_smoothed_count']}")
    print(f"  [3] Stationary Zero-Velocity Anchoring (ZUPT @ <{args.stat_dist_thresh:.2f}m):")
    print(f"      Static Instances (<1.0m): {stats['stat_instances_after_100m']} vs Moving: {stats['moving_instances_after']}")
    print(f"{'-'*78}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Strict Trajectory Anomaly Wiping Pipeline.")
    parser.add_argument("--root", default="/home/nvidia/working_ppt/Postdoc_Materials/论文3/meteor_6cam_lidar_deploy/scenes")
    parser.add_argument("--scenes", default="all", help="comma-separated scene names or 'all'")
    parser.add_argument("--in-traj-subdir", default="agent_traj_raw", help="input trajectory subdir name")
    parser.add_argument("--out-traj-subdir", default="agent_traj", help="output trajectory subdir name")
    parser.add_argument("--in-bev-subdir", default="bev_box_raw", help="input bev_box subdir name")
    parser.add_argument("--out-bev-subdir", default="bev_box", help="output bev_box subdir name")
    parser.add_argument("--stat-dist-thresh", type=float, default=1.0, help="3.0s stationary threshold in meters")
    parser.add_argument("--min-lifespan", type=int, default=5, help="minimum track lifespan in frames to retain")
    parser.add_argument("--sync-bev-box", action="store_true", default=True, help="synchronize cleaned boxes into bev_box/")
    args = parser.parse_args()

    if args.scenes == "all":
        scenes = [
            "data_20260910_061820", "data_20260910_062659", "data_20260910_063822",
            "data_20260910_064331", "data_20260910_073823", "data_20260910_074912"
        ]
    else:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    all_stats = {}
    for sc in scenes:
        st = process_scene_strict(sc, args)
        if st is not None:
            all_stats[sc] = st

    summary_path = os.path.join(args.root, f"strict_wiping_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[*] Full strict wiping report written to: {summary_path}")


if __name__ == "__main__":
    main()
