#!/usr/bin/env python3
"""Build 3D Bounding Box GT with Multi-Sweep Pose-Fused LiDAR Points and Camera Confirmation.

Key Innovations:
  1. Multi-sweep Pose Fusion (--n-sweeps 5, default 5):
     Compensates ego-motion across historical frames (t-4..t) via ENU global pose (ego_motion.npz)
     to accumulate dense 3D point clouds (up to 1.15M points per frame).
  2. Ground Plane Asphalt Removal:
     Filters road asphalt points (z < 0.15m) to strictly prevent adjacent vehicles from merging across lanes.
  3. 2D Multi-Camera Semantic Back-Projection (seg2d21):
     Accurately segments Vehicle (Car/Truck/Bus/Moto) and VRU (Bike/Ped) points.
  4. Fast Density-Aware 3D Clustering & cv2.boxPoints Principal Heading Calculation.
  5. Cross-Modal Multi-Camera Confirmation (Camera Confirmation):
     Projects 8 3D box corners to all 6 surround cameras to verify 2D semantic mask overlap.
     Prunes tree leaves, roadside buildings, and flying point noise.
  6. Outputs:
     - bev_box/<fi:04d>.png: uint8 [800, 500] @ 0.2m (1: vehicle, 2: VRU)
     - bev_box/<fi:04d>.npz:
         - boxes:    float32 [N, 6] (cls, cx, cy, l, w, yaw) for METEOR BevLaneDataset
         - boxes_3d: float32 [N, 8] (cls, cx, cy, zc, l, w, h, yaw) for 3D evaluation/visualization
     - Updates manifest.json with "bev_box" and "bev_box_p"

Usage:
  python3 scripts/build_3d_box_gt.py --scenes all --n-sweeps 5 --workers 16
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# METEOR BEV specifications
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2

# Semantic IDs in seg2d21
VEH_CLASSES = [2, 3, 4, 5]  # car, truck, bus, moto
VRU_CLASSES = [6, 7]        # bike, ped


def read_T_ego_lidar(path):
    rows = []
    in_mat = False
    for raw in open(path):
        line = raw.strip()
        if "4x4" in line and ("齐次" in line or "变换" in line):
            in_mat = True
            continue
        if in_mat:
            vals = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)
            if len(vals) == 4:
                rows.append([float(v) for v in vals])
            if len(rows) == 4:
                break
    assert len(rows) == 4, f"cannot parse 4x4 homogeneous matrix from {path}"
    return np.array(rows, np.float64)


def read_pcd_xyz(path):
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            head += line
            if line.startswith(b"DATA"):
                break
        m = re.search(rb"POINTS (\d+)", head)
        if not m:
            return np.zeros((0, 3), np.float64)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


def box_bev_corners(cx, cy, l, w, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    cors = []
    for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
        px = cx + lx * cb - wy * sb
        py = cy + lx * sb + wy * cb
        row = (BEV_XH - px) / RES
        col = (BEV_YH - py) / RES
        cors.append([col, row])
    return np.array(cors, dtype=np.float32)


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def check_camera_confirmation(corners_ego, target_cls, cams_calib, seg, min_px=10, min_ratio=0.10):
    target_sems = VEH_CLASSES if target_cls == 1 else VRU_CLASSES
    for ci, cname in enumerate(CAMS):
        cal = cams_calib[cname]
        K = np.array(cal["K"], dtype=np.float64)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec, t_ec = T_ec[:3, :3], T_ec[:3, 3]

        p_cam = (corners_ego - t_ec) @ R_ec
        z = p_cam[:, 2]
        if (z > 0.5).sum() < 4:
            continue

        u = (K[0, 0] * p_cam[:, 0] / np.maximum(z, 0.5) + K[0, 2]) * 0.25
        v = (K[1, 1] * p_cam[:, 1] / np.maximum(z, 0.5) + K[1, 2]) * 0.25

        u1, u2 = int(np.clip(u.min(), 0, 191)), int(np.clip(u.max(), 0, 191))
        v1, v2 = int(np.clip(v.min(), 0, 107)), int(np.clip(v.max(), 0, 107))

        if u2 - u1 < 2 or v2 - v1 < 2:
            continue

        crop = seg[ci, v1 : v2 + 1, u1 : u2 + 1]
        hits = np.isin(crop, target_sems).sum()
        ratio = hits / float(crop.size)

        if hits >= min_px and ratio >= min_ratio:
            return True
    return False


def process_single_frame(job):
    fi, h_paths, h_poses, curr_pose, seg2d_path, cams_calib, R_el, t_el, png_dst, npz_dst, overwrite = job

    if not overwrite and os.path.exists(png_dst) and os.path.exists(npz_dst):
        return fi, 0, 0

    curr_xk, curr_yk, curr_yawk = curr_pose
    Rk = np.array([[np.cos(curr_yawk), -np.sin(curr_yawk)],
                   [np.sin(curr_yawk), np.cos(curr_yawk)]])

    sweeps_pts = []
    for h_path, h_pose in zip(h_paths, h_poses):
        pts_l = read_pcd_xyz(h_path)
        if len(pts_l) == 0:
            continue
        pts_e = pts_l @ R_el.T + t_el

        # Filter ROI & remove ground asphalt below 0.15m
        valid = (
            (pts_e[:, 0] >= -79.0)
            & (pts_e[:, 0] <= 79.0)
            & (pts_e[:, 1] >= -49.0)
            & (pts_e[:, 1] <= 49.0)
            & (pts_e[:, 2] >= 0.15)
            & (pts_e[:, 2] <= 4.2)
        )
        pts_e = pts_e[valid]
        if len(pts_e) == 0:
            continue

        h_xi, h_yi, h_yawi = h_pose
        if abs(h_xi - curr_xk) < 1e-4 and abs(h_yi - curr_yk) < 1e-4 and abs(h_yawi - curr_yawk) < 1e-4:
            sweeps_pts.append(pts_e)
        else:
            Ri = np.array([[np.cos(h_yawi), -np.sin(h_yawi)],
                           [np.sin(h_yawi), np.cos(h_yawi)]])
            pts_w2d = pts_e[:, :2] @ Ri.T + np.array([h_xi, h_yi])
            pts_k2d = (pts_w2d - np.array([curr_xk, curr_yk])) @ Rk
            comp_pts = np.zeros_like(pts_e)
            comp_pts[:, :2] = pts_k2d
            comp_pts[:, 2] = pts_e[:, 2]
            sweeps_pts.append(comp_pts)

    if len(sweeps_pts) == 0:
        cv2.imwrite(png_dst, np.zeros((BEV_H, BEV_W), np.uint8))
        np.savez_compressed(npz_dst, boxes=np.zeros((0, 6), np.float32), boxes_3d=np.zeros((0, 8), np.float32))
        return fi, 0, 0

    pts_e = np.vstack(sweeps_pts)

    seg = np.load(seg2d_path)["seg"] if os.path.exists(seg2d_path) else None
    if seg is None:
        cv2.imwrite(png_dst, np.zeros((BEV_H, BEV_W), np.uint8))
        np.savez_compressed(npz_dst, boxes=np.zeros((0, 6), np.float32), boxes_3d=np.zeros((0, 8), np.float32))
        return fi, 0, 0

    # Multi-camera semantic back-projection
    pt_cls = np.zeros(len(pts_e), dtype=np.uint8)
    for ci, cname in enumerate(CAMS):
        cal = cams_calib[cname]
        K = np.array(cal["K"], dtype=np.float64)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec, t_ec = T_ec[:3, :3], T_ec[:3, 3]

        p_cam = (pts_e - t_ec) @ R_ec
        z = p_cam[:, 2]
        front = (z > 0.5) & (z < 75.0)
        if not np.any(front):
            continue

        u = (K[0, 0] * p_cam[:, 0] / z + K[0, 2]) * 0.25
        v = (K[1, 1] * p_cam[:, 1] / z + K[1, 2]) * 0.25
        in_fov = front & (u >= 0) & (u < 192) & (v >= 0) & (v < 108)

        u_idx = u[in_fov].astype(np.int32)
        v_idx = v[in_fov].astype(np.int32)
        sem = seg[ci, v_idx, u_idx]

        veh = np.isin(sem, VEH_CLASSES)
        vru = np.isin(sem, VRU_CLASSES)
        fov_indices = np.where(in_fov)[0]

        pt_cls[fov_indices[veh]] = 1
        pt_cls[fov_indices[vru]] = 2

    bev = np.zeros((BEV_H, BEV_W), np.uint8)
    boxes_out = []
    boxes_3d_out = []
    cand_boxes = []

    # Cluster configs: (target_cls, min_l, max_l, min_w, max_w, min_h, max_h, min_pts, min_px, vres)
    configs = [
        (1, 1.8, 16.0, 1.1, 3.8, 0.8, 4.5, 25, 12, 0.22),
        (2, 0.3, 2.2, 0.3, 1.8, 0.8, 2.3, 8, 8, 0.20),
    ]

    for target_cls, min_l, max_l, min_w, max_w, min_h, max_h, min_pts, min_px, vres in configs:
        t_pts = pts_e[pt_cls == target_cls]
        if len(t_pts) < min_pts:
            continue

        vx = ((t_pts[:, 0] + 80.0) / vres).astype(np.int32)
        vy = ((t_pts[:, 1] + 50.0) / vres).astype(np.int32)
        gh = int(160.0 / vres)
        gw = int(100.0 / vres)
        occ = np.zeros((gh, gw), dtype=np.uint8)
        valid_v = (vx >= 0) & (vx < gh) & (vy >= 0) & (vy < gw)
        occ[vx[valid_v], vy[valid_v]] = 1

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)

        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 3:
                continue
            pt_mask = labels[vx[valid_v], vy[valid_v]] == lbl
            c_pts = t_pts[valid_v][pt_mask]
            if len(c_pts) < min_pts:
                continue

            pts_2d = c_pts[:, :2].astype(np.float32)
            rect = cv2.minAreaRect(pts_2d)
            (cx, cy), _, _ = rect
            box_pts = cv2.boxPoints(rect)
            v01 = box_pts[1] - box_pts[0]
            v12 = box_pts[2] - box_pts[1]
            len01 = np.linalg.norm(v01)
            len12 = np.linalg.norm(v12)
            if len01 > len12:
                major_vec = v01
                l, w = len01, len12
            else:
                major_vec = v12
                l, w = len12, len01
            yaw = float(np.arctan2(major_vec[1], major_vec[0]))

            zmin = float(np.percentile(c_pts[:, 2], 2))
            zmax = float(np.percentile(c_pts[:, 2], 98))
            h = max(zmax - zmin, 1.0)
            zc = (zmin + zmax) / 2.0

            if not (min_l <= l <= max_l and min_w <= w <= max_w and min_h <= h <= max_h):
                continue

            corners_3d = box_3d_corners(cx, cy, zc, l, w, h, yaw)
            if not check_camera_confirmation(corners_3d, target_cls, cams_calib, seg, min_px=min_px):
                continue

            cand_boxes.append({
                "cls": target_cls,
                "cx": cx, "cy": cy, "zc": zc,
                "l": l, "w": w, "h": h,
                "yaw": yaw,
                "pts": len(c_pts),
            })

    # Non-maximum suppression / deduplication
    cand_boxes.sort(key=lambda b: b["pts"], reverse=True)
    kept_boxes = []
    for b in cand_boxes:
        overlap = False
        for kb in kept_boxes:
            dist = np.hypot(b["cx"] - kb["cx"], b["cy"] - kb["cy"])
            if dist < max(b["l"], kb["l"]) * 0.75:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(b)

    for b in kept_boxes:
        cors_b = box_bev_corners(b["cx"], b["cy"], b["l"], b["w"], b["yaw"])
        cv2.fillPoly(bev, [np.round(cors_b).astype(np.int32).reshape(-1, 1, 2)], int(b["cls"]))
        boxes_out.append([b["cls"], b["cx"], b["cy"], b["l"], b["w"], b["yaw"]])
        boxes_3d_out.append([b["cls"], b["cx"], b["cy"], b["zc"], b["l"], b["w"], b["h"], b["yaw"]])

    cv2.imwrite(png_dst, bev)
    np.savez_compressed(
        npz_dst,
        boxes=np.array(boxes_out, np.float32) if boxes_out else np.zeros((0, 6), np.float32),
        boxes_3d=np.array(boxes_3d_out, np.float32) if boxes_3d_out else np.zeros((0, 8), np.float32),
    )

    n_veh = sum(1 for b in boxes_out if b[0] == 1)
    n_vru = sum(1 for b in boxes_out if b[0] == 2)
    return fi, n_veh, n_vru


def process_scene(scene_name, raw_dir, scene_dir, n_sweeps=5, workers=16, limit=0, overwrite=True):
    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        print(f"[-] Missing {mf_path}, skipping.")
        return

    man = json.load(open(mf_path))
    cams_calib = man["cams"]
    frames = man["frames"]
    if limit > 0:
        frames = frames[:limit]
    F = len(frames)

    calib_lidar_path = os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt")
    T_el = read_T_ego_lidar(calib_lidar_path)
    R_el, t_el = T_el[:3, :3], T_el[:3, 3]

    lidar_dir = os.path.join(raw_dir, "lidar")
    pcd_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd"))

    # Load 2D ENU ego poses for motion compensation
    ego_motion_path = os.path.join(scene_dir, "ego_motion.npz")
    if os.path.exists(ego_motion_path) and "pose" in np.load(ego_motion_path):
        poses = np.load(ego_motion_path)["pose"]
    else:
        poses = np.zeros((F, 3), np.float32)

    bev_box_dir = os.path.join(scene_dir, "bev_box")
    os.makedirs(bev_box_dir, exist_ok=True)

    print(f"[*] Processing 3D Box ({n_sweeps}-sweep pose fusion) for {F} frames in {scene_name} ...", flush=True)
    t0 = time.time()

    jobs = []
    for fi in range(F):
        start_fi = max(0, fi - n_sweeps + 1)
        h_paths = [os.path.join(lidar_dir, pcd_files[hi]) for hi in range(start_fi, fi + 1)]
        h_poses = [poses[hi] for hi in range(start_fi, fi + 1)]
        curr_pose = poses[fi]

        seg2d_path = os.path.join(scene_dir, f"seg2d21/{fi:04d}.npz")
        png_dst = os.path.join(bev_box_dir, f"{fi:04d}.png")
        npz_dst = os.path.join(bev_box_dir, f"{fi:04d}.npz")
        jobs.append((fi, h_paths, h_poses, curr_pose, seg2d_path, cams_calib, R_el, t_el, png_dst, npz_dst, overwrite))

    tot_veh, tot_vru = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (fi, n_veh, n_vru) in enumerate(ex.map(process_single_frame, jobs)):
            tot_veh += n_veh
            tot_vru += n_vru
            if (i + 1) % 500 == 0 or (i + 1) == F:
                elapsed = time.time() - t0
                fps = (i + 1) / max(elapsed, 1e-3)
                print(f"  [{i+1}/{F}] {fps:.1f} fps | Total veh: {tot_veh}, vru: {tot_vru}", flush=True)

    # Update manifest.json
    for fr in man["frames"]:
        fi = fr["frame"]
        fr["bev_box"] = f"bev_box/{fi:04d}.png"
        fr["bev_box_p"] = f"bev_box/{fi:04d}.npz"
    json.dump(man, open(mf_path, "w"))

    print(f"[+] Finished {scene_name}: {tot_veh} vehicles, {tot_vru} VRUs in {time.time() - t0:.1f}s\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build 3D Box and BEV Box GT with Multi-Sweep Pose Fusion")
    ap.add_argument("--scenes", default="all", help="all or comma-separated scene names")
    ap.add_argument("--n-sweeps", type=int, default=5, help="Number of historical LiDAR sweeps to fuse (default 5)")
    ap.add_argument("--workers", type=int, default=16, help="Process workers")
    ap.add_argument("--limit", type=int, default=0, help="Limit frames per scene")
    ap.add_argument("--no-overwrite", action="store_true", help="Skip existing files")
    args = ap.parse_args()

    scenes_dir = os.path.join(BASE_DIR, "scenes")
    raw_base = os.path.join(BASE_DIR, "raw_data")

    all_scene_names = sorted(d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d)))

    if args.scenes != "all":
        req = set(s.strip() for s in args.scenes.split(","))
        target_scenes = [s for s in all_scene_names if s in req]
    else:
        target_scenes = all_scene_names

    print(f"Target scenes ({len(target_scenes)}): {target_scenes}")
    print(f"LiDAR fusion sweeps: {args.n_sweeps}")
    t_start = time.time()

    for s_name in target_scenes:
        s_dir = os.path.join(scenes_dir, s_name)
        r_dir = os.path.join(raw_base, s_name)
        process_scene(s_name, r_dir, s_dir, n_sweeps=args.n_sweeps, workers=args.workers, limit=args.limit, overwrite=not args.no_overwrite)

    print(f"==================================================")
    print(f"[ALL COMPLETE] Total time: {(time.time() - t_start)/60:.2f} min")
    print(f"==================================================")


if __name__ == "__main__":
    main()
