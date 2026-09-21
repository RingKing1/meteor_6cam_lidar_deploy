#!/usr/bin/env python3
"""3D LiDAR Point Labeling, Global Map Accumulation & BEV Lane GT Extraction.

Pipelines:
  1. Projects LiDAR points onto 6 cameras, samples 2D surface class (lane, road, etc.)
  2. Accumulates labeled points across the scene in global map frame (0.1m grid)
  3. Stretches road corridor under the ego trajectory (fills blind zones)
  4. Resolves cell classes (thin classes override area classes)
  5. Crops per-frame ego BEV GT (800x500 @ 0.2m) -> gt/<fi:04d>.png
  6. Updates manifest.json with "gt" paths

Usage:
  python3 scripts/build_bev_gt.py \
    --raw METEOR/data/data_20260910_063822 \
    --scene scenes/data_20260910_063822 \
    --workers 16
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# BEV Grid specifications matching METEOR
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH = 80.0, 50.0
RES = 0.2
MAP_RES = 0.1       # 0.1m resolution for map accumulation
EGO_FILL_R = 1.75   # half a lane width (m) for ego blind zone filling

CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

THIN_CLASSES = {4, 5, 6, 7}  # laneline, stopline, road_edge, marking


def read_pcd_xyz(path):
    """Fast binary PCD reader."""
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            head += line
            if line.startswith(b"DATA"):
                break
        m = re.search(rb"POINTS (\d+)", head)
        if not m:
            return np.zeros((0, 3), np.float32)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


def label_single_sweep(args):
    """Worker: project one LiDAR sweep onto 6 camera surface masks."""
    pcd_path, mask_npz_path, cams_calib, pose_xyyaw, R_el, t_el = args
    if not os.path.exists(pcd_path) or not os.path.exists(mask_npz_path):
        return None

    # Load PCD and transform to ego frame
    pts_l = read_pcd_xyz(pcd_path)
    if len(pts_l) == 0:
        return None
    pts_e = pts_l @ R_el.T + t_el

    # Filter near-ground points (-1.5m <= z <= 3.5m, range <= 60m)
    dist2 = pts_e[:, 0]**2 + pts_e[:, 1]**2
    valid_pts = (dist2 <= 3600.0) & (pts_e[:, 2] >= -1.5) & (pts_e[:, 2] <= 3.5)
    pts_e = pts_e[valid_pts]
    if len(pts_e) == 0:
        return None

    # Load 6 camera surface masks [6, 432, 768]
    masks = np.load(mask_npz_path)["mask"]

    # Transform to global frame
    xg0, yg0, yaw0 = pose_xyyaw
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    xg = c0 * pts_e[:, 0] - s0 * pts_e[:, 1] + xg0
    yg = s0 * pts_e[:, 0] + c0 * pts_e[:, 1] + yg0
    zg = pts_e[:, 2]

    # Best label per point (thin class priority > area class)
    point_label = np.zeros(len(pts_e), dtype=np.uint8)

    for ci, c_name in enumerate(CAMS):
        cal = cams_calib[c_name]
        K = np.array(cal["K"], dtype=np.float64)
        T_ego_cam = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec = T_ego_cam[:3, :3]
        t_ec = T_ego_cam[:3, 3]

        # Ego -> Cam: p_cam = (p_ego - t_ec) @ R_ec
        p_cam = (pts_e - t_ec) @ R_ec
        z = p_cam[:, 2]
        front = z > 0.5
        if not np.any(front):
            continue

        u = (K[0, 0] * p_cam[:, 0] / np.maximum(z, 0.5) + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * p_cam[:, 1] / np.maximum(z, 0.5) + K[1, 2]).astype(np.int32)
        in_fov = front & (u >= 0) & (u < 768) & (v >= 0) & (v < 432)
        if not np.any(in_fov):
            continue

        idx = np.where(in_fov)[0]
        lbls = masks[ci, v[idx], u[idx]]

        # Distance gate for thin classes (only sample lane lines within 18m)
        r_cam = np.hypot(p_cam[idx, 0], p_cam[idx, 1])
        is_thin = np.isin(lbls, list(THIN_CLASSES))
        lbls[is_thin & (r_cam > 18.0)] = 0

        # Update point labels (thin overrides area class)
        update_mask = (lbls > 0) & ((point_label[idx] == 0) | np.isin(lbls, list(THIN_CLASSES)))
        point_label[idx[update_mask]] = lbls[update_mask]

    keep = point_label > 0
    if not np.any(keep):
        return None

    return xg[keep], yg[keep], point_label[keep]


def crop_single_frame(args):
    """Worker: crop per-frame 800x500 ego BEV from the global map raster."""
    fi, pose_xyyaw, map_raster, origin_xy, out_png = args
    if os.path.exists(out_png):
        return fi

    xg0, yg0, yaw0 = pose_xyyaw
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    x_min, y_min = origin_xy
    map_h, map_w = map_raster.shape

    # Construct ego grid cell coordinates
    # row 0 is +80m (fwd), row 800 is -80m (rear)
    # col 0 is +50m (left), col 500 is -50m (right)
    xs = np.linspace(BEV_XH - RES / 2.0, -BEV_XH + RES / 2.0, BEV_H)
    ys = np.linspace(BEV_YH - RES / 2.0, -BEV_YH + RES / 2.0, BEV_W)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    # Transform ego (gx, gy) to global map (x_map, y_map)
    x_map = c0 * gx - s0 * gy + xg0
    y_map = s0 * gx + c0 * gy + yg0

    # Convert to map grid pixel indices
    r_map = ((x_map - x_min) / MAP_RES).astype(np.int32)
    c_map = ((y_map - y_min) / MAP_RES).astype(np.int32)

    valid = (r_map >= 0) & (r_map < map_h) & (c_map >= 0) & (c_map < map_w)
    crop = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    crop[valid] = map_raster[r_map[valid], c_map[valid]]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, crop)
    return fi


def process_scene(raw_dir, scene_dir, workers=8, limit=0):
    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        raise RuntimeError(f"manifest.json missing in {scene_dir}")

    man = json.load(open(mf_path))
    cams_calib = man["cams"]
    frames = man["frames"]
    if limit > 0:
        frames = frames[:limit]
    F = len(frames)

    # Read ego motion poses
    ego_npz = np.load(os.path.join(scene_dir, "ego_motion.npz"))
    poses = ego_npz["pose"][:F]  # [F, 3] (x, y, yaw)

    # Read T_ego_lidar
    calib_lidar_path = os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt")
    rows = []
    in_mat = False
    for line in open(calib_lidar_path):
        l = line.strip()
        if "4x4" in l and ("齐次" in l or "变换" in l):
            in_mat = True
            continue
        if in_mat:
            vals = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", l)
            if len(vals) == 4:
                rows.append([float(v) for v in vals])
            if len(rows) == 4:
                break
    T_ego_lidar = np.array(rows, np.float64)
    R_el = T_ego_lidar[:3, :3]
    t_el = T_ego_lidar[:3, 3]

    lidar_dir = os.path.join(raw_dir, "lidar")
    pcd_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd"))

    # Step 1: Label LiDAR sweeps in parallel
    print(f"[*] Step 1: Projecting LiDAR & sampling surface masks across {F} frames ...", flush=True)
    job_args = []
    for fi in range(F):
        pcd_path = os.path.join(lidar_dir, pcd_files[fi])
        mask_npz = os.path.join(scene_dir, f"surface_mask/{fi:04d}.npz")
        job_args.append((pcd_path, mask_npz, cams_calib, poses[fi], R_el, t_el))

    all_xg, all_yg, all_lbl = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(label_single_sweep, job_args)):
            if res is not None:
                xg, yg, lbl = res
                all_xg.append(xg)
                all_yg.append(yg)
                all_lbl.append(lbl)
            if (i + 1) % 500 == 0 or (i + 1) == F:
                print(f"  Processed {i+1}/{F} sweeps", flush=True)

    if not all_xg:
        print("[ERROR] No labeled points found! Check surface_mask output.")
        return

    all_xg = np.concatenate(all_xg)
    all_yg = np.concatenate(all_yg)
    all_lbl = np.concatenate(all_lbl)
    print(f"[+] Total accumulated labeled 3D points: {len(all_xg):,}", flush=True)

    # Step 2: Build global map grid
    print("[*] Step 2: Accumulating into global map frame ...", flush=True)
    x_min, x_max = all_xg.min() - 10.0, all_xg.max() + 10.0
    y_min, y_max = all_yg.min() - 10.0, all_yg.max() + 10.0
    map_h = int(np.ceil((x_max - x_min) / MAP_RES))
    map_w = int(np.ceil((y_max - y_min) / MAP_RES))
    print(f"  Global grid size: {map_h} x {map_w} (Resolution: {MAP_RES}m)", flush=True)

    # Accumulate hits
    r_pts = ((all_xg - x_min) / MAP_RES).astype(np.int32)
    c_pts = ((all_yg - y_min) / MAP_RES).astype(np.int32)
    ok = (r_pts >= 0) & (r_pts < map_h) & (c_pts >= 0) & (c_pts < map_w)
    r_pts, c_pts, lbl_pts = r_pts[ok], c_pts[ok], all_lbl[ok]

    counts = np.zeros((9, map_h, map_w), dtype=np.uint16)
    flat_rc = r_pts * map_w + c_pts
    for c_id in range(1, 9):
        mask_c = lbl_pts == c_id
        if np.any(mask_c):
            bin_cnt = np.bincount(flat_rc[mask_c], minlength=map_h * map_w)
            counts[c_id] = bin_cnt.reshape(map_h, map_w).astype(np.uint16)

    # Step 3: Stamp road corridor along ego trajectory
    print("[*] Step 3: Stamping road into blind zone along trajectory ...", flush=True)
    traj_r = ((poses[:, 0] - x_min) / MAP_RES).astype(np.int32)
    traj_c = ((poses[:, 1] - y_min) / MAP_RES).astype(np.int32)
    fill_rad = int(np.ceil(EGO_FILL_R / MAP_RES))
    
    # Draw trajectory disk into a binary mask
    traj_mask = np.zeros((map_h, map_w), dtype=bool)
    for tr, tc in zip(traj_r, traj_c):
        cv2.circle(traj_mask.view(np.uint8), (int(tc), int(tr)), fill_rad, 1, -1)
    
    # Where unlabeled, stamp as road (1)
    unlabeled = counts.sum(axis=0) == 0
    counts[1, traj_mask & unlabeled] = 5

    # Step 4: Rasterize global semantic map
    # Area classes: argmax
    map_raster = np.argmax(counts, axis=0).astype(np.uint8)
    
    # Thin classes override with threshold >= 2 hits
    for tid in (6, 5, 7, 4):  # road_edge, stopline, marking, laneline (highest priority)
        override = counts[tid] >= 2
        map_raster[override] = tid

    # Step 5: Crop per-frame BEV GT in parallel
    print(f"[*] Step 4: Cropping {F} per-frame BEV images (800x500 @ 0.2m) ...", flush=True)
    gt_dir = os.path.join(scene_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    crop_args = []
    for fi in range(F):
        out_png = os.path.join(gt_dir, f"{fi:04d}.png")
        crop_args.append((fi, poses[fi], map_raster, (x_min, y_min), out_png))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(crop_single_frame, crop_args)):
            if (i + 1) % 500 == 0 or (i + 1) == F:
                print(f"  Cropped {i+1}/{F} frames", flush=True)

    # Step 6: Update manifest.json
    for fr in frames:
        fr["gt"] = f"gt/{fr['frame']:04d}.png"
    json.dump(man, open(mf_path, "w"))
    print(f"[+] Done! BEV Lane GT written to {gt_dir} and manifest updated.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Raw data directory (containing lidar/, calib/)")
    ap.add_argument("--scene", required=True, help="Scene output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Limit frames for testing")
    args = ap.parse_args()

    process_scene(args.raw, args.scene, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
