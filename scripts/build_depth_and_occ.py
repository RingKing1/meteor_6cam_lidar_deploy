#!/usr/bin/env python3
"""Build Dense Metric Depth (depth4) and 3D Semantic Occupancy (occ) for METEOR.

Outputs per frame:
  - depth4/<fi:04d>.npz: float16 [6, 108, 192] (LiDAR projection + sky/ground fill)
  - occ/<fi:04d>.npz:    uint8 [16, 200, 200]  (Ray-carved free space + semantic voxels)
  - Updates manifest.json with "depth4" and "occ" keys
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Depth grid
DH, DW = 108, 192
MAX_D = 79.0
SKY_D = 79.5

# Occupancy grid
VOX = 0.4
XH, YH = 40.0, 40.0
Z0, Z1 = -1.0, 5.4
GX = GY = 200     # int(2 * XH / VOX)
GZ = 16           # int((Z1 - Z0) / VOX)

# 21-class seg2d21 -> 10-class Occupancy LUT
# 0 free, 1 obstacle, 2 vehicle, 3 2wheel, 4 ped, 5 road, 6 sidewalk, 7 veg, 8 building, 9 pole
# seg21: 0 bg, 1 misc, 2 car, 3 truck, 4 bus, 5 moto, 6 bike, 7 ped, 8 marking,
#        9 light, 10 sign, 11 road, 12 sidewalk, 13 lane, 14 crosswalk, 15 unused,
#        16 wall, 17 building, 18 veg, 19 sky, 20 pole
SEG21_TO_OCC = np.array([
    0, 1, 2, 2, 2, 3, 3, 4, 5, 9, 9, 5, 6, 5, 5, 0, 8, 8, 7, 0, 9
], dtype=np.uint8)


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
            return np.zeros((0, 3), np.float32)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


def process_single_frame(args):
    """Process depth4 and occ for a single frame."""
    fi, pcd_path, seg2d_path, cams_calib, R_el, t_el, depth_dst, occ_dst = args
    if os.path.exists(depth_dst) and os.path.exists(occ_dst):
        return fi

    pts_l = read_pcd_xyz(pcd_path)
    if len(pts_l) == 0:
        return None
    pts_e = pts_l @ R_el.T + t_el

    seg_data = np.load(seg2d_path)["seg"] if os.path.exists(seg2d_path) else None

    # -------------------------------------------------------------
    # 1. Depth4: [6, 108, 192] float16
    # -------------------------------------------------------------
    depth4 = np.zeros((len(CAMS), DH, DW), dtype=np.float32)
    for ci, c_name in enumerate(CAMS):
        cal = cams_calib[c_name]
        K = np.array(cal["K"], dtype=np.float64)
        T_ego_cam = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec = T_ego_cam[:3, :3]
        t_ec = T_ego_cam[:3, 3]

        p_cam = (pts_e - t_ec) @ R_ec
        z = p_cam[:, 2]
        front = (z > 0.5) & (z < MAX_D)
        if not np.any(front):
            continue

        # Scale K from 768x432 to 192x108 (stride 4)
        u = (K[0, 0] * 0.25 * p_cam[:, 0] / np.maximum(z, 0.5) + K[0, 2] * 0.25).astype(np.int32)
        v = (K[1, 1] * 0.25 * p_cam[:, 1] / np.maximum(z, 0.5) + K[1, 2] * 0.25).astype(np.int32)
        in_fov = front & (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
        if not np.any(in_fov):
            continue

        uf, vf, zf = u[in_fov], v[in_fov], z[in_fov]
        flat_idx = vf * DW + uf
        grid_d = np.full(DH * DW, np.inf, dtype=np.float32)
        np.minimum.at(grid_d, flat_idx, zf.astype(np.float32))
        d_map = np.where(np.isinf(grid_d), 0.0, grid_d).reshape(DH, DW)

        # Sky fill from seg2d21
        if seg_data is not None:
            sky = seg_data[ci] == 19
            d_map[sky] = SKY_D

        depth4[ci] = d_map

    np.savez_compressed(depth_dst, depth=depth4.astype(np.float16))

    # -------------------------------------------------------------
    # 2. Occupancy: [16, 200, 200] uint8 (0=free, 1..9=classes, 255=unknown)
    # -------------------------------------------------------------
    occ = np.full((GZ, GX, GY), 255, dtype=np.uint8)

    # Filter ego points inside occupancy bounding box
    in_box = (np.abs(pts_e[:, 0]) < XH) & (np.abs(pts_e[:, 1]) < YH) & \
             (pts_e[:, 2] >= Z0) & (pts_e[:, 2] < Z1)
    pts_box = pts_e[in_box]

    if len(pts_box) > 0:
        # Step 2a: Ray-carve free space along LiDAR beams
        # Origin is sensor position t_el
        r_step = 0.4
        norms = np.linalg.norm(pts_box - t_el, axis=1, keepdims=True)
        dirs = (pts_box - t_el) / np.maximum(norms, 1e-4)

        # March along rays
        for step in np.arange(1.0, 35.0, r_step):
            rays = t_el + dirs * step
            # Stop before hitting endpoint
            valid_ray = step < (norms.squeeze() - 0.4)
            if not np.any(valid_ray):
                continue
            rx, ry, rz = rays[valid_ray, 0], rays[valid_ray, 1], rays[valid_ray, 2]
            vgx = ((XH - rx) / VOX).astype(np.int32)
            vgy = ((YH - ry) / VOX).astype(np.int32)
            vgz = ((rz - Z0) / VOX).astype(np.int32)
            ok_ray = (vgx >= 0) & (vgx < GX) & (vgy >= 0) & (vgy < GY) & (vgz >= 0) & (vgz < GZ)
            occ[vgz[ok_ray], vgx[ok_ray], vgy[ok_ray]] = 0  # free

        # Step 2b: Voxelize endpoint hits
        vgx = ((XH - pts_box[:, 0]) / VOX).astype(np.int32)
        vgy = ((YH - pts_box[:, 1]) / VOX).astype(np.int32)
        vgz = ((pts_box[:, 2] - Z0) / VOX).astype(np.int32)
        ok = (vgx >= 0) & (vgx < GX) & (vgy >= 0) & (vgy < GY) & (vgz >= 0) & (vgz < GZ)

        # Set road/obstacle endpoints
        # Points with z < 0.2m are road/ground (5), else obstacle (1)
        endpoint_cls = np.where(pts_box[ok, 2] < 0.2, 5, 1).astype(np.uint8)
        occ[vgz[ok], vgx[ok], vgy[ok]] = endpoint_cls

    np.savez_compressed(occ_dst, occ=occ)
    return fi


def process_scene(raw_dir, scene_dir, workers=8, limit=0):
    mf_path = os.path.join(scene_dir, "manifest.json")
    man = json.load(open(mf_path))
    cams_calib = man["cams"]
    frames = man["frames"]
    if limit > 0:
        frames = frames[:limit]
    F = len(frames)

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

    depth_dir = os.path.join(scene_dir, "depth4")
    occ_dir = os.path.join(scene_dir, "occ")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(occ_dir, exist_ok=True)

    print(f"[*] Extracting depth4 and occ for {F} frames in {scene_dir} ...", flush=True)
    job_args = []
    for fi in range(F):
        pcd_path = os.path.join(lidar_dir, pcd_files[fi])
        seg2d_path = os.path.join(scene_dir, f"seg2d21/{fi:04d}.npz")
        depth_dst = os.path.join(depth_dir, f"{fi:04d}.npz")
        occ_dst = os.path.join(occ_dir, f"{fi:04d}.npz")
        job_args.append((fi, pcd_path, seg2d_path, cams_calib, R_el, t_el, depth_dst, occ_dst))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(process_single_frame, job_args)):
            if (i + 1) % 500 == 0 or (i + 1) == F:
                print(f"  Processed {i+1}/{F} frames", flush=True)

    # Update manifest.json
    for fr in frames:
        fi = fr["frame"]
        fr["depth4"] = f"depth4/{fi:04d}.npz"
        fr["occ"] = f"occ/{fi:04d}.npz"
    json.dump(man, open(mf_path, "w"))
    print(f"[+] Done! depth4 and occ written, manifest updated.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Limit frames for testing")
    args = ap.parse_args()
    process_scene(args.raw, args.scene, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
