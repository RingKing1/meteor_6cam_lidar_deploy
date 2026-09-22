#!/usr/bin/env python3
"""Build 10-Class Semantic Occupancy (occ) and Metric Depth (depth4) for METEOR.

Key Capabilities:
  1. 10-Class Semantic Taxonomy:
     - 0: free, 1: obstacle, 2: vehicle, 3: 2wheel, 4: pedestrian,
     - 5: road, 6: sidewalk, 7: vegetation, 8: building, 9: pole/sign, 255: unknown.
  2. 3D Bounding Box Voxel Injection (FocalFormer3D Ground Truth):
     - Solid-fills the interior voxels of confirmed vehicles (2) and pedestrians (4),
       providing exact 3D orientation, spatial envelope, and protecting from ray carving.
  3. 2D Multi-Camera Semantic Back-Projection (seg2d21):
     - Samples 2D semantic maps across all 6 surround cameras with OCC_PRIO arbitration.
     - Naturally recovers distant and occluded targets missed by 3D bounding boxes.
  4. Temporal Motion-Compensated Accumulation (--acc-sweeps, default 4 = +-4 sweeps):
     - Accumulates static classes (road, sidewalk, vegetation, building, pole) over 9 sweeps
       using ENU global poses from ego_motion.npz, eliminating point cloud sparsity and lattice holes.
     - Restricts dynamic classes (vehicles/VRUs) to +-1 frame (<=0.1s) to prevent ghost trails.
  5. Ray-Carving Free Space:
     - Steps along sensor rays to mark unobserved voxels as 0 (free) while strictly preserving
       all confirmed obstacles (1..9).
  6. Metric Depth Generation:
     - Projects LiDAR points to 6 cameras [6, 108, 192] float16 with sky fill (79.5m).

Usage:
  # Process a single scene
  python3 scripts/autolabel_semantic_occ/build_depth_and_occ.py --scene data_20260910_061820 --workers 16

  # Process all scenes
  python3 scripts/autolabel_semantic_occ/build_depth_and_occ.py --scenes all --workers 16
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

CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Depth grid specifications
DH, DW = 108, 192
MAX_D = 79.0
SKY_D = 79.5

# Occupancy grid specifications
VOX = 0.4
XH, YH = 40.0, 40.0
Z0, Z1 = -1.0, 5.4
GX = GY = 200     # int(2 * XH / VOX)
GZ = 16           # int((Z1 - Z0) / VOX)

# 21-to-10 OCC Mapping & Priority
LUT21 = np.array([0, 1, 2, 2, 2, 3, 3, 4, 5, 9, 9, 5, 6, 5, 5, 0, 8, 8, 7, 0, 9], dtype=np.uint8)
OCC_PRIO = np.array([0, 3, 5, 6, 7, 1, 1, 2, 2, 4], dtype=np.int8)
OCC_NAMES = ["free", "obstacle", "vehicle", "2wheel", "pedestrian", "road", "sidewalk", "vegetation", "building", "pole/sign"]


def read_T_ego_lidar(path):
    rows = []
    in_mat = False
    for line in open(path):
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
    assert len(rows) == 4, f"Cannot parse 4x4 matrix from {path}"
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
            return np.zeros((0, 3), np.float32)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


def label_points_2d(pts_e, seg21, cams_calib):
    """Projects 3D points onto 6 cameras and samples 2D semantic maps with OCC_PRIO."""
    cls_pts = np.zeros(len(pts_e), dtype=np.uint8)
    for ci, c_name in enumerate(CAMS):
        cal = cams_calib[c_name]
        K = np.array(cal["K"], dtype=np.float64)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec, t_ec = T_ec[:3, :3], T_ec[:3, 3]

        pc = (pts_e - t_ec) @ R_ec
        z = pc[:, 2]
        m = z > 0.5
        u = (K[0, 0] * 0.25 * pc[m, 0] / z[m] + K[0, 2] * 0.25).astype(np.int32)
        v = (K[1, 1] * 0.25 * pc[m, 1] / z[m] + K[1, 2] * 0.25).astype(np.int32)
        ok = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)

        s = np.full(int(m.sum()), 255, dtype=np.uint8)
        s[ok] = seg21[ci, v[ok], u[ok]]
        c = np.where(s == 255, 0, LUT21[np.minimum(s, 20)])
        cur = cls_pts[m]
        upd = OCC_PRIO[c] > OCC_PRIO[cur]
        cur[upd] = c[upd]
        cls_pts[m] = cur
    return cls_pts


def inject_3d_boxes(occ, boxes_3d):
    """Solid-fills the interior voxels of 3D boxes."""
    if len(boxes_3d) == 0:
        return 0
    injected_count = 0
    for b in boxes_3d:
        b_cls, cx, cy, zc, l, w, h, yaw = b
        target_occ = 2 if b_cls == 1.0 else 4  # 2: vehicle, 4: ped/VRU

        r_max = np.hypot(l / 2.0, w / 2.0)
        r_min_idx = max(0, int((XH - (cx + r_max)) / VOX))
        r_max_idx = min(GX, int((XH - (cx - r_max)) / VOX) + 1)
        c_min_idx = max(0, int((YH - (cy + r_max)) / VOX))
        c_max_idx = min(GY, int((YH - (cy - r_max)) / VOX) + 1)
        z_min_idx = max(0, int(((zc - h / 2.0) - Z0) / VOX))
        z_max_idx = min(GZ, int(((zc + h / 2.0) - Z0) / VOX) + 1)

        if r_min_idx >= r_max_idx or c_min_idx >= c_max_idx or z_min_idx >= z_max_idx:
            continue

        sub_z, sub_r, sub_c = np.meshgrid(
            np.arange(z_min_idx, z_max_idx),
            np.arange(r_min_idx, r_max_idx),
            np.arange(c_min_idx, c_max_idx),
            indexing="ij"
        )
        vx = XH - (sub_r + 0.5) * VOX
        vy = YH - (sub_c + 0.5) * VOX
        vz = Z0 + (sub_z + 0.5) * VOX

        dx, dy = vx - cx, vy - cy
        cb, sb = np.cos(yaw), np.sin(yaw)
        lx = dx * cb + dy * sb
        ly = -dx * sb + dy * cb
        lz = vz - zc

        inside = (np.abs(lx) <= (l / 2.0)) & (np.abs(ly) <= (w / 2.0)) & (np.abs(lz) <= (h / 2.0))
        occ[sub_z[inside], sub_r[inside], sub_c[inside]] = target_occ
        injected_count += inside.sum()
    return injected_count


def ray_carve_free_space(occ, pts_e, t_el):
    """Carves free space (0) along sensor rays, without overwriting confirmed occupied voxels (1..9)."""
    if len(pts_e) == 0:
        return
    q = pts_e[::2]
    d = np.linalg.norm(q - t_el, axis=1)
    dirs = (q - t_el) / np.maximum(d[:, None], 1e-4)

    for step in np.arange(1.0, 35.0, 0.4):
        rays = t_el + dirs * step
        valid_ray = step < (d - 0.4)
        if not np.any(valid_ray):
            continue
        rx, ry, rz = rays[valid_ray, 0], rays[valid_ray, 1], rays[valid_ray, 2]
        vgx = ((XH - rx) / VOX).astype(np.int32)
        vgy = ((YH - ry) / VOX).astype(np.int32)
        vgz = ((rz - Z0) / VOX).astype(np.int32)
        ok_ray = (vgx >= 0) & (vgx < GX) & (vgy >= 0) & (vgy < GY) & (vgz >= 0) & (vgz < GZ)
        sel = occ[vgz[ok_ray], vgx[ok_ray], vgy[ok_ray]]
        w = (sel == 255)  # only overwrite unobserved/unknown
        occ[vgz[ok_ray][w], vgx[ok_ray][w], vgy[ok_ray][w]] = 0


def process_single_frame(task_args):
    """Worker task to process 1 frame of depth4 and 10-class occ."""
    (
        fi,
        raw_dir,
        scene_dir,
        pcd_files,
        ego_poses,
        cams_calib,
        T_el,
        acc_sweeps,
        depth_dst,
        occ_dst,
        force,
    ) = task_args

    if not force and os.path.exists(depth_dst) and os.path.exists(occ_dst):
        return fi

    R_el, t_el = T_el[:3, :3], T_el[:3, 3]
    F = len(pcd_files)

    # 1. Load current frame PCD and seg2d21
    pcd_curr_path = os.path.join(raw_dir, "lidar", pcd_files[fi])
    if not os.path.exists(pcd_curr_path):
        return None
    pcd_curr = read_pcd_xyz(pcd_curr_path)
    if len(pcd_curr) == 0:
        return None
    pts_curr_e = pcd_curr @ R_el.T + t_el

    seg_curr_path = os.path.join(scene_dir, "seg2d21", f"{fi:04d}.npz")
    seg21_curr = np.load(seg_curr_path)["seg"] if os.path.exists(seg_curr_path) else None

    # Compute depth4
    depth4 = np.zeros((len(CAMS), DH, DW), dtype=np.float32)
    for ci, c_name in enumerate(CAMS):
        cal = cams_calib[c_name]
        K = np.array(cal["K"], dtype=np.float64)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec, t_ec = T_ec[:3, :3], T_ec[:3, 3]

        pc = (pts_curr_e - t_ec) @ R_ec
        z = pc[:, 2]
        front = (z > 0.5) & (z < MAX_D)
        if not np.any(front):
            continue

        u = (K[0, 0] * 0.25 * pc[front, 0] / z[front] + K[0, 2] * 0.25).astype(np.int32)
        v = (K[1, 1] * 0.25 * pc[front, 1] / z[front] + K[1, 2] * 0.25).astype(np.int32)
        zf = z[front]
        in_fov = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
        if not np.any(in_fov):
            continue

        uf, vf, zf_in = u[in_fov], v[in_fov], zf[in_fov]
        flat_idx = vf * DW + uf
        grid_d = np.full(DH * DW, np.inf, dtype=np.float32)
        np.minimum.at(grid_d, flat_idx, zf_in.astype(np.float32))
        d_map = np.where(np.isinf(grid_d), 0.0, grid_d).reshape(DH, DW)

        if seg21_curr is not None:
            sky = seg21_curr[ci] == 19
            d_map[sky] = SKY_D
        depth4[ci] = d_map

    np.savez_compressed(depth_dst, depth=depth4.astype(np.float16))

    # 2. Accumulate labeled points from neighboring sweeps
    x0, y0, yaw0 = ego_poses[fi]
    c0, s0 = np.cos(yaw0), np.sin(yaw0)

    acc_pts = []
    acc_cls = []

    for off in range(-acc_sweeps, acc_sweeps + 1):
        j = fi + off
        if j < 0 or j >= F:
            continue

        pcd_j_path = os.path.join(raw_dir, "lidar", pcd_files[j])
        if not os.path.exists(pcd_j_path):
            continue
        pcd_j = read_pcd_xyz(pcd_j_path)
        if len(pcd_j) == 0:
            continue
        pts_j_e = pcd_j @ R_el.T + t_el

        dist2 = pts_j_e[:, 0]**2 + pts_j_e[:, 1]**2
        valid = (dist2 < 70.0**2) & (pts_j_e[:, 2] > Z0) & (pts_j_e[:, 2] < Z1 + 1.0)
        pts_j_e = pts_j_e[valid]
        if len(pts_j_e) == 0:
            continue

        seg_j_path = os.path.join(scene_dir, "seg2d21", f"{j:04d}.npz")
        if os.path.exists(seg_j_path):
            seg21_j = np.load(seg_j_path)["seg"]
            cls_j = label_points_2d(pts_j_e, seg21_j, cams_calib)
        else:
            cls_j = np.zeros(len(pts_j_e), dtype=np.uint8)

        # Dynamic class filtering for off != 0
        if abs(off) > 1:
            stat_mask = (cls_j != 2) & (cls_j != 3) & (cls_j != 4)
            pts_j_e = pts_j_e[stat_mask]
            cls_j = cls_j[stat_mask]

        if len(pts_j_e) == 0:
            continue

        if off != 0:
            xj, yj, yawj = ego_poses[j]
            cj, sj = np.cos(yawj), np.sin(yawj)
            xw = cj * pts_j_e[:, 0] - sj * pts_j_e[:, 1] + xj
            yw = sj * pts_j_e[:, 0] + cj * pts_j_e[:, 1] + yj
            dx, dy = xw - x0, yw - y0
            pts_j_fi_x = c0 * dx + s0 * dy
            pts_j_fi_y = -s0 * dx + c0 * dy
            pts_j_fi = np.stack([pts_j_fi_x, pts_j_fi_y, pts_j_e[:, 2]], axis=1)
        else:
            pts_j_fi = pts_j_e

        acc_pts.append(pts_j_fi)
        acc_cls.append(cls_j)

    if acc_pts:
        all_pts = np.vstack(acc_pts)
        all_cls = np.concatenate(acc_cls)
    else:
        all_pts = np.zeros((0, 3), dtype=np.float64)
        all_cls = np.zeros(0, dtype=np.uint8)

    # 3. Voxelization
    occ = np.full((GZ, GX, GY), 255, dtype=np.uint8)

    in_grid = (np.abs(all_pts[:, 0]) < XH) & (np.abs(all_pts[:, 1]) < YH) & (all_pts[:, 2] >= Z0) & (all_pts[:, 2] < Z1)
    pts_in = all_pts[in_grid]
    cls_in = all_cls[in_grid]

    # Default unlabeled points to road(5) if below ground threshold else obstacle(1)
    cls_filled = np.where(cls_in == 0, np.where(pts_in[:, 2] < 0.2, 5, 1), cls_in).astype(np.uint8)

    r = ((XH - pts_in[:, 0]) / VOX).astype(np.int32)
    c = ((YH - pts_in[:, 1]) / VOX).astype(np.int32)
    z = ((pts_in[:, 2] - Z0) / VOX).astype(np.int32)
    ok = (r >= 0) & (r < GX) & (c >= 0) & (c < GY) & (z >= 0) & (z < GZ)

    lin = (z[ok] * GX + r[ok]) * GY + c[ok]
    cnt = np.bincount(lin * 10 + cls_filled[ok], minlength=GZ * GX * GY * 10).reshape(-1, 10)
    hit = cnt.sum(1) > 0
    occ.reshape(-1)[hit] = cnt[hit].argmax(1).astype(np.uint8)

    # 4. Inject 3D bounding boxes (solid fill)
    box_path = os.path.join(scene_dir, "bev_box", f"{fi:04d}.npz")
    if os.path.exists(box_path):
        b3d = np.load(box_path)["boxes_3d"]
        inject_3d_boxes(occ, b3d)

    # 5. Ray-carving free space
    in_curr = (np.abs(pts_curr_e[:, 0]) < XH) & (np.abs(pts_curr_e[:, 1]) < YH) & (pts_curr_e[:, 2] >= Z0) & (pts_curr_e[:, 2] < Z1)
    ray_carve_free_space(occ, pts_curr_e[in_curr], t_el)

    np.savez_compressed(occ_dst, occ=occ)
    return fi


def process_scene(scene, raw_base, scene_base, workers=16, acc_sweeps=4, force=False, limit=0):
    raw_dir = os.path.join(raw_base, scene)
    scene_dir = os.path.join(scene_base, scene)

    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        print(f"[-] manifest.json not found in {scene_dir}, skipping.")
        return

    man = json.load(open(mf_path))
    cams_calib = man["cams"]
    frames = man["frames"]
    if limit > 0:
        frames = frames[:limit]
    F = len(frames)

    calib_lidar_path = os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt")
    T_el = read_T_ego_lidar(calib_lidar_path)

    ego_poses = np.load(os.path.join(scene_dir, "ego_motion.npz"))["pose"]
    lidar_dir = os.path.join(raw_dir, "lidar")
    pcd_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd"))

    depth_dir = os.path.join(scene_dir, "depth4")
    occ_dir = os.path.join(scene_dir, "occ")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(occ_dir, exist_ok=True)

    print(f"\n[*] Processing scene {scene}: {F} frames -> depth4/ & occ/")
    t0 = time.time()

    tasks = []
    for fr in frames:
        fi = fr["frame"]
        depth_dst = os.path.join(depth_dir, f"{fi:04d}.npz")
        occ_dst = os.path.join(occ_dir, f"{fi:04d}.npz")
        tasks.append((
            fi,
            raw_dir,
            scene_dir,
            pcd_files,
            ego_poses,
            cams_calib,
            T_el,
            acc_sweeps,
            depth_dst,
            occ_dst,
            force,
        ))

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(process_single_frame, tasks)):
            done += 1
            if done % 200 == 0 or done == F:
                elapsed = time.time() - t0
                fps = done / max(0.001, elapsed)
                eta = (F - done) / max(0.001, fps)
                print(f"  [{done}/{F}] {fps:.1f} fps | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    # Update manifest.json
    for fr in frames:
        fi = fr["frame"]
        fr["depth4"] = f"depth4/{fi:04d}.npz"
        fr["occ"] = f"occ/{fi:04d}.npz"
    json.dump(man, open(mf_path, "w"), indent=2)

    total_time = time.time() - t0
    print(f"[✓] Scene {scene} finished: {F} frames in {total_time:.1f}s ({F/total_time:.1f} fps). Manifest updated.")


def main():
    parser = argparse.ArgumentParser(description="10-Class Semantic Occupancy and Depth Autolabeling")
    parser.add_argument("--raw-base", default="", help="Path to raw_data base directory")
    parser.add_argument("--scene-base", default="", help="Path to scenes base directory")
    parser.add_argument("--scene", default="", help="Specific scene to process")
    parser.add_argument("--scenes", default="", help="'all' or comma-separated scene list")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument("--acc-sweeps", type=int, default=4, help="Number of temporal accumulation sweeps (+- sweeps)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of frames per scene for testing")

    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_base = args.raw_base or os.path.join(repo_dir, "raw_data")
    scene_base = args.scene_base or os.path.join(repo_dir, "scenes")

    if args.scenes == "all":
        scene_list = sorted([s for s in os.listdir(scene_base) if s.startswith("data_")])
    elif args.scenes:
        scene_list = [s.strip() for s in args.scenes.split(",")]
    elif args.scene:
        scene_list = [args.scene]
    else:
        scene_list = ["data_20260910_061820"]

    print(f"==================================================")
    print(f"Launching 10-Class Semantic Occupancy Autolabeling")
    print(f"Scenes: {scene_list}")
    print(f"Workers: {args.workers}, Accumulation Sweeps: +- {args.acc_sweeps}")
    print(f"==================================================")

    t_start = time.time()
    for s in scene_list:
        process_scene(
            s,
            raw_base,
            scene_base,
            workers=args.workers,
            acc_sweeps=args.acc_sweeps,
            force=args.force,
            limit=args.limit,
        )

    print(f"\n[✓] All scenes successfully processed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
