#!/usr/bin/env python3
"""Convert the custom 6-camera + 1-LiDAR dataset to a METEOR scene.

Input layout (per sequence):
  calib/camera/camera_*.json   extrinsic = T_cam_lidar (LiDAR->camera, optical),
                               intrinsic at 1920x1080, images already undistorted
  calib/lidar/lidar2imu_calib.txt  4x4 T_ego_lidar (LiDAR->ego; x-fwd,y-left,z-up)
  camera/camera_*/<ts>.jpg     1920x1080
  lidar/<ts>.pcd               PCD v0.7 binary, x y z intensity float32
  localization/<ts>.yaml       body-frame vel

Output scene:
  manifest.json  cams (K at 768x432, T_ego_cam) + frames (img paths)
  img/<CAM>/<fi>.jpg           resized 768x432
  lidar_bev/<fi>.npz           {"lb": float16 [4,400,250]}
  ego_motion.npz               v0 [F] m/s
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

CAM_MAP = [
    ("camera_0_front100",   "CAM_FRONT_WIDE"),
    ("camera_3_left_front", "CAM_FRONT_LEFT"),
    ("camera_4_right_front","CAM_FRONT_RIGHT"),
    ("camera_5_back",       "CAM_BACK_WIDE"),
    ("camera_6_left_back",  "CAM_BACK_LEFT"),
    ("camera_7_right_back", "CAM_BACK_RIGHT"),
]
GH, GW, RES = 400, 250, 0.4
Z0, Z1 = -1.0, 4.0
SX, SY, OW, OH = 768.0 / 1920.0, 432.0 / 1080.0, 768, 432


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
    T = np.array(rows, np.float64)
    assert abs(T[3, 3] - 1.0) < 1e-6 and abs(T[3, :3]).max() < 1e-6, \
        f"last row is not [0 0 0 1] in {path}: {T[3]}"
    return T


def read_cam(path):
    j = json.load(open(path))
    T_cam_lidar = np.array(j["extrinsic"], np.float64).reshape(4, 4)
    k = j["intrinsic"]
    K = np.array(k, np.float64).reshape(3, 3)
    return T_cam_lidar, K


def read_pcd_xyz(path):
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            head += line
            if line.startswith(b"DATA"):
                break
        n = int(re.search(rb"POINTS (\d+)", head).group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


def raster(pe):
    lb = np.zeros((4, GH, GW), np.float32)
    r = ((80.0 - pe[:, 0]) / RES).astype(np.int32)
    c = ((50.0 - pe[:, 1]) / RES).astype(np.int32)
    ok = (r >= 0) & (r < GH) & (c >= 0) & (c < GW)
    r, c, z = r[ok], c[ok], np.clip(pe[ok, 2], Z0, Z1)
    if len(r) == 0:
        return lb
    z = z.astype(np.float32)
    flat = r * GW + c
    cnt = np.bincount(flat, minlength=GH * GW).astype(np.float32)
    zsum = np.bincount(flat, weights=z, minlength=GH * GW).astype(np.float32)
    zmax = np.full(GH * GW, np.float32(Z0), np.float32)
    np.maximum.at(zmax, flat, z)
    occ = cnt > 0
    lb[0] = np.log1p(cnt).reshape(GH, GW)
    lb[1] = np.where(occ, zmax, 0.0).reshape(GH, GW)
    lb[2] = np.where(occ, zsum / np.maximum(cnt, 1), 0.0).reshape(GH, GW)
    lb[3] = occ.reshape(GH, GW).astype(np.float32)
    return lb


def parse_vel(path):
    txt = open(path).read()
    m = re.search(r"vel:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)", txt)
    return float(m.group(1)), float(m.group(2))


def _img_job(args):
    src, dst = args
    if os.path.exists(dst):
        return
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"read fail {src}")
    im = cv2.resize(im, (OW, OH), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _lb_job(args):
    src, dst, R_flat, t = args
    if os.path.exists(dst):
        return
    R = R_flat.reshape(3, 3)
    p = read_pcd_xyz(src)
    pe = p @ R.T + t
    np.savez_compressed(dst, lb=raster(pe).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symlink", action="store_true",
                    help="link raw 1080p images instead of resizing")
    a = ap.parse_args()

    T_ego_lidar = read_T_ego_lidar(
        os.path.join(a.src, "calib/lidar/lidar2imu_calib.txt"))
    R_el = T_ego_lidar[:3, :3]
    t_el = T_ego_lidar[:3, 3]

    cams = {}
    for d, name in CAM_MAP:
        T_cam_lidar, K = read_cam(os.path.join(
            a.src, f"calib/camera/{d}.json"))
        T_ego_cam = T_ego_lidar @ np.linalg.inv(T_cam_lidar)
        Ks = K.copy()
        Ks[0] *= SX
        Ks[1] *= SY
        cams[name] = {"K": Ks.astype(np.float32).tolist(),
                      "T_ego_cam": T_ego_cam.astype(np.float32).tolist()}

    ts = sorted(f[:-4] for f in os.listdir(os.path.join(a.src, "lidar"))
                if f.endswith(".pcd"))
    if a.limit:
        ts = ts[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    # images
    jobs = []
    for d, name in CAM_MAP:
        for fi, t in enumerate(ts):
            src = os.path.join(a.src, f"camera/{d}/{t}.jpg")
            rel = f"img/{name}/{fi:04d}.jpg"
            dst = os.path.join(a.out, rel)
            if a.symlink:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
            else:
                jobs.append((src, dst))
    if jobs:
        print(f"resizing {len(jobs)} images ...", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, _ in enumerate(ex.map(_img_job, jobs)):
                if (i + 1) % 2000 == 0:
                    print(f"  {i+1}/{len(jobs)}", flush=True)

    # lidar BEV rasters
    lb_dir = os.path.join(a.out, "lidar_bev")
    os.makedirs(lb_dir, exist_ok=True)

    print("rasterising LiDAR sweeps ...", flush=True)
    lb_jobs = [(os.path.join(a.src, f"lidar/{t}.pcd"),
                os.path.join(lb_dir, f"{fi:04d}.npz"),
                R_el.ravel(), t_el) for fi, t in enumerate(ts)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_lb_job, lb_jobs)):
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(ts)}", flush=True)

    # speed
    v0 = []
    for t in ts:
        vx, vy = parse_vel(os.path.join(a.src, f"localization/{t}.yaml"))
        v0.append(float(np.hypot(vx, vy)))
    np.savez(os.path.join(a.out, "ego_motion.npz"),
             v0=np.array(v0, np.float32))

    frames = [{"frame": fi,
               "imgs": {name: f"img/{name}/{fi:04d}.jpg"
                        for _, name in CAM_MAP},
               "lidar_bev": f"lidar_bev/{fi:04d}.npz"}
              for fi in range(len(ts))]
    json.dump({"cams": cams, "frames": frames},
              open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"done: {a.out} ({len(ts)} frames)", flush=True)


if __name__ == "__main__":
    main()
