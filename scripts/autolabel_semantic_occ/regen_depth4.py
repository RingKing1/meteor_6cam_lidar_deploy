#!/usr/bin/env python3
"""Re-generate per-camera dense depth GT (depth4/) with the NEW extrinsics.

Background (2026-09-24): the scene manifests were updated to the XCalib
refined extrinsics (T_ego_cam), but the existing depth4/*.npz files were
projected with the OLD extrinsics (up to ~0.17 m / 0.4 deg of misalignment
with the images). This script re-projects the raw LiDAR sweeps with the
manifest's CURRENT T_cam_ego and re-runs the same densification as
METEOR/bevlane/extract_depth_dense.py.

Inputs per scene:
  raw_data/<scene>/lidar/<ts>.pcd        raw LiDAR sweep (x y z intensity f32)
  raw_data/<scene>/calib/lidar/lidar2imu_calib.txt   T_ego_lidar
  scenes/<scene>/manifest.json           CURRENT (new) extrinsics + frames
  scenes/<scene>/seg2d21/<fi>.npz        [6,108,192] 21-class semantic map
                                          (segment source for densification)

Output: scenes/<scene>/depth4/<fi>.npz  {"depth": float16 [6,108,192]}
  same layout/value conventions as the original generator:
  MAX_DEPTH 79, sky(19)=79.5, near car segments(2)=2.0, ground-plane fill.

Differences vs extract_depth_dense.py (documented for honesty):
  - segment source is the 21-class semantic map (upsampled to 432x768 is NOT
    done; seg is used at 108x192 like seg_map_small's output) instead of
    instance-level panoptic RLEs: same-class objects merge into one segment,
    so the same-segment nearest fill may cross object boundaries;
  - ego_vehicle is approximated by car-class segments whose LiDAR median
    depth is < 3 m (no dedicated ego class in the 21-class map).

  python3 scripts/autolabel_semantic_occ/regen_depth4.py \
      --scenes data_20260910_064331 --workers 8 [--limit 20]
"""
import argparse, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import griddata

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DH, DW = 108, 192          # stride-4 grid for 768x432 input
MAX_DEPTH = 79.0
SKY_D, EGO_D = 79.5, 2.0
GROUND = {11, 12, 13, 14, 8}     # road, sidewalk, lane, crosswalk, marking
ROADPAINT = {13, 14, 8}
SKY, CAR = {19}, {2}


def ground_plane_depth(K4, T_ego_cam):
    """Camera-z depth of each stride-4 pixel's ray ∩ ground plane (ego z=0)."""
    vs, us = np.meshgrid(np.arange(DH), np.arange(DW), indexing="ij")
    pix = np.stack([us + 0.5, vs + 0.5, np.ones((DH, DW), np.float64)], 0).reshape(3, -1)
    r = np.linalg.inv(K4) @ pix                    # cam-frame rays, z=1
    R, t = T_ego_cam[:3, :3], T_ego_cam[:3, 3]
    dz = (R @ r)[2]
    down = dz < -0.02
    radial = np.sqrt(r[0] ** 2 + r[1] ** 2)
    down &= radial <= 1.2
    s = np.where(down, -t[2] / np.where(down, dz, -1.0), 0.0)
    g = np.where(s > 0.5, np.minimum(s, 79.0), 0.0)
    return g.reshape(DH, DW).astype(np.float32)


def densify(sparse, seg, sky, ego, gnd=None, gpl=None):
    d = sparse.copy()
    invalid = d <= 0
    if invalid.any() and (d > 0).any():
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        if gnd is not None:
            src_m = (d > 0) & gnd
            tgt_m = invalid & gnd
            if src_m.sum() >= 16 and tgt_m.any():
                # degenerate guard: griddata's Qhull dies when all source
                # points share one coordinate (e.g. a single LiDAR scan
                # column) -- fall back to skipping the linear fill
                _sy, _sx = yy[src_m], xx[src_m]
                if _sy.max() > _sy.min() and _sx.max() > _sx.min():
                    vals = griddata(np.stack([_sy, _sx], 1), d[src_m],
                                    np.stack([yy[tgt_m], xx[tgt_m]], 1),
                                    method="linear")
                    out = d[tgt_m]; ok = np.isfinite(vals)
                    out[ok] = vals[ok]; d[tgt_m] = out; invalid = d <= 0
        if gnd is not None and gpl is not None and invalid.any():
            fill = invalid & gnd & (gpl > 0)
            d[fill] = gpl[fill]; invalid = d <= 0
        _, (iy, ix) = ndimage.distance_transform_edt(d <= 0, return_indices=True)
        near = d[iy, ix]
        same = (seg[iy, ix] == seg) & (seg > 0)
        fill = invalid & same & (near > 0)
        d[fill] = near[fill]
        still = d <= 0
        if still.any():
            for sid in np.unique(seg[still]):
                if sid == 0:
                    continue
                cells = seg == sid
                vals = d[cells & (d > 0)]
                if len(vals) >= 3:
                    d[cells & (d <= 0)] = np.median(vals)
    elif gnd is not None and gpl is not None and invalid.any():
        fill = invalid & gnd & (gpl > 0)
        d[fill] = gpl[fill]
    d[sky] = SKY_D
    d[ego] = EGO_D
    return d


CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def read_T_ego_lidar(path):
    lines = open(path).read().splitlines()
    idx = [i for i, l in enumerate(lines) if "4x4" in l][0]
    return np.array([[float(x) for x in l.split()]
                     for l in lines[idx + 1:idx + 5]], np.float64)


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


def seg_maps(sg):
    """seg: int32 segment map (class-as-segment), sky/ego/ground booleans."""
    seg = sg.astype(np.int32)
    sky = np.isin(seg, list(SKY))
    ego = np.isin(seg, list(CAR))          # refined below by median depth
    gnd = np.isin(seg, list(GROUND))
    return seg, sky, ego, gnd


def process_scene(args):
    scene, workers, limit, force = args
    raw = os.path.join(BASE, "raw_data", scene)
    outd = os.path.join(BASE, "scenes", scene)
    try:
        T_el = read_T_ego_lidar(os.path.join(raw, "calib/lidar/lidar2imu_calib.txt"))
        R_el, t_el = T_el[:3, :3], T_el[:3, 3]
        man = json.load(open(os.path.join(outd, "manifest.json")))
        ts = sorted(f[:-4] for f in os.listdir(os.path.join(raw, "lidar"))
                    if f.endswith(".pcd"))
        frames = man["frames"][:limit] if limit else man["frames"]
        if len(ts) != len(man["frames"]):
            return f"[warn] {scene}: pcd {len(ts)} != frames {len(man['frames'])}"

        cams = []
        for ch in CAMS:
            K = np.array(man["cams"][ch]["K"], np.float64)
            T_ego_cam = np.array(man["cams"][ch]["T_ego_cam"], np.float64)
            T_cam_ego = np.linalg.inv(T_ego_cam)
            gpl = ground_plane_depth(K / 4.0, T_ego_cam)
            cams.append((K / 4.0, T_cam_ego, gpl))

        os.makedirs(os.path.join(outd, "depth4"), exist_ok=True)
        n_ok = n_fail = 0
        for fr in frames:
            fi = fr["frame"]
            p = os.path.join(outd, f"depth4/{fi:04d}.npz")
            if os.path.exists(p) and not force:
                fr["depth4"] = f"depth4/{fi:04d}.npz"
                continue
            try:
                pts = read_pcd_xyz(os.path.join(raw, "lidar", f"{ts[fi]}.pcd"))
                pe = pts @ R_el.T + t_el
                depth = np.zeros((len(CAMS), DH, DW), np.float32)
                sg = np.load(os.path.join(outd, f"seg2d21/{fi:04d}.npz"))["seg"]
                for ci, ch in enumerate(CAMS):
                    K4, T_cam_ego, gpl = cams[ci]
                    pc = pe @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
                    z = pc[:, 2]
                    m = (z > 0.5) & (z < MAX_DEPTH)
                    u = (K4[0, 0] * pc[m, 0] / z[m] + K4[0, 2]).astype(np.int32)
                    v = (K4[1, 1] * pc[m, 1] / z[m] + K4[1, 2]).astype(np.int32)
                    zm = z[m].astype(np.float32)
                    ok = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
                    d = np.full(DH * DW, np.inf, np.float32)
                    np.minimum.at(d, v[ok] * DW + u[ok], zm[ok])
                    d[np.isinf(d)] = 0.0
                    d = d.reshape(DH, DW)
                    seg, sky, ego, gnd = seg_maps(sg[ci])
                    if ego.any():
                        # approximate ego vehicle: car segments whose LiDAR median < 3 m
                        for sid in np.unique(seg[ego]):
                            cells = seg == sid
                            vals = d[cells & (d > 0)]
                            if len(vals) and np.median(vals) < 3.0:
                                ego |= cells
                            else:
                                ego &= ~cells
                    d = densify(d, seg, sky, ego, gnd, gpl)
                    depth[ci] = d
                np.savez_compressed(p, depth=depth.astype(np.float16))
                fr["depth4"] = f"depth4/{fi:04d}.npz"
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  [warn] {scene} frame {fi}: {e}", flush=True)
        json.dump(man, open(os.path.join(outd, "manifest.json"), "w"))
        return f"[ok] {scene}: regenerated {n_ok} frames ({n_fail} failed)"
    except Exception as e:
        import traceback
        return f"[fail] {scene}: {e}\n{traceback.format_exc()[-800:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = [(s, a.workers, a.limit, a.force) for s in a.scenes]
    if a.limit or len(jobs) == 1:
        for j in jobs:
            print(process_scene(j), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            for r in ex.map(process_scene, jobs):
                print(r, flush=True)


if __name__ == "__main__":
    main()
