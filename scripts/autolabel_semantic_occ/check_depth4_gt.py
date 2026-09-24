#!/usr/bin/env python3
"""Check depth4 GT quality: compare the densified depth4 against the raw
LiDAR sweep re-projected with the CURRENT manifest extrinsics.

For each requested frame:
  - sparse lidar depth: raw pcd -> ego -> per-camera min-pooled depth
    (the authoritative measurement; new extrinsics). LiDAR returns that land
    on the EGO CAR BODY are dropped first (|x|<3 m, |y|<1.1 m, 0.1<z<1.8 m),
    otherwise they become noisy near-range pixels in the depth map;
  - depth4: the densified GT the trainer reads (old or re-projected);
  - per-distance-band metrics on the pixels the LiDAR actually sees, and a
    4-panel visual: image | depth4 (turbo) | sparse lidar (turbo) |
    error heatmap, per camera.

METRICS (all in metres):
  - MAE  = Mean Absolute Error = mean(|depth4 - lidar|). Pure error SIZE
    (ignores direction). ~0.001 m = essentially identical; <0.2 m (half a
    0.4 m voxel) is good; >0.8 m is a densification-fill artifact.
  - bias = mean(depth4 - lidar) SIGNED. +0.1 m means depth4 is systematically
    0.1 m FARTHER than the LiDAR measurement (over-estimate); ~0 = no
    systematic offset. Use with MAE: bias tells direction, MAE tells size.
  - n     = number of LiDAR-covered pixels compared.
  - mae_<lo>_<hi> / bias_<lo>_<hi> = the same metrics restricted to the
    distance band (1-5, 5-10, 10-20, 20-40, 40-70 m). Far bands show large,
    frame-dependent MAE because the LiDAR is sparse there and depth4 is
    densified by interpolation/fill -- the GT itself is noisy far away, which
    is also why fine-tuning the depth head does not improve 40-70 m depth.

Outputs: diagnostics/depth4_check/<scene>/frame_<fi>_cam<N>.png
         diagnostics/depth4_check/<scene>/stats.json   (per-frame + band stats)

  python3 scripts/autolabel_semantic_occ/check_depth4_gt.py \
      --scene data_20260910_064331 --frames 100,200,300 --workers 4
"""
import argparse, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DH, DW = 108, 192

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
BANDS = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 70)]


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


def sparse_depth(pe, K4, T_cam_ego):
    pc = pe @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
    z = pc[:, 2]
    m = (z > 0.5) & (z < 79.0)
    u = (K4[0, 0] * pc[m, 0] / z[m] + K4[0, 2]).astype(np.int32)
    v = (K4[1, 1] * pc[m, 1] / z[m] + K4[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
    d = np.full(DH * DW, np.inf, np.float32)
    np.minimum.at(d, v[ok] * DW + u[ok], z[m][ok].astype(np.float32))
    d[np.isinf(d)] = 0.0
    return d.reshape(DH, DW)


def process_frame(args):
    fi, scene, scene_dir, raw_dir, pcd_files, cams, out_dir = args
    try:
        f = json.load(open(os.path.join(scene_dir, "manifest.json")))["frames"][fi]
        depth4 = np.load(os.path.join(scene_dir, f["depth4"]))["depth"]
        pts = read_pcd_xyz(os.path.join(raw_dir, "lidar", pcd_files[fi] + ".pcd"))
        R_el, t_el = cams["T_el"][:3, :3], cams["T_el"][:3, 3]
        pe = pts @ R_el.T + t_el
        _ego = ((np.abs(pe[:, 0]) < 3.0) & (np.abs(pe[:, 1]) < 1.1)
                & (pe[:, 2] > 0.1) & (pe[:, 2] < 1.8))
        pe = pe[~_ego]

        frame_stats = {}
        for ci, ch in enumerate(CAMS):
            K4, Tce = cams[ch]
            sp = sparse_depth(pe, K4, Tce)
            gt = depth4[ci]
            mask = sp > 0.5
            err = gt[mask] - sp[mask]
            abs_err = np.abs(err)
            stats = {"n": int(mask.sum()), "mae_m": float(abs_err.mean()) if mask.any() else None,
                     "bias_m": float(err.mean()) if mask.any() else None,
                     "gt_cover": float((gt[mask] > 0).mean()) if mask.any() else None}
            for lo, hi in BANDS:
                sel = mask & (sp > lo) & (sp < hi)
                if sel.any():
                    stats[f"mae_{lo}_{hi}"] = float(np.abs(gt[sel] - sp[sel]).mean())
                    stats[f"bias_{lo}_{hi}"] = float((gt[sel] - sp[sel]).mean())
                else:
                    stats[f"mae_{lo}_{hi}"] = stats[f"bias_{lo}_{hi}"] = None
            frame_stats[ch] = stats

            # visual: image | depth4 | sparse | error
            img = cv2.imread(os.path.join(scene_dir, f["imgs"][ch]))
            img = cv2.resize(img, (DW * 4, DH * 4), interpolation=cv2.INTER_LINEAR)
            def turbo(d):
                d8 = (d / 79.0 * 255).clip(0, 255).astype(np.uint8)
                c = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
                return cv2.resize(c, (DW * 4, DH * 4),
                                  interpolation=cv2.INTER_NEAREST)
            e = np.zeros_like(gt)
            e[mask] = np.abs(err) * 10.0
            panel = np.vstack([img, turbo(gt), turbo(sp), turbo(e)])
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"frame_{fi:04d}_cam{ci}.png"), panel)
        return fi, frame_stats
    except Exception as ex:
        import traceback as _tb
        print(f"[worker-err] frame {fi}: {_tb.format_exc()[-800:]}", flush=True)
        return fi, {"error": str(ex)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", default="100,200,300", help="comma frames")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    scene_dir = os.path.join(BASE, "scenes", a.scene)
    raw_dir = os.path.join(BASE, "raw_data", a.scene)
    man = json.load(open(os.path.join(scene_dir, "manifest.json")))
    pcd_files = sorted(f[:-4] for f in os.listdir(os.path.join(raw_dir, "lidar"))
                       if f.endswith(".pcd"))
    T_el = read_T_ego_lidar(os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt"))
    cams = {"T_el": T_el}
    for ch in CAMS:
        K = np.array(man["cams"][ch]["K"], np.float64)
        T_ego_cam = np.array(man["cams"][ch]["T_ego_cam"], np.float64)
        cams[ch] = (K / 4.0, np.linalg.inv(T_ego_cam))
    out_dir = os.path.join(BASE, "diagnostics", "depth4_check", a.scene)
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(int(x), a.scene, scene_dir, raw_dir, pcd_files, cams, out_dir)
            for x in a.frames.split(",")]
    stats = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fi, s in ex.map(process_frame, jobs):
            stats[fi] = s
            print(f"frame {fi}: n={s.get('CAM_FRONT_WIDE',{}).get('n')} "
                  f"mae={s.get('CAM_FRONT_WIDE',{}).get('mae_m')} "
                  f"bias={s.get('CAM_FRONT_WIDE',{}).get('bias_m')}", flush=True)

    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"[done] stats -> {os.path.join(out_dir, 'stats.json')}")


if __name__ == "__main__":
    main()
