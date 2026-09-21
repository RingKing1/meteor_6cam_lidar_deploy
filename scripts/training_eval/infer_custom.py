#!/usr/bin/env python3
"""Run the 6-camera + LiDAR engine on a converted custom scene -> mp4."""
import argparse, json, os, sys, time
import cv2
import numpy as np

os.environ.setdefault("METEOR_OCC_VIEW", "iso")   # 3D isometric voxel OCC panel
os.environ.setdefault("METEOR_OCC_EVERY", "1")    # render every frame on modern GPUs
sys.path.insert(0, os.environ.get("METEOR_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "METEOR")))
from deploy.runtime import MeteorRT
from deploy import orin_render as R

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--no-lidar", action="store_true")
ap.add_argument("--occ-view", default="iso", choices=["iso", "top"], help="OCC view mode: iso (3D) or top (2D)")
ap.add_argument("--occ-every", type=int, default=1, help="Render OCC every N frames (1 for full rate)")
a = ap.parse_args()

os.environ["METEOR_OCC_VIEW"] = a.occ_view
os.environ["METEOR_OCC_EVERY"] = str(a.occ_every)

m = json.load(open(os.path.join(a.scene, "manifest.json")))
K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32))
               for c in CAMS])[None]
v0s = np.load(os.path.join(a.scene, "ego_motion.npz"))["v0"]

rt = MeteorRT(a.engine, skip_outputs=(), n_out_slots=2)
R.CAM_DRAW = list(CAMS)
R.CAMS = list(CAMS)
R.set_bev_extent(int(rt.shapes["lane"][-2]))
print("[render] 6-camera custom layout", flush=True)

frames = m["frames"][::a.stride]
if a.limit:
    frames = frames[:a.limit]
vw = None
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (R.VW, R.VH))

t_start = time.time()
for n, f in enumerate(frames):
    raw = {c: cv2.imread(os.path.join(a.scene, f["imgs"][c])) for c in CAMS}
    imgs = np.stack([raw[c][:, :, ::-1].transpose(2, 0, 1)
                     for c in CAMS])[None]
    lb = None
    if not a.no_lidar:
        lb = np.load(os.path.join(a.scene, f["lidar_bev"]))["lb"] \
            .astype(np.float32)[None]
    v0 = float(v0s[f["frame"]]) if f["frame"] < len(v0s) else 0.0
    t0 = time.time()
    out = rt.infer(imgs, K, Tc, v0=v0, lidar_bev=lb)
    if lb is not None:
        out = dict(out); out["lidar_bev_in"] = lb
    dt = (time.time() - t0) * 1000
    canvas = R.compose_frame(raw, K, Tc, v0, out, dt)
    if vw is not None:
        vw.write(canvas)
    if n % 100 == 0 or n == len(frames) - 1:
        elapsed = time.time() - t_start
        fps_curr = (n + 1) / max(elapsed, 0.001)
        eta_sec = (len(frames) - n - 1) / max(fps_curr, 0.001)
        print(f"[{n:04d}/{len(frames)}] infer+render: {dt:.1f} ms | avg: {fps_curr:.1f} FPS | ETA: {eta_sec/60:.1f} min | v0: {v0:.1f} m/s",
              flush=True)
if vw is not None:
    vw.release()
print("DONE", a.out or "(no video)")
