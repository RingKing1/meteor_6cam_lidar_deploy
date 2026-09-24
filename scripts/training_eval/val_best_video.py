#!/usr/bin/env python3
"""Render the GT-vs-prediction validation dashboard to an MP4 (best-model picker companion).

Reuses build_dashboard() from visualize_val.py (6 camera overlay + BEV GT/pred +
E2E trajectory + 3D OCC GT/pred), but iterates frames continuously and writes a
video instead of a handful of PNGs.

  python3 scripts/training_eval/val_best_video.py \
      --ckpt checkpoints/.../best.pt \
      --scene data_20260910_064331 \
      --out videos/val_best_data_20260910_064331.mp4 \
      --stride 2 --fps 10
"""
import argparse, json, os, sys
import cv2
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "training_eval"))

from bevlane.model import MODELS                              # noqa: E402
from visualize_val import build_dashboard                    # noqa: E402

SCENES_DIR = os.path.join(BASE_DIR, "scenes")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=2, help="frame stride (2 = every other frame)")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--n-cams", type=int, default=6)
a = ap.parse_args()

scene_dir = os.path.join(SCENES_DIR, a.scene)
assert os.path.isdir(scene_dir), f"scene not found: {scene_dir}"

print(f"[val-video] building model v52, loading {a.ckpt}", flush=True)
model = MODELS["v52"](n_seg=21).cuda().eval()
ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
model.load_state_dict(ck["model"], strict=False)
print("[val-video] model loaded", flush=True)

man = json.load(open(os.path.join(scene_dir, "manifest.json")))
v0_all = np.load(os.path.join(scene_dir, "ego_motion.npz"))["v0"]
frames = man["frames"][::a.stride]
if a.limit:
    frames = frames[:a.limit]

# first frame fixes the dashboard size
os.makedirs(os.path.dirname(a.out), exist_ok=True)
vw = None
n_ok = 0
t0 = __import__("time").time()

for n, fr in enumerate(frames):
    fi = fr["frame"] if "frame" in fr else n * a.stride
    from bevlane.dataset import BevLaneDataset
    # dataset __getitem__ does all IO + GT assembly; build a single-item ds per frame is wasteful,
    # so inline via the manifest like visualize_val does.
    sample_ready = False
    try:
        raw_imgs = {c: cv2.imread(os.path.join(scene_dir, fr["imgs"][c]))
                    for c in fr["imgs"]}
        if any(v is None for v in raw_imgs.values()):
            continue
        imgs_t = torch.from_numpy(
            np.stack([raw_imgs[c][:, :, ::-1].transpose(2, 0, 1) for c in fr["imgs"]])[None]
        ).cuda()
        K_t = torch.from_numpy(
            np.stack([np.array(man["cams"][c]["K"], np.float32) for c in fr["imgs"]])[None]
        ).cuda()
        Tc_t = torch.from_numpy(
            np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32))
                      for c in fr["imgs"]])[None]
        ).cuda()
        lidar_bev_t = torch.from_numpy(
            np.load(os.path.join(scene_dir, fr["lidar_bev"]))["lb"]
        )[None].cuda()
        depth4_t = torch.from_numpy(
            np.load(os.path.join(scene_dir, fr["depth4"]))["d"]
        )[None].cuda()
        gt = cv2.imread(os.path.join(scene_dir, fr["gt"]), cv2.IMREAD_GRAYSCALE)
        sample_ready = True
    except KeyError:
        continue

    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = model(imgs_t, K_t, Tc_t, lidar=depth4_t, lidar_bev=lidar_bev_t)

    bev_pred = out[0][0].argmax(0).cpu().numpy()
    seg2d_pred = out[2][0].argmax(1).cpu().numpy()
    ego_pred = out[7][0].cpu().numpy()
    wp = ego_pred[:36].reshape(3, 6, 2)
    conf = ego_pred[36:39]
    best_wp = wp[np.argmax(conf)]
    occ_logits = out[8][0]
    free_margin = occ_logits[1:].max(0).values - occ_logits[0]
    occ_pred = occ_logits.argmax(0).cpu().numpy()
    occ_pred[free_margin.cpu().numpy() <= 0.20] = 0
    occ_gt = np.load(os.path.join(scene_dir, fr["occ"]))["occ"] if os.path.exists(
        os.path.join(scene_dir, fr["occ"])) else np.zeros_like(occ_pred)

    fi = int(fr["frame"])
    v0 = float(v0_all[fi]) if fi < len(v0_all) else 0.0
    meta = {"scene": a.scene, "frame": fi, "v0": v0}
    dashboard = build_dashboard(raw_imgs, seg2d_pred, gt, bev_pred, best_wp,
                                occ_gt, occ_pred, meta)
    if vw is None:
        H, W = dashboard.shape[:2]
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
        print(f"[val-video] dashboard {W}x{H}, {len(frames)} target frames", flush=True)
    vw.write(dashboard)
    n_ok += 1
    if n_ok % 50 == 0:
        print(f"[val-video] {n_ok} frames written, "
              f"{n_ok / max(__import__('time').time() - t0, 1):.1f} fps", flush=True)

if vw is not None:
    vw.release()
print(f"[val-video] DONE {a.out} ({n_ok} frames) in {__import__('time').time() - t0:.0f}s",
      flush=True)
