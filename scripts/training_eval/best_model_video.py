#!/usr/bin/env python3
"""Render the best fine-tuned checkpoint to video in the EXISTING videos/ style.

This is deliberately NOT the GT-comparison dashboard: it loads a PyTorch
checkpoint and feeds it through deploy/orin_render.compose_frame -- the same
renderer infer_custom.py uses for data_20260910_*.mp4 -- so the output has:
  * 6 camera tiles, no segmentation overlay (upstream default since 2026-09-05),
    with 3D/2D boxes and the FRONT_WIDE path ribbon;
  * 6 depth tiles;
  * a large predicted BEV panel with boxes and the E2E trajectory.
No ground truth is shown.

  python3 scripts/training_eval/best_model_video.py \
      --ckpt checkpoints/.../picked_best.pt \
      --scene data_20260910_064331 \
      --out videos/best_model_data_20260910_064331.mp4 \
      --stride 2 --fps 10
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METEOR_DIR = os.environ.get("METEOR_REPO",
                            os.path.join(os.path.dirname(BASE_DIR), "METEOR"))
sys.path.insert(0, METEOR_DIR)
sys.path.insert(0, BASE_DIR)

from bevlane.model import MODELS                              # noqa: E402
from bevlane.ckpt_load import load_net                       # noqa: E402
from deploy.runtime import preprocess_images                 # noqa: E402
from deploy import orin_render as R                           # noqa: E402

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# Real tuple layout for the v52 checkpoint (observed from the model; hm2d/reg2d are nested tuples).
# scalar indices: 0 lane, 1 depth, 2 seg2d, 3 hm, 4 reg, 7 ego, 8 occ,
# 10 stationary, 14 traj. 5 = hm2d tuple(3), 6 = reg2d tuple(3).
# idx from the REAL model tuple (observed shapes):
#   0 lane[1,9,800,500] 1 depth 2 seg2d 3 hm 4 reg | 5 hm2d(t) 6 reg2d(t)
#   7 ego[1,42] 8 occ[1,10,16,200,200] 9 traj[1,39,400,250] 10 stationary[1,1,400,250]
# (14 is [1,24,12,2], a different head -- NOT traj; using it silently skipped
#  all agent-trajectory drawing in compose_frame because rr0<12 was never true)
SCALAR_MAP = {0: "lane", 1: "depth", 2: "seg2d", 3: "hm", 4: "reg",
              7: "ego", 8: "occ", 9: "traj", 10: "stationary"}
HM2D_IDX, REG2D_IDX = 5, 6
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

scene_dir = os.path.join(BASE_DIR, "scenes", a.scene)
assert os.path.isdir(scene_dir), f"scene not found: {scene_dir}"

print(f"[video] loading v52 + checkpoint {a.ckpt}", flush=True)
model = MODELS["v52"](n_seg=21).cuda().eval()
# load_net grows the depth-slim shape (and other branches) from the checkpoint itself
load_net(model, a.ckpt, verbose=True)
print("[video] model loaded", flush=True)

man = json.load(open(os.path.join(scene_dir, "manifest.json")))
v0_all = np.load(os.path.join(scene_dir, "ego_motion.npz"))["v0"]
frames = man["frames"][::a.stride]
if a.limit:
    frames = frames[:a.limit]

# match the 6-camera custom layout used for the existing videos
R.CAM_DRAW = list(CAMS); R.CAMS = list(CAMS)
# BEV extent MUST match the model's lane raster rows (800 = 80 fwd / 80 rear @0.2m).
# A hard-coded 1000 made the renderer crop with a 0.25 m/row assumption while the
# model emits 0.2 m/row: the ego marker was drawn ~16 m forward of the real ego
# position (fixed 2026-09-24; the older videos had this offset).
os.makedirs(os.path.dirname(a.out), exist_ok=True)
vw, n_ok = None, 0
t0 = time.time()

for fr in frames:
    try:
        raw = {c: cv2.imread(os.path.join(scene_dir, fr["imgs"][c])) for c in CAMS}
        if any(v is None for v in raw.values()):
            continue
        imgs = preprocess_images([raw[c] for c in CAMS])
        K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in CAMS])[None]
        Tc = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32))
                       for c in CAMS])[None]
        lb = np.load(os.path.join(scene_dir, fr["lidar_bev"]))["lb"]
        d4 = np.load(os.path.join(scene_dir, fr["depth4"]))["depth"]
    except KeyError:
        continue

    fi = int(fr["frame"])
    v0 = float(v0_all[fi]) if fi < len(v0_all) else 0.0
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out_t = model(torch.from_numpy(imgs).cuda(),
                      torch.from_numpy(K).cuda(),
                      torch.from_numpy(Tc).cuda(),
                      lidar=torch.from_numpy(d4)[None].cuda(),
                      lidar_bev=torch.from_numpy(lb)[None].cuda())
    if not getattr(R, "_bev_set", False):   # set once from the real lane rows
        R.set_bev_extent(int(out_t[0].shape[-2]))
        R._bev_set = True

    # tuple -> named dict in the format compose_frame expects; numpy/cpu, batch kept (MeteorRT style)
    out = {}
    for i, v in enumerate(out_t):
        if i in SCALAR_MAP:
            out[SCALAR_MAP[i]] = v.detach().float().cpu().numpy()
        elif i == HM2D_IDX:
            for k in range(3):
                out[f"hm2d_s{k}"] = v[k].detach().float().cpu().numpy()
        elif i == REG2D_IDX:
            for k in range(3):
                out[f"reg2d_s{k}"] = v[k].detach().float().cpu().numpy()

    canvas = R.compose_frame(raw, K, Tc, v0, out, 0.0)
    if vw is None:
        H, W = canvas.shape[:2]
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (W, H))
        print(f"[video] canvas {W}x{H}, {len(frames)} target frames", flush=True)
    vw.write(canvas)
    n_ok += 1
    if n_ok % 100 == 0:
        print(f"[video] {n_ok} frames, {n_ok / max(time.time() - t0, 1):.1f} fps",
              flush=True)

if vw is not None:
    vw.release()
print(f"[video] DONE {a.out} ({n_ok} frames) in {time.time() - t0:.0f}s",
      flush=True)
