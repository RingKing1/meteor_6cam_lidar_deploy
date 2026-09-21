#!/usr/bin/env python3
import json, os, sys, time
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT

scene, engine = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(scene, "manifest.json")))
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32))
               for c in CAMS])[None]
v0s = np.load(os.path.join(scene, "ego_motion.npz"))["v0"]

rt = MeteorRT(engine, skip_outputs=("flow", "unk", "pl", "lg_pts", "lg_meta",
                                    "lg_adj", "occ", "hm2d_s0", "hm2d_s1",
                                    "hm2d_s2", "reg2d_s0", "reg2d_s1",
                                    "reg2d_s2", "depth", "seg2d", "depth_mean"),
              n_out_slots=1)

f = m["frames"][0]
imgs = np.stack([
    cv2.cvtColor(cv2.imread(os.path.join(scene, f["imgs"][c])),
                 cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    for c in CAMS])[None]
lb = np.load(os.path.join(scene, f["lidar_bev"]))["lb"].astype(np.float32)[None]
print("inputs:", imgs.shape, imgs.dtype, K.shape, Tc.shape,
      "v0=%.2f" % v0s[0], lb.shape, lb.dtype)

for i in range(5):
    t0 = time.time()
    out = rt.infer(imgs, K, Tc, v0=float(v0s[0]), lidar_bev=lb)
    dt = (time.time() - t0) * 1000
    if i == 0:
        for k, v in out.items():
            v = np.asarray(v)
            print(f"  {k:12s} {v.shape} {v.dtype} "
                  f"min={v.min():.3f} max={v.max():.3f}")
    print(f"frame {i}: {dt:.1f} ms")
print("SMOKE OK")
