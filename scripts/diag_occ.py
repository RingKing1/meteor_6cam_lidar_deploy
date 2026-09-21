#!/usr/bin/env python3
"""Diagnostic: compare predicted OCC road voxels vs LiDAR ground points in one top-down view.

OCC grid convention (bevlane/extract_occ.py): row=(40-x)/0.4, col=(40-y)/0.4.
LiDAR raster (lidar_bev ch3 occupancy) shares ego frame x-fwd/y-left.
"""
import json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.environ.get("METEOR_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "METEOR")))
from deploy.runtime import MeteorRT

scene, engine, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
fi = int(sys.argv[4]) if len(sys.argv) > 4 else 10
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
m = json.load(open(os.path.join(scene, "manifest.json")))
K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32))
               for c in CAMS])[None]
v0s = np.load(os.path.join(scene, "ego_motion.npz"))["v0"]
rt = MeteorRT(engine, skip_outputs=(), n_out_slots=1)

f = m["frames"][fi]
raw = {c: cv2.imread(os.path.join(scene, f["imgs"][c])) for c in CAMS}
imgs = np.stack([raw[c][:, :, ::-1].transpose(2, 0, 1) for c in CAMS])[None]
lb = np.load(os.path.join(scene, f["lidar_bev"]))["lb"].astype(np.float32)[None]
out = rt.infer(imgs, K, Tc, v0=float(v0s[f["frame"]]), lidar_bev=lb)

occ = np.asarray(out["occ"])[0]              # [10,16,200,200] logits
free = occ[0]
best = occ.argmax(0)                          # [16,200,200]
margin = occ.max(0) - free                    # confidence over free
# road voxels: class 5 (OCC_NAMES index 5 = road), low z bins
z = margin.argmax(0)
road = (best[z, np.arange(200)[:, None], np.arange(200)[None, :]] == 5) if False else None
# simpler: any height bin classified road with margin>0
road_any = ((best == 5) & (margin > 0.5)).any(0)
obs_any = (np.isin(best, [2, 3, 4]) & (margin > 0.5) &
           (np.arange(16)[:, None, None] >= 1)).any(0)

# canvas: x up (front at top), y left -> image x = (40+y?) ; use col=(40-y)/0.4 -> y=40-c*0.4
S = 8
W = H = int(80 / 0.4 * S)                  # 200 cells * S px = +-40 m
canvas = np.full((H, W, 3), 24, np.uint8)
for g in range(0, 201, 25):                 # 10 m grid (25 cells)
    cv2.line(canvas, (g * S, 0), (g * S, H), (45, 45, 45), 1)
    cv2.line(canvas, (0, g * S), (W, g * S), (45, 45, 45), 1)
def put(mask, col, scale=1):
    rr, cc = np.nonzero(mask)               # row 0 = x+40 (front/top), col 0 = y+40 (left)
    px = np.clip(cc * S, 0, W - S); py = np.clip(rr * S, 0, H - S)
    for x, y in zip(px[::scale], py[::scale]):
        canvas[y:y + S, x:x + S] = col
put(road_any, (60, 180, 60), 1)               # green = predicted road
put(obs_any, (60, 60, 220), 1)                # red = predicted vehicles/VRU

# overlay LiDAR ground (z between -0.2..0.25) directly from raw pcd via raster ch:
l = lb[0]
occ_l = (l[3] > 0) & (l[1] < 0.3)            # ground returns
# lidar grid is 400x250 +-80/+-50; crop +-40 and remap: row=(80-x)/0.4 -> for +-40 rows 100..300
sub = occ_l[100:300, 25:225]                 # x +40..-40 (200), y +40..-40 (200)
rr, cc = np.nonzero(sub)
px = np.clip(cc * S, 0, W - S); py = np.clip(rr * S, 0, H - S)
for x, y in zip(px[::3], py[::3]):
    cv2.rectangle(canvas, (x, y), (x + 3, y + 3), (170, 200, 110), -1)  # teal ground

# ego marker (x=0,y=0 -> row=100,col=100 -> centre of canvas)
cx = cy = 100 * S
cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 3)
# axes: front arrow up
cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 70), (0, 0, 255), 3, tipLength=0.3)
cv2.putText(canvas, "FRONT", (cx - 30, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
cv2.putText(canvas, "green=OCC road  red=OCC obj  teal=LiDAR ground", (10, H-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
cv2.imwrite(out_png, canvas)
print("saved", out_png, "road cells:", int(road_any.sum()), "obj cells:", int(obs_any.sum()))
