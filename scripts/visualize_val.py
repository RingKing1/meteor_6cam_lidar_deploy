#!/usr/bin/env python3
"""Comprehensive Multi-Task Visualization for METEOR Epoch 1 on Validation Set.

Outputs a multi-panel visual dashboard per frame:
  - 6 Surround Camera Views with 2D Panoptic Segmentation Overlay
  - BEV Lane & Road Perception: Ground Truth vs Model Prediction (with E2E Trajectory)
  - 3D Occupancy View Mode: Ground Truth 3D Voxels vs Model Predicted 3D Voxels (Isometric Cube Rendering)

Usage:
  python3 scripts/visualize_val.py [--frames 50,150,300,500,800] [--out-dir diagnostics/val_epoch1_vis]
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "METEOR"))

from bevlane.dataset import BevLaneDataset, CAMS as DATASET_CAMS
from bevlane.model import DepthSegIPMNetV52, enable_depth_slim

CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT_WIDE",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_WIDE",
    "CAM_BACK_RIGHT",
]

# OCC Palette: 0:free, 1:obstacle, 2:vehicle, 3:2wheel, 4:ped, 5:road, 6:sidewalk, 7:veg, 8:building, 9:pole
OCC_PAL = np.array([
    [0, 0, 0],          # 0: free / unobserved
    [230, 230, 235],    # 1: obstacle (white/light grey)
    [0, 0, 235],        # 2: vehicle (red)
    [120, 15, 35],      # 3: 2wheel (dark red)
    [220, 25, 65],      # 4: ped (crimson)
    [130, 65, 130],     # 5: road (slate purple)
    [235, 35, 230],     # 6: sidewalk (magenta/pink)
    [105, 140, 35],     # 7: vegetation (olive green)
    [70, 70, 75],       # 8: building (dark grey)
    [230, 220, 0],      # 9: pole / sign (yellow)
], np.uint8)

# BEV 9-class Palette
BEV_PAL = np.array([
    [18, 18, 22],       # 0: bg
    [75, 75, 80],       # 1: road
    [110, 85, 85],      # 2: sidewalk
    [220, 210, 40],     # 3: crosswalk
    [255, 235, 40],     # 4: laneline
    [220, 40, 40],      # 5: stopline
    [180, 40, 180],     # 6: road_edge
    [0, 210, 210],      # 7: marking
    [50, 140, 210],     # 8: parking
], np.uint8)

# 2D 21-class Palette
SEG21_PAL = np.array([
    [0, 0, 0], [60, 60, 60], [0, 0, 220], [0, 80, 200], [0, 160, 200],
    [140, 20, 40], [200, 30, 70], [220, 20, 60], [255, 230, 40], [255, 140, 0],
    [255, 215, 0], [128, 64, 128], [244, 35, 232], [255, 255, 0], [250, 170, 30],
    [0, 0, 0], [102, 102, 156], [70, 70, 70], [107, 142, 35], [70, 130, 180],
    [220, 220, 0],
], np.uint8)


def cube_render(occ, W=640, H=500, rng_m=24.0, drop=(8,), zmax_m=3.2):
    """High-quality 3D isometric voxel rendering with metric ground grid."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (24, 24, 28)
    occ = np.asarray(occ)
    n = min(int(rng_m / 0.4), occ.shape[1] // 2)
    r0 = occ.shape[1] // 2 - n
    zmax = min(int((zmax_m + 1.0) / 0.4), occ.shape[0])
    occ = occ[:zmax, r0:r0 + 2 * n, r0:r0 + 2 * n]
    su = W / (3.0 * n)
    a, b = su * 0.75, su * 1.15 * 0.375
    sz = su * 0.95
    v0 = H * 0.16

    def pt(r, c, z):
        return (int((c - r) * a + W // 2), int((c + r) * b - z * sz + v0))

    # Metric Ground Grid (every 4m = 10 cells)
    gcol = (45, 45, 52)
    for g in range(0, 2 * n + 1, 10):
        cv2.line(img, pt(g, 0, 0), pt(g, 2 * n, 0), gcol, 1, cv2.LINE_AA)
        cv2.line(img, pt(0, g, 0), pt(2 * n, g, 0), gcol, 1, cv2.LINE_AA)

    FLAT = (5, 6)
    keep = (occ > 0) & (occ != 255) & ~np.isin(occ, drop)
    flat_m = keep & np.isin(occ, FLAT)
    cube_m = keep & ~np.isin(occ, FLAT)

    # Flat carpet (road / sidewalk)
    zz, rr, cc = np.nonzero(flat_m)
    order = np.argsort(rr + cc)
    for k in order:
        z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
        col = (OCC_PAL[occ[z, r, c]][::-1] * 0.65).astype(np.uint8).tolist()
        poly = np.array([pt(r, c, 0), pt(r + 1, c, 0), pt(r + 1, c + 1, 0), pt(r, c + 1, 0)], np.int32)
        cv2.fillPoly(img, [poly], col)

    # 3D Shaded Cubes
    zz, rr, cc = np.nonzero(cube_m)
    if len(zz):
        order = np.argsort((rr + cc) * (occ.shape[0] + 1) + zz)
        for k in order:
            z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
            base = OCC_PAL[occ[z, r, c]][::-1].astype(np.float32)
            shade = 0.65 + 0.35 * z / max(zmax - 1, 1)
            top = np.clip(base * shade, 0, 255).astype(np.uint8).tolist()
            left = np.clip(base * shade * 0.55, 0, 255).astype(np.uint8).tolist()
            right = np.clip(base * shade * 0.75, 0, 255).astype(np.uint8).tolist()
            t00, t10 = pt(r, c, z + 1), pt(r + 1, c, z + 1)
            t11, t01 = pt(r + 1, c + 1, z + 1), pt(r, c + 1, z + 1)
            b10, b11, b01 = pt(r + 1, c, z), pt(r + 1, c + 1, z), pt(r, c + 1, z)
            cv2.fillPoly(img, [np.array([t10, t11, b11, b10], np.int32)], left)
            cv2.fillPoly(img, [np.array([t01, t11, b11, b01], np.int32)], right)
            tp = np.array([t00, t10, t11, t01], np.int32)
            cv2.fillPoly(img, [tp], top)
            cv2.polylines(img, [tp], True, tuple(int(v * 0.40) for v in top), 1)

    # Ego vehicle position marker
    cv2.drawMarker(img, pt(n, n, 0), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


def render_bev(bev_map, traj_wp=None, title=""):
    """Render BEV map [800, 500] with optional trajectory overlay."""
    H, W = bev_map.shape
    # Palette lookup
    rgb = BEV_PAL[np.minimum(bev_map, 8)]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Ego vehicle marker at row 400, col 250 (range +-80m fwd/rear, +-50m left/right)
    cx, cy = 250, 400
    cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)

    # Draw trajectory waypoints: wp shape [6, 2] (forward dx, lateral dy) in metres
    if traj_wp is not None:
        pts = []
        for dx, dy in traj_wp:
            px = int(cx - dy / 0.2)
            py = int(cy - dx / 0.2)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(bgr, pts[i], pts[i + 1], (0, 255, 255), 3, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(bgr, (px, py), 5, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(bgr, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Crop/zoom around ego: show forward 50m (row 150), rear 20m (row 500), lateral +-30m (col 100..400)
    crop = bgr[150:500, 100:400]
    out = cv2.resize(crop, (300, 460), interpolation=cv2.INTER_NEAREST)

    # Border and title banner
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (60, 60, 65), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (35, 35, 40), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def build_dashboard(raw_imgs, seg2d_pred, bev_gt, bev_pred, ego_wp, occ_gt, occ_pred, meta_info):
    """Assemble a unified publication-quality 1920x1080 visual dashboard."""
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    canvas[:] = (18, 18, 22)

    # 1. Header Banner (height 45)
    cv2.rectangle(canvas, (0, 0), (1920, 45), (28, 28, 34), -1)
    cv2.line(canvas, (0, 45), (1920, 45), (55, 55, 65), 1)
    title_text = f"METEOR Epoch 1 Validation Inference | Scene: {meta_info['scene']} | Frame {meta_info['frame']} | v0: {meta_info['v0']:.1f} m/s | Model: DepthSegIPMNetV52"
    cv2.putText(canvas, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 245), 2, cv2.LINE_AA)

    # 2. Top Panel: 6 Surround Cameras (height 480, width 1920)
    # Arrange: Row 1 (FL, FW, FR), Row 2 (BL, BW, BR)
    cam_order = [
        ("CAM_FRONT_LEFT", "Front Left"),
        ("CAM_FRONT_WIDE", "Front Wide (Main)"),
        ("CAM_FRONT_RIGHT", "Front Right"),
        ("CAM_BACK_LEFT", "Back Left"),
        ("CAM_BACK_WIDE", "Back Wide"),
        ("CAM_BACK_RIGHT", "Back Right"),
    ]

    C_W, C_H = 634, 236
    for idx, (c_key, label) in enumerate(cam_order):
        row = idx // 3
        col = idx % 3
        x0 = 8 + col * (C_W + 4)
        y0 = 50 + row * (C_H + 4)

        orig = raw_imgs[c_key]
        small_orig = cv2.resize(orig, (C_W, C_H))

        # 2D segmentation overlay
        c_i = DATASET_CAMS.index(c_key)
        seg_c = seg2d_pred[c_i]  # [108, 192]
        seg_c_up = cv2.resize(seg_c, (C_W, C_H), interpolation=cv2.INTER_NEAREST)
        color_seg = SEG21_PAL[np.minimum(seg_c_up, 20)][:, :, ::-1]  # RGB to BGR
        has_label = seg_c_up > 0
        blend = small_orig.copy()
        blend[has_label] = cv2.addWeighted(small_orig[has_label], 0.65, color_seg[has_label], 0.35, 0)

        # Label tag
        cv2.rectangle(blend, (0, 0), (len(label) * 11 + 10, 24), (20, 20, 25), -1)
        cv2.putText(blend, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 235), 1, cv2.LINE_AA)
        cv2.rectangle(blend, (0, 0), (C_W - 1, C_H - 1), (50, 50, 55), 1)

        canvas[y0:y0 + C_H, x0:x0 + C_W] = blend

    # 3. Bottom Panel (y=534 to 1040, height 506)
    # Column 1: BEV GT & BEV Pred (x: 6 to 626)
    bev_gt_img = render_bev(bev_gt, title="BEV Ground Truth")
    bev_pred_img = render_bev(bev_pred, traj_wp=ego_wp, title="BEV Prediction + 3.0s Trajectory")
    canvas[540:1000, 10:310] = bev_gt_img
    canvas[540:1000, 320:620] = bev_pred_img

    # Column 2: 3D Occupancy Ground Truth (x: 630 to 1270)
    occ_gt_img = cube_render(occ_gt, W=630, H=475, rng_m=24.0)
    cv2.rectangle(occ_gt_img, (0, 0), (occ_gt_img.shape[1], 26), (35, 35, 40), -1)
    cv2.putText(occ_gt_img, "3D Occupancy Ground Truth (LiDAR)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(occ_gt_img, (0, 0), (occ_gt_img.shape[1] - 1, occ_gt_img.shape[0] - 1), (60, 60, 65), 1)
    canvas[540:1015, 634:1264] = occ_gt_img

    # Column 3: 3D Occupancy Model Prediction (x: 1274 to 1914)
    occ_pred_img = cube_render(occ_pred, W=630, H=475, rng_m=24.0)
    cv2.rectangle(occ_pred_img, (0, 0), (occ_pred_img.shape[1], 26), (35, 35, 40), -1)
    cv2.putText(occ_pred_img, "3D Occupancy Model Prediction (3D Mode)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(occ_pred_img, (0, 0), (occ_pred_img.shape[1] - 1, occ_pred_img.shape[0] - 1), (60, 60, 65), 1)
    canvas[540:1015, 1274:1904] = occ_pred_img

    # 4. Footer Legend (y=1045 to 1080)
    cv2.rectangle(canvas, (0, 1045), (1920, 1080), (25, 25, 30), -1)
    cv2.line(canvas, (0, 1045), (1920, 1045), (55, 55, 65), 1)
    legend_items = [
        ("Road", (130, 65, 130)),
        ("Sidewalk", (235, 35, 230)),
        ("Laneline", (40, 235, 255)),
        ("Obstacle", (230, 230, 235)),
        ("Vehicle", (0, 0, 235)),
        ("Pole/Sign", (0, 220, 230)),
        ("Vegetation", (35, 140, 105)),
        ("Future Path (3s)", (0, 255, 255)),
    ]
    lx = 30
    for name, bgr_col in legend_items:
        cv2.rectangle(canvas, (lx, 1055), (lx + 16, 1071), bgr_col, -1)
        cv2.rectangle(canvas, (lx, 1055), (lx + 16, 1071), (200, 200, 200), 1)
        cv2.putText(canvas, name, (lx + 22, 1068), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        lx += len(name) * 9 + 50

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data_20260910_064331", help="Validation scene name")
    ap.add_argument("--ckpt", default=os.path.join(BASE_DIR, "checkpoints/meteor_custom_v52/best.pt"))
    ap.add_argument("--frames", default="50,150,300,500,800", help="Comma-separated frame indices")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "diagnostics/val_epoch1_vis"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_indices = [int(f.strip()) for f in args.frames.split(",") if f.strip()]

    print(f"[*] Initializing dataset for validation scene: {args.scene} ...", flush=True)
    scenes_dir = os.path.join(BASE_DIR, "scenes")
    ds = BevLaneDataset(
        root=scenes_dir,
        scenes=[args.scene],
        with_depth=True,
        depth_hw=(108, 192),
        with_seg2d=True,
        seg2d_key="seg2d21",
        with_ego=True,
        with_occ=True,
        with_lidarbev=True,
        n_cams=8,
    )

    print(f"[*] Loading model from checkpoint: {args.ckpt} ...", flush=True)
    model = DepthSegIPMNetV52(n_seg=21).cuda()
    enable_depth_slim(model, widths=(128, 128, 96, 64))
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print("[+] Model loaded successfully!", flush=True)

    ego_npz = np.load(os.path.join(scenes_dir, args.scene, "ego_motion.npz"))
    v0_arr = ego_npz["v0"]

    for fi in frame_indices:
        print(f"[*] Processing validation frame {fi} ...", flush=True)
        sample = ds[fi]
        imgs_t, K_t, Tc_t, gt_t = (t[None].cuda() for t in sample[:4])
        depth4_t = sample[4][None].cuda()
        lidar_bev_t = sample[8][None].cuda()
        bev_gt = sample[3].cpu().numpy()
        occ_gt = sample[7].cpu().numpy()

        # Raw images
        man_f = json.load(open(os.path.join(scenes_dir, args.scene, "manifest.json")))["frames"][fi]
        raw_imgs = {}
        for c in [
            "CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT"
        ]:
            p = os.path.join(scenes_dir, args.scene, man_f["imgs"][c])
            raw_imgs[c] = cv2.imread(p)

        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = model(imgs_t, K_t, Tc_t, lidar=depth4_t, lidar_bev=lidar_bev_t)

        # 1. BEV Lane map
        bev_pred = out[0][0].argmax(0).cpu().numpy()

        # 2. 2D surround segmentation
        seg2d_pred = out[2][0].argmax(1).cpu().numpy()  # [8, 108, 192]

        # 3. E2E Trajectory
        ego_pred = out[7][0].cpu().numpy()
        wp = ego_pred[:36].reshape(3, 6, 2)
        conf = ego_pred[36:39]
        best_wp = wp[np.argmax(conf)]

        # 4. 3D Occupancy
        occ_logits = out[8][0]  # [10, 16, 200, 200]
        free_margin = occ_logits[1:].max(0).values - occ_logits[0]
        occ_pred = occ_logits.argmax(0).cpu().numpy()
        occ_pred[free_margin.cpu().numpy() <= 0.20] = 0

        meta = {
            "scene": args.scene,
            "frame": fi,
            "v0": float(v0_arr[fi]) if fi < len(v0_arr) else 0.0,
        }

        dashboard = build_dashboard(raw_imgs, seg2d_pred, bev_gt, bev_pred, best_wp, occ_gt, occ_pred, meta)
        out_path = os.path.join(args.out_dir, f"val_epoch1_frame_{fi:04d}.png")
        cv2.imwrite(out_path, dashboard)
        print(f"[+] Saved dashboard to {out_path}", flush=True)

    print(f"[+] All visual dashboards successfully created in {args.out_dir}!", flush=True)


if __name__ == "__main__":
    main()
