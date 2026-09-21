#!/usr/bin/env python3
"""Visualize Multi-Modal E2E Trajectory Modes (Straight, Left, Right)
and Navigation Intent Conditioning on Validation Scenes.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "METEOR"))

from bevlane.dataset import BevLaneDataset, CAMS as DATASET_CAMS
from bevlane.model import DepthSegIPMNetV52, enable_depth_slim
from deploy.occ_iso import cube_render_fast

# Colors (BGR)
COLOR_STRAIGHT = (60, 240, 60)      # Bright Green
COLOR_LEFT = (240, 180, 40)         # Sky Cyan/Blue
COLOR_RIGHT = (40, 120, 255)        # Orange/Red
MODE_COLORS = [COLOR_STRAIGHT, COLOR_LEFT, COLOR_RIGHT]
MODE_NAMES = ["Mode 0 (Straight)", "Mode 1 (Left Turn)", "Mode 2 (Right Turn)"]

# BEV 9-class Palette (matching dataset and validation)
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


def render_bev_modes(bev_map, wps_dict, probs_dict, active_mode=None, title="", W=450, H=500):
    """Render BEV map with trajectories overlaid."""
    rgb = BEV_PAL[np.minimum(bev_map, 8)]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cx, cy = 250, 400
    # Ego marker
    cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)

    for k in [0, 1, 2]:
        wp = wps_dict[k]
        col = MODE_COLORS[k]
        is_active = (active_mode is None) or (k == active_mode)
        thickness = 4 if is_active else 2

        pts = [(cx, cy)]
        for dx, dy in wp:
            px = int(cx - dy / 0.2)
            py = int(cy - dx / 0.2)
            pts.append((px, py))

        for i in range(len(pts) - 1):
            if is_active:
                cv2.line(bgr, pts[i], pts[i + 1], col, thickness, cv2.LINE_AA)
            else:
                cv2.line(bgr, pts[i], pts[i + 1], tuple(int(c * 0.45) for c in col), thickness, cv2.LINE_AA)

        for px, py in pts[1:]:
            r = 6 if is_active else 4
            cv2.circle(bgr, (px, py), r, col, -1, cv2.LINE_AA)
            if is_active:
                cv2.circle(bgr, (px, py), r + 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Zoom around ego: forward 45m (row 175), rear 15m (row 475), lateral +-20m (col 150..350)
    crop = bgr[175:475, 150:350]
    out = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    # Title header
    if title:
        cv2.rectangle(out, (0, 0), (W, 26), (30, 30, 36), -1)
        cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (60, 60, 70), 1)
    return out


def render_cam_view(raw_img, wps_dict, probs_dict, K, Tce, active_mode=None, title="", W=760, H=440):
    """Render front wide camera view with 3D trajectories projected onto road."""
    img = cv2.resize(raw_img, (W, H))
    sx, sy = W / 768.0, H / 432.0

    # Draw non-active first, then active
    draw_order = [k for k in [0, 1, 2] if k != active_mode] + ([active_mode] if active_mode is not None else [])

    for k in draw_order:
        wp = wps_dict[k]
        col = MODE_COLORS[k]
        is_active = (active_mode is None) or (k == active_mode)
        thickness = 4 if is_active else 2

        # 3D waypoints -> camera 2D coordinates
        pts = []
        # include ego origin near bottom front hood
        for dx, dy in np.concatenate([[[2.5, 0.0]], wp], 0):
            p_ego = np.array([dx, dy, -0.65, 1.0])
            p_cam = Tce @ p_ego
            if p_cam[2] > 0.5:
                u = (K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]) * sx
                v = (K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]) * sy
                if -W < u < 2 * W and -H < v < 2 * H:
                    pts.append((int(u), int(v)))

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                if is_active:
                    cv2.line(img, pts[i], pts[i + 1], col, thickness, cv2.LINE_AA)
                else:
                    cv2.line(img, pts[i], pts[i + 1], tuple(int(c * 0.45) for c in col), thickness, cv2.LINE_AA)
            for pt in pts[1:]:
                r = 6 if is_active else 4
                cv2.circle(img, pt, r, col, -1, cv2.LINE_AA)
                if is_active:
                    cv2.circle(img, pt, r + 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (W, 26), (30, 30, 36), -1)
    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (60, 60, 70), 1)
    return img


def build_intent_dashboard(raw_img_fw, bev_pred, occ_pred, results_by_cond, meta_info, K_fw, Tc_fw):
    """Build a unified 1920x1080 multi-panel dashboard comparing autonomous decision
    vs straight, left, and right navigation intent conditioning.
    """
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    canvas[:] = (18, 18, 22)

    # 1. Header Banner
    cv2.rectangle(canvas, (0, 0), (1920, 48), (28, 28, 35), -1)
    cv2.line(canvas, (0, 48), (1920, 48), (60, 60, 75), 1)
    title = f"METEOR Multi-Modal E2E Planning & Intent Conditioning | Scene: {meta_info['scene']} | Frame {meta_info['frame']} | v0: {meta_info['v0']:.1f} m/s"
    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 245), 2, cv2.LINE_AA)

    # 2. Top-Left: Front Wide Camera with all 3 Candidate Modes
    auto_res = results_by_cond["autonomous"]
    cam_img = render_cam_view(
        raw_img_fw, auto_res["wps"], auto_res["probs"], K_fw, Tc_fw,
        active_mode=auto_res["best_k"],
        title="Front Wide Camera: Projected 3-Mode Candidates (Green=Straight, Blue=Left, Orange=Right)",
        W=760, H=440
    )
    canvas[58:498, 16:776] = cam_img

    # 3. Top-Center: Global BEV with All 3 Candidate Branches
    bev_img = render_bev_modes(
        bev_pred, auto_res["wps"], auto_res["probs"],
        active_mode=auto_res["best_k"],
        title="BEV Map: Multi-Modal Trajectory Tree & Probabilities",
        W=550, H=440
    )
    canvas[58:498, 786:1336] = bev_img

    # 4. Top-Right: 3D Occupancy Voxel Grid
    occ_iso = cube_render_fast(occ_pred, W=550, H=440, rng_m=24.0)
    cv2.rectangle(occ_iso, (0, 0), (550, 26), (30, 30, 36), -1)
    cv2.putText(occ_iso, "3D Semantic Occupancy Grid (Voxel Space)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(occ_iso, (0, 0), (549, 439), (60, 60, 70), 1)
    canvas[58:498, 1346:1896] = occ_iso

    # 5. Bottom Panel: 4 Conditions Side-by-Side Comparison (y=520 to 1020, height 490)
    conditions = [
        ("autonomous", "(A) Autonomous Decision (No Intent)", "Auto Selected: Mode 0 (Straight)"),
        ("straight", "(B) Intent Conditioned: [Straight]", "Forced: Mode 0 (Stay in Lane)"),
        ("left", "(C) Intent Conditioned: [Left Turn]", "Commanded: Mode 1 (Turn Left +4.97m)"),
        ("right", "(D) Intent Conditioned: [Right Turn]", "Commanded: Mode 2 (Turn Right -6.74m)"),
    ]

    PANEL_W, PANEL_H = 455, 470
    for idx, (cond_key, banner_title, subtitle) in enumerate(conditions):
        x0 = 16 + idx * (PANEL_W + 16)
        y0 = 516
        res = results_by_cond[cond_key]
        wps = res["wps"]
        probs = res["probs"]
        best_k = res["best_k"]

        panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        panel[:] = (24, 24, 28)

        # Draw miniature BEV crop
        mini_bev = render_bev_modes(bev_pred, wps, probs, active_mode=best_k, title="", W=PANEL_W - 16, H=280)
        panel[10:290, 8:PANEL_W - 8] = mini_bev

        # Info Box at bottom
        cv2.rectangle(panel, (8, 298), (PANEL_W - 8, PANEL_H - 10), (32, 32, 38), -1)
        cv2.rectangle(panel, (8, 298), (PANEL_W - 8, PANEL_H - 10), (55, 55, 65), 1)

        cv2.putText(panel, banner_title, (16, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, subtitle, (16, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.44, MODE_COLORS[best_k], 1, cv2.LINE_AA)

        # Mode probabilities text
        for k in range(3):
            wp_end = wps[k, -1]
            p_val = probs[k] * 100
            prefix = "-> " if k == best_k else "   "
            txt = f"{prefix}{MODE_NAMES[k]}: {p_val:5.1f}% | 3s: ({wp_end[0]:+.2f}m, {wp_end[1]:+.2f}m)"
            col = (240, 240, 240) if k == best_k else (140, 140, 140)
            cv2.putText(panel, txt, (16, 374 + k * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

        # Dynamics line
        steer_deg = np.degrees(res["steer"])
        dyn_txt = f"Dynamics: Steer={steer_deg:+.1f} deg | Accel={res['accel']:+.2f} m/s2 | Brake={res['brake']:.2f}"
        cv2.putText(panel, dyn_txt, (16, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (65, 65, 75), 1)
        canvas[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = panel

    # 6. Legend Banner (bottom 35)
    cv2.rectangle(canvas, (0, 1045), (1920, 1080), (22, 22, 26), -1)
    cv2.line(canvas, (0, 1045), (1920, 1045), (55, 55, 65), 1)
    legends = [
        ("Mode 0: Straight / Lane Keep", COLOR_STRAIGHT),
        ("Mode 1: Left Turn / Left Lane Change", COLOR_LEFT),
        ("Mode 2: Right Turn / Right Lane Change", COLOR_RIGHT),
        ("Active Selected Trajectory (Solid Bold Line)", (255, 255, 255)),
    ]
    lx = 30
    for name, bgr_col in legends:
        cv2.rectangle(canvas, (lx, 1056), (lx + 16, 1070), bgr_col, -1)
        cv2.rectangle(canvas, (lx, 1056), (lx + 16, 1070), (200, 200, 200), 1)
        cv2.putText(canvas, name, (lx + 22, 1068), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        lx += len(name) * 10 + 45

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data_20260910_064331")
    ap.add_argument("--frames", default="50,150,300")
    ap.add_argument("--ckpt", default=os.path.join(BASE_DIR, "checkpoints/meteor_custom_v52/best.pt"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "diagnostics/intent_vis"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_indices = [int(f.strip()) for f in args.frames.split(",") if f.strip()]
    scenes_dir = os.path.join(BASE_DIR, "scenes")

    print(f"[*] Initializing dataset for scene: {args.scene} ...", flush=True)
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

    print(f"[*] Loading model from: {args.ckpt} ...", flush=True)
    model = DepthSegIPMNetV52(n_seg=21).cuda()
    enable_depth_slim(model, widths=(128, 128, 96, 64))
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print("[+] Model loaded successfully!", flush=True)

    ego_npz = np.load(os.path.join(scenes_dir, args.scene, "ego_motion.npz"))
    v0_arr = ego_npz["v0"]
    manifest = json.load(open(os.path.join(scenes_dir, args.scene, "manifest.json")))
    K_fw = np.array(manifest["cams"]["CAM_FRONT_WIDE"]["K"], np.float32)
    Tce_fw = np.linalg.inv(np.array(manifest["cams"]["CAM_FRONT_WIDE"]["T_ego_cam"], np.float32))

    for fi in frame_indices:
        print(f"[*] Generating intent visualization for frame {fi} ...", flush=True)
        sample = ds[fi]
        imgs_t, K_t, Tc_t, gt_t = (t[None].cuda() for t in sample[:4])
        depth4_t = sample[4][None].cuda()
        lidar_bev_t = sample[8][None].cuda()

        # Load raw Front Wide camera image
        man_f = manifest["frames"][fi]
        p_fw = os.path.join(scenes_dir, args.scene, man_f["imgs"]["CAM_FRONT_WIDE"])
        raw_img_fw = cv2.imread(p_fw)

        # Evaluate under 4 conditions
        conditions_to_run = [
            ("autonomous", None),
            ("straight", [1.0, 0.0, 0.0]),
            ("left", [0.0, 1.0, 0.0]),
            ("right", [0.0, 0.0, 1.0]),
        ]

        results_by_cond = {}
        bev_pred_base = None
        occ_pred_base = None

        v0_curr = float(v0_arr[fi]) if fi < len(v0_arr) else 0.0
        v0_t = torch.tensor([v0_curr], dtype=torch.float32, device="cuda")

        for cond_name, it_val in conditions_to_run:
            it_t = torch.tensor([it_val], dtype=torch.float32, device="cuda") if it_val is not None else None
            with torch.no_grad():
                out = model(imgs_t, K_t, Tc_t, lidar=depth4_t, lidar_bev=lidar_bev_t, v0=v0_t, intent=it_t)

            ego_arr = out[7][0].cpu().numpy()
            wps = ego_arr[:36].reshape(3, 6, 2)
            confs = ego_arr[36:39]
            probs = np.exp(confs - np.max(confs))
            probs = probs / np.sum(probs)
            best_k = int(np.argmax(confs))

            steer = float(ego_arr[39])
            accel = float(ego_arr[40])
            brake = float(1.0 / (1.0 + np.exp(-ego_arr[41])))

            results_by_cond[cond_name] = {
                "wps": wps,
                "confs": confs,
                "probs": probs,
                "best_k": best_k,
                "steer": steer,
                "accel": accel,
                "brake": brake,
            }

            if cond_name == "autonomous":
                bev_pred_base = out[0][0].argmax(0).cpu().numpy()
                occ_logits = out[8][0]
                # Drop to 10 height bins, +-24m horizontal
                occ_pred_base = occ_logits[:10].argmax(0).cpu().numpy().astype(np.uint8)

        meta = {
            "scene": args.scene,
            "frame": fi,
            "v0": float(v0_arr[fi]) if fi < len(v0_arr) else 0.0,
        }

        dashboard = build_intent_dashboard(
            raw_img_fw, bev_pred_base, occ_pred_base,
            results_by_cond, meta, K_fw, Tce_fw
        )
        out_path = os.path.join(args.out_dir, f"intent_comparison_frame_{fi:04d}.png")
        cv2.imwrite(out_path, dashboard)
        print(f"[+] Saved dashboard to {out_path}", flush=True)

    print(f"[+] All intent visualization dashboards generated in: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
