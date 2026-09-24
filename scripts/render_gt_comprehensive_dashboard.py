#!/usr/bin/env python3
"""Comprehensive Ground Truth Multimodal Inspection Dashboard Generator.

Visualizes full-stack ground truth across all sensor streams and perception heads:
  - 6 Surround Camera Views:
      * 2D Detection Bounding Boxes (YOLOv8x consensus with fine-grained classes)
      * 3D Bounding Box Wireframe Projections using LATEST XCalib Single-DOF Refined Extrinsics
  - BEV Static & Dynamic Ground Truth:
      * Static Road Surface Map (gt/*.png: road, sidewalk, crosswalk, laneline, stopline, etc.)
      * Purified 3D Bounding Boxes (bev_box/*.npz)
      * Future 3.0s Trajectory Polylines (agent_traj/*.npz)
      * Moving vs Stationary Vehicle Status (Red dynamic polylines vs Cyan parked markers)
  - 3D Occupancy Ground Truth:
      * High-fidelity Isometric 3D Voxel Grid Rendering (occ/*.npz)
      * Metric ground grid and height-shaded voxel cubes

Outputs:
  box3d_artifacts/gt_comprehensive_inspection/<scene>/frame_<fi:04d>_dashboard.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Camera order and layout
CAMS_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT",  "CAM_BACK_WIDE",  "CAM_BACK_RIGHT"],
]
CAMS_ORDER = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAM_NAME_MAP = {
    "CAM_FRONT_WIDE": "front_wide",
    "CAM_FRONT_LEFT": "front_left",
    "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK_WIDE": "back_wide",
    "CAM_BACK_LEFT": "back_left",
    "CAM_BACK_RIGHT": "back_right",
}

IMG_W, IMG_H = 768, 432
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2

CLASS_NAMES_2D = {
    0: "obstacle",
    1: "car",
    2: "truck",
    3: "bus",
    4: "bicycle",
    5: "motorcycle",
    6: "pedestrian",
}

CLASS_COLORS_2D = {
    0: (180, 180, 180), # obstacle: grey
    1: (0, 255, 128),   # car: bright spring green
    2: (0, 200, 255),   # truck: yellow-orange
    3: (255, 128, 0),   # bus: cyan
    4: (255, 255, 0),   # bicycle: cyan-blue
    5: (180, 100, 255), # motorcycle: purple
    6: (0, 100, 255),   # pedestrian: orange-red
}

# 3D Box Wireframe Edges
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]

# OCC Palette: 0:free, 1:obstacle, 2:vehicle, 3:2wheel, 4:ped, 5:road, 6:sidewalk, 7:veg, 8:building, 9:pole
OCC_PAL = np.array([
    [0, 0, 0],          # 0: free
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


def load_lidar2ego(deploy_dir: Path, scene: str) -> np.ndarray:
    """Loads 4x4 lidar2imu transformation matrix."""
    calib_file = deploy_dir / "raw_data" / scene / "calib/lidar/lidar2imu_calib.txt"
    rows = []
    in_block = False
    for line in calib_file.read_text().splitlines():
        if "4x4" in line:
            in_block = True
            continue
        if in_block:
            vals = line.split()
            if len(vals) == 4:
                rows.append([float(v) for v in vals])
            if len(rows) == 4:
                break
    return np.array(rows, dtype=np.float64)


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    """Computes 8 corners of a 3D bounding box in ego frame."""
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def draw_3d_wireframe(img, corners_3d_ego, K, T_lidar_camera, T_ego_lidar, color=(0, 255, 128), thickness=2):
    """Projects 3D corners (in ego frame) onto camera plane via lidar frame using latest refined extrinsics."""
    # Step 1: Ego -> LiDAR
    corners_lidar = corners_3d_ego @ T_ego_lidar[:3, :3].T + T_ego_lidar[:3, 3]
    # Step 2: LiDAR -> Camera
    corners_cam = corners_lidar @ T_lidar_camera[:3, :3].T + T_lidar_camera[:3, 3]
    depth = corners_cam[:, 2]

    # Require box to be in front of camera
    if depth.mean() < 1.0:
        return img

    h, w, _ = img.shape
    sx, sy = w / 768.0, h / 432.0

    pts_2d = []
    for i in range(8):
        if depth[i] > 0.5:
            ui = (K[0, 0] * corners_cam[i, 0] / depth[i] + K[0, 2]) * sx
            vi = (K[1, 1] * corners_cam[i, 1] / depth[i] + K[1, 2]) * sy
            pts_2d.append((int(round(ui)), int(round(vi))))
        else:
            pts_2d.append(None)

    valid_pts = [p for p in pts_2d if p is not None]
    if len(valid_pts) < 4:
        return img

    # Check that at least 2 corners project inside image canvas [0, w) x [0, h)
    in_screen_cnt = sum(1 for p in valid_pts if 0 <= p[0] < w and 0 <= p[1] < h)
    if in_screen_cnt < 2:
        return img

    for p1, p2 in BOX_EDGES:
        if pts_2d[p1] is not None and pts_2d[p2] is not None:
            pt1 = pts_2d[p1]
            pt2 = pts_2d[p2]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Front orientation cross (p0-p5 and p1-p4)
    if pts_2d[0] is not None and pts_2d[1] is not None and pts_2d[4] is not None and pts_2d[5] is not None:
        cv2.line(img, pts_2d[0], pts_2d[5], color, 1, cv2.LINE_AA)
        cv2.line(img, pts_2d[1], pts_2d[4], color, 1, cv2.LINE_AA)

    return img


def render_cube_occ(occ, W=640, H=864, rng_m=24.0, drop=(8,), zmax_m=3.2):
    """High-quality 3D isometric voxel rendering with metric ground grid."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (20, 20, 24)
    occ = np.asarray(occ)
    n = min(int(rng_m / 0.4), occ.shape[1] // 2)
    r0 = occ.shape[1] // 2 - n
    zmax = min(int((zmax_m + 1.0) / 0.4), occ.shape[0])
    occ_sub = occ[:zmax, r0:r0 + 2 * n, r0:r0 + 2 * n]
    su = W / (3.0 * n)
    a, b = su * 0.75, su * 1.15 * 0.375
    sz = su * 0.95
    v0 = H * 0.22

    def pt(r, c, z):
        return (int((c - r) * a + W // 2), int((c + r) * b - z * sz + v0))

    # Metric Ground Grid (every 4m = 10 cells)
    gcol = (38, 38, 46)
    for g in range(0, 2 * n + 1, 10):
        cv2.line(img, pt(g, 0, 0), pt(g, 2 * n, 0), gcol, 1, cv2.LINE_AA)
        cv2.line(img, pt(0, g, 0), pt(2 * n, g, 0), gcol, 1, cv2.LINE_AA)

    FLAT = (5, 6)
    keep = (occ_sub > 0) & (occ_sub != 255) & ~np.isin(occ_sub, drop)
    flat_m = keep & np.isin(occ_sub, FLAT)
    cube_m = keep & ~np.isin(occ_sub, FLAT)

    # Flat carpet (road / sidewalk)
    zz, rr, cc = np.nonzero(flat_m)
    order = np.argsort(rr + cc)
    for k in order:
        z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
        cls_idx = min(int(occ_sub[z, r, c]), len(OCC_PAL) - 1)
        col = (OCC_PAL[cls_idx][::-1] * 0.65).astype(np.uint8).tolist()
        poly = np.array([pt(r, c, 0), pt(r + 1, c, 0), pt(r + 1, c + 1, 0), pt(r, c + 1, 0)], np.int32)
        cv2.fillPoly(img, [poly], col)

    # Isometric Range Rings at 10m and 20m on ground plane (after road)
    for R in [10.0, 20.0]:
        pts = []
        for deg in range(0, 360, 3):
            rad = np.radians(deg)
            x = R * np.cos(rad)
            y = R * np.sin(rad)
            r = n - x / 0.4
            c = n - y / 0.4
            pts.append(pt(r, c, 0))
        poly_pts = np.array(pts, np.int32)
        ring_col = (85, 85, 105) if R == 20.0 else (58, 58, 72)
        cv2.polylines(img, [poly_pts], True, ring_col, 1, cv2.LINE_AA)

    # 3D Shaded Cubes
    zz, rr, cc = np.nonzero(cube_m)
    if len(zz):
        order = np.argsort((rr + cc) * (occ_sub.shape[0] + 1) + zz)
        for k in order:
            z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
            cls_idx = min(int(occ_sub[z, r, c]), len(OCC_PAL) - 1)
            base = OCC_PAL[cls_idx][::-1].astype(np.float32)
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
    p_ego = pt(n, n, 0)
    cv2.drawMarker(img, p_ego, (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)

    # Coordinate tripod at ego: +X (Forward, Red), +Y (Left, Green), +Z (Up, Blue)
    cv2.arrowedLine(img, p_ego, pt(n - 10, n, 0), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(img, p_ego, pt(n, n - 10, 0), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(img, p_ego, pt(n, n, 4), (255, 120, 0), 2, cv2.LINE_AA, tipLength=0.25)

    # Metric distance labels along axes on ground plane
    for R in [10.0, 20.0]:
        pt_fwd = pt(n - R / 0.4, n, 0)
        cv2.putText(img, f"{int(R)}m", (pt_fwd[0] + 4, pt_fwd[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 220), 1, cv2.LINE_AA)
        pt_bwd = pt(n + R / 0.4, n, 0)
        cv2.putText(img, f"-{int(R)}m", (pt_bwd[0] + 4, pt_bwd[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 180), 1, cv2.LINE_AA)

    p_x_max = pt(0, n, 0)
    cv2.putText(img, "+24m (Fwd)", (p_x_max[0] + 4, p_x_max[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 230), 1, cv2.LINE_AA)
    return img


def render_dashboard_frame(scene_dir: Path, deploy_dir: Path, fi: int, out_path: Path, cams_calib: dict, refined_calib: dict, T_ego_lidar: np.ndarray, draw_2d_boxes: bool = False, traj_subdir: str = "agent_traj", stat_thresh: float = 1.0):
    """Renders a single high-resolution multi-modal composite dashboard."""
    manifest_p = scene_dir / "manifest.json"
    with open(manifest_p) as f:
        manifest = json.load(f)
    fr = manifest["frames"][fi]

    # 1. Load 2D Detection Boxes (only if draw_2d_boxes enabled)
    b2d_p = scene_dir / "bbox2d" / f"{fi:04d}.npz"
    boxes_2d, counts_2d = None, None
    if draw_2d_boxes and b2d_p.exists():
        z2d = np.load(b2d_p)
        boxes_2d = z2d["boxes"]
        counts_2d = z2d["counts"]

    # 2. Load 3D Bounding Boxes
    bev_p = scene_dir / "bev_box" / f"{fi:04d}.npz"
    boxes_3d = []
    if bev_p.exists():
        zbev = np.load(bev_p)
        if "boxes_3d" in zbev:
            boxes_3d = zbev["boxes_3d"]

    # 3. Load 3.0s Trajectory Ground Truth
    traj_p = scene_dir / traj_subdir / f"{fi:04d}.npz"
    traj_boxes, trajs, tvalids, n_agents = None, None, None, 0
    if traj_p.exists():
        ztraj = np.load(traj_p)
        traj_boxes = ztraj["boxes"]
        trajs = ztraj["traj"]
        tvalids = ztraj["tvalid"]
        n_agents = int(ztraj["count"])

    # Motion Colors: Vivid Red for Moving, Royal Blue for Stationary
    COLOR_MOVING = (0, 0, 255)       # Red (BGR)
    COLOR_STATIONARY = (255, 60, 0)  # Royal Blue (BGR)

    def get_box_motion_status(cx, cy, threshold=stat_thresh):
        if traj_boxes is None or n_agents == 0:
            return False, []
        best_dist = 1e9
        best_k = -1
        for k in range(n_agents):
            t_cx, t_cy = traj_boxes[k, 1], traj_boxes[k, 2]
            d = (cx - t_cx)**2 + (cy - t_cy)**2
            if d < best_dist:
                best_dist = d
                best_k = k
        if best_k >= 0 and best_dist < 0.25:
            deltas = trajs[best_k]
            valids = tvalids[best_k]
            max_disp = 0.0
            fut_pts = []
            for h in range(6):
                if valids[h] > 0.5:
                    dx, dy = deltas[h]
                    disp = np.sqrt(dx * dx + dy * dy)
                    max_disp = max(max_disp, disp)
                    fut_pts.append((cx + dx, cy + dy))
            is_moving = (max_disp >= threshold)
            return is_moving, (fut_pts if is_moving else [])
        return False, []

    # Pre-compute motion state for each 3D box
    box_motion_info = []
    n_moving, n_stationary = 0, 0
    for b in boxes_3d:
        cls_3d, cx, cy, zc, l, w, h, yaw = b
        is_moving, fut_pts = get_box_motion_status(cx, cy)
        col = COLOR_MOVING if is_moving else COLOR_STATIONARY
        box_motion_info.append((is_moving, fut_pts, col))
        if is_moving:
            n_moving += 1
        else:
            n_stationary += 1

    # 4. Render 6 Camera Views
    cam_imgs = {}
    for ci, cname in enumerate(CAMS_ORDER):
        rel_img = fr["imgs"].get(cname)
        p_img = scene_dir / rel_img if rel_img else None
        if p_img and p_img.exists():
            img = cv2.imread(str(p_img))
        else:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        ckey = CAM_NAME_MAP.get(cname)
        cal = cams_calib.get(cname)
        if cal is not None and ckey in refined_calib:
            K = np.array(cal["K"], dtype=np.float64)
            # Use LATEST Single-DOF Refined Extrinsics:
            T_lc = np.array(refined_calib[ckey]["T_lidar_camera"], dtype=np.float64)

            # Draw 3D wireframe boxes projected onto 2D image (RED: moving, BLUE: stationary)
            for bi, b in enumerate(boxes_3d):
                cls_3d, cx, cy, zc, l, w, h, yaw = b
                cors = box_3d_corners(cx, cy, zc, l, w, h, yaw)
                _, _, col = box_motion_info[bi]
                draw_3d_wireframe(img, cors, K, T_lc, T_ego_lidar, color=col, thickness=2)

        # Draw 2D Detection Boxes
        if draw_2d_boxes and boxes_2d is not None and counts_2d is not None and ci < len(counts_2d):
            cnt = int(counts_2d[ci])
            for k in range(cnt):
                cls_id, cx, cy, w, h = boxes_2d[ci, k]
                cls_int = int(round(cls_id))
                x1 = int(round(cx - w / 2.0))
                y1 = int(round(cy - h / 2.0))
                x2 = int(round(cx + w / 2.0))
                y2 = int(round(cy + h / 2.0))
                col = CLASS_COLORS_2D.get(cls_int, (255, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                label_txt = CLASS_NAMES_2D.get(cls_int, f"cls_{cls_int}")
                # Semi-transparent tag background
                tw, th = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
                cv2.putText(img, label_txt, (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Camera Header Badge
        cv2.rectangle(img, (8, 8), (260, 38), (15, 15, 20), -1)
        cv2.putText(img, cname, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cam_imgs[cname] = img

    row_top = np.hstack([cam_imgs[c] for c in CAMS_LAYOUT[0]])
    row_bot = np.hstack([cam_imgs[c] for c in CAMS_LAYOUT[1]])
    cams_grid = np.vstack([row_top, row_bot])

    # 5. Render BEV Static Map + Dynamic Boxes + 3.0s Trajectories
    gt_p = scene_dir / "gt" / f"{fi:04d}.png"
    gt_png = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) if gt_p.exists() else None
    bev_color = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev_color[:] = (18, 18, 22)

    if gt_png is not None:
        bev_color[gt_png == 1] = (65, 65, 72)       # Road drivable (slate)
        bev_color[gt_png == 2] = (130, 80, 130)     # Sidewalk (purple)
        bev_color[gt_png == 3] = (0, 230, 255)      # Crosswalk (yellow)
        bev_color[gt_png == 4] = (255, 255, 255)    # Laneline (crisp white)
        bev_color[gt_png == 5] = (40, 40, 240)      # Stopline (red)
        bev_color[gt_png == 6] = (0, 140, 255)      # Road edge (orange)
        bev_color[gt_png == 7] = (220, 210, 50)     # Markings (cyan)
        bev_color[gt_png == 8] = (50, 140, 210)     # Parking (blue)

    def to_bev_px(x, y):
        r = int(round((BEV_XH - x) / RES))
        c = int(round((BEV_YH - y) / RES))
        return (c, r)

    ego_px = to_bev_px(0.0, 0.0)

    # 5b. Metric Range Rings & Axis Scale (20m, 40m, 60m, 80m)
    for rng in [20, 40, 60, 80]:
        r_px = int(round(rng / RES))
        is_key = (rng in (40, 80))
        col = (85, 85, 105) if is_key else (50, 50, 62)
        cv2.circle(bev_color, ego_px, r_px, col, 1, cv2.LINE_AA)

        # Forward label along longitudinal axis
        pt_f = to_bev_px(float(rng), 0.0)
        if 15 <= pt_f[1] < BEV_H - 15:
            cv2.putText(bev_color, f"{rng}m", (pt_f[0] + 4, pt_f[1] + (14 if rng == 80 else 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 205), 1, cv2.LINE_AA)

        # Backward label
        pt_b = to_bev_px(-float(rng), 0.0)
        if 15 <= pt_b[1] < BEV_H - 160:
            cv2.putText(bev_color, f"-{rng}m", (pt_b[0] + 4, pt_b[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 175), 1, cv2.LINE_AA)

        # Lateral labels for 40m key range
        if rng == 40:
            pt_l = to_bev_px(0.0, 40.0)
            cv2.putText(bev_color, "40m", (pt_l[0] + 4, pt_l[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 175), 1, cv2.LINE_AA)
            pt_r = to_bev_px(0.0, -40.0)
            cv2.putText(bev_color, "40m", (pt_r[0] - 28, pt_r[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 175), 1, cv2.LINE_AA)

    # Subtle crosshair axis lines through ego
    cv2.line(bev_color, (ego_px[0], 10), (ego_px[0], BEV_H - 160), (42, 42, 52), 1, cv2.LINE_AA)
    cv2.line(bev_color, (15, ego_px[1]), (BEV_W - 15, ego_px[1]), (42, 42, 52), 1, cv2.LINE_AA)

    # Draw 3D oriented boxes & trajectories on BEV (RED border: moving, BLUE border: stationary)
    for bi, b in enumerate(boxes_3d):
        cls_3d, cx, cy, zc, l, w, h, yaw = b
        cb, sb = np.cos(yaw), np.sin(yaw)
        cors = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
            px = cx + lx * cb - wy * sb
            py = cy + lx * sb + wy * cb
            cors.append(to_bev_px(px, py))

        is_moving, fut_pts, col = box_motion_info[bi]

        # Fill box with subtle background tone
        fill_col = (25, 20, 45) if is_moving else (45, 30, 20)
        cv2.fillPoly(bev_color, [np.array(cors, dtype=np.int32)], fill_col)
        # Bounding box border: RED for moving, BLUE for stationary
        cv2.polylines(bev_color, [np.array(cors, dtype=np.int32)], True, col, 2, cv2.LINE_AA)

        # Front indicator line inside box (marks vehicle head position)
        p_FL, p_FR, p_RR, p_RL = [np.array(p, float) for p in cors]
        alpha = 0.28
        p1 = (1.0 - alpha) * p_FL + alpha * p_RL
        p2 = (1.0 - alpha) * p_FR + alpha * p_RR
        cv2.line(bev_color, (int(round(p1[0])), int(round(p1[1]))),
                           (int(round(p2[0])), int(round(p2[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Longitudinal centerline from center to front bumper
        p_center = (p_FL + p_FR + p_RR + p_RL) / 4.0
        p_front = (p_FL + p_FR) / 2.0
        cv2.line(bev_color, (int(round(p_center[0])), int(round(p_center[1]))),
                           (int(round(p_front[0])), int(round(p_front[1]))), (255, 255, 255), 1, cv2.LINE_AA)

        # If moving: draw future 3.0s trajectory polyline with arrow
        if is_moving and fut_pts:
            pts_traj = [to_bev_px(cx, cy)] + [to_bev_px(fx, fy) for fx, fy in fut_pts]
            if len(pts_traj) > 1:
                cv2.polylines(bev_color, [np.array(pts_traj, dtype=np.int32)], False, COLOR_MOVING, 2, cv2.LINE_AA)
                cv2.arrowedLine(bev_color, pts_traj[-2], pts_traj[-1], COLOR_MOVING, 2, cv2.LINE_AA, tipLength=0.35)
            for p in pts_traj[1:]:
                cv2.circle(bev_color, p, 3, COLOR_MOVING, -1, cv2.LINE_AA)

    # Ego vehicle footprint
    cv2.rectangle(bev_color, (ego_px[0] - 6, ego_px[1] - 12), (ego_px[0] + 6, ego_px[1] + 12), (0, 0, 255), -1)
    cv2.polylines(bev_color, [np.array([
        (ego_px[0] - 6, ego_px[1] - 12), (ego_px[0] + 6, ego_px[1] - 12),
        (ego_px[0] + 6, ego_px[1] + 12), (ego_px[0] - 6, ego_px[1] + 12)
    ])], True, (255, 255, 255), 1)

    bev_panel_w = int(cams_grid.shape[0] * (BEV_W / BEV_H))  # 540 px
    bev_panel = cv2.resize(bev_color, (bev_panel_w, cams_grid.shape[0]), interpolation=cv2.INTER_NEAREST)

    # BEV Legend & Overlay at BOTTOM of panel
    leg_h = 148
    leg_y1 = cams_grid.shape[0] - leg_h - 12
    leg_y2 = cams_grid.shape[0] - 12
    cv2.rectangle(bev_panel, (12, leg_y1), (bev_panel_w - 12, leg_y2), (14, 14, 18), -1)
    cv2.rectangle(bev_panel, (12, leg_y1), (bev_panel_w - 12, leg_y2), (50, 50, 60), 1)
    cv2.putText(bev_panel, "BEV Map & 3.0s Trajectory", (22, leg_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bev_panel, f"3D Objects: {len(boxes_3d)} confirmed", (22, leg_y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)
    cv2.putText(bev_panel, f"Moving: {n_moving} (RED Border + Traj)", (22, leg_y1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bev_panel, f"Stationary: {n_stationary} (BLUE Border)", (22, leg_y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 80, 0), 1, cv2.LINE_AA)
    cv2.putText(bev_panel, "Range: 20m, 40m, 60m, 80m | Head: white line", (22, leg_y1 + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 190), 1, cv2.LINE_AA)

    # 6. Render 3D Occupancy Ground Truth Panel
    occ_p = scene_dir / "occ" / f"{fi:04d}.npz"
    occ_arr = np.load(occ_p)["occ"] if occ_p.exists() else np.zeros((16, 200, 200), dtype=np.uint8)
    occ_panel_w = 640
    occ_panel = render_cube_occ(occ_arr, W=occ_panel_w, H=cams_grid.shape[0])

    # OCC Legend & Overlay at BOTTOM of panel
    cv2.rectangle(occ_panel, (12, leg_y1), (occ_panel_w - 12, leg_y2), (14, 14, 18), -1)
    cv2.rectangle(occ_panel, (12, leg_y1), (occ_panel_w - 12, leg_y2), (50, 50, 60), 1)
    cv2.putText(occ_panel, "3D Occupancy Ground Truth", (22, leg_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(occ_panel, "Range: 48m x 48m (-24m~+24m) | Rings: 10m, 20m | Grid: 4m", (22, leg_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    # Class Swatches & Labels (accurate BGR colors matching voxel render)
    swatch_y1 = leg_y1 + 64
    # Vehicle (Blue)
    cv2.rectangle(occ_panel, (24, swatch_y1), (36, swatch_y1 + 12), (235, 0, 0), -1)
    cv2.putText(occ_panel, "Veh (Blue)", (42, swatch_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # Road (Purple)
    cv2.rectangle(occ_panel, (140, swatch_y1), (152, swatch_y1 + 12), (130, 65, 130), -1)
    cv2.putText(occ_panel, "Road (Purple)", (158, swatch_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # Sidewalk (Pink)
    cv2.rectangle(occ_panel, (280, swatch_y1), (292, swatch_y1 + 12), (230, 35, 235), -1)
    cv2.putText(occ_panel, "Sidewalk (Pink)", (298, swatch_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # Obstacle (White)
    cv2.rectangle(occ_panel, (430, swatch_y1), (442, swatch_y1 + 12), (235, 230, 230), -1)
    cv2.putText(occ_panel, "Obstacle (White)", (448, swatch_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)

    swatch_y2 = leg_y1 + 88
    # Veg (Green)
    cv2.rectangle(occ_panel, (24, swatch_y2), (36, swatch_y2 + 12), (35, 140, 105), -1)
    cv2.putText(occ_panel, "Veg (Green)", (42, swatch_y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # Pole (Yellow)
    cv2.rectangle(occ_panel, (140, swatch_y2), (152, swatch_y2 + 12), (0, 220, 230), -1)
    cv2.putText(occ_panel, "Pole (Yellow)", (158, swatch_y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # Pedestrian (Crimson)
    cv2.rectangle(occ_panel, (280, swatch_y2), (292, swatch_y2 + 12), (65, 25, 220), -1)
    cv2.putText(occ_panel, "Ped (Crimson)", (298, swatch_y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)
    # 2-Wheel (Brown)
    cv2.rectangle(occ_panel, (430, swatch_y2), (442, swatch_y2 + 12), (35, 15, 120), -1)
    cv2.putText(occ_panel, "2-Wheel (Brown)", (448, swatch_y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 220), 1, cv2.LINE_AA)

    cv2.putText(occ_panel, "View: Isometric 3D | Ego: yellow triangle | Axes: +X(Red) +Y(Grn) +Z(Blu)",
                (22, leg_y1 + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1, cv2.LINE_AA)

    # 7. Horizontal Stack of 3 Columns
    body = np.hstack([cams_grid, bev_panel, occ_panel])

    # 8. Top Header Banner
    header_h = 50
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    header[:] = (12, 12, 16)
    cv2.line(header, (0, header_h - 1), (body.shape[1], header_h - 1), (60, 60, 75), 1)

    title_txt = f"METEOR MULTIMODAL GROUND TRUTH INSPECTION  |  Scene: {scene_dir.name}  |  Frame: {fi:04d} / {len(manifest['frames'])}"
    meta_txt = f"Calib: XCalib Single-DOF Refined  |  3D Boxes: {len(boxes_3d)}  |  Moving: {n_moving}  |  Stationary: {n_stationary}"
    cv2.putText(header, title_txt, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, meta_txt, (body.shape[1] - 1050, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 1, cv2.LINE_AA)

    full_dashboard = np.vstack([header, body])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full_dashboard, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def worker_render(task_args):
    scene_str, deploy_str, fi, out_str, cams_calib, refined_calib, T_ego_lidar, draw_2d_boxes, traj_subdir, stat_thresh = task_args
    scene_dir = Path(scene_str)
    deploy_dir = Path(deploy_str)
    out_path = Path(out_str)
    render_dashboard_frame(scene_dir, deploy_dir, fi, out_path, cams_calib, refined_calib, T_ego_lidar, draw_2d_boxes, traj_subdir, stat_thresh)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Render comprehensive ground truth dashboard across scenes.")
    parser.add_argument("--deploy-dir", default="/home/nvidia/working_ppt/Postdoc_Materials/论文3/meteor_6cam_lidar_deploy")
    parser.add_argument("--scenes", default="all", help="all or comma-separated scene list")
    parser.add_argument("--frames-per-scene", type=int, default=20, help="number of frames to sample per scene")
    parser.add_argument("--frames", default=None, help="comma-separated specific frame indices")
    parser.add_argument("--traj-subdir", default="agent_traj", help="trajectory subdir (default: agent_traj)")
    parser.add_argument("--stat-thresh", type=float, default=1.0, help="stationary 3s distance threshold (default: 1.0m)")
    parser.add_argument("--out-dir", default=None, help="output directory (default: box3d_artifacts/gt_comprehensive_inspection)")
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    parser.add_argument("--draw-2d-box", action="store_true", default=False, help="whether to draw 2D detection boxes (default: False)")
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir)
    scenes_root = deploy_dir / "scenes"
    out_dir = Path(args.out_dir) if args.out_dir else deploy_dir / "box3d_artifacts" / "gt_comprehensive_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Refined Calibrations
    calib_json = deploy_dir / "tools/xcalib_calibration/refined_single_dof/deploy_extrinsics_final.json"
    scaled_calib_json = deploy_dir / "tools/xcalib_calibration/converted/scaled_calibrations.json"
    refined_data = json.loads(calib_json.read_text())["cameras"]

    # Fallback to scaled calibs if some cameras (e.g. front_wide, back_right) not in refined
    if scaled_calib_json.exists():
        first_scene = list(json.loads(scaled_calib_json.read_text())["scenes"].values())[0]["cameras"]
        for cname, cinfo in first_scene.items():
            if cname not in refined_data:
                refined_data[cname] = {"T_lidar_camera": cinfo["T_lidar_camera"]}

    if args.scenes == "all":
        scene_dirs = sorted([p for p in scenes_root.glob("data_*") if p.is_dir()])
    else:
        scene_dirs = [scenes_root / s.strip() for s in args.scenes.split(",") if s.strip()]

    print("=" * 80)
    print("  METEOR Multimodal Ground Truth Comprehensive Inspection Generator")
    print(f"  Target Scenes:       {len(scene_dirs)}")
    print(f"  Traj Subdir:         {args.traj_subdir}")
    print(f"  Stat Threshold:      {args.stat_thresh}m")
    print(f"  Output Directory:    {out_dir}")
    print(f"  Extrinsics:          XCalib Single-DOF Refined (deploy_extrinsics_final.json)")
    print(f"  Draw 2D Boxes:       {args.draw_2d_box}")
    print(f"  Parallel Workers:    {args.workers}")
    print("=" * 80)

    tasks = []
    for s_dir in scene_dirs:
        scene = s_dir.name
        manifest_p = s_dir / "manifest.json"
        if not manifest_p.exists():
            print(f"[!] {scene}: manifest.json not found, skipping.")
            continue

        manifest = json.loads(manifest_p.read_text())
        total_frames = len(manifest["frames"])
        cams_calib = manifest["cams"]

        # Load scene lidar2ego
        T_lidar_ego = load_lidar2ego(deploy_dir, scene)
        T_ego_lidar = np.linalg.inv(T_lidar_ego)

        if args.frames:
            sample_indices = [int(f.strip()) for f in args.frames.split(",") if f.strip()]
        else:
            sample_indices = np.round(np.linspace(20, total_frames - 45, args.frames_per_scene)).astype(int).tolist()

        for fi in sample_indices:
            out_p = out_dir / scene / f"frame_{fi:04d}_dashboard.jpg"
            tasks.append((
                str(s_dir), str(deploy_dir), fi, str(out_p),
                cams_calib, refined_data, T_ego_lidar,
                args.draw_2d_box, args.traj_subdir, args.stat_thresh
            ))

    print(f"[*] Dispatching {len(tasks)} dashboard rendering tasks across {args.workers} workers...")
    t0 = time.time()
    n_done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res_path in executor.map(worker_render, tasks):
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                elapsed = time.time() - t0
                fps = n_done / max(elapsed, 1e-4)
                remain = (len(tasks) - n_done) / max(fps, 1e-4)
                print(f"  [{n_done:3d}/{len(tasks):3d}] ({100*n_done/len(tasks):5.1f}%) | Speed: {fps:.2f} frames/s | ETA: {remain:.1f}s")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"  [SUCCESS] Finished {n_done} dashboards in {total_time:.1f}s ({n_done/total_time:.2f} fps)!")
    print(f"  Output Directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
