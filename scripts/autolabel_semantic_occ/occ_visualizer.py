#!/usr/bin/env python3
"""Multi-Camera 3D Wireframe & 10-Class Semantic Occupancy BEV Visualizer.

Creates composite visualizations:
  - Left panel: 6 surround cameras (2x3 grid) with 3D wireframe bounding boxes.
  - Right panel: High-resolution 10-Class Occupancy BEV map with range rings,
    clean ego marker, voxel statistics, and color legend.

Usage:
  python3 scripts/autolabel_semantic_occ/occ_visualizer.py \
      --scene data_20260910_061820 \
      --frames 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 \
      --out-dir occ_test_vis
"""
import argparse
import json
import os
import re
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CAMS_ORDER = [
    ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT",  "CAM_BACK_WIDE",  "CAM_BACK_RIGHT"],
]

# OCC specifications
VOX = 0.4
XH, YH = 40.0, 40.0
Z0, Z1 = -1.0, 5.4
GX = GY = 200
GZ = 16

# 10-class Occupancy color palette in BGR:
PAL_BGR = {
    0: (35, 35, 35),       # free -> dark charcoal
    1: (200, 200, 200),   # obstacle -> light grey
    2: (0, 220, 0),       # vehicle -> bright green
    3: (0, 140, 255),     # 2wheel -> orange
    4: (0, 255, 255),     # pedestrian -> bright yellow
    5: (140, 80, 140),    # road -> purple
    6: (220, 100, 220),   # sidewalk -> pink/magenta
    7: (34, 139, 34),     # vegetation -> forest green
    8: (100, 100, 100),   # building -> slate/medium grey
    9: (255, 255, 0),     # pole/sign -> cyan
    255: (10, 10, 10)     # unknown/unobserved -> black
}

OCC_PRIO = np.array([0, 3, 5, 6, 7, 1, 1, 2, 2, 4], dtype=np.int8)

LEGEND_ITEMS = [
    ("Road", (140, 80, 140)),
    ("Sidewalk", (220, 100, 220)),
    ("Vehicle", (0, 220, 0)),
    ("Pedestrian", (0, 255, 255)),
    ("2-Wheel", (0, 140, 255)),
    ("Vegetation", (34, 139, 34)),
    ("Building", (100, 100, 100)),
    ("Pole/Sign", (255, 255, 0)),
    ("Obstacle", (200, 200, 200)),
    ("Free", (60, 60, 60)),
]

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]
FRONT_CROSS = [(0, 5), (1, 4)]


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def draw_3d_box_on_camera(img, corners_3d, K, T_ego_cam, color=(0, 255, 0), is_vru=False):
    R_ec = T_ego_cam[:3, :3]
    t_ec = T_ego_cam[:3, 3]
    p_cam = (corners_3d - t_ec) @ R_ec
    z = p_cam[:, 2]
    if (z > 0.5).sum() < 4:
        return img

    h, w, _ = img.shape
    sx, sy = w / 768.0, h / 432.0
    u = (K[0, 0] * p_cam[:, 0] / np.maximum(z, 0.2) + K[0, 2]) * sx
    v = (K[1, 1] * p_cam[:, 1] / np.maximum(z, 0.2) + K[1, 2]) * sy

    pts_2d = np.stack([u, v], axis=1).astype(np.int32)

    for p1, p2 in EDGES:
        if z[p1] > 0.5 and z[p2] > 0.5:
            pt1 = tuple(pts_2d[p1])
            pt2 = tuple(pts_2d[p2])
            if (-100 < pt1[0] < w + 100 and -100 < pt1[1] < h + 100) or \
               (-100 < pt2[0] < w + 100 and -100 < pt2[1] < h + 100):
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    front_col = (255, 255, 0) if not is_vru else (0, 200, 255)
    for p1, p2 in FRONT_CROSS:
        if z[p1] > 0.5 and z[p2] > 0.5:
            cv2.line(img, tuple(pts_2d[p1]), tuple(pts_2d[p2]), front_col, 1, cv2.LINE_AA)

    return img


def render_occ_bev(occ, target_size=864):
    """Renders 16x200x200 OCC grid into a high-res top-down BEV map with overlay annotations."""
    H, W = GX, GY
    bev_img = np.zeros((H, W, 3), dtype=np.uint8)

    for r_i in range(H):
        for c_i in range(W):
            col_voxels = occ[:, r_i, c_i]
            best_cls = 255
            best_prio = -1
            for v in col_voxels:
                if v == 255:
                    continue
                if v == 0 and best_prio < 0:
                    best_cls = 0
                    best_prio = 0
                elif 0 < v < 10:
                    p = OCC_PRIO[v]
                    if p > best_prio:
                        best_cls = v
                        best_prio = p
            bev_img[r_i, c_i] = PAL_BGR.get(best_cls, (10, 10, 10))

    # Resize to target square
    bev_canvas = cv2.resize(bev_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # Center of ego car (x=0, y=0 -> row=100, col=100)
    cx = cy = int(target_size / 2.0)
    px_per_m = target_size / 80.0  # 80m total span (-40 to +40m)

    # Draw metric range rings (10m, 20m, 30m)
    for dist_m in [10, 20, 30]:
        radius_px = int(dist_m * px_per_m)
        cv2.circle(bev_canvas, (cx, cy), radius_px, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(
            bev_canvas,
            f"{dist_m}m",
            (cx + 5, cy - radius_px + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (140, 140, 140),
            1,
            cv2.LINE_AA
        )

    # Crosshairs through ego origin
    cv2.line(bev_canvas, (cx, 0), (cx, target_size), (45, 45, 45), 1)
    cv2.line(bev_canvas, (0, cy), (target_size, cy), (45, 45, 45), 1)

    # Ego vehicle marker (triangle with heading arrow)
    cv2.drawMarker(bev_canvas, (cx, cy), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.arrowedLine(bev_canvas, (cx, cy), (cx, cy - 35), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(bev_canvas, "EGO", (cx - 15, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return bev_canvas


def render_frame_composite(scene_dir, raw_dir, fi, out_path, cams_calib):
    manifest = json.load(open(os.path.join(scene_dir, "manifest.json")))
    fr = manifest["frames"][fi]

    # 1. Load 3D boxes
    box_path = os.path.join(scene_dir, "bev_box", f"{fi:04d}.npz")
    boxes_3d = []
    if os.path.exists(box_path):
        d_box = np.load(box_path)
        if "boxes_3d" in d_box and len(d_box["boxes_3d"]) > 0:
            boxes_3d = d_box["boxes_3d"]

    # 2. Render 6 cameras
    cam_imgs = {}
    for cname, cal in cams_calib.items():
        rel_img = fr["imgs"].get(cname)
        if not rel_img:
            continue
        im_path = os.path.join(scene_dir, rel_img)
        img = cv2.imread(im_path)
        if img is None:
            continue

        K = np.array(cal["K"], dtype=np.float64)
        T_ego_cam = np.array(cal["T_ego_cam"], dtype=np.float64)

        for b in boxes_3d:
            cls, cx, cy, zc, l, w, h, yaw = b
            corners = box_3d_corners(cx, cy, zc, l, w, h, yaw)
            col = (0, 220, 0) if cls == 1 else (0, 220, 255)
            draw_3d_box_on_camera(img, corners, K, T_ego_cam, color=col, is_vru=(cls == 2))

        cv2.putText(img, cname, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cam_imgs[cname] = img

    row_top = np.hstack([cam_imgs[c] for c in CAMS_ORDER[0] if c in cam_imgs])
    row_bot = np.hstack([cam_imgs[c] for c in CAMS_ORDER[1] if c in cam_imgs])
    cams_grid = np.vstack([row_top, row_bot])  # shape [864, 2304, 3]

    # 3. Load and render Occupancy
    occ_path = os.path.join(scene_dir, "occ", f"{fi:04d}.npz")
    if not os.path.exists(occ_path):
        print(f"[-] Missing OCC file: {occ_path}")
        return

    occ = np.load(occ_path)["occ"]
    bev_panel = render_occ_bev(occ, target_size=cams_grid.shape[0])  # [864, 864, 3]

    # Overlay stats on BEV panel
    n_veh = int((occ == 2).sum())
    n_ped = int((occ == 4).sum())
    n_free = int((occ == 0).sum())

    cv2.putText(bev_panel, f"10-Class OCC BEV (Frame {fi:04d})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bev_panel, f"Veh Voxels: {n_veh} | Ped Voxels: {n_ped} | Free: {n_free}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw color legend at the bottom of BEV panel
    legend_y = bev_panel.shape[0] - 30
    lx = 15
    for name, col in LEGEND_ITEMS[:5]:
        cv2.rectangle(bev_panel, (lx, legend_y - 12), (lx + 14, legend_y + 2), col, -1)
        cv2.putText(bev_panel, name, (lx + 18, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        lx += 85

    legend_y2 = bev_panel.shape[0] - 10
    lx2 = 15
    for name, col in LEGEND_ITEMS[5:]:
        cv2.rectangle(bev_panel, (lx2, legend_y2 - 12), (lx2 + 14, legend_y2 + 2), col, -1)
        cv2.putText(bev_panel, name, (lx2 + 18, legend_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        lx2 += 85

    # 4. Composite final wide image
    composite = np.hstack([cams_grid, bev_panel])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, composite)
    print(f"[+] Saved composite visualization to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data_20260910_061820")
    parser.add_argument("--frames", default="100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    scene_dir = os.path.join(BASE_DIR, "scenes", args.scene)
    raw_dir = os.path.join(BASE_DIR, "raw_data", args.scene)
    manifest = json.load(open(os.path.join(scene_dir, "manifest.json")))
    cams_calib = manifest["cams"]

    frame_list = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    out_dir = args.out_dir or os.path.join(BASE_DIR, "box3d_artifacts/intermediates/occ_test_vis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Visualizing {len(frame_list)} frames for scene {args.scene} -> {out_dir}")
    for fi in frame_list:
        out_file = os.path.join(out_dir, f"frame_{fi:04d}_demo.png")
        render_frame_composite(scene_dir, raw_dir, fi, out_file, cams_calib)

    print(f"[✓] All {len(frame_list)} visualizations generated at: {out_dir}")


if __name__ == "__main__":
    main()
