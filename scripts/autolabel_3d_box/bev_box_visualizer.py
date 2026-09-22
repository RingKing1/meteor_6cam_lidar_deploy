#!/usr/bin/env python3
"""Multi-Camera 3D Bounding Box Wireframe & BEV Map Visualizer.

Projects true 3D bounding boxes (with measured height and elevation)
onto all 6 surround cameras, and renders BEV road + box raster side-by-side.

Usage:
  python3 box3d_artifacts/intermediates/bev_box_visualizer.py --scene data_20260910_063822 --frames 100,200,400,600,800,1000
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

BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2

# 12 edges connecting 8 vertices:
# Vertices 0..3: top face (FL, FR, RR, RL)
# Vertices 4..7: bottom face (FL, FR, RR, RL)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]
# Front face edges (0, 1, 5, 4) - cross to show heading
FRONT_CROSS = [(0, 5), (1, 4)]


def read_T_ego_lidar(path):
    rows = []
    in_mat = False
    for raw in open(path):
        line = raw.strip()
        if "4x4" in line and ("齐次" in line or "变换" in line):
            in_mat = True
            continue
        if in_mat:
            vals = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)
            if len(vals) == 4:
                rows.append([float(v) for v in vals])
            if len(rows) == 4:
                break
    return np.array(rows, np.float64)


def read_pcd_xyz(path):
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            head += line
            if line.startswith(b"DATA"):
                break
        m = re.search(rb"POINTS (\d+)", head)
        if not m:
            return np.zeros((0, 3), np.float64)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)[:, :3].astype(np.float64)


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

    # 12 wireframe edges
    for p1, p2 in EDGES:
        if z[p1] > 0.5 and z[p2] > 0.5:
            pt1 = tuple(pts_2d[p1])
            pt2 = tuple(pts_2d[p2])
            if (-100 < pt1[0] < w + 100 and -100 < pt1[1] < h + 100) or \
               (-100 < pt2[0] < w + 100 and -100 < pt2[1] < h + 100):
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # Front face indicator (cross on front face)
    front_col = (255, 255, 0) if not is_vru else (0, 200, 255)
    for p1, p2 in FRONT_CROSS:
        if z[p1] > 0.5 and z[p2] > 0.5:
            cv2.line(img, tuple(pts_2d[p1]), tuple(pts_2d[p2]), front_col, 1, cv2.LINE_AA)

    # 2D bounding box envelope
    vis = z > 0.5
    if vis.sum() >= 4:
        u_min = int(np.clip(u[vis].min(), 0, w - 1))
        u_max = int(np.clip(u[vis].max(), 0, w - 1))
        v_min = int(np.clip(v[vis].min(), 0, h - 1))
        v_max = int(np.clip(v[vis].max(), 0, h - 1))
        if u_max - u_min > 5 and v_max - v_min > 5:
            cv2.rectangle(img, (u_min, v_min), (u_max, v_max), front_col, 1, cv2.LINE_AA)

    return img


def render_frame(scene_dir, raw_dir, fi, out_path, cams_calib, R_el, t_el, pcd_files, bev_box_dir=None):
    fr = json.load(open(os.path.join(scene_dir, "manifest.json")))["frames"][fi]

    box_folder = bev_box_dir if bev_box_dir else os.path.join(scene_dir, "bev_box")
    npz_path = os.path.join(box_folder, f"{fi:04d}.npz")
    if not os.path.exists(npz_path):
        print(f"[-] Missing {npz_path}")
        return

    npz_data = np.load(npz_path)
    if "boxes_3d" in npz_data and len(npz_data["boxes_3d"]) > 0:
        boxes_3d = npz_data["boxes_3d"]
    else:
        # Reconstruct true zc and h from point cloud
        pts_l = read_pcd_xyz(os.path.join(raw_dir, f"lidar/{pcd_files[fi]}"))
        pts_e = pts_l @ R_el.T + t_el
        raw_boxes = npz_data["boxes"]
        boxes_3d = []
        for b in raw_boxes:
            cls, cx, cy, l, w, yaw = b
            cb, sb = np.cos(-yaw), np.sin(-yaw)
            dx = pts_e[:, 0] - cx
            dy = pts_e[:, 1] - cy
            bx = cb * dx - sb * dy
            by = sb * dx + cb * dy
            in_box = (np.abs(bx) <= l / 2 + 0.3) & (np.abs(by) <= w / 2 + 0.3) & (pts_e[:, 2] >= -0.5) & (pts_e[:, 2] <= 4.0)
            c_pts = pts_e[in_box]
            if len(c_pts) >= 5:
                zmin = float(np.percentile(c_pts[:, 2], 2))
                zmax = float(np.percentile(c_pts[:, 2], 98))
                zc = (zmin + zmax) / 2.0
                h = max(zmax - zmin, 1.2)
            else:
                zc = 1.0
                h = 1.6
            boxes_3d.append([cls, cx, cy, zc, l, w, h, yaw])
        boxes_3d = np.array(boxes_3d, np.float32)

    # Render on 6 cameras
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
            col = (0, 255, 0) if cls == 1 else (0, 220, 255)
            draw_3d_box_on_camera(img, corners, K, T_ego_cam, color=col, is_vru=(cls == 2))

        # Title overlay
        cv2.putText(img, cname, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cam_imgs[cname] = img

    row_top = np.hstack([cam_imgs[c] for c in CAMS_ORDER[0] if c in cam_imgs])
    row_bot = np.hstack([cam_imgs[c] for c in CAMS_ORDER[1] if c in cam_imgs])
    cams_grid = np.vstack([row_top, row_bot])

    # BEV panel
    bev_box_png = cv2.imread(os.path.join(box_folder, f"{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
    gt_png = cv2.imread(os.path.join(scene_dir, f"gt/{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)

    bev_color = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev_color[:] = (20, 20, 24)  # dark background
    if gt_png is not None:
        bev_color[gt_png == 1] = (65, 65, 70)       # Road drivable (dark slate)
        bev_color[gt_png == 2] = (140, 80, 140)     # Sidewalk (pink/purple)
        bev_color[gt_png == 3] = (0, 230, 255)      # Crosswalk (bright yellow)
        bev_color[gt_png == 4] = (255, 255, 255)    # Laneline (pure white)
        bev_color[gt_png == 5] = (40, 40, 240)      # Stopline (red)
        bev_color[gt_png == 6] = (0, 140, 255)      # Road edge (orange)
        bev_color[gt_png == 7] = (220, 210, 50)     # Markings (cyan)
        bev_color[gt_png == 8] = (160, 90, 40)      # Parking (blue)

    if bev_box_png is not None:
        bev_color[bev_box_png == 1] = (0, 220, 0)    # Vehicle (lime green)
        bev_color[bev_box_png == 2] = (0, 220, 255)  # VRU (cyan)

    # Ego vehicle
    ego_col, ego_row = int(BEV_YH / RES), int(BEV_XH / RES)
    cv2.rectangle(bev_color, (ego_col - 5, ego_row - 10), (ego_col + 5, ego_row + 10), (0, 0, 255), -1)
    cv2.putText(bev_color, "Ego", (ego_col - 12, ego_row + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    bev_panel = cv2.resize(bev_color, (int(cams_grid.shape[0] * (BEV_W / BEV_H)), cams_grid.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.putText(bev_panel, f"BEV GT (Frame {fi})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bev_panel, f"Objects: {len(boxes_3d)} confirmed", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

    # Add legend at bottom
    legend_items = [
        ("Road", (65, 65, 70)),
        ("Lane", (255, 255, 255)),
        ("Cross", (0, 230, 255)),
        ("Walk", (140, 80, 140)),
        ("Veh", (0, 220, 0)),
        ("VRU", (0, 220, 255)),
    ]
    lx = 15
    ly = bev_panel.shape[0] - 20
    for name, col in legend_items:
        cv2.rectangle(bev_panel, (lx, ly - 12), (lx + 14, ly + 2), col, -1)
        cv2.putText(bev_panel, name, (lx + 18, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        lx += 86

    canvas = np.hstack([cams_grid, bev_panel])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"[+] Saved Frame {fi:04d} visualization to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data_20260910_061820")
    ap.add_argument("--frames", default="100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bev-box-dir", default=None)
    args = ap.parse_args()

    scene_dir = os.path.join(BASE_DIR, "scenes", args.scene)
    raw_dir = os.path.join(BASE_DIR, "raw_data", args.scene)
    man = json.load(open(os.path.join(scene_dir, "manifest.json")))
    cams_calib = man["cams"]

    T_el = read_T_ego_lidar(os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt"))
    R_el, t_el = T_el[:3, :3], T_el[:3, 3]
    pcd_files = sorted(f for f in os.listdir(os.path.join(raw_dir, "lidar")) if f.endswith(".pcd"))

    frame_list = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(BASE_DIR, "box3d_artifacts/intermediates/multi_frame_demos")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Generating 3D Box visualizations for {len(frame_list)} frames in {args.scene} ...")
    print(f"[*] Destination: {out_dir}")
    for fi in frame_list:
        out_file = os.path.join(out_dir, f"frame_{fi:04d}_demo.png")
        render_frame(scene_dir, raw_dir, fi, out_file, cams_calib, R_el, t_el, pcd_files, bev_box_dir=args.bev_box_dir)

    print(f"[+] All {len(frame_list)} multi-frame visualizations successfully generated at: {out_dir}")


if __name__ == "__main__":
    main()
