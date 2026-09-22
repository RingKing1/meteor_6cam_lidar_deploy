#!/usr/bin/env python3
"""Multi-Modal Visualizer for 2D BBoxes, Purified 3D BBoxes, and Future 3.0s Agent Trajectories.

Renders high-resolution composite dashboard:
  - 6 Surround Cameras: 2D detection boxes (with track ID & class name) + 3D wireframe projection
  - BEV Radar/Map Panel: Semantic road layout + 3D oriented bounding boxes
  - Future 3.0s Trajectory Polyline:
      * Moving agents (disp >= 0.3m): Bright Red polyline with future waypoint dots & heading arrows
      * Stationary agents (disp < 0.3m): Cyan/Blue marker showing parked/stopped state
"""
import argparse
import json
import os
import cv2
import numpy as np

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
    1: (0, 255, 128),   # car: bright spring green
    2: (0, 200, 255),   # truck: yellow-orange
    3: (255, 128, 0),   # bus: cyan
    4: (255, 255, 0),   # bicycle: cyan-blue
    5: (180, 100, 255), # motorcycle: purple
    6: (0, 100, 255),   # pedestrian: orange-red
}

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def draw_3d_wireframe(img, corners_3d, K, T_ego_cam, color=(0, 255, 0)):
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
    return img


def render_dashboard(scene_dir, fi, out_path, cams_calib):
    manifest_p = os.path.join(scene_dir, "manifest.json")
    with open(manifest_p) as f:
        manifest = json.load(f)
    fr = manifest["frames"][fi]

    # 1. Load 2D bounding boxes
    b2d_p = os.path.join(scene_dir, "bbox2d", f"{fi:04d}.npz")
    boxes_2d = None
    counts_2d = None
    if os.path.exists(b2d_p):
        z2d = np.load(b2d_p)
        boxes_2d = z2d["boxes"]
        counts_2d = z2d["counts"]

    # 2. Load 3D bounding boxes
    bev_p = os.path.join(scene_dir, "bev_box", f"{fi:04d}.npz")
    boxes_3d = []
    if os.path.exists(bev_p):
        zbev = np.load(bev_p)
        if "boxes_3d" in zbev:
            boxes_3d = zbev["boxes_3d"]

    # 3. Load Agent Trajectories
    traj_p = os.path.join(scene_dir, "agent_traj", f"{fi:04d}.npz")
    traj_boxes, trajs, tvalids, n_agents = None, None, None, 0
    if os.path.exists(traj_p):
        ztraj = np.load(traj_p)
        traj_boxes = ztraj["boxes"]
        trajs = ztraj["traj"]
        tvalids = ztraj["tvalid"]
        n_agents = int(ztraj["count"])

    # 4. Render 6 Camera Views
    cam_imgs = {}
    for ci, cname in enumerate(CAMS_ORDER):
        rel_img = fr["imgs"].get(cname)
        if not rel_img:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            p = os.path.join(scene_dir, rel_img)
            img = cv2.imread(p) if os.path.exists(p) else np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        cal = cams_calib.get(cname)
        if cal is not None:
            K = np.array(cal["K"], dtype=np.float64)
            T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)

            # Draw 3D wireframes
            for b in boxes_3d:
                cls, cx, cy, zc, l, w, h, yaw = b
                cors = box_3d_corners(cx, cy, zc, l, w, h, yaw)
                col = (0, 255, 0) if cls == 1 else (0, 220, 255)
                draw_3d_wireframe(img, cors, K, T_ec, color=col)

        # Draw 2D Detection Boxes
        if boxes_2d is not None and counts_2d is not None and ci < len(counts_2d):
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
                cv2.putText(img, label_txt, (x1 + 3, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Camera name banner
        cv2.rectangle(img, (10, 10), (250, 42), (0, 0, 0), -1)
        cv2.putText(img, cname, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cam_imgs[cname] = img

    row_top = np.hstack([cam_imgs[c] for c in CAMS_LAYOUT[0]])
    row_bot = np.hstack([cam_imgs[c] for c in CAMS_LAYOUT[1]])
    cams_grid = np.vstack([row_top, row_bot])

    # 5. Render BEV Panel with Map + Boxes + 3.0s Trajectory Polylines
    gt_png = cv2.imread(os.path.join(scene_dir, f"gt/{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
    bev_color = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev_color[:] = (22, 22, 26)  # sleek dark slate

    if gt_png is not None:
        bev_color[gt_png == 1] = (65, 65, 70)       # Road drivable (dark slate)
        bev_color[gt_png == 2] = (140, 80, 140)     # Sidewalk (purple)
        bev_color[gt_png == 3] = (0, 230, 255)      # Crosswalk (yellow)
        bev_color[gt_png == 4] = (255, 255, 255)    # Laneline (white)
        bev_color[gt_png == 5] = (40, 40, 240)      # Stopline (red)
        bev_color[gt_png == 6] = (0, 140, 255)      # Road edge (orange)
        bev_color[gt_png == 7] = (220, 210, 50)     # Markings (cyan)

    def to_bev_px(x, y):
        r = int(round((BEV_XH - x) / RES))
        c = int(round((BEV_YH - y) / RES))
        return (c, r)

    # Render 3D Boxes on BEV
    for b in boxes_3d:
        cls, cx, cy, zc, l, w, h, yaw = b
        cb, sb = np.cos(yaw), np.sin(yaw)
        cors = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
            px = cx + lx * cb - wy * sb
            py = cy + lx * sb + wy * cb
            cors.append(to_bev_px(px, py))
        col = (0, 220, 0) if cls == 1 else (0, 220, 255)
        cv2.fillPoly(bev_color, [np.array(cors, dtype=np.int32)], col)
        cv2.polylines(bev_color, [np.array(cors, dtype=np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)

    # Render Agent Trajectories (Future 3.0s, 6 Steps)
    n_moving, n_stationary = 0, 0
    if traj_boxes is not None and n_agents > 0:
        for k in range(n_agents):
            cls, cx, cy, l, w, yaw = traj_boxes[k]
            agent_px = to_bev_px(cx, cy)
            valid_steps = tvalids[k]
            deltas = trajs[k]

            # Determine stationary vs moving
            max_disp = 0.0
            for h in range(6):
                if valid_steps[h] > 0.5:
                    dx, dy = deltas[h]
                    disp = np.sqrt(dx * dx + dy * dy)
                    max_disp = max(max_disp, disp)

            is_stat = (max_disp < 0.3)
            if is_stat:
                n_stationary += 1
                # Stationary marker: Cyan diamond/circle
                cv2.circle(bev_color, agent_px, 4, (255, 200, 0), -1, cv2.LINE_AA)
            else:
                n_moving += 1
                # Moving trajectory: Red polyline connecting waypoints
                pts_traj = [agent_px]
                for h in range(6):
                    if valid_steps[h] > 0.5:
                        dx, dy = deltas[h]
                        fut_x = cx + dx
                        fut_y = cy + dy
                        fut_px = to_bev_px(fut_x, fut_y)
                        pts_traj.append(fut_px)
                        # Waypoint dot
                        cv2.circle(bev_color, fut_px, 2 + h // 2, (0, 100, 255), -1, cv2.LINE_AA)

                if len(pts_traj) > 1:
                    cv2.polylines(bev_color, [np.array(pts_traj, dtype=np.int32)], False, (0, 50, 255), 2, cv2.LINE_AA)
                    # Heading arrow on last segment
                    cv2.arrowedLine(bev_color, pts_traj[-2], pts_traj[-1], (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.4)

    # Ego vehicle
    ego_px = to_bev_px(0.0, 0.0)
    cv2.rectangle(bev_color, (ego_px[0] - 5, ego_px[1] - 10), (ego_px[0] + 5, ego_px[1] + 10), (0, 0, 255), -1)
    cv2.putText(bev_color, "Ego", (ego_px[0] - 12, ego_px[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    # Resize BEV panel to match camera grid height
    bev_panel_w = int(cams_grid.shape[0] * (BEV_W / BEV_H))
    bev_panel = cv2.resize(bev_color, (bev_panel_w, cams_grid.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Legend & Status Overlay
    cv2.rectangle(bev_panel, (15, 15), (bev_panel_w - 15, 135), (0, 0, 0), -1)
    cv2.putText(bev_panel, f"BEV Trajectory & GT (F#{fi})", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(bev_panel, f"Objects: {len(boxes_3d)} confirmed", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1)
    cv2.putText(bev_panel, f"Moving Trajectories: {n_moving} (Red)", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 50, 255), 1)
    cv2.putText(bev_panel, f"Stationary Agents: {n_stationary} (Blue)", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 1)

    # Combine side-by-side
    combined = np.hstack([cams_grid, bev_panel])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/meteor_6cam_lidar_deploy/scenes")
    parser.add_argument("--scene", default="data_20260910_061820")
    parser.add_argument("--frames", default="0,10,20,50,100,200", help="comma-separated frame indices")
    parser.add_argument("--out-dir", default="/work/meteor_6cam_lidar_deploy/box3d_artifacts/visual_inspection")
    args = parser.parse_args()

    scene_dir = os.path.join(args.root, args.scene)
    manifest_p = os.path.join(scene_dir, "manifest.json")
    with open(manifest_p) as f:
        manifest = json.load(f)
    cams_calib = manifest["cams"]

    frame_indices = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    print(f"[*] Rendering dashboard for scene {args.scene} frames: {frame_indices}")

    for fi in frame_indices:
        out_p = os.path.join(args.out_dir, f"{args.scene}_frame_{fi:04d}_dashboard.jpg")
        render_dashboard(scene_dir, fi, out_p, cams_calib)
        print(f"  [+] Saved {out_p}")


if __name__ == "__main__":
    main()
