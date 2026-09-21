#!/usr/bin/env python3
"""Comparative Benchmark: FocalFormer3D-LC vs CenterPoint on 10 Random Frames.

Features:
  1. 5-Sweep Pose Motion-Compensated LiDAR Point Clouds (ENU global trajectory).
  2. 6 Surround-Camera High-Resolution Synchronized Images.
  3. Model 1: FocalFormer3D-LC (FocalFormer3D_LC.pth) - Multimodal LiDAR + Camera Cross-Attention.
  4. Model 2: CenterPoint (rpn_centerhead_sim.onnx) - LiDAR 3D CenterHead.
  5. 10 Random Frames Comparison with Statistical Metrics & Side-by-Side Visualizations.

Usage (inside docker):
  python3 /work/meteor_6cam_lidar_deploy/box3d_artifacts/intermediates/compare_focalformer_centerpoint.py
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
import projects.mmdet3d_plugin

# Class mapping in nuScenes
NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
VEHICLE_LABEL_IDS = [0, 1, 2, 3, 4]   # car, truck, const, bus, trailer
VRU_LABEL_IDS = [6, 7, 8]             # motorcycle, bicycle, pedestrian

CAMS_ORDER = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# 12 wireframe edges for 8 corners
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]
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
            return np.zeros((0, 4), np.float32)
        n = int(m.group(1))
        data = np.fromfile(f, dtype=np.float32, count=n * 4)
    return data.reshape(n, 4)


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def decode_centerpoint_task_heads(outs, score_thresh=0.25):
    """Decodes 6 task outputs of rpn_centerhead_sim.onnx into 3D bounding boxes."""
    # outs has 36 outputs: 6 tasks * (reg, height, dim, rot, vel, hm)
    # pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], voxel_size = 0.075, out_factor = 8
    voxel_x = 0.075 * 8.0  # 0.6m
    pc_min_x = -54.0
    pc_min_y = -54.0

    task_classes = [
        [0],        # car
        [1, 2],     # truck, construction_vehicle
        [3, 4],     # bus, trailer
        [5],        # barrier
        [6, 7],     # motorcycle, bicycle
        [8, 9],     # pedestrian, traffic_cone
    ]

    all_boxes = []
    all_scores = []
    all_labels = []

    H, W = 180, 180
    ys, xs = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')

    for task_id in range(6):
        base_idx = task_id * 6
        reg = outs[base_idx][0]       # (2, 180, 180)
        height = outs[base_idx + 1][0]  # (1, 180, 180)
        dim = outs[base_idx + 2][0]     # (3, 180, 180)
        rot = outs[base_idx + 3][0]     # (2, 180, 180)
        vel = outs[base_idx + 4][0]     # (2, 180, 180)
        hm = outs[base_idx + 5][0]      # (num_cls, 180, 180)

        # Sigmoid on heatmap
        hm = 1.0 / (1.0 + np.exp(-np.clip(hm, -20.0, 20.0)))
        # Local peak suppression (CenterPoint 3x3 max-pool NMS)
        hm_t = torch.from_numpy(hm).unsqueeze(0)
        hmax = torch.nn.functional.max_pool2d(hm_t, kernel_size=3, stride=1, padding=1)
        keep_peak = (hmax == hm_t).squeeze(0).numpy()
        hm = hm * keep_peak

        # Extract detections for this task
        num_cls = hm.shape[0]
        for c in range(num_cls):
            global_cls = task_classes[task_id][c]
            c_hm = hm[c]
            mask = c_hm > score_thresh
            if not np.any(mask):
                continue
            
            c_scores = c_hm[mask]
            grid_y, grid_x = ys[mask], xs[mask]

            r_x = reg[0][mask]
            r_y = reg[1][mask]
            h_z = height[0][mask]
            
            d_x = np.exp(dim[0][mask])
            d_y = np.exp(dim[1][mask])
            d_z = np.exp(dim[2][mask])

            sin_r = rot[0][mask]
            cos_r = rot[1][mask]
            yaw = np.arctan2(sin_r, cos_r)

            pos_x = (grid_x + r_x) * voxel_x + pc_min_x
            pos_y = (grid_y + r_y) * voxel_x + pc_min_y

            for i in range(len(c_scores)):
                all_boxes.append([pos_x[i], pos_y[i], h_z[i], d_x[i], d_y[i], d_z[i], yaw[i]])
                all_scores.append(float(c_scores[i]))
                all_labels.append(global_cls)

    if len(all_boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Simple Non-Maximum Suppression (BEV distance-based)
    keep_indices = []
    order = np.argsort(-scores)
    while len(order) > 0:
        idx = order[0]
        keep_indices.append(idx)
        if len(order) == 1:
            break
        other = order[1:]
        dx = boxes[other, 0] - boxes[idx, 0]
        dy = boxes[other, 1] - boxes[idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        same_cls = (labels[other] == labels[idx])
        suppress = (dist < 1.5) & same_cls
        order = other[~suppress]

    keep = np.array(keep_indices, dtype=np.int64)
    return boxes[keep], scores[keep], labels[keep]


def draw_wireframe_on_image(img, corners_3d, K, T_ego_cam, color=(0, 255, 255), label_str=""):
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
    pts = np.stack([u, v], axis=1).astype(np.int32)

    for p1, p2 in EDGES:
        if z[p1] > 0.5 and z[p2] > 0.5:
            cv2.line(img, tuple(pts[p1]), tuple(pts[p2]), color, 2, cv2.LINE_AA)
    for p1, p2 in FRONT_CROSS:
        if z[p1] > 0.5 and z[p2] > 0.5:
            cv2.line(img, tuple(pts[p1]), tuple(pts[p2]), color, 1, cv2.LINE_AA)

    top_center = pts[0:4].mean(axis=0).astype(int)
    if label_str and z.mean() > 0.5:
        cv2.putText(img, label_str, (top_center[0] - 15, max(15, top_center[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_bev_canvas(boxes, labels, scores, pts_fused=None, title="", width=500, height=800):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 24)

    # Grid lines every 10m (res = 0.2m -> 50px)
    for r in range(0, height, 50):
        cv2.line(canvas, (0, r), (width, r), (40, 40, 45), 1)
    for c in range(0, width, 50):
        cv2.line(canvas, (c, 0), (c, height), (40, 40, 45), 1)

    # Ego vehicle center at (x=250, y=700) (pointing UP)
    # BEV coord: X in [-50, 50] (width), Y in [0, 100] (length front)
    ego_col, ego_row = width // 2, int(height * 0.85)
    cv2.rectangle(canvas, (ego_col - 5, ego_row - 12), (ego_col + 5, ego_row + 12), (0, 140, 255), -1)

    # Plot fused points in background
    if pts_fused is not None and len(pts_fused) > 0:
        step = max(1, len(pts_fused) // 15000)
        sub_pts = pts_fused[::step]
        px = sub_pts[:, 0]  # Ego X (right) -> canvas X
        py = sub_pts[:, 1]  # Ego Y (front) -> canvas Y (upwards)
        col = (width // 2 + px / 0.2).astype(int)
        row = (ego_row - py / 0.2).astype(int)
        valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        canvas[row[valid], col[valid]] = (90, 90, 95)

    # Plot detected boxes
    for i in range(len(boxes)):
        b = boxes[i]
        cx, cy, cz, l, w, h, yaw = b[:7]
        cls_id = labels[i]
        score = scores[i]
        is_veh = cls_id in VEHICLE_LABEL_IDS

        color = (0, 255, 120) if is_veh else (255, 180, 0)
        # 4 corners in BEV
        cb, sb = np.cos(yaw), np.sin(yaw)
        cors = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
            bx = cx + lx * cb - wy * sb
            by = cy + lx * sb + wy * cb
            c_col = int(width // 2 + bx / 0.2)
            c_row = int(ego_row - by / 0.2)
            cors.append([c_col, c_row])
        cors = np.array(cors, dtype=np.int32)
        cv2.polylines(canvas, [cors], True, color, 2, cv2.LINE_AA)

        # Arrow indicating heading
        front_mid = ((cors[0] + cors[1]) / 2.0).astype(int)
        center_pt = ((cors[0] + cors[2]) / 2.0).astype(int)
        cv2.arrowedLine(canvas, tuple(center_pt), tuple(front_mid), (0, 255, 255), 1, tipLength=0.3)

    cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Detections: {len(boxes)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data_20260910_061820")
    parser.add_argument("--frames", default="120,240,360,480,600,720,840,960,1080,1200")
    parser.add_argument("--n-sweeps", type=int, default=5)
    parser.add_argument("--out-dir", default="/work/meteor_6cam_lidar_deploy/box3d_artifacts/intermediates/comparison_10frames")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_indices = [int(f.strip()) for f in args.frames.split(",")]

    print("================================================================================")
    print("  Comparing FocalFormer3D-LC vs CenterPoint on 10 Random Frames (5-Sweep Fusion)")
    print("================================================================================")
    print(f"Scene:        {args.scene}")
    print(f"Frames (10):  {frame_indices}")
    print(f"Sweeps:       {args.n_sweeps}")
    print(f"Output Dir:   {args.out_dir}")
    print("================================================================================")

    # 1. Load calibration & manifest
    scene_dir = f"/work/meteor_6cam_lidar_deploy/scenes/{args.scene}"
    raw_dir = f"/work/meteor_6cam_lidar_deploy/raw_data/{args.scene}"

    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    cams_calib = manifest["cams"]

    calib_lidar_path = os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt")
    T_el = read_T_ego_lidar(calib_lidar_path)
    R_el, t_el = T_el[:3, :3], T_el[:3, 3]

    ego_motion_path = os.path.join(scene_dir, "ego_motion.npz")
    ego_poses = np.load(ego_motion_path)["pose"]

    lidar_dir = os.path.join(raw_dir, "lidar")
    pcd_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd"))

    # 2. Build Models
    print("\n[1/3] Loading FocalFormer3D-LC (Multimodal)...")
    cfg_focal = Config.fromfile('/work/FocalFormer3D/projects/configs/focalformer3d/FocalFormer3D_LC.py')
    cfg_focal.model.pretrained = None
    focal_model = build_model(cfg_focal.model, test_cfg=cfg_focal.get('test_cfg')).cuda().eval()
    load_checkpoint(focal_model, '/work/meteor_6cam_lidar_deploy/box3d_artifacts/models/FocalFormer3D_LC.pth', map_location='cpu')
    print("  [+] FocalFormer3D-LC loaded successfully.")

    print("\n[2/3] Loading CenterPoint ONNX RPN Engine...")
    cp_sess = ort.InferenceSession(
        '/work/meteor_6cam_lidar_deploy/box3d_artifacts/models/rpn_centerhead_sim.onnx',
        providers=['CUDAExecutionProvider']
    )
    print("  [+] CenterPoint ONNX Engine loaded successfully.")

    # Compute lidar2img matrices (6, 4, 4)
    # Camera scale for (800, 448) input
    lidar2img_list = []
    for cname in CAMS_ORDER:
        cal = cams_calib[cname]
        K = np.array(cal["K"], dtype=np.float64).copy()
        # Scale K from 768x432 to 800x448
        K[0, 0] *= (800.0 / 768.0)
        K[0, 2] *= (800.0 / 768.0)
        K[1, 1] *= (448.0 / 432.0)
        K[1, 2] *= (448.0 / 432.0)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec = T_ec[:3, :3]
        t_ec = T_ec[:3, 3]

        # T_ego_to_cam: P_cam = (P_ego - t_ec) @ R_ec
        # = P_ego @ R_ec - t_ec @ R_ec
        mat4 = np.eye(4, dtype=np.float64)
        mat4[:3, :3] = R_ec.T
        mat4[:3, 3] = -t_ec @ R_ec
        proj = np.eye(4, dtype=np.float64)
        proj[:3, :3] = K
        lidar2img_mat = proj @ mat4
        lidar2img_list.append(lidar2img_mat)
    lidar2img_arr = np.stack(lidar2img_list, axis=0).astype(np.float32)

    # 3. Process 10 Frames
    comparison_stats = []
    print("\n[3/3] Running Inference on 10 Random Frames...")

    for frame_idx in frame_indices:
        print(f"\n---> Frame {frame_idx:04d} ...")
        t_frame_start = time.time()

        # (a) Accumulate 5 sweeps
        curr_pose = ego_poses[frame_idx]
        x0, y0, yaw0 = curr_pose[0], curr_pose[1], curr_pose[2]
        c0, s0 = np.cos(yaw0), np.sin(yaw0)
        R_we0 = np.array([[c0, -s0], [s0, c0]], dtype=np.float64)

        sweeps_pts = []
        for s_off in range(args.n_sweeps):
            f_i = max(0, frame_idx - s_off)
            pcd_path = os.path.join(lidar_dir, pcd_files[f_i])
            raw_pts = read_pcd_xyz(pcd_path)
            if len(raw_pts) == 0:
                continue
            # Sensor to Ego
            p_ego = raw_pts[:, :3] @ R_el.T + t_el
            intensity = raw_pts[:, 3:4]
            if s_off > 0:
                p_i = ego_poses[f_i]
                xi, yi, yawi = p_i[0], p_i[1], p_i[2]
                ci, si = np.cos(yawi), np.sin(yawi)
                R_wei = np.array([[ci, -si], [si, ci]], dtype=np.float64)
                # To world ENU then to current ego
                p_w = p_ego[:, :2] @ R_wei.T + np.array([xi, yi])
                p_ego_curr = (p_w - np.array([x0, y0])) @ R_we0
                p_ego[:, :2] = p_ego_curr

            time_lag = float(-s_off * 0.1)
            sweep_feat = np.hstack([p_ego, intensity, np.full((len(p_ego), 1), time_lag, dtype=np.float32)])
            sweeps_pts.append(sweep_feat)

        fused_pts = np.vstack(sweeps_pts).astype(np.float32)
        # Filter range [-54..54, -54..54, -5..3]
        mask_range = (
            (fused_pts[:, 0] >= -54.0) & (fused_pts[:, 0] <= 54.0) &
            (fused_pts[:, 1] >= -54.0) & (fused_pts[:, 1] <= 54.0) &
            (fused_pts[:, 2] >= -5.0) & (fused_pts[:, 2] <= 3.0)
        )
        fused_pts = fused_pts[mask_range]
        print(f"  Fused points: {len(fused_pts):,} across 5 sweeps")

        # (b) Load 6 camera images
        imgs_list = []
        orig_front_img = None
        for ci, cname in enumerate(CAMS_ORDER):
            img_path = os.path.join(scene_dir, "img", cname, f"{frame_idx:04d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(scene_dir, "img", cname.lower(), f"{frame_idx:04d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(scene_dir, cname.lower(), f"{frame_idx:04d}.jpg")

            if not os.path.exists(img_path):
                print(f"  [!] Warning: image not found for {cname} at {img_path}")
                img_bgr = np.zeros((432, 768, 3), dtype=np.uint8)
            else:
                img_bgr = cv2.imread(img_path)
            if cname == "CAM_FRONT_WIDE":
                orig_front_img = img_bgr.copy()

            # Resize to 800x448
            img_resized = cv2.resize(img_bgr, (800, 448))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Normalize
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img_norm = (img_rgb - mean) / std
            img_chw = img_norm.transpose(2, 0, 1)  # (3, 448, 800)
            imgs_list.append(img_chw)

        imgs_tensor = torch.from_numpy(np.stack(imgs_list, axis=0)).unsqueeze(0).cuda().float()  # (1, 6, 3, 448, 800)
        pts_tensor = torch.from_numpy(fused_pts).cuda().float()

        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        img_metas = [{
            "lidar2img": lidar2img_arr,
            "box_type_3d": LiDARInstance3DBoxes,
            "img_shape": [(448, 800, 3)] * 6,
        }]

        # (c) Model 1: FocalFormer3D-LC Inference
        torch.cuda.synchronize()
        t_focal_0 = time.time()
        with torch.no_grad():
            voxels, num_points, coors = focal_model.voxelize([pts_tensor], voxel_type='voxel')
            voxel_features = focal_model.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            x_middle = focal_model.pts_middle_encoder(voxel_features, coors, batch_size)
            focal_res = focal_model.simple_test([pts_tensor], img_metas, img=imgs_tensor, rescale=False)
        torch.cuda.synchronize()
        focal_latency = (time.time() - t_focal_0) * 1000.0

        focal_bboxes = focal_res[0]["pts_bbox"]["boxes_3d"].tensor.cpu().numpy()
        focal_scores = focal_res[0]["pts_bbox"]["scores_3d"].cpu().numpy()
        focal_labels = focal_res[0]["pts_bbox"]["labels_3d"].cpu().numpy()

        focal_mask = focal_scores > 0.25
        focal_b = focal_bboxes[focal_mask]
        focal_s = focal_scores[focal_mask]
        focal_l = focal_labels[focal_mask]

        # (d) Model 2: CenterPoint ONNX Inference
        torch.cuda.synchronize()
        t_cp_0 = time.time()
        bev_input = x_middle.detach().cpu().numpy()  # (1, 256, 180, 180)
        cp_outs = cp_sess.run(None, {"input": bev_input})
        cp_b, cp_s, cp_l = decode_centerpoint_task_heads(cp_outs, score_thresh=0.25)
        torch.cuda.synchronize()
        cp_latency = (time.time() - t_cp_0) * 1000.0

        # Stats calculation
        focal_veh_cnt = int(sum(1 for l in focal_l if l in VEHICLE_LABEL_IDS))
        focal_vru_cnt = int(sum(1 for l in focal_l if l in VRU_LABEL_IDS))
        cp_veh_cnt = int(sum(1 for l in cp_l if l in VEHICLE_LABEL_IDS))
        cp_vru_cnt = int(sum(1 for l in cp_l if l in VRU_LABEL_IDS))

        focal_mean_score = float(np.mean(focal_s[:10])) if len(focal_s) > 0 else 0.0
        cp_mean_score = float(np.mean(cp_s[:10])) if len(cp_s) > 0 else 0.0

        print(f"  [FocalFormer3D-LC] Latency: {focal_latency:.1f}ms | Veh: {focal_veh_cnt}, VRU: {focal_vru_cnt}, Mean Score: {focal_mean_score:.2f}")
        print(f"  [CenterPoint]     Latency: {cp_latency:.1f}ms | Veh: {cp_veh_cnt}, VRU: {cp_vru_cnt}, Mean Score: {cp_mean_score:.2f}")

        stat_record = {
            "frame": frame_idx,
            "focalformer": {
                "latency_ms": round(focal_latency, 1),
                "total_boxes": len(focal_b),
                "vehicles": focal_veh_cnt,
                "vrus": focal_vru_cnt,
                "top10_mean_score": round(focal_mean_score, 3),
            },
            "centerpoint": {
                "latency_ms": round(cp_latency, 1),
                "total_boxes": len(cp_b),
                "vehicles": cp_veh_cnt,
                "vrus": cp_vru_cnt,
                "top10_mean_score": round(cp_mean_score, 3),
            }
        }
        comparison_stats.append(stat_record)

        # (e) Render Side-by-Side Comparison Visualization
        # Top half: FocalFormer3D-LC (Cam Front + BEV)
        # Bottom half: CenterPoint (Cam Front + BEV)
        focal_cam_img = orig_front_img.copy()
        cp_cam_img = orig_front_img.copy()

        cal_front = cams_calib["CAM_FRONT_WIDE"]
        K_f = np.array(cal_front["K"], dtype=np.float64)
        T_ec_f = np.array(cal_front["T_ego_cam"], dtype=np.float64)

        # Draw FocalFormer boxes
        for i in range(len(focal_b)):
            b = focal_b[i]
            l_id = focal_l[i]
            s = focal_s[i]
            cx, cy, cz, l, w, h, yaw = b[:7]
            corners = box_3d_corners(cx, cy, cz, l, w, h, yaw)
            is_veh = l_id in VEHICLE_LABEL_IDS
            color = (0, 255, 120) if is_veh else (255, 200, 0)
            name = NUSC_CLASSES[l_id] if l_id < len(NUSC_CLASSES) else "obj"
            draw_wireframe_on_image(focal_cam_img, corners, K_f, T_ec_f, color, f"{name}:{s:.2f}")

        # Draw CenterPoint boxes
        for i in range(len(cp_b)):
            b = cp_b[i]
            l_id = cp_l[i]
            s = cp_s[i]
            cx, cy, cz, l, w, h, yaw = b[:7]
            corners = box_3d_corners(cx, cy, cz, l, w, h, yaw)
            is_veh = l_id in VEHICLE_LABEL_IDS
            color = (0, 220, 255) if is_veh else (255, 120, 0)
            name = NUSC_CLASSES[l_id] if l_id < len(NUSC_CLASSES) else "obj"
            draw_wireframe_on_image(cp_cam_img, corners, K_f, T_ec_f, color, f"{name}:{s:.2f}")

        # BEV maps
        bev_focal = render_bev_canvas(focal_b, focal_l, focal_s, fused_pts, f"FocalFormer3D-LC (Multimodal)", width=500, height=432)
        bev_cp = render_bev_canvas(cp_b, cp_l, cp_s, fused_pts, f"CenterPoint (LiDAR-Only)", width=500, height=432)

        # Overlay model title tags on cameras
        cv2.putText(focal_cam_img, f"FocalFormer3D-LC [LiDAR 5-Sweep + 6 Cams] (Veh:{focal_veh_cnt}, VRU:{focal_vru_cnt}, Latency:{focal_latency:.1f}ms)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(cp_cam_img, f"CenterPoint [LiDAR 5-Sweep] (Veh:{cp_veh_cnt}, VRU:{cp_vru_cnt}, Latency:{cp_latency:.1f}ms)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

        row_focal = np.hstack([focal_cam_img, bev_focal])
        row_cp = np.hstack([cp_cam_img, bev_cp])
        combined = np.vstack([row_focal, row_cp])

        out_img_path = os.path.join(args.out_dir, f"frame_{frame_idx:04d}_compare.png")
        cv2.imwrite(out_img_path, combined)
        print(f"  [+] Saved comparison visualization: {out_img_path}")

    # Save summary stats
    summary_path = os.path.join(args.out_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_stats, f, indent=2)
    print(f"\n[+] Full 10-frame comparison stats saved to: {summary_path}")


if __name__ == "__main__":
    main()
