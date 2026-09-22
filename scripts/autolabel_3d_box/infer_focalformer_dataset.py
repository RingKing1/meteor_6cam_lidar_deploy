#!/usr/bin/env python3
"""Full Dataset Inference with FocalFormer3D-LC (5-Sweep Pose Fusion + 6 Surround Cams).

Exports 3D bounding boxes and BEV masks into standard bev_box format:
  box3d_artifacts/focalformer_bev_box/{scene}/bev_box/{frame:04d}.npz
  box3d_artifacts/focalformer_bev_box/{scene}/bev_box/{frame:04d}.png

Coordinate Conversion:
  MMDetection3D / SECOND convention -> Standard Right-Handed ROS REP-103 convention
  (as documented in focalformer_deploy/docs/COORDINATE_AND_ORIENTATION_CONVENTION.md):
    true_yaw = box.yaw + pi/2
    true_length = box.dy
    true_width  = box.dx
    true_height = box.dz
    true_zc     = z_bottom + box.dz / 2.0
"""
import argparse
import glob
import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import projects.mmdet3d_plugin

NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
VEHICLE_LABEL_IDS = {0, 1, 2, 3, 4}  # car, truck, const, bus, trailer -> cls = 1
VRU_LABEL_IDS = {6, 7, 8}            # motorcycle, bicycle, pedestrian -> cls = 2

CAMS_ORDER = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2


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


def box_bev_corners(cx, cy, l, w, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    cors = []
    for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
        px = cx + lx * cb - wy * sb
        py = cy + lx * sb + wy * cb
        row = (BEV_XH - px) / RES
        col = (BEV_YH - py) / RES
        cors.append([col, row])
    return np.array(cors, dtype=np.float32)


def is_ego_vehicle_box(cx, cy, l, w, yaw):
    """Prunes self-detection false positives where the ego vehicle body is detected as an object."""
    if abs(cx) > 2.0 or abs(cy) > 0.8:
        return False
    # Check if box contains ego origin (0.0, 0.0)
    dx, dy = -cx, -cy
    c, s = np.cos(yaw), np.sin(yaw)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return abs(lx) <= (l / 2.0) and abs(ly) <= (w / 2.0)


def load_frame_inputs(scene_dir, raw_dir, lidar_dir, pcd_files, frame_idx, n_sweeps, ego_poses, R_el, t_el):
    """Loads 5-sweep motion-compensated point cloud and 6 camera images."""
    curr_pose = ego_poses[frame_idx]
    x0, y0, yaw0 = curr_pose[0], curr_pose[1], curr_pose[2]
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    R_we0 = np.array([[c0, -s0], [s0, c0]], dtype=np.float64)

    sweeps_pts = []
    for s_off in range(n_sweeps):
        f_i = max(0, frame_idx - s_off)
        pcd_path = os.path.join(lidar_dir, pcd_files[f_i])
        raw_pts = read_pcd_xyz(pcd_path)
        if len(raw_pts) == 0:
            continue
        p_ego = raw_pts[:, :3] @ R_el.T + t_el
        intensity = raw_pts[:, 3:4]
        if s_off > 0:
            p_i = ego_poses[f_i]
            xi, yi, yawi = p_i[0], p_i[1], p_i[2]
            ci, si = np.cos(yawi), np.sin(yawi)
            R_wei = np.array([[ci, -si], [si, ci]], dtype=np.float64)
            p_w = p_ego[:, :2] @ R_wei.T + np.array([xi, yi])
            p_ego_curr = (p_w - np.array([x0, y0])) @ R_we0
            p_ego[:, :2] = p_ego_curr

        time_lag = float(-s_off * 0.1)
        sweep_feat = np.hstack([p_ego, intensity, np.full((len(p_ego), 1), time_lag, dtype=np.float32)])
        sweeps_pts.append(sweep_feat)

    if sweeps_pts:
        fused_pts = np.vstack(sweeps_pts).astype(np.float32)
        mask_range = (
            (fused_pts[:, 0] >= -54.0) & (fused_pts[:, 0] <= 54.0) &
            (fused_pts[:, 1] >= -54.0) & (fused_pts[:, 1] <= 54.0) &
            (fused_pts[:, 2] >= -5.0) & (fused_pts[:, 2] <= 3.0)
        )
        fused_pts = fused_pts[mask_range]
    else:
        fused_pts = np.zeros((0, 5), dtype=np.float32)

    # 6 camera images
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    imgs_list = []
    for ci, cname in enumerate(CAMS_ORDER):
        img_path = os.path.join(scene_dir, "img", cname, f"{frame_idx:04d}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(scene_dir, "img", cname.lower(), f"{frame_idx:04d}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(scene_dir, cname.lower(), f"{frame_idx:04d}.jpg")

        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
        else:
            img_bgr = np.zeros((432, 768, 3), dtype=np.uint8)

        img_resized = cv2.resize(img_bgr, (800, 448))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_norm = (img_rgb - mean) / std
        imgs_list.append(img_norm.transpose(2, 0, 1))

    imgs_arr = np.stack(imgs_list, axis=0).astype(np.float32)
    return frame_idx, fused_pts, imgs_arr


def save_frame_result(out_dir, frame_idx, converted_boxes_3d, converted_scores=None):
    """Saves .npz and .png matching scenes/{scene}/bev_box/ schema with optional scores."""
    npz_path = os.path.join(out_dir, f"{frame_idx:04d}.npz")
    png_path = os.path.join(out_dir, f"{frame_idx:04d}.png")

    boxes_out = []
    boxes_3d_out = []
    scores_out = []
    bev_canvas = np.zeros((BEV_H, BEV_W), dtype=np.uint8)

    for i, b in enumerate(converted_boxes_3d):
        cls, cx, cy, zc, l, w, h, yaw = b
        cors_b = box_bev_corners(cx, cy, l, w, yaw)
        cv2.fillPoly(bev_canvas, [np.round(cors_b).astype(np.int32).reshape(-1, 1, 2)], int(cls))
        boxes_out.append([cls, cx, cy, l, w, yaw])
        boxes_3d_out.append([cls, cx, cy, zc, l, w, h, yaw])
        if converted_scores is not None and i < len(converted_scores):
            scores_out.append(converted_scores[i])

    save_dict = {
        "boxes": np.array(boxes_out, dtype=np.float32).reshape(-1, 6) if boxes_out else np.zeros((0, 6), dtype=np.float32),
        "boxes_3d": np.array(boxes_3d_out, dtype=np.float32).reshape(-1, 8) if boxes_3d_out else np.zeros((0, 8), dtype=np.float32)
    }
    if converted_scores is not None:
        save_dict["scores"] = np.array(scores_out, dtype=np.float32) if scores_out else np.zeros((0,), dtype=np.float32)

    np.savez_compressed(npz_path, **save_dict)
    cv2.imwrite(png_path, bev_canvas)


def process_scene(scene, args, model, lidar2img_arr):
    scene_dir = f"/work/meteor_6cam_lidar_deploy/scenes/{scene}"
    raw_dir = f"/work/meteor_6cam_lidar_deploy/raw_data/{scene}"
    lidar_dir = os.path.join(raw_dir, "lidar")

    calib_lidar_path = os.path.join(raw_dir, "calib/lidar/lidar2imu_calib.txt")
    T_el = read_T_ego_lidar(calib_lidar_path)
    R_el, t_el = T_el[:3, :3], T_el[:3, 3]

    ego_motion_path = os.path.join(scene_dir, "ego_motion.npz")
    ego_poses = np.load(ego_motion_path)["pose"]

    pcd_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd"))
    num_frames = len(pcd_files)

    scene_out_dir = os.path.join(args.out_dir, scene, "bev_box")
    os.makedirs(scene_out_dir, exist_ok=True)

    print(f"\n[*] Processing {scene}: {num_frames} frames -> {scene_out_dir}")

    # Determine frames to process (support resume & specific frames)
    if args.frames is not None:
        target_indices = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        frames_to_run = [fi for fi in target_indices if 0 <= fi < num_frames]
    else:
        frames_to_run = []
        for fi in range(num_frames):
            npz_f = os.path.join(scene_out_dir, f"{fi:04d}.npz")
            png_f = os.path.join(scene_out_dir, f"{fi:04d}.png")
            if not (os.path.exists(npz_f) and os.path.exists(png_f)) or args.force:
                frames_to_run.append(fi)

        if args.max_frames > 0:
            frames_to_run = frames_to_run[:args.max_frames]

    print(f"[*] Frames to process: {len(frames_to_run)} / {num_frames} (already completed: {num_frames - len(frames_to_run)})")
    if not frames_to_run:
        print(f"[+] Scene {scene} already 100% completed. Skipping.")
        return

    # Background async writer
    writer_pool = ThreadPoolExecutor(max_workers=4)

    # Prefetch Queue
    data_queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()

    def prefetch_worker():
        for fi in frames_to_run:
            if stop_event.is_set():
                break
            item = load_frame_inputs(scene_dir, raw_dir, lidar_dir, pcd_files, fi, args.n_sweeps, ego_poses, R_el, t_el)
            data_queue.put(item)
        data_queue.put(None)  # Sentinel

    fetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    fetch_thread.start()

    img_metas = [{
        "box_type_3d": LiDARInstance3DBoxes,
        "lidar2img": lidar2img_arr,
        "img_shape": [(448, 800, 3)] * 6,
        "input_shape": (448, 800)
    }]

    t_start = time.time()
    total_veh, total_vru = 0, 0
    processed_count = 0

    while True:
        item = data_queue.get()
        if item is None:
            break
        frame_idx, fused_pts, imgs_arr = item

        pts_tensor = torch.from_numpy(fused_pts).cuda().float()
        imgs_tensor = torch.from_numpy(imgs_arr).unsqueeze(0).cuda().float()

        with torch.no_grad():
            focal_res = model.simple_test([pts_tensor], img_metas, img=imgs_tensor)

        focal_bboxes = focal_res[0]["pts_bbox"]["boxes_3d"].tensor.cpu().numpy()
        focal_scores = focal_res[0]["pts_bbox"]["scores_3d"].cpu().numpy()
        focal_labels = focal_res[0]["pts_bbox"]["labels_3d"].cpu().numpy()

        mask = focal_scores >= args.score_thresh
        focal_b = focal_bboxes[mask]
        focal_s = focal_scores[mask]
        focal_l = focal_labels[mask]

        converted_boxes = []
        converted_scores = []
        for i in range(len(focal_b)):
            l_id = int(focal_l[i])
            if l_id in VEHICLE_LABEL_IDS:
                target_cls = 1.0  # Vehicle
                total_veh += 1
            elif l_id in VRU_LABEL_IDS:
                target_cls = 2.0  # VRU
                total_vru += 1
            else:
                continue

            cx, cy, z_bottom, dx, dy, dz, yaw_sec = focal_b[i][:7]

            # Coordinate & Orientation Conversion
            true_yaw = yaw_sec + np.pi / 2.0
            true_yaw = (true_yaw + np.pi) % (2.0 * np.pi) - np.pi

            true_length = float(dy)
            true_width = float(dx)
            true_height = float(dz)
            true_zc = float(z_bottom + dz / 2.0)

            # Filter out ego-vehicle self-detection
            if target_cls == 1.0 and is_ego_vehicle_box(cx, cy, true_length, true_width, true_yaw):
                total_veh -= 1
                continue

            converted_boxes.append([target_cls, cx, cy, true_zc, true_length, true_width, true_height, true_yaw])
            converted_scores.append(float(focal_s[i]))

        writer_pool.submit(save_frame_result, scene_out_dir, frame_idx, converted_boxes, converted_scores)

        processed_count += 1
        if processed_count % 10 == 0 or processed_count == len(frames_to_run):
            elapsed = time.time() - t_start
            fps = processed_count / max(0.001, elapsed)
            eta_s = (len(frames_to_run) - processed_count) / max(0.001, fps)
            eta_m = eta_s / 60.0
            print(f"  [{processed_count}/{len(frames_to_run)}] {fps:.2f} fps | Veh: {total_veh}, VRU: {total_vru} | ETA: {eta_m:.1f}m")

    stop_event.set()
    fetch_thread.join()
    writer_pool.shutdown(wait=True)
    total_time = time.time() - t_start
    print(f"[+] Scene {scene} finished: {processed_count} frames in {total_time:.1f}s ({total_veh} veh, {total_vru} vru)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default="all", help="all or comma-separated scene list")
    parser.add_argument("--n-sweeps", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.15)
    parser.add_argument("--frames", default=None, help="comma-separated specific frame indices")
    parser.add_argument("--max-frames", type=int, default=-1, help="-1 for all frames, or N for smoke test")
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    parser.add_argument("--out-dir", default="/work/meteor_6cam_lidar_deploy/box3d_artifacts/focalformer_bev_box_015")
    args = parser.parse_args()

    all_scenes = [
        "data_20260910_061820",
        "data_20260910_062659",
        "data_20260910_063822",
        "data_20260910_064331",
        "data_20260910_073823",
        "data_20260910_074912"
    ]
    if args.scenes == "all":
        target_scenes = all_scenes
    else:
        target_scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print("================================================================================")
    print("  FocalFormer3D-LC Full Dataset 3D Box & BEV Generation Pipeline")
    print("================================================================================")
    print(f"Scenes:        {target_scenes}")
    print(f"Sweeps:        {args.n_sweeps}")
    print(f"Score Thresh:  {args.score_thresh}")
    print(f"Max Frames:    {args.max_frames}")
    print(f"Output Dir:    {args.out_dir}")
    print("================================================================================")

    # 1. Build & Load FocalFormer3D-LC Model
    print("[1/2] Loading FocalFormer3D-LC Model...")
    cfg = Config.fromfile('/work/FocalFormer3D/projects/configs/focalformer3d/FocalFormer3D_LC.py')
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda().eval()
    ckpt_path = "/work/meteor_6cam_lidar_deploy/box3d_artifacts/models/FocalFormer3D_LC.pth"
    load_checkpoint(model, ckpt_path, map_location="cuda")
    print("[+] Model loaded successfully.")

    # 2. Compute lidar2img projection matrix from first scene manifest
    first_scene = target_scenes[0]
    with open(f"/work/meteor_6cam_lidar_deploy/scenes/{first_scene}/manifest.json") as f:
        manifest = json.load(f)
    cams_calib = manifest["cams"]

    lidar2img_list = []
    for cname in CAMS_ORDER:
        cal = cams_calib[cname]
        K = np.array(cal["K"], dtype=np.float64).copy()
        K[0, 0] *= (800.0 / 768.0)
        K[0, 2] *= (800.0 / 768.0)
        K[1, 1] *= (448.0 / 432.0)
        K[1, 2] *= (448.0 / 432.0)
        T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)
        R_ec = T_ec[:3, :3]
        t_ec = T_ec[:3, 3]

        mat4 = np.eye(4, dtype=np.float64)
        mat4[:3, :3] = R_ec.T
        mat4[:3, 3] = -t_ec @ R_ec
        proj = np.eye(4, dtype=np.float64)
        proj[:3, :3] = K
        lidar2img_list.append(proj @ mat4)
    lidar2img_arr = np.stack(lidar2img_list, axis=0).astype(np.float32)

    # 3. Process scenes
    print("\n[2/2] Running Inference...")
    for scene in target_scenes:
        process_scene(scene, args, model, lidar2img_arr)

    print("\n================================================================================")
    print("  All requested scenes processed successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
