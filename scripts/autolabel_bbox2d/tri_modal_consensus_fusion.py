#!/usr/bin/env python3
"""Tri-Modal Consensus Fusion Engine for 2D Bounding Boxes and Purified 3D BEV Boxes.

Implements the Tri-Modal Consensus Arbiter (3D LiDAR Geometry + 2D YOLOv8x + 2D Panoptic Segmentation):
  - Dimension 1: 3D LiDAR geometry (FocalFormer3D score >= 0.15 candidates)
  - Dimension 2: 2D Vision (YOLOv8x on 6 surround cameras @ 768x432)
  - Dimension 3: 2D Pixel Semantics (Mask2Former seg2d21 @ 108x192)

Outputs:
  - scenes/<scene>/bbox2d/<fi:04d>.npz: float32 [8, 96, 5] (cls, cx, cy, w, h), uint8 counts [8]
  - scenes/<scene>/bev_box/<fi:04d>.npz: verified high-fidelity 3D boxes
  - scenes/<scene>/bev_box/<fi:04d>.png: rendered BEV mask (800x500 @ 0.2m)
  - Updates manifest.json with "bbox2d" and "bev_box"
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# METEOR 10-Class Taxonomy
# 0: obstacle_others, 1: vehicle_car, 2: vehicle_truck, 3: vehicle_bus,
# 4: bicycle, 5: motorcycle, 6: pedestrian
COCO_TO_METEOR = {
    2: 1,  # car -> vehicle_car
    7: 2,  # truck -> vehicle_truck
    5: 3,  # bus -> vehicle_bus
    1: 4,  # bicycle -> bicycle
    3: 5,  # motorcycle -> motorcycle
    0: 6,  # person -> pedestrian
}

# Macro-categories for 3D/2D consensus matching
# 3D: 1.0 = Vehicle, 2.0 = VRU
# seg2d21: 2:car, 3:truck, 4:bus, 5:moto -> veh; 6:bike, 7:ped -> vru
SEG_VEH_CLASSES = {2, 3, 4, 5}
SEG_VRU_CLASSES = {6, 7}

METEOR_TO_MACRO = {
    1: 1.0, 2: 1.0, 3: 1.0,   # vehicles
    4: 2.0, 5: 2.0, 6: 2.0,   # VRU
}

CAMS_ORDER = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Total camera slots expected by METEOR dataset loader (8 slots, slots 6 & 7 reserved for narrow)
N_CAMS_TOTAL = 8
KMAX = 96
IMG_W, IMG_H = 768, 432
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2


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


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def project_3d_box_to_camera(corners_3d, K, T_ego_cam):
    """Projects 8 3D corners to camera image plane. Returns [x1, y1, x2, y2] or None if outside."""
    R_ec = T_ego_cam[:3, :3]
    t_ec = T_ego_cam[:3, 3]
    p_cam = (corners_3d - t_ec) @ R_ec
    z = p_cam[:, 2]
    # Object must be significantly in front of camera
    if (z > 0.5).sum() < 4:
        return None

    z_clip = np.maximum(z, 0.2)
    u = K[0, 0] * p_cam[:, 0] / z_clip + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / z_clip + K[1, 2]

    # Filter out points that are far behind or unreasonable
    valid = z > 0.5
    u_v = u[valid]
    v_v = v[valid]
    if len(u_v) == 0:
        return None

    x1, y1 = float(np.min(u_v)), float(np.min(v_v))
    x2, y2 = float(np.max(u_v)), float(np.max(v_v))

    # Clamp to image frame
    cx1 = max(0.0, min(float(IMG_W), x1))
    cy1 = max(0.0, min(float(IMG_H), y1))
    cx2 = max(0.0, min(float(IMG_W), x2))
    cy2 = max(0.0, min(float(IMG_H), y2))

    if (cx2 - cx1) < 4.0 or (cy2 - cy1) < 4.0:
        return None

    return [cx1, cy1, cx2, cy2]


def compute_iou_2d(boxA, boxB):
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def check_mask_coverage(proj_box, seg_cam, macro_cls):
    """Computes foreground semantic mask ratio within the projected 2D box in seg2d21."""
    # seg_cam has shape [108, 192] (stride 4 of 432x768)
    x1, y1, x2, y2 = proj_box
    sx1 = max(0, min(191, int(round(x1 * 0.25))))
    sy1 = max(0, min(107, int(round(y1 * 0.25))))
    sx2 = max(0, min(192, int(round(x2 * 0.25))))
    sy2 = max(0, min(108, int(round(y2 * 0.25))))

    if sx2 <= sx1 or sy2 <= sy1:
        return 0.0

    crop = seg_cam[sy1:sy2, sx1:sx2]
    total_px = crop.size
    if total_px == 0:
        return 0.0

    if macro_cls == 1.0:
        match_count = np.isin(crop, list(SEG_VEH_CLASSES)).sum()
    elif macro_cls == 2.0:
        match_count = np.isin(crop, list(SEG_VRU_CLASSES)).sum()
    else:
        return 0.0

    return float(match_count) / float(total_px)


def is_ego_hood_box(cname, box):
    """Filters out self-detected ego vehicle hood at the bottom of front cameras."""
    if "FRONT" not in cname:
        return False
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cy = (y1 + y2) / 2.0
    # Bottom area of front camera covering wide width
    if cy > 320.0 and (w > 400.0 or y2 >= (IMG_H - 5)):
        return True
    return False


def process_scene_consensus(scene, args, yolo_model, cams_calib):
    scene_dir = os.path.join(args.root, scene)
    manifest_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found for scene {scene}, skipping.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.frames is not None:
        target_fids = set(int(x.strip()) for x in args.frames.split(",") if x.strip())
        frames = [fr for fr in manifest["frames"] if fr["frame"] in target_fids]
    elif args.max_frames > 0:
        frames = manifest["frames"][:args.max_frames]
    else:
        frames = manifest["frames"]

    out_bbox2d_dir = os.path.join(scene_dir, "bbox2d")
    out_bev_dir = os.path.join(scene_dir, "bev_box")
    os.makedirs(out_bbox2d_dir, exist_ok=True)
    os.makedirs(out_bev_dir, exist_ok=True)

    focal_scene_dir = os.path.join(args.focal_dir, scene, "bev_box")

    t_start = time.time()
    n_frames = len(frames)
    confirmed_3d_count = 0
    rescued_count = 0
    rejected_noise_count = 0

    print(f"\n[*] Starting Tri-Modal Consensus for scene: {scene} ({n_frames} frames)")

    for fi, fr in enumerate(frames):
        frame_idx = fr["frame"]
        npz_2d_path = os.path.join(out_bbox2d_dir, f"{frame_idx:04d}.npz")
        npz_3d_path = os.path.join(out_bev_dir, f"{frame_idx:04d}.npz")
        png_3d_path = os.path.join(out_bev_dir, f"{frame_idx:04d}.png")

        if not args.force and os.path.exists(npz_2d_path) and os.path.exists(npz_3d_path):
            fr["bbox2d"] = f"bbox2d/{frame_idx:04d}.npz"
            fr["bev_box"] = f"bev_box/{frame_idx:04d}.png"
            continue

        # 1. Load 3D candidates (from FocalFormer3D 0.15)
        focal_npz = os.path.join(focal_scene_dir, f"{frame_idx:04d}.npz")
        if not os.path.exists(focal_npz):
            # Fallback to existing bev_box if 0.15 artifact not found
            focal_npz = npz_3d_path

        cand_3d_boxes = []
        cand_scores = []
        if os.path.exists(focal_npz):
            z3d = np.load(focal_npz)
            if "boxes_3d" in z3d and len(z3d["boxes_3d"]) > 0:
                cand_3d_boxes = z3d["boxes_3d"]
                if "scores" in z3d and len(z3d["scores"]) == len(cand_3d_boxes):
                    cand_scores = z3d["scores"]
                else:
                    cand_scores = np.full(len(cand_3d_boxes), 0.35, dtype=np.float32)

        # 2. Load 6 camera images
        imgs_bgr = []
        img_paths = []
        for cname in CAMS_ORDER:
            p = os.path.join(scene_dir, "img", cname, f"{frame_idx:04d}.jpg")
            if not os.path.exists(p):
                p = os.path.join(scene_dir, "img", cname.lower(), f"{frame_idx:04d}.jpg")
            img_paths.append(p)
            if os.path.exists(p):
                imgs_bgr.append(cv2.imread(p))
            else:
                imgs_bgr.append(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))

        # 3. Run YOLOv8x on 6 cameras
        yolo_dets_by_cam = [[] for _ in range(len(CAMS_ORDER))]
        with torch.no_grad():
            yolo_results = yolo_model(imgs_bgr, conf=0.20, verbose=False)

        for ci, res in enumerate(yolo_results):
            cname = CAMS_ORDER[ci]
            for b in res.boxes:
                coco_cls = int(b.cls[0].item())
                if coco_cls not in COCO_TO_METEOR:
                    continue
                meteor_cls = COCO_TO_METEOR[coco_cls]
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy().tolist()

                if is_ego_hood_box(cname, xyxy):
                    continue

                yolo_dets_by_cam[ci].append({
                    "cls": meteor_cls,
                    "macro_cls": METEOR_TO_MACRO[meteor_cls],
                    "conf": conf,
                    "box": xyxy,
                    "matched_3d": False,
                })

        # 4. Load 2D segmentation (seg2d21)
        seg2d_path = os.path.join(scene_dir, "seg2d21", f"{frame_idx:04d}.npz")
        seg_cam_arr = np.zeros((len(CAMS_ORDER), 108, 192), dtype=np.uint8)
        if os.path.exists(seg2d_path):
            zseg = np.load(seg2d_path)
            if "seg" in zseg:
                seg_cam_arr = zseg["seg"][:len(CAMS_ORDER)]

        # 5. Tri-Modal Arbiter: Vote on each 3D candidate
        confirmed_3d = []
        confirmed_3d_np = []
        per_cam_confirmed_2d = [[] for _ in range(N_CAMS_TOTAL)]

        for bi in range(len(cand_3d_boxes)):
            b3d = cand_3d_boxes[bi]
            s3d = float(cand_scores[bi])
            cls_3d, cx, cy, zc, l, w, h, yaw = b3d
            macro_cls = float(cls_3d)

            # Rule 1: 3D Score Vote
            votes = 0
            if s3d >= 0.40:
                votes += 2
            elif s3d >= 0.15:
                votes += 1

            corners = box_3d_corners(cx, cy, zc, l, w, h, yaw)
            yolo_voted = False
            mask_voted = False
            cam_matches = []  # (ci, best_yolo_det, proj_box)

            # Project across 6 cameras
            for ci, cname in enumerate(CAMS_ORDER):
                cal = cams_calib[cname]
                K = np.array(cal["K"], dtype=np.float64)
                T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)

                proj_box = project_3d_box_to_camera(corners, K, T_ec)
                if proj_box is None:
                    continue

                # Rule 2: YOLO IoU Match
                best_iou = 0.0
                best_ydet = None
                for ydet in yolo_dets_by_cam[ci]:
                    if ydet["macro_cls"] == macro_cls:
                        iou = compute_iou_2d(proj_box, ydet["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_ydet = ydet

                if best_iou >= 0.25 and best_ydet is not None:
                    if not yolo_voted:
                        votes += 1
                        yolo_voted = True
                    best_ydet["matched_3d"] = True
                    cam_matches.append((ci, best_ydet, proj_box))
                else:
                    # Rule 3: Mask2Former Semantics
                    cov = check_mask_coverage(proj_box, seg_cam_arr[ci], macro_cls)
                    if cov >= 0.20:
                        if not mask_voted:
                            votes += 1
                            mask_voted = True
                        cam_matches.append((ci, None, proj_box))

            # Consensus Decision
            if votes >= 2:
                confirmed_3d_count += 1
                if s3d < 0.25:
                    rescued_count += 1  # Rescued by tri-modal consensus!

                confirmed_3d.append((macro_cls, cx, cy, l, w, yaw))
                confirmed_3d_np.append([macro_cls, cx, cy, zc, l, w, h, yaw])

                # Register 2D bounding boxes for visible cameras
                for ci, ydet, proj_box in cam_matches:
                    if ydet is not None:
                        # Adopt fine-grained YOLO boundary
                        bx1, by1, bx2, by2 = ydet["box"]
                        fine_cls = ydet["cls"]
                    else:
                        # Adopt tight projected 3D box
                        bx1, by1, bx2, by2 = proj_box
                        fine_cls = 1 if macro_cls == 1.0 else 6

                    w_px = max(2.0, bx2 - bx1)
                    h_px = max(2.0, by2 - by1)
                    cx_px = (bx1 + bx2) / 2.0
                    cy_px = (by1 + by2) / 2.0
                    per_cam_confirmed_2d[ci].append((w_px * h_px, fine_cls, cx_px, cy_px, w_px, h_px))
            else:
                rejected_noise_count += 1

        # 6. Additional 2D handling: Preserve far-field high-conf YOLO detections (>60m)
        for ci, cname in enumerate(CAMS_ORDER):
            for ydet in yolo_dets_by_cam[ci]:
                if ydet["matched_3d"]:
                    continue
                # If unmatched 2D box has high confidence and is far-field (small box in distant view)
                bx1, by1, bx2, by2 = ydet["box"]
                w_px = bx2 - bx1
                h_px = by2 - by1
                if ydet["conf"] >= 0.75 and (w_px < 35.0 and h_px < 35.0 and by1 < 250.0):
                    cx_px = (bx1 + bx2) / 2.0
                    cy_px = (by1 + by2) / 2.0
                    per_cam_confirmed_2d[ci].append((w_px * h_px, ydet["cls"], cx_px, cy_px, w_px, h_px))

        # 7. Pack and Save bbox2d/<fi>.npz
        boxes_2d = np.zeros((N_CAMS_TOTAL, KMAX, 5), dtype=np.float32)
        counts_2d = np.zeros(N_CAMS_TOTAL, dtype=np.uint8)

        for ci in range(N_CAMS_TOTAL):
            cands = per_cam_confirmed_2d[ci]
            cands.sort(reverse=True)  # Largest boxes first
            for k, (_, cls, cx, cy, w, h) in enumerate(cands[:KMAX]):
                boxes_2d[ci, k] = (cls, cx, cy, w, h)
            counts_2d[ci] = min(len(cands), KMAX)

        np.savez_compressed(npz_2d_path, boxes=boxes_2d, counts=counts_2d)
        fr["bbox2d"] = f"bbox2d/{frame_idx:04d}.npz"

        # 8. Pack and Save bev_box/<fi>.npz and bev_box/<fi>.png
        bev_canvas = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
        boxes_bev_out = []
        for mcls, cx, cy, l, w, yaw in confirmed_3d:
            cors_b = box_bev_corners(cx, cy, l, w, yaw)
            cv2.fillPoly(bev_canvas, [np.round(cors_b).astype(np.int32).reshape(-1, 1, 2)], int(mcls))
            boxes_bev_out.append([mcls, cx, cy, l, w, yaw])

        np.savez_compressed(
            npz_3d_path,
            boxes=np.array(boxes_bev_out, dtype=np.float32).reshape(-1, 6) if boxes_bev_out else np.zeros((0, 6), dtype=np.float32),
            boxes_3d=np.array(confirmed_3d_np, dtype=np.float32).reshape(-1, 8) if confirmed_3d_np else np.zeros((0, 8), dtype=np.float32)
        )
        cv2.imwrite(png_3d_path, bev_canvas)
        fr["bev_box"] = f"bev_box/{frame_idx:04d}.png"

        if (fi + 1) % 50 == 0 or (fi + 1) == n_frames:
            fps = (fi + 1) / max(0.001, time.time() - t_start)
            print(f"  [{fi+1}/{n_frames}] {fps:.1f} fps | Confirmed 3D: {confirmed_3d_count} (Rescued: {rescued_count}, Rejected noise: {rejected_noise_count})")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print(f"[+] Scene {scene} finished in {total_time:.1f}s. Confirmed: {confirmed_3d_count}, Rescued: {rescued_count}, Filtered: {rejected_noise_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/meteor_6cam_lidar_deploy/scenes", help="Scenes root directory")
    parser.add_argument("--scenes", default="all", help="all or comma-separated scene list")
    parser.add_argument("--focal-dir", default="/work/meteor_6cam_lidar_deploy/box3d_artifacts/focalformer_bev_box_015")
    parser.add_argument("--model-path", default="/work/meteor_6cam_lidar_deploy/box3d_artifacts/models/yolov8x.pt")
    parser.add_argument("--frames", default=None, help="comma-separated specific frame indices")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_scenes = [
        "data_20260910_061820",
        "data_20260910_062659",
        "data_20260910_063822",
        "data_20260910_064331",
        "data_20260910_073823",
        "data_20260910_074912",
    ]
    if args.scenes == "all":
        target_scenes = all_scenes
    else:
        target_scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print("================================================================================")
    print("  Tri-Modal Consensus Fusion Pipeline (BBox2D + Purified BEV Box)")
    print("================================================================================")
    print(f"Scenes:      {target_scenes}")
    print(f"Focal Dir:   {args.focal_dir}")
    print(f"YOLO Model:  {args.model_path}")
    print(f"Max Frames:  {args.max_frames}")
    print(f"Force:       {args.force}")
    print("================================================================================")

    print("[1/2] Loading YOLOv8x model...")
    yolo_model = YOLO(args.model_path)
    print("[+] YOLOv8x loaded successfully.")

    first_manifest_p = os.path.join(args.root, target_scenes[0], "manifest.json")
    with open(first_manifest_p) as f:
        first_manifest = json.load(f)
    cams_calib = first_manifest["cams"]

    print("\n[2/2] Running Consensus Fusion...")
    for scene in target_scenes:
        process_scene_consensus(scene, args, yolo_model, cams_calib)

    print("\n================================================================================")
    print("  All requested scenes processed successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
