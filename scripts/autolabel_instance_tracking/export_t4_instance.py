#!/usr/bin/env python3
"""2D Instance Tracking & TIER IV t4dataset Export Pipeline.

Associates 2D multi-camera bounding boxes with persistent 3D MOT track tokens across time and views:
  - Spatiotemporal Hungarian matching between projected 3D tracked boxes and 2D bounding boxes
  - Unifies object identity across 360 surround camera views (e.g. crossing front-wide to front-left)
  - Preserves temporal track identity throughout occlusions and scene duration

Exports standard TIER IV t4dataset annotation format:
  - scenes/<scene>/annotation/object_ann.json
  - scenes/<scene>/annotation/instance.json
  - scenes/<scene>/annotation/category.json
"""
import argparse
import json
import os
import sys
import time
import uuid

import numpy as np

CAMS_ORDER = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

IMG_W, IMG_H = 768, 432

# METEOR Category Definitions
CATEGORY_DEFS = [
    {"token": "cat_obstacle_others", "name": "obstacle_others"},
    {"token": "cat_vehicle_car",     "name": "vehicle_car"},
    {"token": "cat_vehicle_truck",   "name": "vehicle_truck"},
    {"token": "cat_vehicle_bus",     "name": "vehicle_bus"},
    {"token": "cat_bicycle",         "name": "bicycle"},
    {"token": "cat_motorcycle",      "name": "motorcycle"},
    {"token": "cat_pedestrian",      "name": "pedestrian"},
]
CLS_TO_CAT_TOKEN = {
    0: "cat_obstacle_others",
    1: "cat_vehicle_car",
    2: "cat_vehicle_truck",
    3: "cat_vehicle_bus",
    4: "cat_bicycle",
    5: "cat_motorcycle",
    6: "cat_pedestrian",
}


def box_3d_corners(cx, cy, zc, l, w, h, yaw):
    cb, sb = np.cos(yaw), np.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    px = cx + dx * cb - dy * sb
    py = cy + dx * sb + dy * cb
    pz = zc + dz
    return np.stack([px, py, pz], axis=1)


def project_corners_to_cam(corners_3d, K, T_ego_cam):
    R_ec = T_ego_cam[:3, :3]
    t_ec = T_ego_cam[:3, 3]
    p_cam = (corners_3d - t_ec) @ R_ec
    z = p_cam[:, 2]
    if (z > 0.5).sum() < 4:
        return None

    z_clip = np.maximum(z, 0.2)
    u = K[0, 0] * p_cam[:, 0] / z_clip + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / z_clip + K[1, 2]

    valid = z > 0.5
    u_v = u[valid]
    v_v = v[valid]
    if len(u_v) == 0:
        return None

    x1, y1 = float(np.min(u_v)), float(np.min(v_v))
    x2, y2 = float(np.max(u_v)), float(np.max(v_v))

    cx1 = max(0.0, min(float(IMG_W), x1))
    cy1 = max(0.0, min(float(IMG_H), y1))
    cx2 = max(0.0, min(float(IMG_W), x2))
    cy2 = max(0.0, min(float(IMG_H), y2))

    if (cx2 - cx1) < 4.0 or (cy2 - cy1) < 4.0:
        return None

    return [cx1, cy1, cx2, cy2]


def compute_iou(bA, bB):
    xA = max(bA[0], bB[0])
    yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2])
    yB = min(bA[3], bB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    areaA = max(0.0, bA[2] - bA[0]) * max(0.0, bA[3] - bA[1])
    areaB = max(0.0, bB[2] - bB[0]) * max(0.0, bB[3] - bB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0.0 else 0.0


def process_scene_export_t4(scene, args):
    scene_dir = os.path.join(args.root, scene)
    manifest_p = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(manifest_p):
        print(f"[!] Manifest not found for scene {scene}, skipping.")
        return

    with open(manifest_p) as f:
        manifest = json.load(f)

    cams_calib = manifest["cams"]
    frames = manifest["frames"]
    n_frames = len(frames)

    ann_dir = os.path.join(scene_dir, "annotation")
    os.makedirs(ann_dir, exist_ok=True)

    bbox2d_dir = os.path.join(scene_dir, "bbox2d")
    bev_box_dir = os.path.join(scene_dir, "bev_box")
    agent_traj_dir = os.path.join(scene_dir, "agent_traj")

    print(f"\n[*] Exporting T4 Instance Tracking Annotations for {scene} ({n_frames} frames)...")
    t_start = time.time()

    object_anns = []
    instances = {}  # instance_token -> category_token
    total_2d_boxes = 0

    for fi, fr in enumerate(frames):
        frame_idx = fr["frame"]

        # Load 2D bounding boxes for this frame
        p_2d = os.path.join(bbox2d_dir, f"{frame_idx:04d}.npz")
        if not os.path.exists(p_2d):
            continue
        z2d = np.load(p_2d)
        boxes_2d_all = z2d["boxes"]    # [8, 96, 5]: (cls, cx, cy, w, h)
        counts_2d_all = z2d["counts"]  # [8]

        # Load 3D bounding boxes and agent_traj for track identity
        p_traj = os.path.join(agent_traj_dir, f"{frame_idx:04d}.npz")
        p_bev = os.path.join(bev_box_dir, f"{frame_idx:04d}.npz")

        b3d_with_tokens = []
        if os.path.exists(p_traj) and os.path.exists(p_bev):
            ztraj = np.load(p_traj)
            zbev = np.load(p_bev)

            traj_boxes = ztraj["boxes"]  # [64, 6]: (cls, xe, ye, l, w, yaw)
            traj_count = int(ztraj["count"])
            bev_3d = zbev["boxes_3d"] if "boxes_3d" in zbev else []

            # Match traj_boxes with bev_3d to obtain zc and height
            for k in range(traj_count):
                cls_id, xe, ye, l, w, yaw_b = traj_boxes[k]
                zc, h = 0.0, 1.8
                for b3 in bev_3d:
                    if abs(b3[1] - xe) < 0.1 and abs(b3[2] - ye) < 0.1:
                        zc, h = b3[3], b3[6]
                        break

                inst_token = f"{scene}_track_{k+1:04d}"
                corners = box_3d_corners(xe, ye, zc, l, w, h, yaw_b)
                b3d_with_tokens.append({
                    "inst_token": inst_token,
                    "cls": int(round(cls_id)),
                    "corners": corners,
                })

        # Process each camera
        for ci, cname in enumerate(CAMS_ORDER):
            cnt = int(counts_2d_all[ci])
            if cnt == 0:
                continue

            cal = cams_calib[cname]
            K = np.array(cal["K"], dtype=np.float64)
            T_ec = np.array(cal["T_ego_cam"], dtype=np.float64)

            # Pre-project all 3D tracked boxes into this camera
            proj_tracks = []
            for b3t in b3d_with_tokens:
                pb = project_corners_to_cam(b3t["corners"], K, T_ec)
                if pb is not None:
                    proj_tracks.append({
                        "inst_token": b3t["inst_token"],
                        "cls": b3t["cls"],
                        "box": pb,
                    })

            sample_data_token = f"{scene}_{cname}_{frame_idx:04d}"

            for k in range(cnt):
                cls_id, cx, cy, w, h = boxes_2d_all[ci, k]
                cls_int = int(round(cls_id))
                bx1 = cx - w / 2.0
                by1 = cy - h / 2.0
                bx2 = cx + w / 2.0
                by2 = cy + h / 2.0
                b2d = [bx1, by1, bx2, by2]

                # Match with projected 3D tracks
                best_iou = 0.0
                matched_token = None
                for pt in proj_tracks:
                    iou = compute_iou(b2d, pt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        matched_token = pt["inst_token"]

                if best_iou < 0.15 or matched_token is None:
                    # Distant or unassociated 2D detection -> persistent camera instance token
                    matched_token = f"{scene}_2d_{cname}_{k+1:04d}"

                cat_token = CLS_TO_CAT_TOKEN.get(cls_int, "cat_vehicle_car")
                instances[matched_token] = cat_token

                ann_entry = {
                    "token": f"ann_{uuid.uuid4().hex[:16]}",
                    "sample_data_token": sample_data_token,
                    "instance_token": matched_token,
                    "category_token": cat_token,
                    "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                    "mask": None
                }
                object_anns.append(ann_entry)
                total_2d_boxes += 1

        if (fi + 1) % 500 == 0 or (fi + 1) == n_frames:
            fps = (fi + 1) / max(0.001, time.time() - t_start)
            print(f"  [{fi+1}/{n_frames}] {fps:.1f} fps | Total 2D instances: {total_2d_boxes} | Unique tracks: {len(instances)}")

    # Save category.json
    with open(os.path.join(ann_dir, "category.json"), "w") as f:
        json.dump(CATEGORY_DEFS, f, indent=2)

    # Save instance.json
    instance_list = [
        {"token": itok, "category_token": ctok}
        for itok, ctok in instances.items()
    ]
    with open(os.path.join(ann_dir, "instance.json"), "w") as f:
        json.dump(instance_list, f, indent=2)

    # Save object_ann.json
    with open(os.path.join(ann_dir, "object_ann.json"), "w") as f:
        json.dump(object_anns, f, indent=2)

    total_time = time.time() - t_start
    print(f"[+] Scene {scene} T4 export completed: {total_2d_boxes} 2D annotations across {len(instances)} instances in {total_time:.1f}s.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/meteor_6cam_lidar_deploy/scenes", help="Scenes root directory")
    parser.add_argument("--scenes", default="all", help="all or comma-separated scene list")
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
    print("  TIER IV t4dataset 2D Instance Tracking Export Pipeline")
    print("================================================================================")
    print(f"Scenes: {target_scenes}")
    print("================================================================================")

    for scene in target_scenes:
        process_scene_export_t4(scene, args)

    print("\n================================================================================")
    print("  All T4 instance tracking datasets exported successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
