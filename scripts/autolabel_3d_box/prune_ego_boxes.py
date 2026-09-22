#!/usr/bin/env python3
"""Prune ego-vehicle self-detection boxes from full dataset bev_box GT.

Iterates across all scenes in:
  1. meteor_6cam_lidar_deploy/scenes/{scene}/bev_box/
  2. meteor_6cam_lidar_deploy/box3d_artifacts/focalformer_bev_box/{scene}/bev_box/

For any frame where the ego vehicle body was mistakenly detected:
  - Removes ego vehicle box from boxes_3d and boxes
  - Re-renders BEV mask PNG
  - Saves updated .npz and .png
"""
import os
import sys
import time
import cv2
import numpy as np

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


def is_ego_vehicle_box(b):
    cls = int(b[0])
    if cls != 1:
        return False
    cx, cy = float(b[1]), float(b[2])
    l, w = float(b[4]), float(b[5])
    yaw = float(b[7])
    if abs(cx) > 2.0 or abs(cy) > 0.8:
        return False
    dx, dy = -cx, -cy
    c, s = np.cos(yaw), np.sin(yaw)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return abs(lx) <= (l / 2.0) and abs(ly) <= (w / 2.0)


def process_target_dir(base_root):
    print(f"\n==================================================")
    print(f"[*] Processing base directory: {base_root}")
    print(f"==================================================")
    scenes = sorted([s for s in os.listdir(base_root) if s.startswith("data_")])
    total_pruned_boxes = 0
    total_pruned_frames = 0
    total_frames_checked = 0

    t0 = time.time()
    for s in scenes:
        bdir = os.path.join(base_root, s, "bev_box")
        if not os.path.exists(bdir):
            continue
        files = sorted([f for f in os.listdir(bdir) if f.endswith(".npz")])
        s_pruned_frames = 0
        s_pruned_boxes = 0

        for f in files:
            total_frames_checked += 1
            npz_path = os.path.join(bdir, f)
            png_path = os.path.join(bdir, f.replace(".npz", ".png"))

            d = np.load(npz_path)
            b3d = d["boxes_3d"]
            if len(b3d) == 0:
                continue

            has_ego = any(is_ego_vehicle_box(b) for b in b3d)
            if not has_ego:
                continue

            # Filter
            filtered_3d = [b for b in b3d if not is_ego_vehicle_box(b)]
            n_removed = len(b3d) - len(filtered_3d)
            s_pruned_boxes += n_removed
            s_pruned_frames += 1

            if len(filtered_3d) > 0:
                new_b3d = np.array(filtered_3d, dtype=np.float32)
                new_boxes = np.array([[b[0], b[1], b[2], b[4], b[5], b[7]] for b in new_b3d], dtype=np.float32)
            else:
                new_b3d = np.zeros((0, 8), dtype=np.float32)
                new_boxes = np.zeros((0, 6), dtype=np.float32)

            # Re-render PNG canvas
            bev_canvas = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
            for b in new_b3d:
                cls, cx, cy, zc, l, w, h, yaw = b
                cors_b = box_bev_corners(cx, cy, l, w, yaw)
                cv2.fillPoly(bev_canvas, [np.round(cors_b).astype(np.int32).reshape(-1, 1, 2)], int(cls))

            np.savez_compressed(npz_path, boxes=new_boxes, boxes_3d=new_b3d)
            cv2.imwrite(png_path, bev_canvas)

        print(f"  [✓] Scene {s}: checked {len(files)} frames, pruned {s_pruned_boxes} ego boxes in {s_pruned_frames} frames.")
        total_pruned_boxes += s_pruned_boxes
        total_pruned_frames += s_pruned_frames

    elapsed = time.time() - t0
    print(f"[+] Finished {base_root}: {total_frames_checked} frames checked, pruned {total_pruned_boxes} ego boxes across {total_pruned_frames} frames in {elapsed:.2f}s.")
    return total_pruned_boxes, total_pruned_frames


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    scenes_dir = os.path.join(root, "scenes")
    artifacts_dir = os.path.join(root, "box3d_artifacts/focalformer_bev_box")

    process_target_dir(scenes_dir)
    if os.path.exists(artifacts_dir):
        process_target_dir(artifacts_dir)
    print("\n[✓] All ego vehicle boxes successfully pruned across full dataset!")


if __name__ == "__main__":
    main()
