#!/usr/bin/env python3
"""Automated Traffic Light State & Ego Corridor Extraction for METEOR.

Extracts frame-by-frame ego-relevant traffic light state (0: none, 1: green, 2: yellow, 3: red)
using 2D panoptic segmentation (seg2d21 class 9) + HSV lamp luminescence detection + ego travel corridor heuristic.

Outputs:
  - <scene>/tl_state.npz: {"label": int64 [F], "conf": float32 [F]}
  - Updates <scene>/manifest.json with "tl_state": "tl_state.npz"

Usage:
  python3 scripts/build_tl_gt.py [--scenes all] [--workers 8]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENES_DIR = os.path.join(BASE_DIR, "scenes")

# Ego travel corridor horizontal bounds (central 50% ~ 56% of front wide camera)
CORRIDOR_X_MIN = 0.22
CORRIDOR_X_MAX = 0.78


def analyze_frame_tl(seg_path, img_path):
    """Analyze a single front-camera frame for ego-relevant traffic light."""
    if not os.path.exists(seg_path) or not os.path.exists(img_path):
        return 0, 0.0

    seg = np.load(seg_path)["seg"][0]  # CAM_FRONT_WIDE, shape [108, 192]
    if not (seg == 9).any():
        return 0, 0.0

    tl_mask = (seg == 9).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tl_mask)
    if num_labels <= 1:
        return 0, 0.0

    img = cv2.imread(img_path)
    if img is None:
        return 0, 0.0
    h, w = img.shape[:2]  # [432, 768]

    best_area = 0
    best_color = 0
    best_conf = 0.0

    for lbl in range(1, num_labels):
        cx = centroids[lbl][0] / 192.0
        area = stats[lbl, cv2.CC_STAT_AREA]

        # Filter to central travel corridor
        if not (CORRIDOR_X_MIN <= cx <= CORRIDOR_X_MAX):
            continue

        x_min = int(stats[lbl, cv2.CC_STAT_LEFT] * 4)
        x_max = int((stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH]) * 4)
        y_min = int(stats[lbl, cv2.CC_STAT_TOP] * 4)
        y_max = int((stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT]) * 4)

        # Pad 2 pixels
        x_min, x_max = max(0, x_min - 2), min(w, x_max + 2)
        y_min, y_max = max(0, y_min - 2), min(h, y_max + 2)

        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # S > 65, V > 100 identifies active illuminated light elements
        lamp_mask = (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 100)
        hues = hsv[:, :, 0][lamp_mask]

        if len(hues) < 3:
            continue

        r_count = int(np.sum((hues < 14) | (hues > 166)))
        g_count = int(np.sum((hues >= 35) & (hues <= 90)))
        y_count = int(np.sum((hues >= 14) & (hues < 35)))

        counts = [0, g_count, y_count, r_count]  # 1: green, 2: yellow, 3: red
        c_idx = int(np.argmax(counts))

        if counts[c_idx] > 0 and area > best_area:
            best_area = area
            best_color = c_idx
            best_conf = float(counts[c_idx] / len(hues))

    return best_color, best_conf


def process_scene(scene_name):
    scene_dir = os.path.join(SCENES_DIR, scene_name)
    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        return f"[ERROR] manifest.json missing in {scene_dir}"

    with open(mf_path, "r") as f:
        manifest = json.load(f)

    frames = manifest["frames"]
    F = len(frames)
    raw_labels = np.zeros(F, dtype=np.int64)
    confs = np.zeros(F, dtype=np.float32)

    for i, fr in enumerate(frames):
        fi = fr["frame"]
        seg_p = os.path.join(scene_dir, f"seg2d21/{fi:04d}.npz")
        img_p = os.path.join(scene_dir, f"img/CAM_FRONT_WIDE/{fi:04d}.jpg")
        c, cf = analyze_frame_tl(seg_p, img_p)
        raw_labels[i] = c
        confs[i] = cf

    # 3-frame temporal median filter (removes 1-frame flickers per METEOR spec)
    sm_labels = raw_labels.copy()
    for j in range(1, F - 1):
        a, b, c = raw_labels[j - 1], raw_labels[j], raw_labels[j + 1]
        if a == c and b != a:
            sm_labels[j] = a

    # Save tl_state.npz
    out_npz = os.path.join(scene_dir, "tl_state.npz")
    np.savez_compressed(out_npz, label=sm_labels, conf=confs)

    # Register in manifest
    manifest["tl_state"] = "tl_state.npz"
    with open(mf_path, "w") as f:
        json.dump(manifest, f, indent=2)

    counts = np.bincount(sm_labels, minlength=4)
    res = (f"[OK] {scene_name} ({F} frames) -> "
           f"None: {counts[0]}, Green: {counts[1]}, Yellow: {counts[2]}, Red: {counts[3]}")
    return res


def main():
    parser = argparse.ArgumentParser(description="Extract traffic light ground truth")
    parser.add_argument("--scenes", default="all", help="comma-separated scene names or 'all'")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads/processes")
    args = parser.parse_args()

    if args.scenes == "all":
        scene_names = sorted(
            d for d in os.listdir(SCENES_DIR)
            if os.path.isdir(os.path.join(SCENES_DIR, d)) and
            os.path.exists(os.path.join(SCENES_DIR, d, "manifest.json"))
        )
    else:
        scene_names = [s.strip() for s in args.scenes.split(",")]

    print(f"[*] Starting Traffic Light State Extraction for {len(scene_names)} scene(s)...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=min(args.workers, len(scene_names))) as ex:
        results = list(ex.map(process_scene, scene_names))

    for r in results:
        print(r)

    print(f"[*] All complete in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
