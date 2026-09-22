#!/usr/bin/env python3
"""Promote solid continuous crosswalk across all 6 scenes (17,845 frames).

In BEV GT (scenes/{scene}/gt/<fi:04d>.png):
For any frame with crosswalk (Class 3), applies connected-component morphology
to bridge zebra gaps and seal Class 4 laneline override into solid Class 3 yellow crosswalks.

Usage:
  python3 scripts/autolabel_semantic_occ/promote_solid_crosswalk.py [--workers 16]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCENES = [
    "data_20260910_061820",
    "data_20260910_062659",
    "data_20260910_063822",
    "data_20260910_064331",
    "data_20260910_073823",
    "data_20260910_074912",
]

# Morphological kernel for 0.2m resolution: 7x7 corresponds to 1.4m x 1.4m
KERNEL_7X7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))


def process_single_gt_file(png_path_str):
    """Worker: update a single GT file if it contains crosswalks."""
    p = Path(png_path_str)
    gt = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return 0, 0, 0

    cw_seed = (gt == 3).astype(np.uint8)
    if not np.any(cw_seed):
        return 0, 0, 0  # No crosswalk, skipped

    cw_before = int(np.sum(cw_seed))

    # Morphological closing
    cw_closed = cv2.morphologyEx(cw_seed, cv2.MORPH_CLOSE, KERNEL_7X7)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cw_closed)

    solid_cw = np.zeros_like(cw_seed, dtype=bool)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 40:  # >= 1.6 m^2
            solid_cw[labels == lbl] = True

    if not np.any(solid_cw):
        return 0, cw_before, cw_before

    white_sealed = int(np.sum(solid_cw & (gt == 4)))
    gt_tuned = gt.copy()
    gt_tuned[solid_cw] = 3
    cw_after = int(np.sum(gt_tuned == 3))

    # Write back
    cv2.imwrite(str(p), gt_tuned)
    return 1, cw_before, cw_after, white_sealed


def process_scene_crosswalk(scene_name, workers=16):
    gt_dir = os.path.join(BASE_DIR, "scenes", scene_name, "gt")
    if not os.path.exists(gt_dir):
        print(f"[-] Missing gt dir for {scene_name}")
        return

    png_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".png")])
    total_files = len(png_files)

    t0 = time.time()
    updated_frames = 0
    total_white_sealed = 0
    total_cw_before = 0
    total_cw_after = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(process_single_gt_file, png_files):
            if res[0] == 1:
                _, b, a, w = res
                updated_frames += 1
                total_cw_before += b
                total_cw_after += a
                total_white_sealed += w

    elapsed = time.time() - t0
    print(f"[✓] {scene_name}: {total_files} frames processed in {elapsed:.1f}s | "
          f"Updated: {updated_frames} frames | "
          f"White Sealed: {total_white_sealed:,} px | "
          f"CW Voxel/Pixel Gain: {total_cw_before:,} -> {total_cw_after:,} px")
    return {
        "scene": scene_name,
        "total_frames": total_files,
        "updated_frames": updated_frames,
        "white_sealed": total_white_sealed,
        "cw_before": total_cw_before,
        "cw_after": total_cw_after,
        "time": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    print("==================================================")
    print(f"Promoting Solid Crosswalk across all 6 scenes (17,845 frames)")
    print(f"Workers: {args.workers}")
    print("==================================================")

    t_all = time.time()
    summary = []
    for s in SCENES:
        res = process_scene_crosswalk(s, workers=args.workers)
        if res:
            summary.append(res)

    tot_frames = sum(r["total_frames"] for r in summary)
    tot_updated = sum(r["updated_frames"] for r in summary)
    tot_white = sum(r["white_sealed"] for r in summary)
    tot_time = time.time() - t_all

    print("\n==================================================")
    print(f"ALL SCENES COMPLETED in {tot_time:.1f}s ({tot_frames / tot_time:.1f} frames/s)")
    print(f"Total Scenes: {len(summary)} | Total Frames: {tot_frames:,}")
    print(f"Frames with Crosswalks Solidified: {tot_updated:,} ({tot_updated / tot_frames * 100:.1f}%)")
    print(f"Total White Laneline Streaks Cleaned: {tot_white:,} pixels")
    print("==================================================")


if __name__ == "__main__":
    main()
