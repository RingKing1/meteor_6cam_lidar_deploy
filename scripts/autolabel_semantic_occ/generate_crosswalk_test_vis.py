#!/usr/bin/env python3
"""Generate 20 test cases evaluating solid continuous crosswalk fine-tuning.

In BEV GT, inside the connected components of Crosswalk (Class 3),
the forced override of Laneline (Class 4) is shielded, and morphological closing
bridges zebra gaps, transforming the crosswalk into a continuous solid yellow rectangle.

Generates:
  1. 20 full multi-camera + fine-tuned BEV GT images in crosswalk_test_vis/
  2. Direct Before-vs-After comparison panels highlighting crosswalk areas.
"""
import os
import json
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCENE = "data_20260910_061820"
FRAMES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

OUT_DIR = os.path.join(BASE_DIR, "crosswalk_test_vis")
COMPARE_DIR = os.path.join(OUT_DIR, "comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

# Standard METEOR 9-Class BGR Palette
PAL = np.zeros((10, 3), dtype=np.uint8)
PAL[0] = [20, 20, 24]       # background
PAL[1] = [65, 65, 70]       # road (slate)
PAL[2] = [140, 80, 140]     # sidewalk (pink/purple)
PAL[3] = [0, 230, 255]      # crosswalk (bright yellow)
PAL[4] = [255, 255, 255]    # laneline (pure white)
PAL[5] = [40, 40, 240]      # stopline (red)
PAL[6] = [0, 140, 255]      # road edge (orange)
PAL[7] = [220, 210, 50]     # markings (cyan)
PAL[8] = [160, 90, 40]      # parking (blue)

BEV_H, BEV_W = 800, 500
RES = 0.2


def solidify_crosswalk(gt):
    """Bridge zebra gaps and shield crosswalk from Class 4 override."""
    cw_seed = (gt == 3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cw_closed = cv2.morphologyEx(cw_seed, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cw_closed)

    solid_cw = np.zeros_like(cw_seed, dtype=bool)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 40:  # >= 1.6 m^2
            solid_cw[labels == lbl] = True

    gt_tuned = gt.copy()
    gt_tuned[solid_cw] = 3
    return gt_tuned, solid_cw


def render_bev(gt, box, title=""):
    bev_color = PAL[gt].copy()
    if box is not None:
        bev_color[box == 1] = [0, 220, 0]    # Vehicle lime green
        bev_color[box == 2] = [0, 220, 255]  # VRU cyan

    # Ego vehicle (red)
    ego_col, ego_row = int(50.0 / RES), int(80.0 / RES)
    cv2.rectangle(bev_color, (ego_col - 5, ego_row - 10), (ego_col + 5, ego_row + 10), (0, 0, 255), -1)
    cv2.putText(bev_color, "Ego", (ego_col - 12, ego_row + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    panel = cv2.resize(bev_color, (540, 864), interpolation=cv2.INTER_NEAREST)
    if title:
        cv2.putText(panel, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend at bottom
    legend_items = [
        ("Road", (65, 65, 70)),
        ("Lane", (255, 255, 255)),
        ("Cross", (0, 230, 255)),
        ("Walk", (140, 80, 140)),
        ("Veh", (0, 220, 0)),
        ("VRU", (0, 220, 255)),
    ]
    lx = 15
    ly = panel.shape[0] - 20
    for name, col in legend_items:
        cv2.rectangle(panel, (lx, ly - 12), (lx + 14, ly + 2), col, -1)
        cv2.putText(panel, name, (lx + 18, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        lx += 86

    return panel


def main():
    scene_dir = os.path.join(BASE_DIR, "scenes", SCENE)
    gt_dir = os.path.join(scene_dir, "gt")
    box_dir = os.path.join(scene_dir, "bev_box")
    focal_vis_dir = os.path.join(BASE_DIR, "focalformer_test_vis")

    print(f"[*] Processing 20 test frames in {SCENE} for solid crosswalk evaluation ...")

    stats_list = []
    for fi in FRAMES:
        gt_path = os.path.join(gt_dir, f"{fi:04d}.png")
        box_path = os.path.join(box_dir, f"{fi:04d}.png")
        focal_vis_path = os.path.join(focal_vis_dir, f"frame_{fi:04d}_demo.png")

        if not os.path.exists(gt_path):
            continue

        gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        box = cv2.imread(box_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(box_path) else None
        focal_vis = cv2.imread(focal_vis_path) if os.path.exists(focal_vis_path) else None

        cams_grid = focal_vis[:, :2304] if focal_vis is not None else np.zeros((864, 2304, 3), dtype=np.uint8)

        # Apply solidification
        gt_tuned, solid_cw = solidify_crosswalk(gt_orig)

        cw_before = int(np.sum(gt_orig == 3))
        cw_after = int(np.sum(gt_tuned == 3))
        white_replaced = int(np.sum(solid_cw & (gt_orig == 4)))
        stats_list.append((fi, cw_before, cw_after, white_replaced))

        # Render After BEV Panel
        panel_after = render_bev(gt_tuned, box, title=f"BEV GT (Solid Crosswalk Frame {fi:04d})")

        # Full composite (2844 x 864)
        full_demo = np.hstack([cams_grid, panel_after])
        out_path = os.path.join(OUT_DIR, f"frame_{fi:04d}_demo.png")
        cv2.imwrite(out_path, full_demo)

        # Before vs After direct comparison
        panel_before = render_bev(gt_orig, box, title=f"BEFORE (Class 4 White Override)")
        panel_comp_after = render_bev(gt_tuned, box, title=f"AFTER (Solid Yellow Crosswalk)")
        cv2.putText(panel_before, f"Crosswalk: {cw_before} px", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(panel_comp_after, f"Solid CW: {cw_after} px (+{white_replaced} white sealed)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        comp_card = np.hstack([panel_before, panel_comp_after])
        comp_path = os.path.join(COMPARE_DIR, f"compare_frame_{fi:04d}.png")
        cv2.imwrite(comp_path, comp_card)

    print(f"[+] All 20 test cases generated in: {OUT_DIR}")
    print(f"[+] Comparison cards in: {COMPARE_DIR}")
    print("\nFrame | CW Before (px) | CW After (px) | White Streaks Sealed (px)")
    print("-" * 60)
    for fi, b, a, w in stats_list:
        print(f"{fi:5d} | {b:14d} | {a:13d} | {w:25d}")


if __name__ == "__main__":
    main()
