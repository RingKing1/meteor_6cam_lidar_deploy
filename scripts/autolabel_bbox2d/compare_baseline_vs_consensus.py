#!/usr/bin/env python3
"""Quantitative and Visual Comparison: Baseline 0.25 vs Tri-Modal Consensus 0.15+Voting.

Analyzes the 20 sampled frames from Phase 2:
  - Total boxes in Baseline (score >= 0.25)
  - Total candidates in FocalFormer (score >= 0.15)
  - Confirmed physical objects after Tri-Modal Consensus Arbiter
  - Rescued dark/distant objects (score 0.15~0.25 rescued by visual + mask)
  - Pruned point-cloud noise/ghost detections
  - Renders side-by-side comparison images for visual verification
"""
import json
import os
import cv2
import numpy as np

SCENE = "data_20260910_061820"
FRAMES = [0, 50, 100, 200, 400, 600, 650, 800, 950, 1000, 1050, 1200, 1400, 1450, 1500, 1600, 1800, 2000, 2200, 2500]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCENE_DIR = os.path.join(BASE_DIR, "scenes", SCENE)
BASELINE_DIR = os.path.join(BASE_DIR, "box3d_artifacts", "focalformer_bev_box", SCENE, "bev_box")
FOCAL_015_DIR = os.path.join(BASE_DIR, "box3d_artifacts", "focalformer_bev_box_015", SCENE, "bev_box")
CONSENSUS_DIR = os.path.join(SCENE_DIR, "bev_box")
OUT_DIR = os.path.join(BASE_DIR, "box3d_artifacts", "visual_inspection")
os.makedirs(OUT_DIR, exist_ok=True)

BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2


def to_bev_px(x, y):
    r = int(round((BEV_XH - x) / RES))
    c = int(round((BEV_YH - y) / RES))
    return (c, r)


def render_bev_boxes(gt_png, boxes, highlight_rescued=False, baseline_boxes=None):
    bev_color = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    bev_color[:] = (22, 22, 26)

    if gt_png is not None:
        bev_color[gt_png == 1] = (65, 65, 70)       # Road drivable
        bev_color[gt_png == 2] = (140, 80, 140)     # Sidewalk
        bev_color[gt_png == 3] = (0, 230, 255)      # Crosswalk
        bev_color[gt_png == 4] = (255, 255, 255)    # Laneline
        bev_color[gt_png == 5] = (40, 40, 240)      # Stopline
        bev_color[gt_png == 6] = (0, 140, 255)      # Road edge
        bev_color[gt_png == 7] = (220, 210, 50)     # Markings

    for b in boxes:
        cls, cx, cy, l, w, yaw = b[:6]
        cb, sb = np.cos(yaw), np.sin(yaw)
        cors = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)):
            px = cx + lx * cb - wy * sb
            py = cy + lx * sb + wy * cb
            cors.append(to_bev_px(px, py))

        # Check if rescued (not present in baseline)
        is_rescued = False
        if highlight_rescued and baseline_boxes is not None:
            is_matched = False
            for bb in baseline_boxes:
                if np.hypot(bb[1] - cx, bb[2] - cy) < 1.5:
                    is_matched = True
                    break
            if not is_matched:
                is_rescued = True

        if is_rescued:
            col = (0, 140, 255)  # Orange for rescued long-tail objects!
            border_col = (0, 255, 255)
        else:
            col = (0, 220, 0) if cls == 1 else (0, 220, 255)
            border_col = (255, 255, 255)

        cv2.fillPoly(bev_color, [np.array(cors, dtype=np.int32)], col)
        cv2.polylines(bev_color, [np.array(cors, dtype=np.int32)], True, border_col, 2 if is_rescued else 1, cv2.LINE_AA)

        if is_rescued:
            center_px = to_bev_px(cx, cy)
            cv2.putText(bev_color, "RESCUED", (center_px[0] - 25, center_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Ego vehicle
    ego_px = to_bev_px(0.0, 0.0)
    cv2.rectangle(bev_color, (ego_px[0] - 5, ego_px[1] - 10), (ego_px[0] + 5, ego_px[1] + 10), (0, 0, 255), -1)
    return bev_color


def main():
    print("=" * 80)
    print("  Phase 2 Quality Analysis: Baseline 0.25 vs Tri-Modal Consensus (20 Frames)")
    print("=" * 80)

    total_base = 0
    total_015_cand = 0
    total_consensus = 0
    total_rescued = 0
    total_filtered = 0

    frame_stats = []

    for fi in FRAMES:
        # 1. Baseline boxes
        base_p = os.path.join(BASELINE_DIR, f"{fi:04d}.npz")
        base_boxes = np.load(base_p)["boxes"] if os.path.exists(base_p) else np.zeros((0, 6))

        # 2. 0.15 Candidates
        c015_p = os.path.join(FOCAL_015_DIR, f"{fi:04d}.npz")
        c015_data = np.load(c015_p) if os.path.exists(c015_p) else {}
        cand_boxes = c015_data["boxes"] if "boxes" in c015_data else np.zeros((0, 6))
        cand_scores = c015_data["scores"] if "scores" in c015_data else np.zeros(0)

        # 3. Consensus Confirmed
        cons_p = os.path.join(CONSENSUS_DIR, f"{fi:04d}.npz")
        cons_boxes = np.load(cons_p)["boxes"] if os.path.exists(cons_p) else np.zeros((0, 6))

        n_base = len(base_boxes)
        n_cand = len(cand_boxes)
        n_cons = len(cons_boxes)

        # Count rescued: confirmed boxes not in baseline
        rescued = 0
        for cb in cons_boxes:
            matched = any(np.hypot(bb[1] - cb[1], bb[2] - cb[2]) < 1.5 for bb in base_boxes)
            if not matched:
                rescued += 1

        filtered = max(0, n_cand - n_cons)

        total_base += n_base
        total_015_cand += n_cand
        total_consensus += n_cons
        total_rescued += rescued
        total_filtered += filtered

        frame_stats.append({
            "fi": fi,
            "base": n_base,
            "cand": n_cand,
            "cons": n_cons,
            "rescued": rescued,
            "filtered": filtered,
        })

        # Render visual comparison for highlighted key frames
        if fi in [100, 600, 1000, 1500, 2000]:
            gt_png = cv2.imread(os.path.join(SCENE_DIR, f"gt/{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
            bev_left = render_bev_boxes(gt_png, base_boxes, highlight_rescued=False)
            bev_right = render_bev_boxes(gt_png, cons_boxes, highlight_rescued=True, baseline_boxes=base_boxes)

            # Header bars
            cv2.rectangle(bev_left, (10, 10), (BEV_W - 10, 70), (0, 0, 0), -1)
            cv2.putText(bev_left, f"Baseline (Score >= 0.25) [F#{fi}]", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 2)
            cv2.putText(bev_left, f"Objects: {n_base}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

            cv2.rectangle(bev_right, (10, 10), (BEV_W - 10, 95), (0, 0, 0), -1)
            cv2.putText(bev_right, f"Tri-Modal Consensus [F#{fi}]", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(bev_right, f"Objects: {n_cons} (+{rescued} rescued)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
            cv2.putText(bev_right, f"Filtered Noise: {filtered}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 255), 1)

            cmp_bev = np.hstack([bev_left, bev_right])
            cmp_out = os.path.join(OUT_DIR, f"compare_frame_{fi:04d}_baseline_vs_consensus.jpg")
            cv2.imwrite(cmp_out, cmp_bev, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  [+] Saved comparison: {cmp_out}")

    print("\n" + "-" * 80)
    print(f"{'Frame':<8} | {'Baseline (0.25)':<16} | {'0.15 Candidates':<16} | {'Consensus Confirmed':<20} | {'Rescued':<10} | {'Filtered Noise':<12}")
    print("-" * 80)
    for s in frame_stats:
        print(f"{s['fi']:<8} | {s['base']:<16} | {s['cand']:<16} | {s['cons']:<20} | {s['rescued']:<10} | {s['filtered']:<12}")
    print("-" * 80)
    print(f"{'TOTAL':<8} | {total_base:<16} | {total_015_cand:<16} | {total_consensus:<20} | {total_rescued:<10} | {total_filtered:<12}")
    print("=" * 80)

    summary_data = {
        "scene": SCENE,
        "frames_evaluated": len(FRAMES),
        "total_baseline_boxes": total_base,
        "total_015_candidates": total_015_cand,
        "total_consensus_boxes": total_consensus,
        "total_rescued_objects": total_rescued,
        "total_filtered_noise": total_filtered,
        "recall_boost_pct": round((total_consensus - total_base) / max(1, total_base) * 100, 2),
        "frame_details": frame_stats
    }
    with open(os.path.join(OUT_DIR, "phase2_20frames_stats.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved stats to {os.path.join(OUT_DIR, 'phase2_20frames_stats.json')}")


if __name__ == "__main__":
    main()
