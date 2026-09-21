#!/usr/bin/env python3
"""Master Production Pipeline for METEOR Full-Stack Autolabel Factory.

Executes end-to-end data generation across custom scenes:
  Stage 1: 2D Panoptic Segmentation & Road Surface Extraction (GPU, Mask2Former)
           -> seg2d21/<fi:04d>.npz [6, 108, 192] uint8
           -> surface_mask/<fi:04d>.npz [6, 432, 768] uint8
  Stage 2: 3D LiDAR Projection, Map Accumulation & BEV Lane GT (Multi-core CPU)
           -> gt/<fi:04d>.png [800, 500] uint8 (0.2m resolution)
  Stage 3: Dense Metric Depth & 3D Occupancy Grid (Multi-core CPU)
           -> depth4/<fi:04d>.npz [6, 108, 192] float16
           -> occ/<fi:04d>.npz [16, 200, 200] uint8
  Stage 4: Complete Integrity Validation across all frames & modalities.

Usage:
  python3 scripts/run_production_pipeline.py
"""
import argparse
import json
import os
import sys
import time

# Relative paths from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DEFAULT_SCENES = [
    {
        "name": "data_20260910_063822",
        "raw_dir": os.path.join(ROOT_DIR, "METEOR/data/data_20260910_063822"),
        "scene_dir": os.path.join(BASE_DIR, "scenes/data_20260910_063822"),
    },
    {
        "name": "data_20260910_064331",
        "raw_dir": os.path.join(ROOT_DIR, "METEOR/data/data_20260910_064331"),
        "scene_dir": os.path.join(BASE_DIR, "scenes/data_20260910_064331"),
    },
]


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def run_stage1_2d(scene_cfg, limit=0):
    log(f"==================================================")
    log(f"[STAGE 1/3] 2D Panoptic Segmentation: {scene_cfg['name']}")
    log(f"==================================================")
    from batch_2d_panoptic import process_scene
    process_scene(
        scene_cfg["scene_dir"],
        model_name="facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        device="cuda",
        limit=limit,
    )


def run_stage2_bev(scene_cfg, workers=16, limit=0):
    log(f"==================================================")
    log(f"[STAGE 2/3] BEV Map Accumulation & Lane GT: {scene_cfg['name']}")
    log(f"==================================================")
    from build_bev_gt import process_scene
    process_scene(
        scene_cfg["raw_dir"],
        scene_cfg["scene_dir"],
        workers=workers,
        limit=limit,
    )


def run_stage3_depth_occ(scene_cfg, workers=16, limit=0):
    log(f"==================================================")
    log(f"[STAGE 3/3] Dense Depth & 3D Occupancy: {scene_cfg['name']}")
    log(f"==================================================")
    from build_depth_and_occ import process_scene
    process_scene(
        scene_cfg["raw_dir"],
        scene_cfg["scene_dir"],
        workers=workers,
        limit=limit,
    )


def run_stage4_verify(scene_cfg):
    log(f"==================================================")
    log(f"[STAGE 4] Integrity Verification: {scene_cfg['name']}")
    log(f"==================================================")
    scene_dir = scene_cfg["scene_dir"]
    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        log(f"[ERROR] manifest.json missing in {scene_dir}")
        return False

    man = json.load(open(mf_path))
    frames = man["frames"]
    total = len(frames)
    
    missing = {"imgs": 0, "gt": 0, "seg2d21": 0, "depth4": 0, "occ": 0, "lidar_bev": 0}
    for fr in frames:
        fi = fr["frame"]
        # imgs
        for c, rel in fr.get("imgs", {}).items():
            if not os.path.exists(os.path.join(scene_dir, rel)):
                missing["imgs"] += 1
        # gt
        gt_p = fr.get("gt")
        if not gt_p or not os.path.exists(os.path.join(scene_dir, gt_p)):
            missing["gt"] += 1
        # seg2d21
        s_p = fr.get("seg2d21")
        if not s_p or not os.path.exists(os.path.join(scene_dir, s_p)):
            missing["seg2d21"] += 1
        # depth4
        d_p = fr.get("depth4")
        if not d_p or not os.path.exists(os.path.join(scene_dir, d_p)):
            missing["depth4"] += 1
        # occ
        o_p = fr.get("occ")
        if not o_p or not os.path.exists(os.path.join(scene_dir, o_p)):
            missing["occ"] += 1
        # lidar_bev
        l_p = fr.get("lidar_bev")
        if not l_p or not os.path.exists(os.path.join(scene_dir, l_p)):
            missing["lidar_bev"] += 1

    log(f"Scene: {scene_cfg['name']} | Total Frames: {total}")
    all_ok = True
    for k, v in missing.items():
        status = "OK" if v == 0 else f"MISSING {v}"
        if v > 0:
            all_ok = False
        log(f"  - Modality '{k}': {status}")
    
    # Check ego_motion.npz
    ego_path = os.path.join(scene_dir, "ego_motion.npz")
    if os.path.exists(ego_path):
        import numpy as np
        ego = np.load(ego_path)
        val = ego["valid"].sum()
        log(f"  - Ego motion: OK ({val}/{len(ego['valid'])} valid waypoints)")
    else:
        log(f"  - Ego motion: MISSING")
        all_ok = False

    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Master autolabel batch production pipeline")
    ap.add_argument("--scenes", default="all", help="all or comma-separated scene names")
    ap.add_argument("--skip-2d", action="store_true", help="Skip 2D panoptic segmentation")
    ap.add_argument("--skip-bev", action="store_true", help="Skip BEV map accumulation")
    ap.add_argument("--skip-depth-occ", action="store_true", help="Skip depth and occupancy")
    ap.add_argument("--workers", type=int, default=16, help="CPU workers for stages 2 & 3")
    ap.add_argument("--limit", type=int, default=0, help="Limit frames for testing")
    args = ap.parse_args()

    selected_scenes = DEFAULT_SCENES
    if args.scenes != "all":
        names = set(s.strip() for s in args.scenes.split(","))
        selected_scenes = [s for s in DEFAULT_SCENES if s["name"] in names]

    t_global_start = time.time()
    log(f"==================================================")
    log(f"Starting Production Pipeline for {len(selected_scenes)} scene(s)")
    log(f"Workers: {args.workers} | Limit: {args.limit}")
    log(f"==================================================")

    for s_cfg in selected_scenes:
        log(f"\n>>>>>>> Processing Scene: {s_cfg['name']} <<<<<<<")
        if not args.skip_2d:
            run_stage1_2d(s_cfg, limit=args.limit)
        if not args.skip_bev:
            run_stage2_bev(s_cfg, workers=args.workers, limit=args.limit)
        if not args.skip_depth_occ:
            run_stage3_depth_occ(s_cfg, workers=args.workers, limit=args.limit)
        run_stage4_verify(s_cfg)

    log(f"\n==================================================")
    log(f"[ALL COMPLETE] Total pipeline time: {(time.time() - t_global_start)/60:.1f} minutes")
    log(f"==================================================")


if __name__ == "__main__":
    main()
