#!/usr/bin/env python3
"""Batch 2D Panoptic Segmentation & Surface Mask Extraction for METEOR.

Runs Mask2Former (Mapillary Vistas) in batches of 6 cameras per frame.
Produces:
  - seg2d21/<fi:04d>.npz: uint8 [6, 108, 192] 21-class segmentation with coverage-based pooling
  - surface_mask/<fi:04d>.npz: uint8 [6, 432, 768] 9 BEV road surface classes
  - Updates manifest.json with "seg2d21" keys

Resumable: skips frames whose outputs already exist.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Downsampled resolution (stride 4 of 768x432)
SW, SH = 192, 108
THIN_IDS = (9, 10, 20, 8, 13)    # light, sign, pole, marking, lane
THIN_FRAC = 0.12

# Mapillary Vistas (65 classes) -> METEOR BEV 9-class mapping
MAPILLARY_TO_BEV9 = {
    13: 1,  # Road -> road
    7:  1,  # Bike Lane -> road
    14: 1,  # Service Lane -> road
    15: 2,  # Sidewalk -> sidewalk
    2:  2,  # Curb -> sidewalk
    9:  2,  # Curb Cut -> sidewalk
    11: 2,  # Pedestrian Area -> sidewalk
    8:  3,  # Crosswalk - Plain -> crosswalk
    23: 3,  # Lane Marking - Crosswalk -> crosswalk
    24: 4,  # Lane Marking - General -> laneline
    4:  6,  # Guard Rail -> road_edge
    5:  6,  # Barrier -> road_edge
    10: 8,  # Parking -> parking_lot
}

# Mapillary Vistas (65 classes) -> METEOR 21-class 2D mapping
MAPILLARY_TO_SEG21 = {
    55: 2, 56: 2, 64: 2, # Car, Caravan, Ego Vehicle -> car
    61: 3, 60: 3,        # Truck, Trailer -> truck
    54: 4,               # Bus -> bus
    57: 5,               # Motorcycle -> moto
    52: 6, 20: 6,        # Bicycle, Bicyclist -> bicycle
    19: 7, 21: 7, 22: 7, # Person, Motorcyclist, Other Rider -> ped
    24: 8, 41: 8, 43: 8, # Lane Marking - General, Manhole, Pothole -> marking
    48: 9,               # Traffic Light -> light
    49: 10, 50: 10, 46: 10, # Traffic Sign -> sign
    13: 11, 7: 11, 14: 11,  # Road, Bike Lane, Service Lane -> road
    15: 12, 2: 12, 9: 12, 11: 12, # Sidewalk, Curb -> sidewalk
    24: 13,              # Lane Marking -> lane
    8: 14, 23: 14,       # Crosswalk -> crosswalk
    6: 16,               # Wall -> wall
    17: 17, 16: 17, 18: 17, # Building, Bridge, Tunnel -> building
    30: 18, 29: 18, 25: 18, # Vegetation, Terrain, Mountain -> vegetation
    27: 19,              # Sky -> sky
    45: 20, 44: 20, 47: 20, # Pole, Street Light, Utility Pole -> pole
}


def downsample_seg21(seg21_full):
    """Downsample 768x432 21-class map to 192x108 with coverage-based pooling."""
    small = cv2.resize(seg21_full, (SW, SH), interpolation=cv2.INTER_NEAREST)
    keep = small != 255
    for tid in THIN_IDS:
        frac = cv2.resize((seg21_full == tid).astype(np.float32), (SW, SH),
                          interpolation=cv2.INTER_AREA)
        small[(frac > THIN_FRAC) & keep] = tid
    return small


def process_scene(scene_dir, model_name, device="cuda", limit=0):
    mf_path = os.path.join(scene_dir, "manifest.json")
    if not os.path.exists(mf_path):
        print(f"[ERROR] {mf_path} not found")
        return

    man = json.load(open(mf_path))
    frames = man["frames"]
    if limit > 0:
        frames = frames[:limit]
    total_frames = len(frames)

    seg_dir = os.path.join(scene_dir, "seg2d21")
    mask_dir = os.path.join(scene_dir, "surface_mask")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Check pending frames
    pending = []
    for fr in frames:
        fi = fr["frame"]
        p_seg = os.path.join(seg_dir, f"{fi:04d}.npz")
        p_mask = os.path.join(mask_dir, f"{fi:04d}.npz")
        if not (os.path.exists(p_seg) and os.path.exists(p_mask)):
            pending.append(fr)

    print(f"[*] Scene: {os.path.basename(scene_dir)} | Total frames: {total_frames} | Pending: {len(pending)}", flush=True)
    if not pending:
        print("[+] All frames already processed! Updating manifest...", flush=True)
        for fr in man["frames"]:
            fr["seg2d21"] = f"seg2d21/{fr['frame']:04d}.npz"
        json.dump(man, open(mf_path, "w"))
        return

    # Load Model (try local cache first for fast offline startup)
    print(f"[*] Loading model {model_name} on {device} (fp16) ...", flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print("[+] Model loaded!", flush=True)

    t_start = time.time()
    n_done = 0

    for idx, fr in enumerate(pending):
        fi = fr["frame"]
        p_seg = os.path.join(seg_dir, f"{fi:04d}.npz")
        p_mask = os.path.join(mask_dir, f"{fi:04d}.npz")

        # Load 6 camera images
        imgs_pil = []
        for c in CAMS:
            rel = fr["imgs"].get(c)
            img_bgr = cv2.imread(os.path.join(scene_dir, rel))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            imgs_pil.append(Image.fromarray(img_rgb))

        # Batch inference across 6 cameras
        inputs = processor(images=imgs_pil, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [(432, 768)] * len(CAMS)
        results = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes, label_ids_to_fuse=[]
        )

        bev9_stack = np.zeros((len(CAMS), 432, 768), dtype=np.uint8)
        seg21_stack = np.zeros((len(CAMS), SH, SW), dtype=np.uint8)

        for ci, res in enumerate(results):
            panoptic_map = res["segmentation"].cpu().numpy()
            segments_info = res["segments_info"]
            id_to_cat = {s["id"]: s["label_id"] for s in segments_info}

            cat_map = np.zeros_like(panoptic_map, dtype=np.int32)
            for seg_id, cat_id in id_to_cat.items():
                cat_map[panoptic_map == seg_id] = cat_id

            # BEV 9 classes
            bev9 = np.zeros((432, 768), dtype=np.uint8)
            for m_id, b_id in MAPILLARY_TO_BEV9.items():
                bev9[cat_map == m_id] = b_id
            bev9_stack[ci] = bev9

            # 2D 21 classes
            seg21_full = np.zeros((432, 768), dtype=np.uint8)
            for m_id, s_id in MAPILLARY_TO_SEG21.items():
                seg21_full[cat_map == m_id] = s_id

            seg21_stack[ci] = downsample_seg21(seg21_full)

        # Save compressed outputs
        np.savez_compressed(p_seg, seg=seg21_stack)
        np.savez_compressed(p_mask, mask=bev9_stack)
        fr["seg2d21"] = f"seg2d21/{fi:04d}.npz"
        n_done += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(pending):
            elapsed = time.time() - t_start
            fps = (idx + 1) / elapsed
            remain_sec = (len(pending) - (idx + 1)) / max(fps, 1e-4)
            print(f"  [{idx+1:4d}/{len(pending):4d}] ({100*(idx+1)/len(pending):5.1f}%) | "
                  f"Speed: {fps:.2f} frames/s (6cams/fr) | "
                  f"ETA: {remain_sec/60:.1f} min", flush=True)

    # Update manifest
    for fr in man["frames"]:
        fr["seg2d21"] = f"seg2d21/{fr['frame']:04d}.npz"
    json.dump(man, open(mf_path, "w"))
    print(f"[+] Successfully finished {scene_dir} in {(time.time()-t_start)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scenes/data_20260910_063822,scenes/data_20260910_064331",
                    help="Comma-separated paths to scenes")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-panoptic")
    ap.add_argument("--limit", type=int, default=0, help="Limit frames for testing")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for s in scenes:
        process_scene(s, args.model, limit=args.limit)


if __name__ == "__main__":
    main()
