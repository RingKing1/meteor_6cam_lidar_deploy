#!/usr/bin/env python3
"""2D Panoptic Segmentation Selection & Diagnostic Script for METEOR.

Evaluates pre-trained panoptic segmentation models (e.g. Mask2Former trained on
Mapillary Vistas or Cityscapes) on the custom 6-camera surround setup.

Outputs:
  - 2D Segmentation maps mapped to METEOR's 21-class taxonomy
  - Road surface masks mapped to METEOR's 9-class BEV taxonomy (lane, road, crosswalk, sidewalk, etc.)
  - Diagnostic side-by-side visualizations in diagnostics/panoptic_test/
  - Per-camera inference latency benchmarks on RTX 4090

Usage:
  python3 scripts/test_2d_panoptic.py \
    --scene scenes/data_20260910_063822 \
    --frame 0 \
    --model facebook/mask2former-swin-large-mapillary-vistas-panoptic
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Camera order matching METEOR rig
CAMS = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# -------------------------------------------------------------------------
# METEOR Taxonomy Definitions
# -------------------------------------------------------------------------
# 1) BEV 9-class Palette (BGR)
# 0: unlabeled, 1: road, 2: sidewalk, 3: crosswalk, 4: laneline, 5: stopline, 6: road_edge, 7: marking, 8: parking
BEV_CLASSES = [
    "unlabeled", "road", "sidewalk", "crosswalk", "laneline",
    "stopline", "road_edge", "marking", "parking_lot"
]
BEV_COLORS_BGR = np.array([
    [0, 0, 0],         # 0: unlabeled
    [100, 100, 100],   # 1: road (gray)
    [160, 90, 140],    # 2: sidewalk (purple)
    [200, 200, 0],     # 3: crosswalk (cyan/yellow)
    [255, 255, 255],   # 4: laneline (white)
    [40, 40, 255],     # 5: stopline (red)
    [0, 140, 255],     # 6: road_edge (orange)
    [60, 220, 240],    # 7: marking (yellow)
    [140, 60, 40],     # 8: parking_lot (blue-ish)
], dtype=np.uint8)

# 2) METEOR 21-class 2D Segmentation Palette (BGR)
# 0: bg, 1: misc, 2: car, 3: truck, 4: bus, 5: moto, 6: bicycle, 7: ped, 8: marking,
# 9: light, 10: sign, 11: road, 12: sidewalk, 13: lane, 14: crosswalk, 15: unused,
# 16: wall, 17: building, 18: vegetation, 19: sky, 20: pole
SEG21_COLORS_BGR = np.array([
    [0, 0, 0],         # 0: bg
    [110, 110, 110],   # 1: misc
    [142, 0, 0],       # 2: car (dark blue)
    [70, 0, 0],        # 3: truck
    [100, 60, 0],      # 4: bus
    [230, 0, 0],       # 5: moto
    [32, 11, 119],     # 6: bicycle
    [60, 20, 220],     # 7: ped (red)
    [0, 255, 255],     # 8: marking (yellow)
    [30, 170, 250],    # 9: light (orange)
    [0, 220, 220],     # 10: sign
    [128, 64, 128],    # 11: road (purple)
    [232, 35, 244],    # 12: sidewalk (pink)
    [255, 255, 255],   # 13: lane (white)
    [100, 150, 200],   # 14: crosswalk
    [0, 0, 0],         # 15: unused
    [156, 102, 102],   # 16: wall
    [70, 70, 70],      # 17: building
    [35, 142, 107],    # 18: vegetation (green)
    [180, 130, 70],    # 19: sky (light blue)
    [153, 153, 153],   # 20: pole
], dtype=np.uint8)

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


def load_model(model_name: str, device: str = "cuda"):
    """Load Mask2Former or OneFormer model and image processor from HuggingFace."""
    try:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError:
        print("[ERROR] Required packages not found. Please install inside the environment:")
        print("        pip install torch torchvision transformers")
        sys.exit(1)

    print(f"[*] Loading model: {model_name} on {device} ...", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("[+] Model loaded successfully!", flush=True)
    return processor, model


def run_inference(processor, model, bgr_img, device="cuda"):
    """Run panoptic inference on a single BGR image."""
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_ms = (time.time() - t0) * 1000

    # Post-process to original image dimensions
    target_sizes = [bgr_img.shape[:2]]
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=target_sizes, label_ids_to_fuse=[]
    )[0]

    panoptic_map = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    # Remap segmentation IDs to category IDs
    id_to_cat = {s["id"]: s["label_id"] for s in segments_info}
    cat_map = np.zeros_like(panoptic_map, dtype=np.int32)
    for seg_id, cat_id in id_to_cat.items():
        cat_map[panoptic_map == seg_id] = cat_id

    return cat_map, segments_info, inference_ms


def convert_to_meteor_classes(cat_map):
    """Convert Mapillary 65-class output into METEOR BEV 9-class and 2D 21-class."""
    h, w = cat_map.shape
    bev9 = np.zeros((h, w), dtype=np.uint8)
    seg21 = np.zeros((h, w), dtype=np.uint8)

    for mapillary_id, bev_id in MAPILLARY_TO_BEV9.items():
        bev9[cat_map == mapillary_id] = bev_id

    for mapillary_id, s21_id in MAPILLARY_TO_SEG21.items():
        seg21[cat_map == mapillary_id] = s21_id

    return bev9, seg21


def render_diagnostic_card(orig_bgr, bev9, seg21, cam_name, latency_ms):
    """Render a 3-column diagnostic visual card: Original | BEV Road Classes | 2D Seg21."""
    h, w = orig_bgr.shape[:2]

    # Render BEV 9-class color mask
    bev_color = BEV_COLORS_BGR[bev9]
    bev_overlay = cv2.addWeighted(orig_bgr, 0.4, bev_color, 0.6, 0)

    # Render 2D 21-class color mask
    seg21_color = SEG21_COLORS_BGR[np.clip(seg21, 0, 20)]
    seg21_overlay = cv2.addWeighted(orig_bgr, 0.4, seg21_color, 0.6, 0)

    # Put headers
    header_h = 36
    canvas = np.zeros((h + header_h, w * 3, 3), dtype=np.uint8)
    canvas[header_h:, 0:w] = orig_bgr
    canvas[header_h:, w:2*w] = bev_overlay
    canvas[header_h:, 2*w:3*w] = seg21_overlay

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{cam_name} (Raw Input)", (15, 24), font, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"BEV Surface Classes (Lane/Road/Crosswalk)", (w + 15, 24), font, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"2D Multi-Task (21 Classes) | {latency_ms:.1f}ms", (2 * w + 15, 24), font, 0.7, (0, 255, 0), 2)

    return canvas


def main():
    ap = argparse.ArgumentParser(description="Test 2D Panoptic Segmentation on Custom METEOR Scenes")
    ap.add_argument("--scene", default="scenes/data_20260910_063822",
                    help="Path to converted scene directory")
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame index to test")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-panoptic",
                    help="HuggingFace model identifier")
    ap.add_argument("--device", default="cuda",
                    help="Inference device (cuda or cpu)")
    ap.add_argument("--out-dir", default="diagnostics/panoptic_test",
                    help="Output directory for visual comparisons")
    args = ap.parse_args()

    # Locate manifest
    mf_path = os.path.join(args.scene, "manifest.json")
    if not os.path.exists(mf_path):
        print(f"[ERROR] manifest.json not found at {mf_path}")
        sys.exit(1)

    mf = json.load(open(mf_path))
    if args.frame >= len(mf["frames"]):
        print(f"[ERROR] Frame {args.frame} exceeds scene frame count ({len(mf['frames'])})")
        sys.exit(1)

    frame_info = mf["frames"][args.frame]
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize model
    processor, model = load_model(args.model, device=args.device)

    print(f"\n[*] Evaluating Frame {args.frame:04d} across all 6 surround cameras...")
    total_time = 0.0
    latencies = []

    for cam_idx, cam_name in enumerate(CAMS):
        rel_img = frame_info["imgs"].get(cam_name)
        if not rel_img:
            print(f"[WARN] Camera {cam_name} not found in manifest frame {args.frame}")
            continue

        img_path = os.path.join(args.scene, rel_img)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to read {img_path}")
            continue

        # Inference
        cat_map, segments, lat_ms = run_inference(processor, model, img, device=args.device)
        bev9, seg21 = convert_to_meteor_classes(cat_map)
        latencies.append(lat_ms)
        total_time += lat_ms

        # Check detected road features
        n_lanepx = np.sum(bev9 == 4)
        n_crosswalk = np.sum(bev9 == 3)
        n_road = np.sum(bev9 == 1)
        print(f"  [{cam_idx+1}/6] {cam_name:18s} | Latency: {lat_ms:5.1f} ms | "
              f"Road px: {n_road:6d} | Lane px: {n_lanepx:5d} | Crosswalk: {n_crosswalk:4d}")

        # Render diagnostic image
        card = render_diagnostic_card(img, bev9, seg21, cam_name, lat_ms)
        out_file = os.path.join(args.out_dir, f"frame_{args.frame:04d}_{cam_name}.jpg")
        cv2.imwrite(out_file, card)

    print(f"\n[+] Diagnostic results saved to: {args.out_dir}/")
    print(f"[+] Total 6-cam inference time: {total_time:.1f} ms (Mean: {np.mean(latencies):.1f} ms/cam)")
    print(f"[+] Theoretical throughput: {1000.0 / total_time:.1f} FPS (Surround 6-Cam Full System)")


if __name__ == "__main__":
    main()
