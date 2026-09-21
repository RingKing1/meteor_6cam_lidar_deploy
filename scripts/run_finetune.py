#!/usr/bin/env python3
"""Launcher for METEOR Multi-Task Fine-Tuning on Custom Dataset.

Fine-tunes DepthSegIPMNetV52 initialized from meteor_v157.pt across:
  - BEV Lane & Road Segmentation (--seg-w 1.0)
  - 2D 21-Class Surround Segmentation (--seg2d-w 0.4 --seg2d-key seg2d21 --n-seg2d 21)
  - 3D Metric Depth Guidance (--depth-w 0.3 --freeze-depth)
  - 3D Semantic Occupancy Grid (--occ-w 0.5)
  - End-to-End Planning Trajectory (--ego-w 1.0)

Usage:
  python3 scripts/run_finetune.py [--epochs 10] [--batch 2] [--lr 1e-4] [--smoke-test]
"""
import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


def main():
    ap = argparse.ArgumentParser(description="METEOR custom fine-tuning launcher")
    ap.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    ap.add_argument("--batch", type=int, default=1, help="Batch size per GPU")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids shm limits)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "checkpoints/meteor_custom_v52"),
                    help="Checkpoint output directory")
    ap.add_argument("--smoke-test", action="store_true", help="Run 4 samples only to verify pipeline")
    args = ap.parse_args()

    meteor_dir = os.path.join(ROOT_DIR, "METEOR")
    init_ckpt = os.path.join(meteor_dir, "models/meteor_v157.pt")
    train_py = os.path.join(meteor_dir, "bevlane/train.py")
    scenes_dir = os.path.join(BASE_DIR, "scenes")
    train_list = os.path.join(BASE_DIR, "train_scenes.txt")
    val_list = os.path.join(BASE_DIR, "val_scenes.txt")

    epochs = 1 if args.smoke_test else args.epochs
    cmd = [
        sys.executable, train_py,
        "--root", scenes_dir,
        "--model", "v52",
        "--init-ckpt", init_ckpt,
        "--train-list", train_list,
        "--val-scenes-file", val_list,
        "--out", args.out,
        "--epochs", str(epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--workers", str(args.workers),
        "--trim-start", "0",
        "--trim-end", "0",
        # Multi-task loss weights
        "--seg-w", "1.0",
        "--seg2d-w", "0.4",
        "--seg2d-key", "seg2d21",
        "--n-seg2d", "21",
        "--depth-w", "0.3",
        "--freeze-depth",
        "--occ-w", "0.5",
        "--ego-w", "1.0",
        # Zero out tasks absent from custom 6-camera rig
        "--box-w", "0.0",
        "--bbox2d-w", "0.0",
        "--traj-w", "0.0",
        "--tl-w", "0.0",
        "--stat-w", "0.0",
        "--lanegraph-w", "0.0",
        "--n-cams", "8",
        "--val-batch", "1",
    ]

    if args.smoke_test:
        cmd.extend(["--limit-train", "4"])

    print("==================================================")
    print("Launching METEOR Multi-Task Fine-Tuning")
    print(f"Model Architecture: DepthSegIPMNetV52")
    print(f"Init Checkpoint:    {init_ckpt}")
    print(f"Scenes Directory:   {scenes_dir}")
    print(f"Train List:         {train_list}")
    print(f"Val List:           {val_list}")
    print(f"Output Directory:   {args.out}")
    print(f"Command:            {' '.join(cmd)}")
    print("==================================================", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = meteor_dir + ":" + env.get("PYTHONPATH", "")
    ret = subprocess.run(cmd, env=env)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
