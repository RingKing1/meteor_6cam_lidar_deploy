#!/bin/bash
set -e

ENGINE="/work/meteor_6cam_lidar_deploy/engine/meteor_6cam_lidar_fp16.engine"
SCENES_DIR="/work/meteor_6cam_lidar_deploy/scenes"
VIDEOS_DIR="/work/meteor_6cam_lidar_deploy/videos"

mkdir -p "$VIDEOS_DIR"

echo "========================================================"
echo ">>> [1/2] Processing Scene: data_20260910_063822 (3D OCC Mode) ..."
echo "========================================================"
python3 /work/meteor_6cam_lidar_deploy/scripts/infer_custom.py \
  --scene "$SCENES_DIR/data_20260910_063822" \
  --engine "$ENGINE" \
  --out "$VIDEOS_DIR/data_20260910_063822.mp4" \
  --fps 10 \
  --occ-view iso \
  --occ-every 1

echo "========================================================"
echo ">>> [2/2] Processing Scene: data_20260910_064331 (3D OCC Mode) ..."
echo "========================================================"
python3 /work/meteor_6cam_lidar_deploy/scripts/infer_custom.py \
  --scene "$SCENES_DIR/data_20260910_064331" \
  --engine "$ENGINE" \
  --out "$VIDEOS_DIR/data_20260910_064331.mp4" \
  --fps 10 \
  --occ-view iso \
  --occ-every 1

echo "========================================================"
echo ">>> ALL 3D OCC VIDEOS COMPLETED SUCCESSFULLY! <<<"
echo "========================================================"
