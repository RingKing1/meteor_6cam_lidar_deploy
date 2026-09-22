#!/bin/bash
set -e

CONTAINER_NAME="meteor_full_5ep"
OUT_DIR="/work/meteor_6cam_lidar_deploy/checkpoints/meteor_custom_v52_full5ep"
LOG_FILE="/work/meteor_6cam_lidar_deploy/train_full_5ep.log"

# Remove existing stopped container with the same name if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[Info] Removing existing container ${CONTAINER_NAME}..."
    docker rm -f ${CONTAINER_NAME}
fi

echo "=================================================="
echo "Starting METEOR Multi-Task 5-Epoch Fine-Tuning"
echo "Date: $(date)"
echo "GPU:  RTX 4090 (24GB VRAM)"
echo "Image: meteor_training:v1"
echo "Output Directory: ${OUT_DIR}"
echo "Log File:         ${LOG_FILE}"
echo "=================================================="

# Run in background via detached Docker container
docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --ipc=host \
  --net=host \
  --restart unless-stopped \
  -v /home/nvidia/working_ppt/Postdoc_Materials/论文3:/work \
  -w /work/meteor_6cam_lidar_deploy \
  meteor_training:v1 \
  bash -c "python3 /work/METEOR/bevlane/train.py \
    --root /work/meteor_6cam_lidar_deploy/scenes \
    --model v52 \
    --init-ckpt /work/METEOR/models/meteor_v157.pt \
    --train-list /work/meteor_6cam_lidar_deploy/train_scenes.txt \
    --val-scenes-file /work/meteor_6cam_lidar_deploy/val_scenes.txt \
    --out ${OUT_DIR} \
    --epochs 5 \
    --batch 1 \
    --lr 0.0001 \
    --workers 2 \
    --trim-start 0 \
    --trim-end 0 \
    --seg-w 1.0 \
    --seg2d-w 0.4 \
    --seg2d-key seg2d21 \
    --n-seg2d 21 \
    --depth-w 0.3 \
    --freeze-depth \
    --occ-w 0.5 \
    --ego-w 1.0 \
    --box-w 1.0 \
    --tl-w 0.5 \
    --stat-w 0.0 \
    --bbox2d-w 0.0 \
    --traj-w 0.0 \
    --lanegraph-w 0.0 \
    --n-cams 8 \
    --val-batch 1 2>&1 | tee ${LOG_FILE}"

echo "[Success] Docker container ${CONTAINER_NAME} launched in background."
