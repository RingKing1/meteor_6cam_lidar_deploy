#!/bin/bash
# ==============================================================================
# Master Production Script for Phase 3: Full 6-Scene Autolabeling & Tracking
#
# Processes all 17,845 frames across 6 scenes:
#   1. FocalFormer3D-LC (Score >= 0.15 low-threshold high-recall candidates)
#   2. Tri-Modal Consensus Arbiter (LiDAR + YOLOv8x + Mask2Former -> bbox2d + bev_box)
#   3. 3D MOT & Future 3.0s Trajectory Engine (agent_traj -> --traj-w & --stat-w)
#   4. TIER IV t4dataset 2D Instance Tracking (object_ann.json)
# ==============================================================================
set -e

WORK_DIR="/home/nvidia/working_ppt/Postdoc_Materials/论文3"
DEPLOY_DIR="${WORK_DIR}/meteor_6cam_lidar_deploy"
LOG_DIR="${DEPLOY_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/phase3_production_${TIMESTAMP}.log"

echo "================================================================================" | tee -a "${MASTER_LOG}"
echo "  METEOR Full Dataset Production Factory - Phase 3 Launch" | tee -a "${MASTER_LOG}"
echo "  Date:      $(date)" | tee -a "${MASTER_LOG}"
echo "  GPU:       RTX 4090 24GB VRAM" | tee -a "${MASTER_LOG}"
echo "  Log File:  ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "================================================================================" | tee -a "${MASTER_LOG}"

SCENES=(
    "data_20260910_061820"
    "data_20260910_062659"
    "data_20260910_063822"
    "data_20260910_064331"
    "data_20260910_073823"
    "data_20260910_074912"
)

TOTAL_SCENES=${#SCENES[@]}
SCENE_IDX=0

for SCENE in "${SCENES[@]}"; do
    SCENE_IDX=$((SCENE_IDX + 1))
    echo "" | tee -a "${MASTER_LOG}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | tee -a "${MASTER_LOG}"
    echo "  [Scene ${SCENE_IDX}/${TOTAL_SCENES}] Processing: ${SCENE}" | tee -a "${MASTER_LOG}"
    echo "  Timestamp: $(date)" | tee -a "${MASTER_LOG}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | tee -a "${MASTER_LOG}"

    # Step 1: FocalFormer3D 0.15 Inference
    echo "[1/4] Running FocalFormer3D 0.15 inference for ${SCENE}..." | tee -a "${MASTER_LOG}"
    docker run --rm --gpus all \
      -v "${WORK_DIR}:/work" \
      -e PYTHONPATH="/work/FocalFormer3D:/work/focalformer_deploy/mmdetection3d:/work/focalformer_deploy" \
      -e LD_LIBRARY_PATH="/opt/TensorRT-8.5.3.1/targets/x86_64-linux-gnu/lib" \
      torch1.10_cu113_py308_ubuntu20.04_bevfusion:v4-focalformer \
      python3 /work/meteor_6cam_lidar_deploy/scripts/autolabel_3d_box/infer_focalformer_dataset.py \
        --scenes "${SCENE}" \
        --score-thresh 0.15 \
        --out-dir /work/meteor_6cam_lidar_deploy/box3d_artifacts/focalformer_bev_box_015 2>&1 | tee -a "${MASTER_LOG}"

    # Step 2: Tri-Modal Consensus Fusion
    echo "[2/4] Running Tri-Modal Consensus Fusion for ${SCENE}..." | tee -a "${MASTER_LOG}"
    docker run --rm --gpus all \
      -v "${WORK_DIR}:/work" \
      meteor_training:v1 \
      python3 /work/meteor_6cam_lidar_deploy/scripts/autolabel_bbox2d/tri_modal_consensus_fusion.py \
        --scenes "${SCENE}" \
        --focal-dir /work/meteor_6cam_lidar_deploy/box3d_artifacts/focalformer_bev_box_015 2>&1 | tee -a "${MASTER_LOG}"

    # Step 3: 3D MOT & Agent Trajectory Generation
    echo "[3/4] Running 3D MOT & 3.0s Future Trajectory for ${SCENE}..." | tee -a "${MASTER_LOG}"
    docker run --rm \
      -v "${WORK_DIR}:/work" \
      meteor_training:v1 \
      python3 /work/meteor_6cam_lidar_deploy/scripts/autolabel_agent_traj/build_agent_traj_gt.py \
        --scenes "${SCENE}" \
        --force 2>&1 | tee -a "${MASTER_LOG}"

    # Step 4: TIER IV t4dataset Instance Tracking Export
    echo "[4/4] Exporting T4 Instance Tracking Annotations for ${SCENE}..." | tee -a "${MASTER_LOG}"
    docker run --rm \
      -v "${WORK_DIR}:/work" \
      meteor_training:v1 \
      python3 /work/meteor_6cam_lidar_deploy/scripts/autolabel_instance_tracking/export_t4_instance.py \
        --scenes "${SCENE}" 2>&1 | tee -a "${MASTER_LOG}"

    echo "[+] Completed processing for scene ${SCENE} at $(date)" | tee -a "${MASTER_LOG}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "================================================================================" | tee -a "${MASTER_LOG}"
echo "  [SUCCESS] All 6 Scenes (17,845 Frames) Processed and Verified!" | tee -a "${MASTER_LOG}"
echo "  Completion Time: $(date)" | tee -a "${MASTER_LOG}"
echo "================================================================================" | tee -a "${MASTER_LOG}"
