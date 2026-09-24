#!/usr/bin/env bash
# ==============================================================================
# Post-training automation (2026-09-24):
#   wait for the running fine-tune container to exit, then
#   1) pick the best epoch by PARSING the val output already in the train logs
#      (each epoch ran the full-set val during training -- no re-evaluation);
#   2) render a video for the best checkpoint over EVERY scene, in the existing
#      videos/ style (no segmentation overlay, no ground truth).
#
# Idempotent: stages whose outputs already exist are skipped.
# All output is appended to logs/post_train_<ts>.log.
# Run detached:
#   nohup bash scripts/post_train_pipeline.sh >/dev/null 2>&1 &
# ==============================================================================
set -u

REPO_DIR="/home/nvidia/working_ppt/Postdoc_Materials/论文3/meteor_6cam_lidar_deploy"
TRAIN_CONTAINER="${TRAIN_CONTAINER:-beautiful_kepler}"
CKPT_NAME="meteor_custom_v52_st5ep_20260923"
VIDEOS_DIR="$REPO_DIR/videos"
LOG_DIR="$REPO_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/post_train_${TS}.log"

# --- container-internal paths (the parent of REPO_DIR is mounted at /work) ---
C_REPO="/work/meteor_6cam_lidar_deploy"
C_CKPT="$C_REPO/checkpoints/$CKPT_NAME"
C_VIDEOS="$C_REPO/videos"

# scenes to render (override with SCENES="a b c"); default: every scene dir
SCENES="${SCENES:-}"

mkdir -p "$LOG_DIR" "$VIDEOS_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "[post-train] started $(date)"
echo "[post-train] waiting for training container '${TRAIN_CONTAINER}' to exit ..."

# ---- 0. wait for training to finish (container gone from `docker ps`) ----
while docker ps --format '{{.Names}}' | grep -qx "${TRAIN_CONTAINER}"; do
    sleep 60
done
echo "[post-train] training container exited at $(date)"

CKPT_DIR="$REPO_DIR/checkpoints/$CKPT_NAME"
if [ ! -f "$CKPT_DIR/ep2.pt" ] || [ ! -f "$CKPT_DIR/ep3.pt" ] || [ ! -f "$CKPT_DIR/ep4.pt" ]; then
    echo "[post-train] ERROR: expected ep2/ep3/ep4 in $CKPT_DIR"; ls -la "$CKPT_DIR"
    exit 1
fi

DOCKER_IMG="meteor_training:v1"
# single mount: parent of REPO_DIR (= 论文3) -> /work. Container paths are all under it.
run_in_docker() {
    docker run --rm --gpus all --ipc=host \
        -v "$(dirname "$REPO_DIR"):/work" \
        "$DOCKER_IMG" "$@"
}

# ---- 1. pick the best epoch by parsing val output in the train logs ----
PICK_JSON="$C_CKPT/pick_best_scores.json"
PICKED="$C_CKPT/picked_best.pt"
if [ -f "$CKPT_DIR/picked_best.pt" ]; then
    echo "[post-train] picked_best.pt already exists, skipping pick"
else
    EP4_LOG="$(grep -l "\[val ep4\]" "$LOG_DIR"/train_*.log 2>/dev/null | head -1)"
    if [ -z "$EP4_LOG" ]; then
        echo "[post-train] ERROR: no train log with [val ep4] in $LOG_DIR"; exit 1
    fi
    EP4_LOG_REL="${EP4_LOG#$REPO_DIR/}"
    echo "[post-train] epoch logs -> ep2: train_ep5_fullval_20260923.log; ep3: train_ep5_resume_ep2_ipc_20260923.log; ep4: $EP4_LOG_REL"
    # ALL paths passed into the container are container-internal (no host /home/... paths)
    run_in_docker bash -c "cd '$C_REPO' && \
        python3 scripts/training_eval/pick_best.py \
            --epoch-logs '2:logs/train_ep5_fullval_20260923.log' \
                         '3:logs/train_ep5_resume_ep2_ipc_20260923.log' \
                         '4:$EP4_LOG_REL' \
            --ckpt-dir '$C_CKPT' --score-json '$PICK_JSON'"
fi
[ -f "$CKPT_DIR/picked_best.pt" ] || { echo "[post-train] ERROR: pick produced no checkpoint"; exit 1; }
echo "[post-train] best checkpoint: $CKPT_DIR/picked_best.pt"

# ---- 2. render a video for every scene (existing videos/ style, no GT) ----
if [ -z "$SCENES" ]; then
    SCENES="$(ls "$REPO_DIR/scenes")"
fi
for SCENE in $SCENES; do
    VIDEO_OUT="$VIDEOS_DIR/best_model_${SCENE}.mp4"
    if [ -f "$VIDEO_OUT" ]; then
        echo "[post-train] exists, skipping: $VIDEO_OUT"; continue
    fi
    echo "[post-train] rendering $SCENE -> $VIDEO_OUT ..."
    run_in_docker bash -c "cd '$C_REPO' && \
        python3 scripts/training_eval/best_model_video.py \
            --ckpt '$PICKED' --scene '$SCENE' \
            --out '$C_VIDEOS/best_model_${SCENE}.mp4' --stride 2 --fps 10"
done

echo "============================================================"
echo "[post-train] ALL DONE $(date)"
echo "[post-train] videos:"
ls -la "$VIDEOS_DIR"/best_model_*.mp4 2>/dev/null
