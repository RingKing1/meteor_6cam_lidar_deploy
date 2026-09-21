#!/bin/bash
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="meteor_trt10_cu124_py310:v1"
docker run --gpus all --rm -it \
  --ipc=host \
  --net=host \
  -v "${REPO_DIR}:/workspace/METEOR" \
  -w /workspace/METEOR \
  "${IMAGE_NAME}" "$@"
