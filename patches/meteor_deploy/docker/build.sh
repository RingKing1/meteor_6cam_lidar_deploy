#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="meteor_trt10_cu124_py310:v1"
echo "=== Building ${IMAGE_NAME} ==="
docker build --network=host -t "${IMAGE_NAME}" -f "${DIR}/Dockerfile" "${DIR}"
echo "=== Build done ==="
