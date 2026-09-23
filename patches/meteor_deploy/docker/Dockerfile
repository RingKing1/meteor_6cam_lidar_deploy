# METEOR TensorRT fp16 引擎构建 + 12 任务演示镜像
# CUDA 12.4 runtime + Python 3.10 + TensorRT 10 (pip) + opencv
# GPU 调用走 cuda-bindings (预编译 wheel)，deploy/runtime.py 内置该回退
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY=

# ---------- 1. 国内 apt 源 (清华 Ubuntu 22.04 Jammy) ----------
COPY sources.list.jammy /etc/apt/sources.list

# ---------- 2. 系统依赖 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git curl ca-certificates \
    libgl1 libglib2.0-0 ffmpeg \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---------- 3. 国内 pip 源 ----------
COPY pip.conf /etc/pip.conf
RUN python -m pip install --no-cache-dir --upgrade pip

# ---------- 4. 主体依赖 (稳定、体大，独立成层便于缓存) ----------
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "tensorrt==10.3.0" \
    "onnx==1.16.2" \
    "onnxruntime==1.19.2" \
    "opencv-python-headless==4.10.0.84"

# ---------- 5. CUDA 运行时绑定 (替代 pycuda，免本地编译) ----------
RUN pip install --no-cache-dir "cuda-bindings==12.8.0"

WORKDIR /workspace/METEOR
CMD ["/bin/bash"]
