# ============================================================
# VACE Dockerfile (requirements-based, no conda)
# 根拠:
# - README: tested with Python 3.10.13 / CUDA 12.4 / PyTorch >= 2.5.1
# - README: pip install torch==2.5.1 torchvision==0.20.1 --index-url .../cu124
# - requirements.txt -> requirements/framework.txt
# - requirements/annotator.txt: preprocessing tools
# - build.sh は docker build の context を docker ディレクトリの "." に固定しているため、
#   VACE 側 requirements.txt を COPY せず、その中身を Dockerfile に展開している
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ==============================
# Environment variables
# ==============================
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6" \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    PIP_NO_CACHE_DIR=1

# ============================================================
# System packages
# 根拠:
# - ffmpeg: README の video inference / pipeline 実行に必要
# - build-essential, ninja-build, cmake, pkg-config: flash_attn ビルド用
# - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: opencv-python 実行用
# - python3 / python3-pip / python3-dev: README の Python>=3.10 条件を満たす実行系
# ============================================================
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    zip \
    tree \
    ffmpeg \
    ca-certificates \
    build-essential \
    pkg-config \
    ninja-build \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip setuptools wheel

# ============================================================
# PyTorch
# 根拠:
# - README の install 手順をそのまま反映
# ============================================================
RUN python3 -m pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# VACE framework requirements
# 根拠:
# - requirements.txt -> requirements/framework.txt
# - install 順は build 安定化のため 2 段に分割
# ============================================================
RUN python3 -m pip install \
    opencv-python>=4.9.0.80 \
    diffusers>=0.31.0 \
    transformers>=4.49.0 \
    tokenizers>=0.20.3 \
    accelerate>=1.1.1 \
    gradio>=5.0.0 \
    "numpy>=1.23.5,<2" \
    tqdm \
    imageio \
    easydict \
    ftfy \
    dashscope \
    imageio-ffmpeg \
    decord \
    einops \
    scikit-image \
    scikit-learn \
    pycocotools \
    timm \
    onnxruntime-gpu \
    beautifulsoup4

# requirements/framework.txt に flash_attn が含まれる
RUN python3 -m pip install flash-attn --no-build-isolation

# ============================================================
# 追加の実行補助
# 根拠:
# - VACE コード中で PIL が直接 import されているため Pillow を追加
#   (requirements には未記載だが、実装側 import に合わせる)
# ============================================================
RUN python3 -m pip install pillow

# ============================================================
# VACE annotator requirements
# 根拠:
# - README: preprocessing tools が必要なら requirements/annotator.txt を install
# - vace_pipeline.py は preprocess + inference を順に実行する想定
# ============================================================
#RUN python3 -m pip install \
#    insightface \
#    git+https://github.com/facebookresearch/sam2.git \
#    git+https://github.com/facebookresearch/segment-anything.git \
#    git+https://github.com/IDEA-Research/GroundingDINO.git \
#    git+https://github.com/xinyu1205/recognize-anything.git \
#    git+https://github.com/martin-chobanyan-sdc/RAFT.git
#
## ============================================================
## Base model runtime
## 根拠:
## - README: Wan2.1-based VACE を使う場合は wan を install
## - README: LTX-Video は Wan と conflict する可能性があるため同居させない
## ============================================================
#RUN python3 -m pip install wan@git+https://github.com/Wan-Video/Wan2.1

# README の Multi GPU acceleration 用 optional package
RUN python3 -m pip install "xfuser>=0.4.1"

# 不足分の追加インストール
RUN python3 -m pip install \
    jupyterlab \
    notebook \
    ipykernel \
    supervision \
    addict yapf pyarrow

# Codexのインストール（修正版）
RUN apt update && \
    apt install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt install -y nodejs && \
    npm install -g @openai/codex

# WAN 2.1 用
RUN apt update && \
    python3 -m pip install "huggingface_hub[cli]" modelscope

# Style Transfer 用
RUN apt update && python3 -m pip install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    opencv-python \
    pillow
RUN apt update && python3 -m pip install -U peft

# YOLOv8 用
RUN apt update && python3 -m pip install ultralytics

# Hugging Face Hub CLI 用
RUN apt update && python3 -m pip install huggingface_hub git-lfs

# Google Drive からのダウンロード用
RUN apt update && python3 -m pip install gdown

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspace/third_party/VACE:$PYTHONPATH
ENV PYTHONPATH=/workspace/third_party/GroundingDINO:$PYTHONPATH
ENV PIP_NO_BUILD_ISOLATION=1
ENV PATH=$HOME/.local/bin:$PATH

# ==============================
# User settings
# ==============================
ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

RUN if ! getent group ${GROUP_ID} >/dev/null; then groupadd -g ${GROUP_ID} ${USER_NAME}; fi && \
    useradd -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ==============================
# Permissions / runtime dirs
# ==============================
RUN mkdir -p /workspace /workspace/.cache/huggingface && \
    chown -R ${USER_NAME}:${GROUP_ID} /workspace

# ホスト bind mount 後に VACE を import できるようにする
#ENV PYTHONPATH=/workspace/third_party/VACE:${PYTHONPATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

USER ${USER_NAME}

# RUN echo "export PYTHONPATH=/workspace/third_party/VACE:\$PYTHONPATH" >> /home/${USER_NAME}/.bashrc && \
#     echo "export HF_HOME=/workspace/.cache/huggingface" >> /home/${USER_NAME}/.bashrc && \
#     echo "export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub" >> /home/${USER_NAME}/.bashrc && \
#     echo "export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers" >> /home/${USER_NAME}/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
