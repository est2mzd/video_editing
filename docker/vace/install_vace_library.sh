#!/bin/bash
set -e

# =========================================
# 0. 前提
# =========================================
cd /workspace

export PIP_NO_BUILD_ISOLATION=1

# =========================================
# 1. torch（最重要）
# =========================================
python3 -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------
# 意図：
# ・全依存の基盤
# ・再インストール防止
# -----------------------------------------

# =========================================
# 2. core（framework）
# =========================================
python3 -m pip install \
  opencv-python diffusers transformers tokenizers \
  accelerate gradio numpy tqdm imageio easydict ftfy \
  dashscope imageio-ffmpeg decord einops scikit-image \
  scikit-learn pycocotools timm onnxruntime-gpu beautifulsoup4

python3 -m pip install flash-attn --no-build-isolation

python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------------------
# 意図：
# ・VACE本体 + annotatorの共通基盤
# -----------------------------------------

# =========================================
# 3. Wan backend
# =========================================
cd /workspace/third_party/Wan2.1
python3 -m pip install -e .

# -----------------------------------------
# 意図：
# ・ローカルcloneをそのまま使う
# ・git+installより安定
# -----------------------------------------

# =========================================
# 4. annotator
# =========================================

# SAM
cd /workspace/third_party/segment-anything
python3 -m pip install -e .

# SAM2
cd /workspace/third_party/sam2
python3 -m pip install -e .

# RAFT
cd /workspace/third_party/RAFT
python3 -m pip install -e .

# recognize-anything
cd /workspace/third_party/recognize-anything
python3 -m pip install -e .

# GroundingDINO（最後）
cd /workspace/third_party/GroundingDINO
python3 setup.py build_ext --inplace
# -----------------------------------------
# 意図：
# ・editable install（-e）で依存衝突回避
# ・GroundingDINOは最後（壊れやすい）
# -----------------------------------------

cd /workspace/third_party

# ① clone（バージョン固定）
# git clone https://github.com/Lightricks/LTX-Video.git
cd LTX-Video
# git checkout ltx-video-0.9.1

# ② ローカルインストール
sudo python3 -m pip install -e . --no-deps --no-build-isolation

# ③ 必須依存のみ個別に
python3 -m pip install sentencepiece


# =========================================
# 5. VACE
# =========================================
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/third_party/VACE:$PYTHONPATH

# -----------------------------------------
# 意図：
# ・install不要（コード参照のみ）
# -----------------------------------------

# =========================================
# 6. XMem
# =========================================
cd /workspace/third_party/XMem

python3 -m pip install -r requirements.txt
export PYTHONPATH=/workspace/third_party/XMem:$PYTHONPATH


# =========================================
# 7. Style transfer
# =========================================
# python3 -m pip install opencv-python
mkdir -p /workspace/weights/lora/
wget -O /workspace/weights/lora/anime.safetensors \
https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors