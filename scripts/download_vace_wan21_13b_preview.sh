#!/bin/bash
set -e

FILE_PATH=$(realpath "$0")
DIR_PATH=$(dirname "$FILE_PATH")
PARENT_DIR=$(dirname "$DIR_PATH")
cd "$PARENT_DIR"

PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
MODEL_DIR="third_party/VACE/models/VACE-Wan2.1-1.3B-Preview"
REPO_ID="ali-vilab/VACE-Wan2.1-1.3B-Preview"

echo "[INFO] Downloading model: $REPO_ID"
echo "[INFO] Target dir: $MODEL_DIR"

"$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO_ID', local_dir='$MODEL_DIR', local_dir_use_symlinks=False)"

echo "[INFO] Download completed"
ls -lh "$MODEL_DIR" | sed -n '1,20p'
