#!/bin/bash
set -e

FILE_PATH=$(realpath "$0")
DIR_PATH=$(dirname "$FILE_PATH")
PARENT_DIR=$(dirname "$DIR_PATH")
cd "$PARENT_DIR"

TARGET_PATH="/workspace/weights/groundingdino_swint_ogc.pth"
URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$(dirname "$TARGET_PATH")"

if [[ -f "$TARGET_PATH" && "$FORCE" -ne 1 ]]; then
  echo "[INFO] Checkpoint already exists: $TARGET_PATH"
  echo "[INFO] Use --force to re-download."
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  echo "[INFO] Downloading with curl..."
  curl -L --fail --retry 3 --retry-delay 2 -o "$TARGET_PATH" "$URL"
elif command -v wget >/dev/null 2>&1; then
  echo "[INFO] Downloading with wget..."
  wget -O "$TARGET_PATH" "$URL"
else
  echo "[ERROR] Neither curl nor wget is available."
  exit 1
fi

if [[ ! -s "$TARGET_PATH" ]]; then
  echo "[ERROR] Download failed or file is empty: $TARGET_PATH"
  exit 1
fi

echo "[INFO] Download completed: $TARGET_PATH"
ls -lh "$TARGET_PATH"
