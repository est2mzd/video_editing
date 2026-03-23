#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
ANNOTATIONS=${ANNOTATIONS:-/workspace/data/annotations.jsonl}
SCHEMA=${SCHEMA:-/workspace/configs/command_schema_ver04.yaml}
CONFIG=${CONFIG:-/workspace/configs/submit_baseline_ver03.yaml}
VIDEO_DIR=${VIDEO_DIR:-/workspace/data/videos}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/data/submission_ver04_videos}
OUTPUT_ZIP=${OUTPUT_ZIP:-/workspace/data/submission_ver04.zip}
LIMIT=${LIMIT:-0}
START_INDEX=${START_INDEX:-0}
VIDEO_CODEC=${VIDEO_CODEC:-libx264}

exec "$PYTHON_BIN" /workspace/src/submit_baseline_ver04.py \
  --annotations "$ANNOTATIONS" \
  --schema "$SCHEMA" \
  --config "$CONFIG" \
  --video-dir "$VIDEO_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --output-zip "$OUTPUT_ZIP" \
  --limit "$LIMIT" \
  --start-index "$START_INDEX" \
  --codec "$VIDEO_CODEC" \
  --overwrite
