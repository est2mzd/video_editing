#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}

"$PYTHON_BIN" src/export_parquet_videos.py \
  --parquet /workspace/data/0000.parquet \
  --annotations /workspace/data/annotations.jsonl \
  --output-dir /workspace/data/videos \
  --metadata-csv /workspace/data/metadata.csv
