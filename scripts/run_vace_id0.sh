#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

ENTRY_PY="src/run_experiment.py"
CONFIG_OVERRIDE="configs/vace_test_only_1_case.yaml"
INPUT_DIR="data/default/train/videos"
OUTPUT_ZIP="submission_vace_id0.zip"

# id=0 の instruction を CSV から取得（見つからなければ既定値）
ID0_PROMPT=$(grep '^0,' "$INPUT_DIR/test_instructions.csv" | head -1 | cut -d',' -f2- || true)
PROMPT=${ID0_PROMPT:-"Use instruction set"}

# rows=1 なので数値ソート先頭の 0.mp4 のみ処理
# strict=1 で失敗時は即停止
PYTHON_BIN=/usr/bin/python3 bash scripts/run.sh \
  "$ENTRY_PY" \
  "$CONFIG_OVERRIDE" \
  "$INPUT_DIR" \
  "$PROMPT" \
  "$OUTPUT_ZIP" \
  1 \
  1

echo "[INFO] Done: $OUTPUT_ZIP (saved under /workspace/logs/<stage>/<exp_id>/)"
