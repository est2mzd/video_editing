#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/test"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/gt_unit_test_02_${TS}.log"

FRAME_STRIDE="${FRAME_STRIDE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/logs/test/notebook_match_v3}"

CMD=(
  python3 "$ROOT/src/test/gt_unit_test_02.py"
  --run-default-scenarios
  --frame-stride "$FRAME_STRIDE"
  --output-dir "$OUTPUT_DIR"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] Running: ${CMD[*]}" | tee "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Log saved: $LOG_FILE" | tee -a "$LOG_FILE"
