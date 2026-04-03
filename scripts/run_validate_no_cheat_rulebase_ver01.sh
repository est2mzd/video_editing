#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/validate_no_cheat_rulebase_ver01_${TS}.log"

CMD=(
  python3 "$ROOT/src/parse/validate_no_cheat_rulebase_ver01.py"
)

echo "[INFO] Running no-cheat validation v01" | tee "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Command: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Done" | tee -a "$LOG_FILE"
