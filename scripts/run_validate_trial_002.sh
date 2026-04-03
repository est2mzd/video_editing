#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TRIAL_NAME="trial_002_v3_preserve_fix"
LOG_FILE="$LOG_DIR/${TRIAL_NAME}_${TS}.log"

CMD=(
  python3 "$ROOT/src/parse/validate_rulebase_single_trial.py"
  --parser-file "$ROOT/src/parse/prototype_instruction_parser_v3_improved_trial002.py"
  --trial-name "$TRIAL_NAME"
)

echo "[INFO] Running ${TRIAL_NAME}" | tee "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Command: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[INFO] Done" | tee -a "$LOG_FILE"
