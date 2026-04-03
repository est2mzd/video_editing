#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

PARSER_FILE="${PARSER_FILE:-$ROOT/src/parse/prototype_instruction_parser_v3_improved.py}"
TRIAL_NAME="${TRIAL_NAME:-trial_001}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TRIAL_NAME}_${TS}.log"

CMD=(
  python3 "$ROOT/src/parse/validate_rulebase_single_trial.py"
  --parser-file "$PARSER_FILE"
  --trial-name "$TRIAL_NAME"
)

echo "[INFO] Running single trial" | tee "$LOG_FILE"
echo "[INFO] Trial: $TRIAL_NAME" | tee -a "$LOG_FILE"
echo "[INFO] Parser: $PARSER_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[INFO] Done" | tee -a "$LOG_FILE"
