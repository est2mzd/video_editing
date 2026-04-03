#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

PARSER_FILE="${PARSER_FILE:-$ROOT/src/parse/instruction_parser_v3_rulebase_trial013_singlefile.py}"
VALIDATOR_FILE="${VALIDATOR_FILE:-$ROOT/src/parse/other_trials/validate_rulebase_single_trial.py}"
TRIAL_NAME="${TRIAL_NAME:-trial_rulebase_013_singlefile_check}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TRIAL_NAME}_${TS}.log"

CMD=(
  python3 "$VALIDATOR_FILE"
  --parser-file "$PARSER_FILE"
  --trial-name "$TRIAL_NAME"
  --eval-batch-size "$EVAL_BATCH_SIZE"
)

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  CMD+=(--show-progress)
fi

echo "[INFO] Running rulebase trial013 singlefile check" | tee "$LOG_FILE"
echo "[INFO] Trial: $TRIAL_NAME" | tee -a "$LOG_FILE"
echo "[INFO] Parser: $PARSER_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Validator: $VALIDATOR_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Command: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[INFO] Done" | tee -a "$LOG_FILE"