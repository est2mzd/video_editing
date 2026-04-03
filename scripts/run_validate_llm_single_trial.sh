#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

PARSER_FILE="${PARSER_FILE:-$ROOT/src/parse/prototype_instruction_parser_v3_llm_trial001.py}"
TRIAL_NAME="${TRIAL_NAME:-llm_trial_001}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TRIAL_NAME}_${TS}.log"

CMD=(
  /usr/bin/python "$ROOT/src/parse/validate_rulebase_single_trial.py"
  --parser-file "$PARSER_FILE"
  --trial-name "$TRIAL_NAME"
  --eval-batch-size "$EVAL_BATCH_SIZE"
)

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  CMD+=(--show-progress)
fi

echo "[INFO] Running single LLM trial" | tee "$LOG_FILE"
echo "[INFO] Trial: $TRIAL_NAME" | tee -a "$LOG_FILE"
echo "[INFO] Parser: $PARSER_FILE" | tee -a "$LOG_FILE"
echo "[INFO] Eval batch size: $EVAL_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "[INFO] Progress: $SHOW_PROGRESS" | tee -a "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[INFO] Done" | tee -a "$LOG_FILE"
