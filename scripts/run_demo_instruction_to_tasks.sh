#!/usr/bin/env bash
set -euo pipefail

# Instruction → Tasks 予測パイプラインの動作確認

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/demo_instruction_to_tasks_${TS}.log"

echo "[INFO] Running instruction → tasks prediction demo" | tee "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$ROOT"
python3 src/parse/demo_instruction_to_tasks.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[INFO] Demo completed successfully" | tee -a "$LOG_FILE"
echo "[INFO] Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
