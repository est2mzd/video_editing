#!/usr/bin/env bash
set -euo pipefail

# InstructionParser v2: チート情報完全排除版

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/prototype_v2_no_cheat_${TS}.log"

echo "[INFO] Running InstructionParser v2 (no cheat information)" | tee "$LOG_FILE"
echo "[INFO] Date: $(date)" | tee -a "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$ROOT"
python3 src/parse/prototype_instruction_parser_class_v2_no_cheat.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[INFO] Prototype v2 validation completed successfully" | tee -a "$LOG_FILE"
echo "[INFO] Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
