#!/usr/bin/env bash
set -euo pipefail

# InstructionParser クラスベース設計の検証

ROOT="/workspace"
LOG_DIR="$ROOT/logs/analysis"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/prototype_instruction_parser_class_${TS}.log"

echo "[INFO] Running InstructionParser class-based design prototype" | tee "$LOG_FILE"
echo "[INFO] Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$ROOT"
python3 src/parse/prototype_instruction_parser_class.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[INFO] Prototype validation completed successfully" | tee -a "$LOG_FILE"
echo "[INFO] Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
