#!/usr/bin/env bash
set -euo pipefail

TRIAL_A="${1:?usage: run_instruction_target_compare.sh <trial_a_dir> <trial_b_dir>}"
TRIAL_B="${2:?usage: run_instruction_target_compare.sh <trial_a_dir> <trial_b_dir>}"
OUT="${OUT:-/workspace/logs/analysis/instruction_target_eval/trial_compare.md}"

PYTHONPATH=/workspace:/workspace/src /usr/bin/python \
  /workspace/src/compare_instruction_target_trials.py \
  --trial-a "$TRIAL_A" \
  --trial-b "$TRIAL_B" \
  --output "$OUT" \
  | tee "${OUT%.md}.log"

echo "done: compare -> $OUT"
