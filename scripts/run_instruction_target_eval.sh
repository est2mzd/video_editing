#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-trial}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/workspace/data/annotations_gt_task_ver10.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logs/analysis/instruction_target_eval}"

mkdir -p "$OUTPUT_ROOT"

PYTHONPATH=/workspace:/workspace/src /usr/bin/python \
  /workspace/src/eval_instruction_target_accuracy.py \
  --annotation-path "$ANNOTATION_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --tag "$TAG" \
  | tee "$OUTPUT_ROOT/last_run_${TAG}.log"

echo "done: tag=$TAG"
