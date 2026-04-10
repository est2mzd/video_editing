#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-trial}"
GT_PATH="${GT_PATH:-/workspace/data/annotations_gt_task_ver10.json}"
LIMIT="${LIMIT:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logs/analysis/action_target_eval100}"

mkdir -p "$OUTPUT_ROOT"

PYTHONPATH=/workspace:/workspace/src /usr/bin/python \
  /workspace/src/eval_action_target_accuracy_100.py \
  --gt-path "$GT_PATH" \
  --limit "$LIMIT" \
  --output-root "$OUTPUT_ROOT" \
  --tag "$TAG" \
  | tee "$OUTPUT_ROOT/last_run_${TAG}.log"

echo "done: tag=$TAG"
