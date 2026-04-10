#!/bin/bash
# Trial 12 evaluation wrapper
# Usage: ./scripts/run_action_target_eval_100_trial12.sh <tag>

set -e

TAG="${1:-trial12}"
PYTHONPATH=/workspace:/workspace/src

echo "Running trial12 evaluation with tag=$TAG"
/usr/bin/python /workspace/src/eval_action_target_accuracy_100_trial12.py "$TAG"
