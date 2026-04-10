#!/usr/bin/env bash
# run_test_replace_background_gt.sh
# replace_background を GT の対象行で1件ずつ実行する。
#
# Usage:
#   ./scripts/run_test_replace_background_gt.sh
#   ./scripts/run_test_replace_background_gt.sh --max-frames 12
#   ./scripts/run_test_replace_background_gt.sh --rows 3,15,16 --max-frames 12
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

EXTRA_ARGS=()
if [[ "${1:-}" == "--rows" && -n "${2:-}" ]]; then
  EXTRA_ARGS+=("--rows" "$2")
  shift 2
fi
if [[ "${1:-}" == "--max-frames" && -n "${2:-}" ]]; then
  EXTRA_ARGS+=("--max-frames" "$2")
fi

echo "=== replace_background GT case test ==="
echo "workspace: $WORKSPACE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "args     : ${EXTRA_ARGS[*]}"
else
  echo "args     : (default rows, one-third frames per video)"
fi
echo ""

PYTHONPATH="$WORKSPACE:$WORKSPACE/src" \
  /usr/bin/python "$WORKSPACE/src/test/test_replace_background_gt_cases.py" \
  "${EXTRA_ARGS[@]}"
