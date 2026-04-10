#!/usr/bin/env bash
# run_test_apply_style_ver6.sh
# apply_style_ver6 を GT apply_style 3件（row 5/17/29）で実行して出力動画を保存する。
#
# Usage:
#   ./scripts/run_test_apply_style_ver6.sh
#   ./scripts/run_test_apply_style_ver6.sh --max-frames N
#   ./scripts/run_test_apply_style_ver6.sh --rows 5
#   ./scripts/run_test_apply_style_ver6.sh --rows 5,17 --max-frames 50
#
# Output:
#   /workspace/logs/test/apply_style_ver6_3cases/<timestamp>/
#     row005_ukiyo-e_*.mp4
#     row017_pixel_art_*.mp4
#     row029_anime_*.mp4
#     report.json
#     run.log
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

EXTRA_ARGS=()
if [[ "${1:-}" == "--max-frames" && -n "${2:-}" ]]; then
  EXTRA_ARGS=("--max-frames" "$2")
fi

if [[ "${1:-}" == "--rows" && -n "${2:-}" ]]; then
  EXTRA_ARGS=("--rows" "$2")
  if [[ "${3:-}" == "--max-frames" && -n "${4:-}" ]]; then
    EXTRA_ARGS+=("--max-frames" "$4")
  fi
fi

echo "=== apply_style_ver6 3-case test ==="
echo "workspace : $WORKSPACE"
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  echo "max-frames: auto (1/3 of each video)"
  echo "rows      : default (5,17,29)"
else
  if [[ "${EXTRA_ARGS[0]}" == "--rows" ]]; then
    echo "rows      : ${EXTRA_ARGS[1]}"
    if [[ ${#EXTRA_ARGS[@]} -ge 4 ]]; then
      echo "max-frames: ${EXTRA_ARGS[3]}"
    else
      echo "max-frames: auto (1/3 of each video)"
    fi
  else
    echo "max-frames: ${EXTRA_ARGS[1]}"
    echo "rows      : default (5,17,29)"
  fi
fi
echo ""

PYTHONPATH="$WORKSPACE:$WORKSPACE/src" \
  /usr/bin/python "$WORKSPACE/src/test/test_apply_style_ver6_3cases.py" \
  "${EXTRA_ARGS[@]}"
