#!/usr/bin/env bash
set -euo pipefail

# User-tunable parameters (can be overridden via environment variables)
ACTIONS="${ACTIONS:-dolly_in,change_color}"
DOLLY_END_SCALE="${DOLLY_END_SCALE:-1.85}"
COLOR_GRADUAL_MODE="${COLOR_GRADUAL_MODE:-auto}"
COLOR_GRADUAL_KEYWORDS="${COLOR_GRADUAL_KEYWORDS:-gradual,gradually,徐々,徐々に,だんだん,少しずつ}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  /usr/bin/python /workspace/src/run_adjustment_pass_v2.py
  --config /workspace/configs/base_config_v2.yaml
  --actions "$ACTIONS"
  --dolly-end-scale "$DOLLY_END_SCALE"
  --color-gradual-mode "$COLOR_GRADUAL_MODE"
  --color-gradual-keywords "$COLOR_GRADUAL_KEYWORDS"
  --output-prefix adjustments
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

export PYTHONPATH=/workspace:/workspace/src
"${CMD[@]}"
