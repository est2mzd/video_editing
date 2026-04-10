#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/workspace/config/submit.yaml}"
exec /workspace/scripts/run_submit_package.sh "$CONFIG_PATH"
