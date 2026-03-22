#!/usr/bin/env bash
set -e
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

ENTRY_PY=${1:-src/run_experiment.py}
CONFIG_PATH=${2:-configs/base.yaml}
shift $(( $# >= 2 ? 2 : $# ))

exec ${PYTHON_BIN:-/usr/bin/python3} "$ENTRY_PY" "$CONFIG_PATH" "$@"