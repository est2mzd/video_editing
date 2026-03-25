#!/bin/bash
set -e

FILE_PATH=$(realpath "$0")
DIR_PATH=$(dirname "$FILE_PATH")
PARENT_DIR=$(dirname "$DIR_PATH")
cd "$PARENT_DIR"

PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}

echo "[INFO] Installing pyarrow with: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install pyarrow

echo "[INFO] pyarrow installation completed"
"$PYTHON_BIN" -m pip show pyarrow | sed -n '1,6p'
