#!/bin/bash
set -e

echo "=== Get Script and Root Directories ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "ROOT_DIR  : $ROOT_DIR"

echo "=== Step1: Download weights ==="
cd "$SCRIPT_DIR/scripts"
./download_weights.sh

echo "=== Step2: Build & Start Docker ==="
cd "$SCRIPT_DIR/docker/vace"
./build.sh
./start.sh

echo "=== Step3: Get container ==="
CONTAINER_ID=$(docker ps -lq)
echo "Container ID: $CONTAINER_ID"

echo "=== Step4: Run inside container ==="

docker exec -it $CONTAINER_ID bash -c "
    set -e

    echo '=== Download Weights ==='
    bash /workspace/scripts/download_weight_from_gdrive.sh

    echo '=== Install Third Party Libraries ==='
    if [ -f /workspace/docker/vace/install_vace_library.sh ]; then
        bash /workspace/docker/vace/install_vace_library.sh
    else
        echo 'install_vace_library.sh not found'
    fi

    echo '=== Run Video Editor ==='
    bash /workspace/scripts/run_submit.sh
"

echo "================================="
echo "✅ ALL PROCESS COMPLETED SUCCESSFULLY"
echo "================================="