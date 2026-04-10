#!/bin/bash
set -e

echo "=== Get Script and Root Directories ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "ROOT_DIR  : $ROOT_DIR"

#echo "=== Step1: Download weights ==="
#cd "$SCRIPT_DIR/scripts"
#./download_weight_from_gdrive.sh

echo "=== Step2: Build & Start Docker ==="
cd "$SCRIPT_DIR/docker/vace"

echo "=== Step2.1: Build Docker Image ==="
./build.sh
echo "=== Step2.2: Remove Docker Container ==="
./remove.sh
echo "=== Step2.3: Start Docker Container ==="
./start.sh

echo "=== Step3: Get container ==="
CONTAINER_ID=$(docker ps -lq)
echo "Container ID: $CONTAINER_ID"

echo "=== Step4: Run inside container ==="

docker exec -it $CONTAINER_ID bash -c "
    set -e

    echo '=== Run Video Editor ==='
    bash /workspace/scripts/run_submit.sh
"

echo "================================="
echo "✅ ALL PROCESS COMPLETED SUCCESSFULLY"
echo "================================="