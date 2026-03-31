#!/bin/bash

set -e

BASE_DIR="/workspace/weights"
MODEL_DIR="${BASE_DIR}/controlnet-canny"

mkdir -p "${MODEL_DIR}"

echo "== Download ControlNet (canny) =="

hf download lllyasviel/sd-controlnet-canny \
  --local-dir "${MODEL_DIR}"

echo "== Done =="