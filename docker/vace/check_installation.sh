#!/bin/bash
set -e

echo "========================================"
echo "VACE Environment Check"
echo "========================================"

# =========================================

# 1. Python / pip

# =========================================

echo "[1] Python / pip"
which python3
python3 --version

python3 -m pip --version

# =========================================

# 2. torch

# =========================================

echo "[2] torch"
python3 - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
EOF

# =========================================

# 3. torch lib path（重要）

# =========================================

echo "[3] torch lib"
TORCH_LIB="/usr/local/lib/python3.10/dist-packages/torch/lib"
ls $TORCH_LIB/libc10.so

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# =========================================

# 4. GroundingDINO

# =========================================

echo "[4] GroundingDINO import"
python3 - << 'EOF'
import groundingdino
print("groundingdino import OK")
EOF

echo "[4-2] GroundingDINO CUDA op"
python3 - << 'EOF'
from groundingdino import _C
print("groundingdino _C OK")
EOF

# =========================================

# 5. SAM / SAM2

# =========================================

echo "[5] SAM"
python3 - << 'EOF'
import segment_anything
print("SAM OK")
EOF

echo "[5-2] SAM2"
python3 - << 'EOF'
import sam2
print("SAM2 OK")
EOF

# =========================================

# 6. RAFT

# =========================================

echo "[6] RAFT"
python3 - << 'EOF'
import raft
print("RAFT import OK")
EOF

# =========================================

# 7. recognize-anything

# =========================================

echo "[7] recognize-anything"
python3 - << 'EOF'
import ram
print("recognize-anything OK")
EOF

# =========================================

# 8. Wan

# =========================================

echo "[8] Wan"
python3 - << 'EOF'
import wan
print("Wan OK")
EOF

# =========================================

# 9. VACE path

# =========================================

echo "[9] VACE path"
python3 - << 'EOF'
import sys
print("PYTHONPATH contains VACE:",
any("VACE" in p for p in sys.path))
EOF

# =========================================

# 10. ldd check（重要）

# =========================================

echo "[10] GroundingDINO ldd"
ldd /workspace/third_party/GroundingDINO/groundingdino/_C*.so | grep "not found" || echo "ldd OK"

echo "========================================"
echo "ALL CHECKS PASSED"
echo "========================================"
