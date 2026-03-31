#!/bin/bash

echo "=== GPU使用プロセス ==="
nvidia-smi

echo "=== kill開始 ==="

# GPU使っているPID抽出
PIDS=$(nvidia-smi | awk '/python/ {print $5}' | sed 's/[^0-9]//g')

for pid in $PIDS; do
    echo "Killing PID: $pid"
    kill -9 $pid
done

echo "=== 完了 ==="
sleep 1
nvidia-smi