#!/usr/bin/env python3
"""
シンプルな instruction → tasks 予測デモ

使用例:
  python3 demo_instruction_to_tasks.py

このスクリプトは以下を実行：
1. GT データ読み込み
2. 名詞バンク構築
3. 予測生成（最初の3件）
"""

import sys
import json
from pathlib import Path

# パス定義
WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"

# PythonパスにWORKSPACEを追加
sys.path.insert(0, str(WORKSPACE))

from src.parse.instruction_parser_ver19 import (
    build_noun_bank,
    build_predictions,
    MULTI_CFG_BEST,
)

# ============================================================================
# Step 1: GT データ読み込み
# ============================================================================
print("=" * 70)
print("Step 1: GT データ読み込み")
print("=" * 70)

with open(GT_PATH, encoding="utf-8") as f:
    gt_data = json.load(f)

print(f"✓ GT データ読み込み: {len(gt_data)} 件")
print(f"  Path: {GT_PATH}")

# サンプル表示
sample = gt_data[0]
print(f"\n  サンプル：")
print(f"    video_path: {sample['video_path']}")
print(f"    instruction: {sample['instruction'][:60]}...")
print(f"    tasks: {len(sample['tasks'])} 個")


# ============================================================================
# Step 2: 名詞バンク構築
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: 名詞バンク構築")
print("=" * 70)

base_records = [
    {
        "video_path": row["video_path"],
        "instruction": row.get("instruction", ""),
        "gt_tasks": row.get("tasks", []),
    }
    for row in gt_data
]

noun_bank, video_to_tasks = build_noun_bank(base_records)

print(f"✓ 名詞バンク構築完了")
print(f"  (video, action) ペア数: {len(noun_bank)}")
print(f"  動画数: {len(video_to_tasks)}")

# サンプル表示
example_key = list(noun_bank.keys())[0]
print(f"\n  サンプル (video, action) → target 候補:")
print(f"    {example_key}")
print(f"    → {noun_bank[example_key]}")


# ============================================================================
# Step 3: 予測生成
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: 予測生成（最初の3件）")
print("=" * 70)

# Step 3a: 簡易版 - 各レコードから直接予測
print("\n--- 簡易版：GT から直接予測 ---")
for i, row in enumerate(gt_data[:3]):
    video_path = row["video_path"]
    instruction = row["instruction"]
    gt_tasks = row.get("tasks", [])
    
    print(f"\n[{i+1}] {video_path}")
    print(f"    Instruction: {instruction[:70]}...")
    print(f"    予測タスク:")
    for task in gt_tasks:
        print(f"      • {task['action']:20} → {task['target']}")


# ============================================================================
# Step 4: 完全パイプライン（build_predictions 使用）
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: 完全パイプライン（build_predictions）")
print("=" * 70)

# GT データから annotations のように見えるレコード群を作成
# (実際には GT データそのものを使用)
records = [
    {
        "video_path": row["video_path"],
        "instruction": row.get("instruction", ""),
    }
    for row in gt_data[:3]  # 最初の3件のみ
]

# 予測生成
predictions = build_predictions(
    records,
    gt_path=GT_PATH,
    mode="multi",
    cfg=MULTI_CFG_BEST
)

print(f"\n✓ 予測生成完了: {len(predictions)} 件\n")

for pred in predictions:
    print(f"  Video: {pred['video_path']}")
    print(f"  Instruction: {pred['instruction'][:65]}...")
    print(f"  Tasks:")
    for t in pred['prediction']['tasks']:
        print(f"    • {t['action']:20s} target={t['target']!r:20s} params_keys={list(t['params'].keys())}")
    print()


print("=" * 70)
print("完了！")
print("=" * 70)
