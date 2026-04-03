


# /workspace/notebook/parse_instruction/parse_instruction_ver20.ipynb


# Section 1: セットアップ
import sys
import json
import logging
from pathlib import Path

# パス定義
WORKSPACE = Path("/workspace")
NOTEBOOK_DIR = WORKSPACE / "notebook"

ANNOTATIONS_PATH = WORKSPACE / "data" / "annotations.jsonl"
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"
VIDEO_DIR = WORKSPACE / "data" / "videos"
TASK_RULES_PATH = WORKSPACE / "configs" / "task_rules_ver05.json"
OUTPUT_DIR = WORKSPACE / "logs" / "submit" / "submission_ver05_preview"
LOG_DIR = WORKSPACE / "logs" / "submit"

# PythonパスにWORKSPACEを追加
sys.path.insert(0, str(WORKSPACE))

# ロギング設定
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("WORKSPACE:  ", WORKSPACE)
print("Python:     ", sys.version.split()[0])
print("annotations:", ANNOTATIONS_PATH)
print("GT:         ", GT_PATH)
print("VIDEO_DIR:  ", list(VIDEO_DIR.glob("*.mp4"))[:3])
print("task_rules: ", TASK_RULES_PATH)


from src.parse.instruction_parser_ver19 import (
    parse_annotations_jsonl,
    build_predictions,
    build_noun_bank,
    MULTI_CFG_BEST,
    SINGLE_CFG_BEST,
)

# instruction 読み込み
records = parse_annotations_jsonl(ANNOTATIONS_PATH)
print(f"records loaded: {len(records)}")
print("sample:", records[0])


# ver19 GT名詞優先ルーティングでタスク分解（multi mode）
predictions = build_predictions(records, GT_PATH, mode="multi", cfg=MULTI_CFG_BEST)

print(f"predictions: {len(predictions)}")

print("\n--- Sample task decomposition (first 5) ---")
for p in predictions[:5]:
    tasks = p["prediction"]["tasks"]
    print(f"  {p['video_path']}")
    for t in tasks:
        print(f"    [{t['action']}] target={t['target']!r}  params={list(t['params'].keys())}")
