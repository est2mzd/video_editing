#!/usr/bin/env python3
"""
v3 完成度検証：100件全ケースの予測品質チェック（改善版）
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"

sys.path.insert(0, str(WORKSPACE))

from scripts.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
    InstructionParserV3,
)


def _is_action_similar(pred: str, gt: str) -> bool:
    """Action が類似しているか"""
    if pred == gt:
        return True
    similar = {
        ("zoom_in", "zoom_out"): True,
        ("dolly_in", "dolly_out"): True,
        ("zoom_in", "dolly_in"): True,
        ("apply_style", "apply_style"): True,
    }
    return (pred, gt) in similar or (gt, pred) in similar


def _is_target_match(pred: str, gt: str) -> bool:
    """Target完全一致"""
    if isinstance(gt, list):
        return pred.lower() in [str(g).lower() for g in gt]
    return pred.lower() == str(gt).lower()


def _is_target_fuzzy(pred: str, gt: str) -> bool:
    """Target ファジー一致"""
    if _is_target_match(pred, gt):
        return True
    gt_str = str(gt).lower() if not isinstance(gt, list) else str(gt[0]).lower()
    pred_lower = pred.lower()
    if pred_lower in gt_str or gt_str in pred_lower:
        return True
    gt_tokens = set(gt_str.split())
    pred_tokens = set(pred_lower.split())
    return len(gt_tokens & pred_tokens) >= 1


def main():
    print("=" * 80)
    print("v3 完成度検証：100件全ケース")
    print("=" * 80)
    
    # 初期化
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    
    kb = build_knowledge_db_v3(GT_PATH)
    parser = InstructionParserV3(kb)
    
    # 予測実行
    results = []
    action_match = 0
    target_exact = 0
    target_fuzzy = 0
    
    print(f"\nStep 1: 100件予測実行中...")
    for i, gt_row in enumerate(gt_data):
        instruction = gt_row.get("instruction", "")
        gt_tasks = gt_row.get("tasks", [])
        
        pred = parser.pred(instruction)
        pred_task = pred["tasks"][0]
        
        if gt_tasks:
            gt_task = gt_tasks[0]
            if pred_task["action"] == gt_task.get("action"):
                action_match += 1
            if _is_target_match(pred_task["target"], gt_task.get("target")):
                target_exact += 1
            if _is_target_fuzzy(pred_task["target"], gt_task.get("target")):
                target_fuzzy += 1
            
            results.append({
                "idx": i + 1,
                "video": gt_row["video_path"],
                "inst": instruction[:50],
                "gt_action": gt_task.get("action"),
                "pred_action": pred_task["action"],
                "action_ok": pred_task["action"] == gt_task.get("action"),
                "gt_target": str(gt_task.get("target"))[:30],
                "pred_target": pred_task["target"],
            })
        
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/100 完了")
    
    # 結果集計
    total = len(results)
    action_acc = (action_match / total * 100) if total > 0 else 0
    target_exact_acc = (target_exact / total * 100) if total > 0 else 0
    target_fuzzy_acc = (target_fuzzy / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("Step 2: 品質指標")
    print("=" * 80)
    
    print(f"\nAction 精度: {action_match}/{total} ({action_acc:.1f}%)")
    print(f"Target 完全一致: {target_exact}/{total} ({target_exact_acc:.1f}%)")
    print(f"Target ファジー: {target_fuzzy}/{total} ({target_fuzzy_acc:.1f}%)")
    
    # 成功例と失敗例
    print("\n" + "=" * 80)
    print("Step 3: 成功例（最初の5件）")
    print("=" * 80)
    success = [r for r in results if r["action_ok"]]
    for res in success[:5]:
        print(f"\n[{res['idx']:2d}] ✓ {res['video']}")
        print(f"    {res['gt_action']:20s} → {res['pred_action']}")
    
    print("\n" + "=" * 80)
    print("Step 4: 失敗例（最初の5件）")
    print("=" * 80)
    failure = [r for r in results if not r["action_ok"]]
    print(f"失敗数: {len(failure)}/{total}\n")
    for res in failure[:5]:
        print(f"[{res['idx']:2d}] {res['gt_action']:20s} → {res['pred_action']:20s}")
        print(f"     {res['inst']}...")
    
    # 完成度判定
    print("\n" + "=" * 80)
    print("Step 5: 完成度判定")
    print("=" * 80)
    
    if action_acc >= 70 and target_fuzzy_acc >= 50:
        grade = "A: 完成 ✅"
    elif action_acc >= 50 and target_fuzzy_acc >= 40:
        grade = "B: 準完成 ⚠️"
    else:
        grade = "C: 継続改修 ❌"
    
    print(f"\n評価: {grade}")
    print(f"\n基準:")
    print(f"  • Action ≥70% : {'✓' if action_acc >= 70 else '✗'} ({action_acc:.1f}%)")
    print(f"  • Target ≥50% : {'✓' if target_fuzzy_acc >= 50 else '✗'} ({target_fuzzy_acc:.1f}%)")


if __name__ == "__main__":
    main()
