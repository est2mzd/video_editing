#!/usr/bin/env python3
"""
v2 完成度検証：100件全ケースの予測品質チェック

実施項目:
  1. 100件全てのGT instruction を入力
  2. v2 の予測結果（action, target）を出力
  3. 実際のGT値と比較
  4. 品質指標を計算
"""

import sys
import json
import re
from pathlib import Path
from typing import Any
from collections import defaultdict

# パス定義
WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"

sys.path.insert(0, str(WORKSPACE))

from scripts.prototype_instruction_parser_class_v2_no_cheat import (
    build_knowledge_db,
    InstructionParser,
)


def main():
    print("=" * 80)
    print("v2 完成度検証：100件全ケースの予測品質チェック")
    print("=" * 80)
    
    # Step 1: GT読み込み
    print("\nStep 1: GT データ読み込み")
    print("-" * 80)
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    
    print(f"✓ GT データ読み込み：{len(gt_data)} 件")
    
    # Step 2: v2 の初期化
    print("\nStep 2: InstructionParser v2 初期化")
    print("-" * 80)
    kb = build_knowledge_db(GT_PATH)
    parser = InstructionParser(kb)
    print(f"✓ Parser 初期化完了")
    
    # Step 3: 全100件を予測
    print("\nStep 3: 全100件にて予測実行")
    print("-" * 80)
    
    results = []
    action_matches = 0
    action_partial = 0
    target_exact = 0
    target_fuzzy = 0
    
    for i, gt_row in enumerate(gt_data):
        video_path = gt_row["video_path"]
        instruction = gt_row.get("instruction", "")
        gt_tasks = gt_row.get("tasks", [])
        
        # v2 で予測
        pred = parser.pred(instruction)
        pred_task = pred["tasks"][0]
        
        # GT の主タスク（一番目）と比較
        if gt_tasks:
            gt_task = gt_tasks[0]
            gt_action = gt_task.get("action", "")
            gt_target = gt_task.get("target", "")
            
            pred_action = pred_task["action"]
            pred_target = pred_task["target"]
            
            # 評価
            action_match = (pred_action == gt_action)
            action_similar = _is_action_similar(pred_action, gt_action)
            target_match = _is_target_match(pred_target, gt_target)
            target_fuzzy_match = _is_target_fuzzy_match(pred_target, gt_target)
            
            if action_match:
                action_matches += 1
            if action_similar:
                action_partial += 1
            if target_match:
                target_exact += 1
            if target_fuzzy_match:
                target_fuzzy += 1
            
            results.append({
                "idx": i + 1,
                "video": video_path,
                "instruction": instruction[:50],
                "gt_action": gt_action,
                "pred_action": pred_action,
                "action_match": action_match,
                "gt_target": str(gt_target)[:40],
                "pred_target": pred_target,
                "target_match": target_match,
            })
        
        # 進捗表示
        if (i + 1) % 20 == 0:
            print(f"  進捗: {i+1}/100")
    
    print(f"✓ 予測完了")
    
    # Step 4: 品質指標の計算
    print("\n" + "=" * 80)
    print("Step 4: 品質指標")
    print("=" * 80)
    
    total = len(results)
    action_accuracy = (action_matches / total * 100) if total > 0 else 0
    action_partial_rate = (action_partial / total * 100) if total > 0 else 0
    target_exact_rate = (target_exact / total * 100) if total > 0 else 0
    target_fuzzy_rate = (target_fuzzy / total * 100) if total > 0 else 0
    
    print(f"\n📊 Action 予測精度:")
    print(f"  • 完全一致: {action_matches}/{total} ({action_accuracy:.1f}%)")
    print(f"  • 部分一致（similar）: {action_partial}/{total} ({action_partial_rate:.1f}%)")
    
    print(f"\n📊 Target 抽出精度:")
    print(f"  • 完全一致: {target_exact}/{total} ({target_exact_rate:.1f}%)")
    print(f"  • ファジー一致: {target_fuzzy}/{total} ({target_fuzzy_rate:.1f}%)")
    
    # Step 5: 詳細結果表示（サンプル）
    print("\n" + "=" * 80)
    print("Step 5: 詳細結果（最初の15件）")
    print("=" * 80)
    
    for res in results[:15]:
        status = "✓" if res["action_match"] else "✗"
        target_status = "✓" if res["target_match"] else "~"
        print(f"\n[{res['idx']:2d}] {status} {res['video']}")
        print(f"    Inst: {res['instruction']}...")
        print(f"    Action: {res['gt_action']:20s} → {res['pred_action']:20s}")
        print(f"    Target: {res['gt_target']:20s} → {res['pred_target']} {target_status}")
    
    # Step 6: 失敗ケースの分析
    print("\n" + "=" * 80)
    print("Step 6: Action 失敗ケース（最初の5件）")
    print("=" * 80)
    
    failures = [r for r in results if not r["action_match"]]
    print(f"失敗ケース数: {len(failures)}/{total}")
    
    for res in failures[:5]:
        print(f"\n[{res['idx']:2d}] {res['video']}")
        print(f"    Expected: {res['gt_action']}")
        print(f"    Got:      {res['pred_action']}")
        print(f"    Inst:     {res['instruction']}...")
    
    # Step 7: 完成度判定
    print("\n" + "=" * 80)
    print("Step 7: 完成度判定")
    print("=" * 80)
    
    if action_accuracy >= 80 and target_fuzzy_rate >= 60:
        status = "✅ 完成 - コンペ対応可能"
        grade = "A"
    elif action_accuracy >= 70 and target_fuzzy_rate >= 50:
        status = "⚠️ 準完成 - 調整が必要"
        grade = "B"
    else:
        status = "❌ 未完成 - 大幅改修が必要"
        grade = "C"
    
    print(f"\n評価: {grade}")
    print(f"  {status}")
    print(f"\n理由:")
    print(f"  • Action 精度: {action_accuracy:.1f}% （目標≥80%）")
    print(f"  • Target 精度: {target_fuzzy_rate:.1f}% （目標≥60%）")
    
    # Step 8: 改善提案
    if grade != "A":
        print(f"\n💡 改善提案:")
        if action_accuracy < 70:
            print(f"  1. action_patterns を拡充（現在: {len(parser.action_patterns)}個）")
        if target_fuzzy_rate < 50:
            print(f"  2. noun_patterns の辞書を拡張")
            failed_targets = [r["gt_target"] for r in failures]
            most_common = defaultdict(int)
            for t in failed_targets:
                most_common[t.split()[0]] += 1
            print(f"  3. よく失敗する target: {sorted(most_common.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    print("\n" + "=" * 80)
    print("検証完了")
    print("=" * 80)


def _is_action_similar(pred: str, gt: str) -> bool:
    """Action が類似しているか判定"""
    if pred == gt:
        return True
    # 関連アクション
    similar_pairs = [
        ("zoom_in", "zoom_out"),
        ("dolly_in", "dolly_out"),
        ("change_color", "change_color"),
        ("apply_style", "apply_style"),
        ("preserve_framing", "preserve_focus"),
    ]
    for pair in similar_pairs:
        if (pred in pair and gt in pair):
            return True
    return False


def _is_target_match(pred: str, gt: str) -> bool:
    """Target が完全に一致するか"""
    if isinstance(gt, list):
        return pred.lower() in [str(g).lower() for g in gt]
    return pred.lower() == str(gt).lower()


def _is_target_fuzzy_match(pred: str, gt: str) -> bool:
    """Target が部分的に一致するか"""
    if _is_target_match(pred, gt):
        return True
    
    gt_str = str(gt).lower() if not isinstance(gt, list) else str(gt[0]).lower()
    pred_lower = pred.lower()
    
    # 部分一致
    if pred_lower in gt_str or gt_str in pred_lower:
        return True
    
    # トークン重複
    gt_tokens = set(gt_str.split())
    pred_tokens = set(pred_lower.split())
    overlap = len(gt_tokens & pred_tokens)
    
    return overlap >= 1


if __name__ == "__main__":
    main()
