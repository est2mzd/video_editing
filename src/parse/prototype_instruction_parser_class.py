#!/usr/bin/env python3
"""
InstructionParser クラスベース設計の検証プロトタイプ

DB には「チートにならない情報」のみ保持：
  - action_patterns: {action → regex}
  - action_target_vocab: {action → [target候補]}
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

# パス定義
WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"

sys.path.insert(0, str(WORKSPACE))


# ============================================================================
# Step 1: 知識DB の構築
# ============================================================================

def build_knowledge_db(gt_path: Path) -> dict[str, Any]:
    """
    GT データから「チートにならない知識」を抽出。
    
    Returns:
        {
          "action_patterns": {action → regex},
          "action_target_vocab": {action → [target候補]},
          "multi_mode": bool
        }
    """
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    
    # Step 1a: action patterns（正規表現を定義）
    action_patterns = {
        "replace_background": r"\bbackground\b|\breplace the .*background\b",
        "replace_object": r"\breplace\b.*\bwith\b",
        "change_color": r"\bchange\b.*\bcolor\b|\brecolor\b",
        "add_object": r"\badd\b|\binsert\b|\bplace\b",
        "remove_object": r"\bremove\b|\bdelete\b|\berase\b",
        "apply_style": r"\bstyle\b|\bcyberpunk\b|\bpixel art\b",
        "edit_motion": r"\bmotion\b|\bwave\b|\bwalk\b|\braise\b",
        "zoom_in": r"\bzoom in\b|\bclose[- ]?up\b",
        "zoom_out": r"\bzoom out\b|\bwider\b",
        "dolly_in": r"\bdolly in\b",
        "change_camera_angle": r"\bcamera angle\b|\blow angle\b|\bhigh angle\b",
        "preserve_framing": r"\bcentered\b|\bframing\b",
        "preserve_focus": r"\bfocus\b|\bsharp\b",
    }
    
    # Step 1b: action ごとに「出現した target」を集計
    #          ⚠️  video_path は含めない（統計情報のみ）
    #          target は str または list の場合がある
    action_target_vocab = defaultdict(set)
    for row in gt_data:
        for task in row.get("tasks", []):
            action = task.get("action", "")
            target = task.get("target", "")
            if action and target:
                if isinstance(target, list):
                    for t in target:
                        if t:
                            action_target_vocab[action].add(str(t))
                else:
                    action_target_vocab[action].add(str(target))
    
    # Step 1c: set を list に変換
    vocab = {
        action: sorted(list(targets))  # ソートして一貫性を確保
        for action, targets in action_target_vocab.items()
    }
    
    print(f"✓ Knowledge DB 構築完了")
    print(f"  Actions: {len(action_patterns)}")
    print(f"  Vocab entries: {len(vocab)}")
    
    return {
        "action_patterns": action_patterns,
        "action_target_vocab": vocab,
        "multi_mode": False,  # 簡略版：単一タスク
    }


# ============================================================================
# Step 2: InstructionParser クラス実装
# ============================================================================

class InstructionParser:
    """
    instruction のみから tasks を予測するクラス。
    DB には「統計的な知識」のみを保持。
    """
    
    def __init__(self, knowledge_db: dict[str, Any]):
        """
        Args:
            knowledge_db: {
              "action_patterns": {action → regex},
              "action_target_vocab": {action → [target]},
              "multi_mode": bool
            }
        """
        self.patterns = knowledge_db["action_patterns"]
        self.vocab = knowledge_db["action_target_vocab"]
        self.multi_mode = knowledge_db.get("multi_mode", False)
    
    def pred(self, instruction: str) -> dict[str, Any]:
        """
        instruction のみから tasks を予測。
        
        Args:
            instruction: テキスト説明
        
        Returns:
            {
              "tasks": [
                {
                  "action": str,
                  "target": str,
                  "constraints": [],
                  "params": {}
                }
              ]
            }
        """
        # Step 1: instruction から action を推定
        action = self._infer_action(instruction)
        
        # Step 2: action に対応する target を選択
        target = self._choose_target(instruction, action)
        
        # Step 3: タスク生成
        tasks = [
            {
                "action": action,
                "target": target,
                "constraints": [],
                "params": {}  # ⚠️  コンペではここに詳細値を生成する必要があるが、
                              #     instruction のみからは難しい
            }
        ]
        
        return {"tasks": tasks}
    
    def _infer_action(self, instruction: str) -> str:
        """Instruction から action パターンマッチで推定"""
        inst = instruction.lower()
        for action, pattern in self.patterns.items():
            if re.search(pattern, inst):
                return action
        return "edit_motion"  # デフォルト
    
    def _choose_target(self, instruction: str, action: str) -> str:
        """
        Instruction テキストと action から target を選択。
        Strategy: token overlap スコア（多いほど高スコア）
        """
        vocab = self.vocab.get(action, [])
        if not vocab:
            return "object"  # デフォルト
        
        inst_tokens = set(instruction.lower().split())
        best_target = vocab[0]
        best_score = 0
        
        for target in vocab:
            target_tokens = set(target.lower().split())
            overlap = len(inst_tokens & target_tokens)
            if overlap > best_score:
                best_score = overlap
                best_target = target
        
        return best_target


# ============================================================================
# Step 3: テスト実行
# ============================================================================

def main():
    print("=" * 70)
    print("InstructionParser クラスベース設計の検証")
    print("=" * 70)
    
    # Step 1: 知識DB構築
    print("\nStep 1: 知識DB構築")
    print("-" * 70)
    kb = build_knowledge_db(GT_PATH)
    
    # Step 2: Parser初期化
    print("\nStep 2: InstructionParser 初期化")
    print("-" * 70)
    parser = InstructionParser(kb)
    print(f"✓ Parser initialized")
    print(f"  Action patterns: {len(parser.patterns)}")
    print(f"  Vocab entries: {len(parser.vocab)}")
    
    # Step 3: テスト予測
    print("\nStep 3: テスト予測（instruction のみ）")
    print("-" * 70)
    
    test_cases = [
        "Apply a smooth dolly in effect toward the man's face",
        "Increase the number of exhausted animals by adding more rhino",
        "Apply an oil painting style to the entire video",
        "Zoom in on the face of the person",
    ]
    
    for instruction in test_cases:
        print(f"\nInstruction: {instruction[:60]}...")
        pred = parser.pred(instruction)
        task = pred["tasks"][0]
        print(f"  → action={task['action']:<20} target={task['target']}")
    
    # Step 4: 知識DBの内容確認
    print("\n" + "=" * 70)
    print("Step 4: 知識DB の内容確認（チート情報なし）")
    print("=" * 70)
    
    print("\n✓ action_patterns（正規表現）:")
    for action, pattern in list(parser.patterns.items())[:5]:
        print(f"  {action}: {pattern[:50]}...")
    
    print("\n✓ action_target_vocab（統計的候補、video_path なし）:")
    for action, targets in list(parser.vocab.items())[:3]:
        print(f"  {action}:")
        print(f"    → {targets[:3]}")  # 最初の3候補のみ表示
    
    print("\n" + "=" * 70)
    print("検証完了！")
    print("=" * 70)
    print("\n📋 設計上の注意点:")
    print("  ✅ DB には video_path なし")
    print("  ✅ params は空 dict（コンペでは生成ロジック追加が必要）")
    print("  ✅ Train/Test 分離が可能")
    print("  ✅ チート情報なし")


if __name__ == "__main__":
    main()
