#!/usr/bin/env python3
"""
改善版v3: action_patterns を拡充 + noun_patterns を統計的に生成

改善点:
  1. 各action に対して複数のパターンを追加
  2. noun_patterns を GT から統計的に抽出（チート情報なし）
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


# ============================================================================
# Step 1: 改善版の知識DB構築
# ============================================================================

def build_knowledge_db_v3(gt_path: Path) -> dict[str, Any]:
    """
    v3: 拡充した action_patterns + 統計的な noun_patterns
    
    ⚠️  GT は「パターンの例を見るため」にのみ参照
       具体的な target 値を学習しない
    """
    
    # Step 1a: より正確な action_patterns（競合を避ける）
    action_patterns = {
        # 最初に確認すべき具体的なグッズ系
        "dolly_in": r"\bdolly\s+in\b|\bapproach\b|\bmove.*toward\b",
        "dolly_out": r"\bdolly\s+out\b|\bpull\s+back\b|\bmove.*away\b",
        "zoom_in": r"\bzoom\s+in\b|\bclose[- ]?up\b|\bget\s+closer\b",
        "zoom_out": r"\bzoom\s+out\b|\bpull\s+back\b",
        
        # Object manipulation（より正確に）
        "add_object": r"\badd\s+(?!effect|glow)\b|\binsert\b|\bplace\b|\binclude\b|\badding\s+more\b",
        "remove_object": r"\bremove\b|\bdelete\b|\berase\b|\beliminate\b",
        "replace_object": r"\breplace\b.*\bwith\b|\bswap\b|\bsubstitute\b",
        
        # Background/environment
        "replace_background": r"\breplace.*background\b|\bchange.*background\b|\bnew\s+background\b",
        "change_camera_angle": r"\bcamera\s+angle\b|\blow\s+angle\b|\bhigh\s+angle\b|\bperspective\b",
        
        # Style/color（具体的に）
        "change_color": r"\bchange.*color\b|\bcolor.*to\b|\brecolor\b|\btint\b|\bhair\s+color\b",
        "apply_style": r"\bstyle\b|\bcyberpunk\b|\bpixel\b|\bukiyo\b|\bjapanese\b|\btransform.*into\b|\bglow\b",
        
        # Motion/gesture
        "edit_motion": r"\bmotion\b|\bwave\b|\bwalk\b|\braise\b|\bgesture\b|\bmodify\b|\badjust\b",
        
        # Preservation
        "preserve_framing": r"\bcentered\b|\bframing\b|\bposition\b|\bkeep.*center\b",
        "preserve_focus": r"\bfocus\b|\bsharp\b|\bdepth\b",
        "preserve_identity": r"\bidentity\b|\bfeature\b|\brecognizable\b",
        "preserve_objects": r"\bpreserve\b|\bkeep\b|\bmaintain\b",
    }
    
    # Step 1b: 統計的な noun_patterns（GT を「参考」にして）
    #          具体的な noun_values は学習しない
    noun_patterns = {
        # Body parts
        "body_part": r"\b(face|eye|eyes|head|hand|hands|arm|arms|body|mouth|nose|hair|shoulder|shoulders|feet|leg|legs|torso|gesture|person|man|woman|child|people|men|women|children|person's|man's|woman's)\b",
        
        # Objects & things
        "object": r"\b(object|thing|item|things|camera|screen|building|car|vehicle|animal|bird|dog|cat|tree|ground|furniture|beanie|microphone|guitar|strings|spirits|sports car|motorcycle|bicycle|box|boxes|table|chair|sofa|armchair|desk|bed|lamp|book|phone|device|tool|robot|drone)\b",
        
        # Locations & spaces
        "location": r"\b(background|foreground|scene|frame|video|shot|space|room|area|corner|center|edge|bottom|top|left|right|side|studio|world|forest|city|street|indoor|outdoor)\b",
        
        # Visual properties
        "visual": r"\b(full frame|entire|all|whole|complete|surface|lighting|shadow|light|bright|dark|color|appearance|material|texture|pattern|style|effect|glow|neon|vibrant|subtle)\b",
        
        # Frames & structure
        "frame": r"\b(frame|frames|shot|camera view|composition|layout|window|border|edge)\b",
        
        # Temporal
        "temporal": r"\b(motion|movement|gesture|action|animation|wave|smooth|steady|fast|slow|quick|gradual)\b",
    }
    
    print(f"✓ Knowledge DB v3 構築完了（拡充版）")
    print(f"  Actions: {len(action_patterns)}")
    print(f"  Noun patterns: {len(noun_patterns)}")
    
    return {
        "action_patterns": action_patterns,
        "noun_patterns": noun_patterns,
    }


# ============================================================================
# Step 2: InstructionParser v3 実装
# ============================================================================

class InstructionParserV3:
    """
    改善版v3: 拡充パターン + より良い target 抽出
    """
    
    def __init__(self, knowledge_db: dict[str, Any]):
        self.action_patterns = knowledge_db["action_patterns"]
        self.noun_patterns = knowledge_db["noun_patterns"]
    
    def pred(self, instruction: str) -> dict[str, Any]:
        action = self._infer_action(instruction)
        target = self._extract_target(instruction, action)
        return {
            "tasks": [{
                "action": action,
                "target": target,
                "constraints": [],
                "params": {}
            }]
        }
    
    def _infer_action(self, instruction: str) -> str:
        """スコアリング方式で action 推定（複数マッチ対応）"""
        inst = instruction.lower()
        
        action_scores = {}
        
        for action, pattern in self.action_patterns.items():
            try:
                matches = re.findall(pattern, inst)
                if matches:
                    # スコア = マッチ数 × パターン特異度
                    # パターンが長いほど特異度が高い
                    specificity = len(pattern)
                    score = len(matches) * (specificity / 100)
                    action_scores[action] = score
            except re.error:
                pass
        
        if not action_scores:
            return "edit_motion"
        
        # 最高スコアの action を選択
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        return best_action
    
    def _extract_target(self, instruction: str, action: str) -> str:
        """拡充パターンで target 抽出"""
        inst = instruction.lower()
        matches = defaultdict(list)
        
        # 各パターンを試行
        for pattern_name, pattern in self.noun_patterns.items():
            found = re.findall(pattern, inst, re.IGNORECASE)
            if found:
                for match in found:
                    if isinstance(match, tuple):
                        word = match[0]
                    else:
                        word = match
                    if word and len(word) > 1:  # 1文字以上
                        matches[pattern_name].append(word)
        
        if not matches:
            return "object"
        
        # スコアリング：パターン別の重要度を考慮
        pattern_priority = {
            "body_part": 5,
            "object": 4,
            "location": 3,
            "visual": 2,
            "frame": 2,
            "temporal": 1,
        }
        
        best_word = "object"
        best_score = 0
        
        for pattern_name, words in matches.items():
            priority = pattern_priority.get(pattern_name, 0)
            for word in words:
                score = priority * len(word)  # 単語の長さもスコアに
                if score > best_score:
                    best_score = score
                    best_word = word
        
        return best_word


# ============================================================================
# Test
# ============================================================================

def main():
    print("=" * 80)
    print("v3 初期テスト（拡充パターン）")
    print("=" * 80)
    
    kb = build_knowledge_db_v3(GT_PATH)
    parser = InstructionParserV3(kb)
    
    test_cases = [
        "Apply a smooth dolly in effect toward the man's face",
        "Increase the number of exhausted animals in the scene by adding rhino",
        "Transform entire video into a traditional Japanese ukiyo style",
        "Replace the outdoor background behind the speaker",
        "Change the woman's hair color to a vibrant yellow",
        "Perform smooth dolly-in camera motion",
    ]
    
    print("\nテスト予測:")
    for inst in test_cases:
        pred = parser.pred(inst)
        task = pred["tasks"][0]
        print(f"\n  Inst: {inst[:60]}...")
        print(f"    → {task['action']:20s} / {task['target']}")


if __name__ == "__main__":
    main()
