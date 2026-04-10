#!/usr/bin/env python3
"""
Trial 13: GT target の妥当性検証（100件全数）

目的：
- 各 instruction の action に対して、GT target が関数実行可能かを判定
- 不適切な場合は、instruction 分析から推奨 target を提示
- 前後比較で .md に記載
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.append("/workspace/src")

from normalize_target_for_functions import (
    extract_effect_type,
    normalize_for_groundingdino,
    normalize_for_cv2_color,
)


def load_gt_data(gt_file: str, max_rows: int = 100) -> List[Dict]:
    """Load GT data from JSON."""
    with open(gt_file, 'r') as f:
        data = json.load(f)
    return data[:max_rows] if max_rows else data


def build_parser():
    """Build parser using trial020."""
    from parse.instruction_parser_v3_rulebase_trial020_singlefile import build_parser
    return build_parser()


def normalize_text(text):
    """Normalize text."""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    return text


def assess_target_feasibility(
    action: str, 
    gt_target: str, 
    instruction: str
) -> Dict:
    """
    評価：GT target が関数実行可能か
    
    Returns:
        {
            'action': action,
            'gt_target': GT target,
            'feasible': 実行可能か (bool),
            'issues': 問題リスト,
            'recommended_target': 推奨 target,
            'reasoning': 推論説明
        }
    """
    
    action_lower = action.lower().strip()
    gt_target_lower = normalize_text(gt_target)
    
    issues = []
    concerns = []
    recommended_target = gt_target
    reasoning = ""
    
    # === add_effect ===
    if action_lower == 'add_effect':
        # 理想：effect 種類が明確（glow, lighting, blur 等）
        effect_type = extract_effect_type(instruction)
        
        # GT target が複雑・抽象的でないか
        if len(gt_target_lower) > 50:
            issues.append("GT target が過度に長い（複合文形式の可能性）")
        
        if any(word in gt_target_lower for word in ['region', 'area', 'scene', 'part of']):
            issues.append("GT target が抽象的な領域指定（GroundingDINO 検出困難）")
            recommended_target = effect_type if effect_type else 'effect'
        
        if effect_type:
            concerns.append(f"instruction から effect={effect_type} を抽出可能")
            if effect_type not in gt_target_lower:
                issues.append(f"GT target に effect 種類 '{effect_type}' が含まれない")
        else:
            issues.append("instruction から effect 種類を抽出できない")
    
    # === change_color ===
    elif action_lower == 'change_color':
        # 理想：単一の対象物（armchair, person, object 等）
        if any(word in gt_target_lower for word in ['left', 'right']) and not 'armchair' in gt_target_lower:
            issues.append("GT target が複数部位指定（左右分離）- 統合推奨")
        
        if len(gt_target_lower) > 100:
            issues.append("GT target が過度に長い（複雑な色指定記述が含まれている）")
            # 色指定部分を除去
            color_keywords = ['to', 'shade of', 'hue', 'color to']
            for keyword in color_keywords:
                if keyword in gt_target_lower:
                    parts = gt_target_lower.split(keyword)
                    if parts:
                        recommended_target = parts[0].strip()
                        break
        
        # 対象物が明確か
        clear_objects = ['armchair', 'person', 'hair', 'object', 'background', 'clothing']
        if not any(obj in gt_target_lower for obj in clear_objects):
            concerns.append("GT target が標準的な対象物ではない")
    
    # === zoom_in ===
    elif action_lower == 'zoom_in':
        # 理想：ズーム焦点が具体的（face, person, hand, object 等）
        if 'camera_view' in gt_target_lower or 'entire' in gt_target_lower:
            issues.append("GT target が 'camera_view' - ズーム焦点が不明確")
            # instruction から焦点を推定
            focus_keywords = ['face', 'person', 'hand', 'group', 'object']
            for keyword in focus_keywords:
                if keyword in instruction.lower():
                    recommended_target = keyword
                    break
        
        if any(word in gt_target_lower for word in ['region', 'area', 'scene']):
            issues.append("GT target が抽象的な領域")
    
    # === replace_background ===
    elif action_lower == 'replace_background':
        # 理想：前景 prompt が YOLO+GroundingDINO で検出可能
        if 'background_behind' in gt_target_lower or '_' in gt_target_lower:
            issues.append("GT target が複合表現（underscore or 複数単語）")
        
        if 'entire' in gt_target_lower or 'full' in gt_target_lower:
            concerns.append("GT target が背景全体の指定 - 妥当")
    
    # === dolly_in / zoom_out ===
    elif action_lower in ['dolly_in', 'zoom_out']:
        # 理想：ズーム焦点が具体的（person, face, object 等）
        if 'camera' in gt_target_lower or 'view' in gt_target_lower:
            issues.append("GT target がカメラ視点指定 - ズーム焦点が不明確")
    
    # === add_object / remove_object ===
    elif action_lower in ['add_object', 'remove_object']:
        # 理想：対象物が明確
        if gt_target_lower == 'object' or gt_target_lower == 'new_object':
            issues.append("GT target が汎用ラベル 'object/new_object' - 具体物が不明確")
    
    # === apply_style ===
    elif action_lower == 'apply_style':
        # 理想：full_frame or person
        if 'full_frame' not in gt_target_lower and 'person' not in gt_target_lower:
            concerns.append("GT target が標準的な apply_style 領域ではない")
    
    # 総合判定
    feasible = len(issues) == 0
    
    if issues:
        reasoning = f"Issues: {'; '.join(issues)}"
    else:
        reasoning = "GT target は実行可能"
    
    return {
        'action': action,
        'gt_target': gt_target,
        'feasible': feasible,
        'issues': issues,
        'concerns': concerns,
        'recommended_target': recommended_target,
        'reasoning': reasoning,
    }


def validate_100rows(output_dir: str) -> Dict:
    """Validate all 100 rows."""
    
    gt_file = "/workspace/data/annotations_gt_task_ver10.json"
    gt_data = load_gt_data(gt_file, max_rows=100)
    
    parser = build_parser()
    
    results = {
        'total': len(gt_data),
        'feasible': 0,
        'infeasible': 0,
        'assessments': [],
    }
    
    for row_idx, gt_row in enumerate(gt_data):
        instruction = gt_row.get('instruction', '')
        gt_tasks = gt_row.get('tasks', [])
        
        if not gt_tasks:
            continue
        
        gt_task = gt_tasks[0]
        gt_action = gt_task.get('action', '')
        gt_target = gt_task.get('target', '')
        
        # Assess feasibility
        assessment = assess_target_feasibility(gt_action, gt_target, instruction)
        assessment['row_index'] = row_idx
        assessment['instruction'] = instruction
        
        results['assessments'].append(assessment)
        
        if assessment['feasible']:
            results['feasible'] += 1
        else:
            results['infeasible'] += 1
    
    return results


def write_results(results: Dict, output_dir: str):
    """Write validation results."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary_file = f"{output_dir}/summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GT Target Feasibility Validation\n")
        f.write(f"==============================\n\n")
        f.write(f"Total rows: {results['total']}\n")
        f.write(f"Feasible: {results['feasible']}\n")
        f.write(f"Infeasible: {results['infeasible']}\n")
        f.write(f"Feasibility rate: {results['feasible']/results['total']:.2%}\n\n")
    
    # Detailed CSV
    csv_file = f"{output_dir}/assessments.csv"
    with open(csv_file, 'w') as f:
        header = "row_index,action,gt_target,feasible,issues,recommended_target,reasoning\n"
        f.write(header)
        for a in results['assessments']:
            issues_str = '; '.join(a['issues']) if a['issues'] else 'none'
            row = (
                f"{a['row_index']},{a['action']},"
                f'"{a["gt_target"]}",'
                f"{a['feasible']},"
                f'"{issues_str}",'
                f'"{a["recommended_target"]}",'
                f'"{a["reasoning"]}"\n'
            )
            f.write(row)
    
    # Markdown report
    md_file = f"{output_dir}/validation_report.md"
    with open(md_file, 'w') as f:
        f.write(f"# Trial 13: GT Target Feasibility Validation\n\n")
        f.write(f"## Summary\n")
        f.write(f"- **Total rows**: {results['total']}\n")
        f.write(f"- **Feasible**: {results['feasible']}\n")
        f.write(f"- **Infeasible**: {results['infeasible']}\n")
        f.write(f"- **Feasibility rate**: {results['feasible']/results['total']:.2%}\n\n")
        
        f.write(f"## Infeasible Cases (with Recommendations)\n\n")
        
        infeasible_count = 0
        for a in results['assessments']:
            if not a['feasible']:
                infeasible_count += 1
                f.write(f"### {infeasible_count}. Row {a['row_index']} - {a['action']}\n\n")
                f.write(f"**Instruction**:\n")
                f.write(f"{a['instruction'][:200]}...\n\n")
                f.write(f"**Issues**:\n")
                for issue in a['issues']:
                    f.write(f"- {issue}\n")
                f.write(f"\n**GT target** (before):\n")
                f.write(f"`{a['gt_target']}`\n\n")
                f.write(f"**Recommended target** (after):\n")
                f.write(f"`{a['recommended_target']}`\n\n")
                f.write(f"**Reasoning**:\n")
                f.write(f"{a['reasoning']}\n\n")
                f.write(f"---\n\n")
    
    print(f"saved: {output_dir}")
    print(f"feasible={results['feasible']}/{results['total']} ({results['feasible']/results['total']:.2%})")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/logs/analysis/gt_target_validation/val_{timestamp}"
    
    print(f"Validating 100 GT targets...", flush=True)
    results = validate_100rows(output_dir)
    write_results(results, output_dir)
    print(f"done", flush=True)


if __name__ == '__main__':
    main()
