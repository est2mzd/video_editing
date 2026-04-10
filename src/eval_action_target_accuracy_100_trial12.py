#!/usr/bin/env python3
"""
Trial 12: 関数実行適性に基づく target 正規化後の精度評価

Trial 11 との比較用。normalize_target_for_functions を導入。
- add_effect: instruction から effect 種類を抽出
- change_color: 複数対象を統合
- zoom_in: camera_view を entire_frame に
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.append("/workspace/src")

from normalize_target_for_functions import normalize_for_function


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
    """Normalize text for comparison."""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    text = re.sub(r'[._-]', ' ', text)
    return text


def jaccard_similarity(s1: str, s2: str, threshold: float = 0.5) -> bool:
    """Compute Jaccard similarity ≥ threshold."""
    tokens1 = set(normalize_text(s1).split())
    tokens2 = set(normalize_text(s2).split())
    
    if not tokens1 and not tokens2:
        return True
    if not tokens1 or not tokens2:
        return False
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / len(union) if union else 0
    return jaccard >= threshold


def normalize_person_words(text: str) -> str:
    """Normalize person-related words to 'person'."""
    person_words = ['man', 'woman', 'boy', 'girl', 'person', 'face', 'head', 'body', 'his', 'her', 'subject']
    text_lower = normalize_text(text)
    if any(word in text_lower for word in person_words):
        return 'person'
    return text


def target_match_relaxed(gt_target: str, extracted_target: str) -> Tuple[bool, str]:
    """
    Relaxed matching with normalize_target_for_functions integration.
    
    Returns: (matched, reason)
    """
    if not gt_target or not extracted_target:
        return (gt_target == extracted_target, "empty_check")
    
    # Exact match
    if normalize_text(gt_target) == normalize_text(extracted_target):
        return (True, "exact_match")
    
    # Person normalization
    gt_person = normalize_person_words(gt_target)
    ex_person = normalize_person_words(extracted_target)
    if gt_person != normalize_text(gt_target) and gt_person == ex_person:
        return (True, "person_normalized")
    
    # Jaccard ≥ 0.5
    if jaccard_similarity(gt_target, extracted_target, threshold=0.5):
        return (True, "jaccard_0.5")
    
    return (False, "mismatch")


def evaluate_100rows(output_dir: str) -> Dict:
    """Evaluate 100 rows with trial12 strategy."""
    
    # Load GT
    gt_file = "/workspace/data/annotations_gt_task_ver10.json"
    gt_data = load_gt_data(gt_file, max_rows=100)
    
    # Build parser
    parser = build_parser()
    
    results = {
        'rows_evaluated': 0,
        'action_accuracy': 0.0,
        'target_accuracy_strict': 0.0,
        'target_accuracy_relaxed': 0.0,
        'joint_accuracy_strict': 0.0,
        'joint_accuracy_relaxed': 0.0,
        'per_action': {},
        'details': [],
        'mismatches': [],
    }
    
    action_stats = {}
    action_action_match = 0
    action_target_strict_match = 0
    action_target_relaxed_match = 0
    joint_strict_match = 0
    joint_relaxed_match = 0
    
    for row_idx, gt_row in enumerate(gt_data):
        instruction = gt_row.get('instruction', '')
        gt_tasks = gt_row.get('tasks', [])
        
        if not gt_tasks:
            continue
        
        # Use first task (primary action)
        gt_task = gt_tasks[0]
        gt_action = gt_task.get('action', '')
        gt_target = gt_task.get('target', '')
        
        # Parse instruction
        pred = parser.infer(instruction)
        pred_tasks = pred.get("tasks", []) if isinstance(pred, dict) else []
        if pred_tasks:
            extracted_action = str(pred_tasks[0].get('action', 'unknown')).strip()
            extracted_target = str(pred_tasks[0].get('target', '')).strip()
        else:
            extracted_action = 'unknown'
            extracted_target = ''
        
        # === Trial 12: normalize_target_for_functions を導入 ===
        norm_info = normalize_for_function(
            extracted_action,
            extracted_target,
            instruction,
            color_value=None
        )
        normalized_target = norm_info.get('normalized_target')
        effect_type = norm_info.get('effect_type')
        
        # For add_effect, use effect_type as target if available
        if extracted_action == 'add_effect' and effect_type and effect_type != 'unknown':
            normalized_target = effect_type
        
        # Comparisons
        action_match = (normalize_text(gt_action) == normalize_text(extracted_action))
        target_strict_match = (normalize_text(gt_target) == normalize_text(normalized_target or extracted_target))
        target_relaxed_match, match_reason = target_match_relaxed(gt_target, normalized_target or extracted_target)
        
        if action_match:
            action_action_match += 1
        if target_strict_match:
            action_target_strict_match += 1
        if target_relaxed_match:
            action_target_relaxed_match += 1
        if action_match and target_strict_match:
            joint_strict_match += 1
        if action_match and target_relaxed_match:
            joint_relaxed_match += 1
        
        # Track per-action
        if extracted_action not in action_stats:
            action_stats[extracted_action] = {
                'total': 0,
                'action_correct': 0,
                'target_relaxed_correct': 0,
                'joint_relaxed_correct': 0,
            }
        action_stats[extracted_action]['total'] += 1
        if action_match:
            action_stats[extracted_action]['action_correct'] += 1
        if target_relaxed_match:
            action_stats[extracted_action]['target_relaxed_correct'] += 1
        if action_match and target_relaxed_match:
            action_stats[extracted_action]['joint_relaxed_correct'] += 1
        
        # Record details
        detail = {
            'row_index': row_idx,
            'instruction': instruction,
            'gt_action': gt_action,
            'extracted_action': extracted_action,
            'gt_target': gt_target,
            'extracted_target': extracted_target,
            'normalized_target': normalized_target,
            'effect_type': effect_type,
            'action_match': action_match,
            'target_strict_match': target_strict_match,
            'target_relaxed_match': target_relaxed_match,
            'match_reason': match_reason,
        }
        results['details'].append(detail)
        
        if not (action_match and target_relaxed_match):
            results['mismatches'].append(detail)
    
    # Calculate overall metrics
    rows_eval = len(gt_data)
    results['rows_evaluated'] = rows_eval
    results['action_accuracy'] = action_action_match / rows_eval if rows_eval else 0
    results['target_accuracy_strict'] = action_target_strict_match / rows_eval if rows_eval else 0
    results['target_accuracy_relaxed'] = action_target_relaxed_match / rows_eval if rows_eval else 0
    results['joint_accuracy_strict'] = joint_strict_match / rows_eval if rows_eval else 0
    results['joint_accuracy_relaxed'] = joint_relaxed_match / rows_eval if rows_eval else 0
    
    # Per-action breakdown
    for action_name in sorted(action_stats.keys()):
        stats = action_stats[action_name]
        results['per_action'][action_name] = {
            'total': stats['total'],
            'action_correct': stats['action_correct'],
            'target_relaxed_correct': stats['target_relaxed_correct'],
            'joint_relaxed_correct': stats['joint_relaxed_correct'],
            'joint_relaxed_accuracy': stats['joint_relaxed_correct'] / stats['total'] if stats['total'] else 0,
        }
    
    return results


def write_results(results: Dict, output_dir: str, tag: str = "trial12"):
    """Write results to JSON, CSV, and MD."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Summary JSON
    summary_file = f"{output_dir}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'rows_evaluated': results['rows_evaluated'],
            'action_accuracy': results['action_accuracy'],
            'target_accuracy_strict': results['target_accuracy_strict'],
            'target_accuracy_relaxed': results['target_accuracy_relaxed'],
            'joint_accuracy_strict': results['joint_accuracy_strict'],
            'joint_accuracy_relaxed': results['joint_accuracy_relaxed'],
        }, f, indent=2)
    
    # Details CSV
    details_file = f"{output_dir}/details.csv"
    with open(details_file, 'w') as f:
        header = "row_index,gt_action,extracted_action,gt_target,extracted_target,normalized_target,action_match,target_relaxed_match,match_reason\n"
        f.write(header)
        for detail in results['details']:
            row = (
                f"{detail['row_index']},"
                f"{detail['gt_action']},"
                f"{detail['extracted_action']},"
                f'"{detail["gt_target"]}",'
                f'"{detail["extracted_target"]}",'
                f'"{detail["normalized_target"] or ""}",'
                f"{detail['action_match']},"
                f"{detail['target_relaxed_match']},"
                f"{detail['match_reason']}\n"
            )
            f.write(row)
    
    # Report MD
    report_file = f"{output_dir}/report.md"
    with open(report_file, 'w') as f:
        f.write(f"# Trial 12 Evaluation Report (Trial020 + normalize_target_for_functions)\n\n")
        f.write(f"**rows**: {results['rows_evaluated']}\n")
        f.write(f"**action_accuracy**: {results['action_accuracy']:.4f} ({int(results['action_accuracy'] * results['rows_evaluated'])}/{results['rows_evaluated']})\n")
        f.write(f"**target_accuracy_strict**: {results['target_accuracy_strict']:.4f}\n")
        f.write(f"**target_accuracy_relaxed**: {results['target_accuracy_relaxed']:.4f}\n")
        f.write(f"**joint_accuracy_strict**: {results['joint_accuracy_strict']:.4f}\n")
        f.write(f"**joint_accuracy_relaxed**: {results['joint_accuracy_relaxed']:.4f}\n\n")
        
        f.write("## Per-Action Breakdown (joint_accuracy_relaxed)\n\n")
        for action_name in sorted(results['per_action'].keys()):
            stats = results['per_action'][action_name]
            f.write(
                f"- **{action_name}**: "
                f"{stats['joint_relaxed_correct']}/{stats['total']} "
                f"({stats['joint_relaxed_accuracy']:.4f})\n"
            )
        
        f.write(f"\n## Mismatches ({len(results['mismatches'])} cases)\n\n")
        for i, mismatch in enumerate(results['mismatches'][:20]):  # Show first 20
            f.write(
                f"### {i+1}. {mismatch['gt_action']} vs {mismatch['extracted_action']}\n"
                f"- instruction: {mismatch['instruction'][:100]}\n"
                f"- GT target: {mismatch['gt_target']}\n"
                f"- extracted: {mismatch['extracted_target']}\n"
                f"- normalized: {mismatch['normalized_target']}\n"
                f"- reason: {mismatch['match_reason']}\n\n"
            )
    
    print(f"saved: {output_dir}")
    print(f"rows={results['rows_evaluated']} action={results['action_accuracy']:.4f} target_relaxed={results['target_accuracy_relaxed']:.4f} joint_relaxed={results['joint_accuracy_relaxed']:.4f}")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "eval100_trial12"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/logs/analysis/action_target_eval100/{tag}_{timestamp}"
    
    print(f"Evaluating 100 rows with trial12 strategy...", flush=True)
    results = evaluate_100rows(output_dir)
    write_results(results, output_dir, tag)
    print(f"done: tag={tag}", flush=True)


if __name__ == '__main__':
    main()
