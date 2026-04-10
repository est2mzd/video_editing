#!/usr/bin/env python3
"""
Trial 14: 初期 dispatcher.py の action/target/params 処理を検証
- 目的: 初期 dispatcher.py が1行目（row 0）の action/target/params をどう解釈するか確認
- 方法: instruction → action/target 抽出 → params 構築 → dispatcher flow 再現
- 出力: 検証結果を .md に記録
"""

import json
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / ".."))

from parse.instruction_parser_v3_rulebase_trial020_singlefile import (
    build_parser,
)

def normalize_target_text(targets):
    """Replicate dispatcher._normalize_target_text()"""
    if targets is None:
        return ""
    if isinstance(targets, (list, tuple, set)):
        parts = [str(t).strip() for t in targets if str(t).strip()]
        return ", ".join(parts)
    return str(targets).strip()


def resolve_action(action, params):
    """Replicate dispatcher._resolve_action()"""
    direct = str(action or "").strip()
    if direct:
        return direct

    for key in ("action", "_action"):
        value = str(params.get(key, "")).strip()
        if value:
            return value

    method = str(params.get("method", "")).strip().lower()
    method_to_action = {
        "stylize": "apply_style",
        "object_zoom_in": "dolly_in",
        "stable_zoom_in": "zoom_in",
        "zoom_out": "zoom_out",
        "perspective_warp": "change_camera_angle",
        "horizontal_shift": "orbit_camera",
        "blur_or_brightness": "add_effect",
        "replace_background": "replace_background",
        "change_background_color": "change_color",
        "inpaint": "remove_object",
    }
    return method_to_action.get(method, "")


def main():
    gt_path = Path("/workspace/data/annotations_gt_task_ver10.json")
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}")
        return

    # Load first row
    with open(gt_path, "r") as f:
        all_rows = json.load(f)
    
    if not all_rows:
        print("GT file is empty")
        return

    row_data = all_rows[0]
    row_idx = 0

    print(f"\n{'='*80}")
    print(f"Trial 14 Validation: dispatcher.py behavior on row {row_idx}")
    print(f"{'='*80}\n")

    # Parse instruction
    instruction = row_data.get("instruction", "")
    parser = build_parser()
    result_dict = parser.infer(instruction)
    action_parsed = result_dict["tasks"][0]["action"]
    target_parsed = result_dict["tasks"][0]["target"]

    # Extract GT values from tasks
    gt_task = row_data.get("tasks", [{}])[0] if row_data.get("tasks") else {}
    gt_action = gt_task.get("action", "N/A")
    gt_target = gt_task.get("target", "N/A")

    print(f"[Input]")
    print(f"  instruction: {instruction[:100]}..." if len(instruction) > 100 else f"  instruction: {instruction}")
    print(f"  GT action:   {gt_action}")
    print(f"  GT target:   {gt_target}")
    print()

    # Dispatcher flow
    print(f"[Parser Output (trial020)]")
    print(f"  parsed_action: {action_parsed}")
    print(f"  parsed_target: {target_parsed}")
    print()

    # Replicate dispatcher flow
    print(f"[dispatcher._normalize_target_text()]")
    normalized_target = normalize_target_text(target_parsed)
    print(f"  input:  {repr(target_parsed)}")
    print(f"  output: {repr(normalized_target)}")
    print()

    # Build params dict (minimal)
    params = {
        "video_id": row_data.get("video_id", ""),
        "instruction": instruction,
    }
    
    print(f"[dispatcher._resolve_action()]")
    resolved_action = resolve_action(action_parsed, params)
    print(f"  input_action: {repr(action_parsed)}")
    print(f"  input_params: {repr(params)}")
    print(f"  output_action: {repr(resolved_action)}")
    print()

    # Final params after dispatcher processing
    params["action"] = resolved_action
    if normalized_target:
        params["target"] = normalized_target
    params.setdefault("instruction", instruction)

    print(f"[Final params (after dispatcher.run_method setup)]")
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, str) and len(val) > 80:
            print(f"  {key}: {val[:80]}...")
        else:
            print(f"  {key}: {repr(val)}")
    print()

    # Comparison
    print(f"[Comparison: GT vs Dispatcher Output]")
    print(f"  GT target:           {repr(gt_target)}")
    print(f"  normalized_target:   {repr(normalized_target)}")
    print(f"  match (strict):      {gt_target == normalized_target}")
    print()

    print(f"  GT action:           {repr(gt_action)}")
    print(f"  resolved_action:     {repr(resolved_action)}")
    print(f"  match (direct):      {gt_action == resolved_action}")
    print()

    # Summary
    report = {
        "row_index": row_idx,
        "instruction": instruction,
        "gt_action": gt_action,
        "gt_target": gt_target,
        "parsed_action": action_parsed,
        "parsed_target": target_parsed,
        "normalized_target": normalized_target,
        "resolved_action": resolved_action,
        "final_params": params,
        "action_match": gt_action == resolved_action,
        "target_match": gt_target == normalized_target,
    }

    return report


if __name__ == "__main__":
    try:
        result = main()
        if result:
            print(f"\n[JSON Report]")
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
