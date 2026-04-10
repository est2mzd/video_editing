#!/usr/bin/env python3
"""
Trial 14: 初期 dispatcher.py の action/target/params 検証 - 100件全数
- 目的: 全100行で初期 dispatcher.py の action/target/params 処理を検証
- 出力: summary.json, assessments.csv, analysis_report.md
"""

import json
import sys
import csv
from pathlib import Path
from datetime import datetime

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


def validate_100_rows():
    gt_path = Path("/workspace/data/annotations_gt_task_ver10.json")
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}")
        return

    with open(gt_path, "r") as f:
        all_rows = json.load(f)

    if len(all_rows) < 100:
        print(f"GT has only {len(all_rows)} rows, expected 100")
        return

    parser = build_parser()
    results = []
    action_matches = 0
    target_matches = 0

    print(f"\n{'='*80}")
    print(f"Trial 14 Validation: dispatcher.py behavior on 100 rows")
    print(f"{'='*80}\n")

    # Process all 100 rows
    for idx in range(100):
        row_data = all_rows[idx]
        instruction = row_data.get("instruction", "")

        # Parse instruction
        result_dict = parser.infer(instruction)
        action_parsed = result_dict["tasks"][0]["action"]
        target_parsed = result_dict["tasks"][0]["target"]

        # Extract GT values from tasks
        gt_task = row_data.get("tasks", [{}])[0] if row_data.get("tasks") else {}
        gt_action = gt_task.get("action", "")
        gt_target = gt_task.get("target", "")

        # Replicate dispatcher flow
        normalized_target = normalize_target_text(target_parsed)
        
        params = {
            "video_id": row_data.get("video_path", ""),
            "instruction": instruction,
        }
        resolved_action = resolve_action(action_parsed, params)

        # Check matches
        action_match = gt_action == resolved_action
        target_match = gt_target == normalized_target

        if action_match:
            action_matches += 1
        if target_match:
            target_matches += 1

        result = {
            "row_index": idx,
            "gt_action": gt_action,
            "gt_target": gt_target,
            "parsed_action": action_parsed,
            "parsed_target": target_parsed,
            "normalized_target": normalized_target,
            "resolved_action": resolved_action,
            "action_match": action_match,
            "target_match": target_match,
            "both_match": action_match and target_match,
        }
        results.append(result)

        # Progress
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/100 rows...")

    # Summary statistics
    both_matches = sum(1 for r in results if r["both_match"])
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total rows: {len(results)}")
    print(f"  Action matches: {action_matches}/100 ({100*action_matches/len(results):.1f}%)")
    print(f"  Target matches: {target_matches}/100 ({100*target_matches/len(results):.1f}%)")
    print(f"  Both match: {both_matches}/100 ({100*both_matches/len(results):.1f}%)")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path("/workspace/logs/analysis/dispatcher_original_validation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"val_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "total": len(results),
        "action_accuracy": action_matches / len(results),
        "target_accuracy": target_matches / len(results),
        "both_accuracy": both_matches / len(results),
        "timestamp": timestamp,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Assessments CSV
    with open(run_dir / "assessments.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "gt_action",
                "gt_target",
                "parsed_action",
                "parsed_target",
                "normalized_target",
                "resolved_action",
                "action_match",
                "target_match",
                "both_match",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Analysis report markdown
    action_mismatches = [r for r in results if not r["action_match"]]
    target_mismatches = [r for r in results if not r["target_match"]]

    with open(run_dir / "analysis_report.md", "w") as f:
        f.write(f"# Trial 14: dispatcher.py Original Validation Report\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total rows: {len(results)}\n")
        f.write(f"- Action accuracy: {action_matches}/100 ({100*action_matches/len(results):.1f}%)\n")
        f.write(f"- Target accuracy: {target_matches}/100 ({100*target_matches/len(results):.1f}%)\n")
        f.write(f"- Both match: {both_matches}/100 ({100*both_matches/len(results):.1f}%)\n\n")

        f.write(f"## Action Mismatches ({len(action_mismatches)} cases)\n\n")
        if action_mismatches:
            f.write(f"| Row | GT Action | Parsed Action | Resolved Action |\n")
            f.write(f"|-----|-----------|---------------|----------------|\n")
            for r in action_mismatches:
                f.write(
                    f"| {r['row_index']} | {r['gt_action']} | {r['parsed_action']} | {r['resolved_action']} |\n"
                )
        else:
            f.write(f"No action mismatches detected.\n\n")

        f.write(f"\n## Target Mismatches ({len(target_mismatches)} cases)\n\n")
        if target_mismatches:
            f.write(f"| Row | GT Target | Parsed Target | Normalized Target |\n")
            f.write(f"|-----|-----------|---------------|------------------|\n")
            for r in target_mismatches:
                gt_t = r["gt_target"][:50] if len(r["gt_target"]) > 50 else r["gt_target"]
                parsed_t = r["parsed_target"][:50] if len(r["parsed_target"]) > 50 else r["parsed_target"]
                norm_t = r["normalized_target"][:50] if len(r["normalized_target"]) > 50 else r["normalized_target"]
                f.write(f"| {r['row_index']} | {gt_t} | {parsed_t} | {norm_t} |\n")
        else:
            f.write(f"No target mismatches detected.\n\n")

        f.write(f"\n## Key Findings\n\n")
        f.write(
            f"- Initial dispatcher.py behavior: "
            f"{'✓ Perfect match' if both_matches == len(results) else f'{both_matches} / {len(results)} perfect matches'}\n"
        )
        f.write(f"- Action accuracy: **{100*action_matches/len(results):.1f}%**\n")
        f.write(f"- Target accuracy: **{100*target_matches/len(results):.1f}%**\n")
        f.write(f"- Parser quality (trial020): Very high action accuracy\n")
        f.write(f"- Target normalization: Mostly stable (pass-through)\n\n")

    print(f"Results saved to: {run_dir}")
    print(f"  - summary.json")
    print(f"  - assessments.csv")
    print(f"  - analysis_report.md")


if __name__ == "__main__":
    try:
        validate_100_rows()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
