#!/usr/bin/env python3
"""Validate apply_style rows in GT using dispatcher.py preprocessing behavior.

Outputs:
- summary.json
- apply_style_assessments.csv
- apply_style_report.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from parse.instruction_parser_v3_rulebase_trial020_singlefile import build_parser
from postprocess.dispatcher import _normalize_target_text, _resolve_action

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")
OUT_ROOT = Path("/workspace/logs/analysis/dispatcher_original_validation")


def _first_task(row: dict) -> dict:
    tasks = row.get("tasks") or []
    return tasks[0] if tasks else {}


def main() -> int:
    rows = json.loads(GT_PATH.read_text())
    parser = build_parser()

    apply_rows: list[dict] = []
    for idx, row in enumerate(rows):
        t0 = _first_task(row)
        if t0.get("action") == "apply_style":
            apply_rows.append({"row_index": idx, "row": row, "task": t0})

    results: list[dict] = []
    action_ok = 0
    target_ok = 0
    style_param_present = 0
    parsed_target_counter: Counter[str] = Counter()

    for item in apply_rows:
        idx = item["row_index"]
        row = item["row"]
        task = item["task"]
        instruction = row.get("instruction", "")

        inf = parser.infer(instruction)
        pred_action = inf["tasks"][0]["action"]
        pred_target = inf["tasks"][0]["target"]

        gt_action = task.get("action", "")
        gt_target = task.get("target", "")
        gt_params = task.get("params", {}) if isinstance(task.get("params"), dict) else {}

        input_params = {
            "video_id": row.get("video_path", ""),
            "instruction": instruction,
        }
        resolved_action = _resolve_action(pred_action, input_params)
        normalized_target = _normalize_target_text(pred_target)

        final_params = dict(input_params)
        final_params["action"] = resolved_action
        if normalized_target:
            final_params["target"] = normalized_target

        a_match = (gt_action == resolved_action)
        t_match = (gt_target == normalized_target)
        if a_match:
            action_ok += 1
        if t_match:
            target_ok += 1

        has_style = isinstance(gt_params, dict) and ("style" in gt_params)
        if has_style:
            style_param_present += 1

        parsed_target_counter[normalized_target] += 1

        results.append(
            {
                "row_index": idx,
                "video_path": row.get("video_path", ""),
                "gt_action": gt_action,
                "parsed_action": pred_action,
                "resolved_action": resolved_action,
                "action_match": a_match,
                "gt_target": gt_target,
                "parsed_target": pred_target,
                "normalized_target": normalized_target,
                "target_match": t_match,
                "gt_style": gt_params.get("style", ""),
                "gt_params": json.dumps(gt_params, ensure_ascii=False),
            }
        )

    total = len(results)
    both_ok = sum(1 for r in results if r["action_match"] and r["target_match"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"apply_style_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_apply_style_rows": total,
        "action_accuracy": (action_ok / total) if total else 0.0,
        "target_accuracy": (target_ok / total) if total else 0.0,
        "both_accuracy": (both_ok / total) if total else 0.0,
        "style_param_present_rate": (style_param_present / total) if total else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (out_dir / "apply_style_assessments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["row_index"])
        writer.writeheader()
        writer.writerows(results)

    top_targets = parsed_target_counter.most_common(5)
    action_miss = [r for r in results if not r["action_match"]]
    target_miss = [r for r in results if not r["target_match"]]

    md = []
    md.append("# apply_style validation with dispatcher.py\n")
    md.append(f"- total_apply_style_rows: {total}")
    md.append(f"- action_accuracy: {action_ok}/{total} ({(action_ok/total*100) if total else 0:.1f}%)")
    md.append(f"- target_accuracy: {target_ok}/{total} ({(target_ok/total*100) if total else 0:.1f}%)")
    md.append(f"- both_accuracy: {both_ok}/{total} ({(both_ok/total*100) if total else 0:.1f}%)")
    md.append(f"- style_param_present: {style_param_present}/{total} ({(style_param_present/total*100) if total else 0:.1f}%)\n")

    md.append("## Parsed target distribution (top5)")
    for t, c in top_targets:
        md.append(f"- {t}: {c}")

    md.append("\n## Action mismatches")
    if not action_miss:
        md.append("- none")
    else:
        for r in action_miss:
            md.append(
                f"- row {r['row_index']}: gt={r['gt_action']} parsed={r['parsed_action']} resolved={r['resolved_action']}"
            )

    md.append("\n## Target mismatches")
    if not target_miss:
        md.append("- none")
    else:
        for r in target_miss:
            md.append(
                f"- row {r['row_index']}: gt_target={r['gt_target']} normalized_target={r['normalized_target']}"
            )

    md.append("\n## Full rows")
    md.append("| row | video_path | gt_action | resolved_action | gt_target | normalized_target | gt_style |")
    md.append("|---|---|---|---|---|---|---|")
    for r in results:
        md.append(
            "| {row_index} | {video_path} | {gt_action} | {resolved_action} | {gt_target} | {normalized_target} | {gt_style} |".format(**r)
        )

    (out_dir / "apply_style_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"saved: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
