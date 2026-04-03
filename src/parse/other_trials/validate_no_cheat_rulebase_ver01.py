#!/usr/bin/env python3
"""Validation for no-cheat rule parser (ver01).

Outputs metrics for:
- GT: /workspace/data/annotations_gt_task_ver10.json
- grouped validation: ver01/ver02 (keys: ver2, ver3, ver4)
"""

from __future__ import annotations

import json
from pathlib import Path

from parse.other_trials.parser_no_cheat_rulebase_ver01 import NoCheatRuleParserV01

WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"
GROUPED_PATHS = [
    WORKSPACE / "data" / "annotations_grouped_ver01.json",
    WORKSPACE / "data" / "annotations_grouped_ver02.json",
]
INSTRUCTION_KEYS = ("ver2", "ver3", "ver4")


def _norm(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    text = str(value).lower().replace("_", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _target_match(pred: str, gt) -> bool:
    p = _norm(pred)
    g = _norm(gt)
    if not p or not g:
        return False
    return p == g or p in g or g in p


def evaluate_gt(parser: NoCheatRuleParserV01, gt_rows: list[dict]) -> dict:
    total = 0
    action_ok = 0
    target_ok = 0
    for row in gt_rows:
        tasks = row.get("tasks", [])
        if not tasks:
            continue
        gt_task = tasks[0]
        pred = parser.pred(row.get("instruction", ""))
        pred_task = pred["tasks"][0]
        total += 1
        if pred_task.get("action") == gt_task.get("action"):
            action_ok += 1
        if _target_match(
            pred_task.get("target", ""),
            gt_task.get("target", ""),
        ):
            target_ok += 1
    return {
        "count": total,
        "action_accuracy": (action_ok / total) if total else 0.0,
        "target_accuracy": (target_ok / total) if total else 0.0,
    }


def evaluate_grouped(
    parser: NoCheatRuleParserV01,
    gt_rows: list[dict],
    grouped_paths: list[Path],
) -> dict:
    gt_by_video = {row["video_path"]: row for row in gt_rows}
    total = 0
    action_ok = 0
    target_ok = 0

    for grouped_path in grouped_paths:
        grouped_rows = json.loads(grouped_path.read_text(encoding="utf-8"))
        for row in grouped_rows:
            video = row.get("video_path")
            if not video or video not in gt_by_video:
                continue
            tasks = gt_by_video[video].get("tasks", [])
            if not tasks:
                continue
            gt_task = tasks[0]
            for key in INSTRUCTION_KEYS:
                instruction = row.get(key)
                if not instruction:
                    continue
                pred = parser.pred(instruction)
                pred_task = pred["tasks"][0]
                total += 1
                if pred_task.get("action") == gt_task.get("action"):
                    action_ok += 1
                if _target_match(
                    pred_task.get("target", ""),
                    gt_task.get("target", ""),
                ):
                    target_ok += 1

    return {
        "count": total,
        "action_accuracy": (action_ok / total) if total else 0.0,
        "target_accuracy": (target_ok / total) if total else 0.0,
    }


def main() -> None:
    print("=" * 80)
    print("No-Cheat Rulebase Validation v01")
    print("=" * 80)

    gt_rows = json.loads(GT_PATH.read_text(encoding="utf-8"))
    parser = NoCheatRuleParserV01()

    gt_result = evaluate_gt(parser, gt_rows)
    grouped_result = evaluate_grouped(parser, gt_rows, GROUPED_PATHS)

    print("\n[GT metrics]")
    print(f"count           : {gt_result['count']}")
    print(f"action_accuracy : {gt_result['action_accuracy'] * 100:.2f}%")
    print(f"target_accuracy : {gt_result['target_accuracy'] * 100:.2f}%")

    print("\n[grouped metrics]")
    print(f"count           : {grouped_result['count']}")
    print(f"action_accuracy : {grouped_result['action_accuracy'] * 100:.2f}%")
    print(f"target_accuracy : {grouped_result['target_accuracy'] * 100:.2f}%")

    gt_goal_ok = (
        gt_result["action_accuracy"] >= 0.80
        and gt_result["target_accuracy"] >= 0.80
    )
    grouped_goal_ok = (
        grouped_result["action_accuracy"] >= 0.70
        and grouped_result["target_accuracy"] >= 0.70
    )

    print("\n[goal check]")
    gt_status = "PASS" if gt_goal_ok else "FAIL"
    grouped_status = "PASS" if grouped_goal_ok else "FAIL"
    print(f"GT goal (action>80%, target>80%)             : {gt_status}")
    print(
        "Grouped goal (action>=70%, target>=70%)      : "
        f"{grouped_status}"
    )


if __name__ == "__main__":
    main()
