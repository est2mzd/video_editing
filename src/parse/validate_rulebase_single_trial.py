#!/usr/bin/env python3
"""Validate one parser trial at a time.

Usage:
    python3 src/parse/validate_rulebase_single_trial.py \
        --parser-file /workspace/src/parse/
            prototype_instruction_parser_v3_improved.py \
        --trial-name trial_001
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"
GROUPED_PATHS = [
    WORKSPACE / "data" / "annotations_grouped_ver01.json",
    WORKSPACE / "data" / "annotations_grouped_ver02.json",
]
GROUPED_KEYS = ("ver2", "ver3", "ver4")


def _norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    s = str(v).lower().replace("_", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _target_ok(pred: str, gt) -> bool:
    p = _norm(pred)
    g = _norm(gt)
    if not p or not g:
        return False
    return p == g or p in g or g in p


def _load_parser(parser_file: Path):
    spec = importlib.util.spec_from_file_location("trial_module", parser_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser file: {parser_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if (
        hasattr(mod, "build_knowledge_db_v3")
        and hasattr(mod, "InstructionParserV3")
    ):
        kb = mod.build_knowledge_db_v3(GT_PATH)
        return mod.InstructionParserV3(kb)

    if hasattr(mod, "build_parser"):
        return mod.build_parser()

    raise RuntimeError(
        "Parser file must provide build_parser() or "
        "(build_knowledge_db_v3 + InstructionParserV3)."
    )


def _iter_progress(items, show_progress: bool, desc: str):
    if show_progress and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def _predict_tasks(
    parser,
    instructions: list[str],
    batch_size: int,
) -> list[dict]:
    if not instructions:
        return []

    if hasattr(parser, "pred_batch") and batch_size > 1:
        try:
            preds = parser.pred_batch(instructions, batch_size=batch_size)
        except TypeError:
            preds = parser.pred_batch(instructions)
        out = []
        for pred in preds:
            tasks = pred.get("tasks", []) if isinstance(pred, dict) else []
            out.append(tasks[0] if tasks else {})
        return out

    out = []
    for inst in instructions:
        tasks = parser.pred(inst).get("tasks", [])
        out.append(tasks[0] if tasks else {})
    return out


def _evaluate_gt(
    parser,
    gt_rows: list[dict],
    eval_batch_size: int,
    show_progress: bool,
) -> tuple[dict, list[dict]]:
    total = 0
    action_ok = 0
    target_ok = 0
    failures = []

    valid_rows = []
    for row in gt_rows:
        if row.get("tasks", []):
            valid_rows.append(row)

    step = max(1, int(eval_batch_size))
    chunks = range(0, len(valid_rows), step)
    chunks = _iter_progress(chunks, show_progress, "GT eval")

    for i in chunks:
        batch_rows = valid_rows[i:i + step]
        instructions = [r.get("instruction", "") for r in batch_rows]
        pred_tasks = _predict_tasks(parser, instructions, step)

        for row, pred_task in zip(batch_rows, pred_tasks):
            gt_task = row["tasks"][0]

            total += 1
            action_match = pred_task.get("action") == gt_task.get("action")
            target_match = _target_ok(
                pred_task.get("target", ""),
                gt_task.get("target", ""),
            )

            action_ok += int(action_match)
            target_ok += int(target_match)

            if not action_match or not target_match:
                failures.append(
                    {
                        "video_path": row.get("video_path", ""),
                        "instruction": row.get("instruction", "")[:200],
                        "gt_action": gt_task.get("action", ""),
                        "pred_action": pred_task.get("action", ""),
                        "gt_target": str(gt_task.get("target", ""))[:120],
                        "pred_target": str(pred_task.get("target", ""))[:120],
                    }
                )

    result = {
        "count": total,
        "action_accuracy": action_ok / total if total else 0.0,
        "target_accuracy": target_ok / total if total else 0.0,
    }
    return result, failures


def _evaluate_grouped(
    parser,
    gt_rows: list[dict],
    eval_batch_size: int,
    show_progress: bool,
) -> dict:
    gt_by_video = {row["video_path"]: row for row in gt_rows}

    total = 0
    action_ok = 0
    target_ok = 0

    eval_items = []
    for grouped_path in GROUPED_PATHS:
        rows = json.loads(grouped_path.read_text(encoding="utf-8"))
        for row in rows:
            video = row.get("video_path")
            if not video or video not in gt_by_video:
                continue

            gt_tasks = gt_by_video[video].get("tasks", [])
            if not gt_tasks:
                continue
            gt_task = gt_tasks[0]

            for key in GROUPED_KEYS:
                inst = row.get(key)
                if not inst:
                    continue
                eval_items.append((inst, gt_task))

    step = max(1, int(eval_batch_size))
    chunks = range(0, len(eval_items), step)
    chunks = _iter_progress(chunks, show_progress, "Grouped eval")

    for i in chunks:
        batch = eval_items[i:i + step]
        instructions = [x[0] for x in batch]
        pred_tasks = _predict_tasks(parser, instructions, step)
        for (_, gt_task), pred_task in zip(batch, pred_tasks):
            total += 1
            action_ok += int(pred_task.get("action") == gt_task.get("action"))
            target_ok += int(
                _target_ok(
                    pred_task.get("target", ""),
                    gt_task.get("target", ""),
                )
            )

    return {
        "count": total,
        "action_accuracy": action_ok / total if total else 0.0,
        "target_accuracy": target_ok / total if total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser-file", required=True)
    ap.add_argument("--trial-name", required=True)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--show-progress", action="store_true")
    args = ap.parse_args()

    parser_file = Path(args.parser_file)
    if not parser_file.exists():
        raise SystemExit(f"Parser file not found: {parser_file}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = WORKSPACE / "logs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = _load_parser(parser_file)
    gt_rows = json.loads(GT_PATH.read_text(encoding="utf-8"))

    gt_result, failures = _evaluate_gt(
        parser,
        gt_rows,
        eval_batch_size=max(1, int(args.eval_batch_size)),
        show_progress=bool(args.show_progress),
    )
    grouped_result = _evaluate_grouped(
        parser,
        gt_rows,
        eval_batch_size=max(1, int(args.eval_batch_size)),
        show_progress=bool(args.show_progress),
    )

    payload = {
        "timestamp": ts,
        "trial_name": args.trial_name,
        "parser_file": str(parser_file),
        "eval_batch_size": max(1, int(args.eval_batch_size)),
        "show_progress": bool(args.show_progress),
        "gt": gt_result,
        "grouped": grouped_result,
        "goals": {
            "gt_goal": (
                gt_result["action_accuracy"] > 0.80
                and gt_result["target_accuracy"] > 0.80
            ),
            "grouped_goal": (
                grouped_result["action_accuracy"] >= 0.70
                and grouped_result["target_accuracy"] >= 0.70
            ),
        },
        "failure_samples": failures[:20],
    }

    out_json = out_dir / f"{args.trial_name}_{ts}.json"
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Trial: {args.trial_name}")
    print(f"Parser: {parser_file}")
    print("=" * 80)
    print("[GT]")
    print(f"count: {gt_result['count']}")
    print(f"action_accuracy: {gt_result['action_accuracy'] * 100:.2f}%")
    print(f"target_accuracy: {gt_result['target_accuracy'] * 100:.2f}%")
    print("[Grouped]")
    print(f"count: {grouped_result['count']}")
    print(f"action_accuracy: {grouped_result['action_accuracy'] * 100:.2f}%")
    print(f"target_accuracy: {grouped_result['target_accuracy'] * 100:.2f}%")
    print("[Goal]")
    print(f"gt_goal: {'PASS' if payload['goals']['gt_goal'] else 'FAIL'}")
    grouped_status = "PASS" if payload["goals"]["grouped_goal"] else "FAIL"
    print(f"grouped_goal: {grouped_status}")
    print(f"result_json: {out_json}")


if __name__ == "__main__":
    main()
