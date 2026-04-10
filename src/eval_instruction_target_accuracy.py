from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
import re
from typing import Any

from postprocess.add_effect import parse_add_effect_instruction
from postprocess.add_object import parse_add_object_instruction
from postprocess.change_color import parse_color_change_instruction
from postprocess.dolly_in import extract_dolly_in_target
from postprocess.zoom_in import parse_zoom_instruction_rulebase


HUMAN_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "face",
    "head",
    "speaker",
    "businessman",
    "profile",
    "subject",
    "human",
}


@dataclass
class EvalRecord:
    video_path: str
    action: str
    instruction: str
    gt_target_raw: str
    pred_target_raw: str
    gt_norm: str
    pred_norm: str
    matched: bool
    reason: str


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _strip_prompt_suffix(text: str) -> str:
    t = str(text).strip()
    t = re.sub(r"\s*\.\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize(text: str) -> str:
    t = _strip_prompt_suffix(text).lower()
    t = t.replace("_", " ")
    t = t.replace("'s", "")
    t = re.sub(r"[^a-z0-9\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _looks_like_human(text: str) -> bool:
    tokens = set(_normalize(text).split())
    return any(w in tokens for w in HUMAN_WORDS)


def _token_jaccard(a: str, b: str) -> float:
    sa = set(_normalize(a).split())
    sb = set(_normalize(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _predict_target(action: str, instruction: str) -> str:
    if action == "dolly_in":
        return _strip_prompt_suffix(extract_dolly_in_target(instruction))

    if action == "zoom_in":
        parsed = parse_zoom_instruction_rulebase(instruction)
        return _strip_prompt_suffix(parsed.target_object)

    if action == "add_effect":
        parsed = parse_add_effect_instruction(instruction)
        return _strip_prompt_suffix(parsed.grounding_target)

    if action in {"add_object", "increase_amount"}:
        parsed = parse_add_object_instruction(instruction)
        return _strip_prompt_suffix(parsed.target_object)

    if action == "change_color":
        parsed = parse_color_change_instruction(instruction)
        return _strip_prompt_suffix(parsed.target_object)

    if action == "replace_background":
        return "background"

    if action == "apply_style":
        return "full frame"

    return ""


def _match(action: str, gt: str, pred: str) -> tuple[bool, str]:
    gt_n = _normalize(gt)
    pr_n = _normalize(pred)

    if not gt_n and not pr_n:
        return True, "both_empty"
    if not gt_n or not pr_n:
        return False, "one_empty"

    if pr_n == gt_n:
        return True, "exact"

    if action in {"dolly_in", "zoom_in"}:
        if _looks_like_human(gt_n) and pr_n == "person":
            return True, "human_collapsed_to_person"

    jac = _token_jaccard(gt_n, pr_n)
    if jac >= 0.5:
        return True, f"token_jaccard_{jac:.2f}"

    if pr_n in gt_n and len(pr_n.split()) >= 1:
        return True, "pred_in_gt"
    if gt_n in pr_n and len(gt_n.split()) >= 1:
        return True, "gt_in_pred"

    return False, f"mismatch_jaccard_{jac:.2f}"


def evaluate(annotation_path: Path) -> list[EvalRecord]:
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    records: list[EvalRecord] = []

    for row in rows:
        video_path = str(row.get("video_path", ""))
        instruction = str(row.get("instruction", ""))
        for task in row.get("tasks", []):
            action = str(task.get("action", "")).strip()
            if action not in {
                "dolly_in",
                "zoom_in",
                "add_effect",
                "add_object",
                "increase_amount",
                "change_color",
                "replace_background",
                "apply_style",
            }:
                continue

            gt_target_raw = _to_text(task.get("target", ""))
            pred_target_raw = _predict_target(action, instruction)
            matched, reason = _match(action, gt_target_raw, pred_target_raw)

            records.append(
                EvalRecord(
                    video_path=video_path,
                    action=action,
                    instruction=instruction,
                    gt_target_raw=gt_target_raw,
                    pred_target_raw=pred_target_raw,
                    gt_norm=_normalize(gt_target_raw),
                    pred_norm=_normalize(pred_target_raw),
                    matched=matched,
                    reason=reason,
                )
            )

    return records


def write_reports(records: list[EvalRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_action: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_action[r.action].append(r)

    summary = {
        "total": len(records),
        "matched": sum(1 for r in records if r.matched),
        "accuracy": (sum(1 for r in records if r.matched) / len(records))
        if records
        else 0.0,
        "per_action": {},
    }

    for action, rows in sorted(by_action.items()):
        ok = sum(1 for r in rows if r.matched)
        summary["per_action"][action] = {
            "total": len(rows),
            "matched": ok,
            "accuracy": ok / len(rows) if rows else 0.0,
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "details.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video_path",
                "action",
                "matched",
                "reason",
                "gt_target_raw",
                "pred_target_raw",
                "gt_norm",
                "pred_norm",
                "instruction",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.video_path,
                    r.action,
                    int(r.matched),
                    r.reason,
                    r.gt_target_raw,
                    r.pred_target_raw,
                    r.gt_norm,
                    r.pred_norm,
                    r.instruction,
                ]
            )

    mismatches = [r for r in records if not r.matched]
    with (output_dir / "mismatches.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video_path",
                "action",
                "reason",
                "gt_target_raw",
                "pred_target_raw",
                "instruction",
            ]
        )
        for r in mismatches:
            w.writerow(
                [
                    r.video_path,
                    r.action,
                    r.reason,
                    r.gt_target_raw,
                    r.pred_target_raw,
                    r.instruction,
                ]
            )

    lines: list[str] = []
    lines.append("# Instruction Target Accuracy Report")
    lines.append("")
    lines.append(f"- total: {summary['total']}")
    lines.append(f"- matched: {summary['matched']}")
    lines.append(f"- accuracy: {summary['accuracy']:.4f}")
    lines.append("")
    lines.append("## Per Action")
    lines.append("")
    for action, info in summary["per_action"].items():
        lines.append(
            f"- {action}: {info['matched']}/{info['total']} ({info['accuracy']:.4f})"
        )
    lines.append("")
    lines.append(f"- mismatches: {len(mismatches)}")

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Evaluate instruction target extraction vs GT")
    parser.add_argument(
        "--annotation-path",
        default="/workspace/data/annotations_gt_task_ver10.json",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/logs/analysis/instruction_target_eval",
    )
    parser.add_argument(
        "--tag",
        default="manual",
        help="Run tag stored in output path",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{args.tag}_{ts}"

    records = evaluate(Path(args.annotation_path))
    write_reports(records, out_dir)

    print(f"saved: {out_dir}")
    print(f"total={len(records)} matched={sum(1 for r in records if r.matched)}")


if __name__ == "__main__":
    main()
