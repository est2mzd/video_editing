from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
import re
from typing import Any

from parse.instruction_parser_v3_rulebase_trial020_singlefile import build_parser


HUMAN_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "face",
    "head",
    "speaker",
    "profile",
    "subject",
    "human",
}


def _to_text(v: Any) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def _normalize_target(text: str) -> str:
    t = str(text).strip().lower()
    t = re.sub(r"\s*\.\s*$", "", t)
    t = t.replace("_", " ")
    t = t.replace("'s", "")
    t = re.sub(r"[^a-z0-9\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _looks_human(text: str) -> bool:
    tokens = set(_normalize_target(text).split())
    return any(w in tokens for w in HUMAN_WORDS)


def _token_jaccard(a: str, b: str) -> float:
    sa = set(_normalize_target(a).split())
    sb = set(_normalize_target(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _target_match_strict(gt: str, pred: str) -> bool:
    return _normalize_target(gt) == _normalize_target(pred)


def _target_match_relaxed(gt: str, pred: str) -> tuple[bool, str]:
    g = _normalize_target(gt)
    p = _normalize_target(pred)

    if g == p:
        return True, "exact"

    if not g or not p:
        return False, "empty"

    if _looks_human(g) and p == "person":
        return True, "human_collapsed_to_person"

    jac = _token_jaccard(g, p)
    if jac >= 0.5:
        return True, f"jaccard_{jac:.2f}"

    if p in g and len(p.split()) >= 1:
        return True, "pred_in_gt"
    if g in p and len(g.split()) >= 1:
        return True, "gt_in_pred"

    return False, f"mismatch_jaccard_{jac:.2f}"


def main() -> None:
    ap = ArgumentParser(description="Evaluate action+target accuracy on 100 GT rows")
    ap.add_argument(
        "--gt-path",
        default="/workspace/data/annotations_gt_task_ver10.json",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of rows to evaluate",
    )
    ap.add_argument(
        "--output-root",
        default="/workspace/logs/analysis/action_target_eval100",
    )
    ap.add_argument(
        "--tag",
        default="run",
    )
    args = ap.parse_args()

    rows = json.loads(Path(args.gt_path).read_text(encoding="utf-8"))
    rows = rows[: args.limit]

    parser = build_parser()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = out_dir / "details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "row_index",
                "video_path",
                "gt_action",
                "pred_action",
                "action_correct",
                "gt_target",
                "pred_target",
                "target_strict",
                "target_relaxed",
                "target_relaxed_reason",
                "joint_strict",
                "joint_relaxed",
            ]
        )

        total = 0
        action_ok = 0
        target_strict_ok = 0
        target_relaxed_ok = 0
        joint_strict_ok = 0
        joint_relaxed_ok = 0

        action_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "total": 0,
                "action_ok": 0,
                "target_relaxed_ok": 0,
                "joint_relaxed_ok": 0,
            }
        )

        for i, row in enumerate(rows, start=1):
            instruction = str(row.get("instruction", ""))
            video_path = str(row.get("video_path", ""))
            gt_tasks = row.get("tasks", [])
            if not gt_tasks:
                continue

            gt_action = str(gt_tasks[0].get("action", "")).strip()
            gt_target = _to_text(gt_tasks[0].get("target", "")).strip()

            pred = parser.infer(instruction)
            pred_tasks = pred.get("tasks", []) if isinstance(pred, dict) else []
            if pred_tasks:
                pred_action = str(pred_tasks[0].get("action", "")).strip()
                pred_target = _to_text(pred_tasks[0].get("target", "")).strip()
            else:
                pred_action = ""
                pred_target = ""

            action_correct = pred_action == gt_action
            target_strict = _target_match_strict(gt_target, pred_target)
            target_relaxed, reason = _target_match_relaxed(gt_target, pred_target)

            joint_strict = action_correct and target_strict
            joint_relaxed = action_correct and target_relaxed

            total += 1
            action_ok += int(action_correct)
            target_strict_ok += int(target_strict)
            target_relaxed_ok += int(target_relaxed)
            joint_strict_ok += int(joint_strict)
            joint_relaxed_ok += int(joint_relaxed)

            st = action_stats[gt_action]
            st["total"] += 1
            st["action_ok"] += int(action_correct)
            st["target_relaxed_ok"] += int(target_relaxed)
            st["joint_relaxed_ok"] += int(joint_relaxed)

            w.writerow(
                [
                    i,
                    video_path,
                    gt_action,
                    pred_action,
                    int(action_correct),
                    gt_target,
                    pred_target,
                    int(target_strict),
                    int(target_relaxed),
                    reason,
                    int(joint_strict),
                    int(joint_relaxed),
                ]
            )

    summary = {
        "rows_evaluated": total,
        "action_accuracy": action_ok / total if total else 0.0,
        "target_accuracy_strict": target_strict_ok / total if total else 0.0,
        "target_accuracy_relaxed": target_relaxed_ok / total if total else 0.0,
        "joint_accuracy_strict": joint_strict_ok / total if total else 0.0,
        "joint_accuracy_relaxed": joint_relaxed_ok / total if total else 0.0,
        "counts": {
            "action_ok": action_ok,
            "target_strict_ok": target_strict_ok,
            "target_relaxed_ok": target_relaxed_ok,
            "joint_strict_ok": joint_strict_ok,
            "joint_relaxed_ok": joint_relaxed_ok,
        },
        "per_action": {},
    }

    for action, st in sorted(action_stats.items()):
        tot = st["total"]
        summary["per_action"][action] = {
            "total": tot,
            "action_accuracy": st["action_ok"] / tot if tot else 0.0,
            "target_relaxed_accuracy": (
                st["target_relaxed_ok"] / tot if tot else 0.0
            ),
            "joint_relaxed_accuracy": (
                st["joint_relaxed_ok"] / tot if tot else 0.0
            ),
        }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines: list[str] = []
    md_lines.append("# Action + Target Accuracy (100 rows)")
    md_lines.append("")
    md_lines.append(f"- rows_evaluated: {summary['rows_evaluated']}")
    md_lines.append(
        f"- action_accuracy: {summary['action_accuracy']:.4f} "
        f"({action_ok}/{total})"
    )
    md_lines.append(
        f"- target_accuracy_strict: {summary['target_accuracy_strict']:.4f} "
        f"({target_strict_ok}/{total})"
    )
    md_lines.append(
        f"- target_accuracy_relaxed: {summary['target_accuracy_relaxed']:.4f} "
        f"({target_relaxed_ok}/{total})"
    )
    md_lines.append(
        f"- joint_accuracy_strict: {summary['joint_accuracy_strict']:.4f} "
        f"({joint_strict_ok}/{total})"
    )
    md_lines.append(
        f"- joint_accuracy_relaxed: {summary['joint_accuracy_relaxed']:.4f} "
        f"({joint_relaxed_ok}/{total})"
    )
    md_lines.append("")
    md_lines.append("## Per Action")
    md_lines.append("")
    for action, info in summary["per_action"].items():
        md_lines.append(
            f"- {action}: "
            f"action={info['action_accuracy']:.4f}, "
            f"target_relaxed={info['target_relaxed_accuracy']:.4f}, "
            f"joint_relaxed={info['joint_relaxed_accuracy']:.4f}, "
            f"n={info['total']}"
        )

    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"saved: {out_dir}")
    print(
        "rows={rows} action={a:.4f} target_relaxed={t:.4f} "
        "joint_relaxed={j:.4f}".format(
            rows=total,
            a=summary["action_accuracy"],
            t=summary["target_accuracy_relaxed"],
            j=summary["joint_accuracy_relaxed"],
        )
    )


if __name__ == "__main__":
    main()
