from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path


def _read_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _read_mismatch_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    with (path / "mismatches.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["video_path"], row["action"]))
    return keys


def main() -> None:
    ap = ArgumentParser(description="Compare two instruction target eval trials")
    ap.add_argument("--trial-a", required=True, help="Path to trial A directory")
    ap.add_argument("--trial-b", required=True, help="Path to trial B directory")
    ap.add_argument(
        "--output",
        default="/workspace/logs/analysis/instruction_target_eval/trial_compare.md",
    )
    args = ap.parse_args()

    a = Path(args.trial_a)
    b = Path(args.trial_b)

    sa = _read_summary(a)
    sb = _read_summary(b)

    ma = _read_mismatch_keys(a)
    mb = _read_mismatch_keys(b)

    fixed = sorted(ma - mb)
    regressed = sorted(mb - ma)
    unchanged = sorted(ma & mb)

    lines: list[str] = []
    lines.append("# Trial Comparison")
    lines.append("")
    lines.append(f"- trial_a: {a}")
    lines.append(f"- trial_b: {b}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- trial_a accuracy: {sa['accuracy']:.4f} ({sa['matched']}/{sa['total']})")
    lines.append(f"- trial_b accuracy: {sb['accuracy']:.4f} ({sb['matched']}/{sb['total']})")
    lines.append(
        f"- delta: {(sb['accuracy'] - sa['accuracy']):+.4f} "
        f"({(sb['matched'] - sa['matched']):+d} matches)"
    )
    lines.append("")
    lines.append("## Per Action Delta")
    lines.append("")

    actions = sorted(set(sa["per_action"].keys()) | set(sb["per_action"].keys()))
    for action in actions:
        ia = sa["per_action"].get(action, {"matched": 0, "total": 0, "accuracy": 0.0})
        ib = sb["per_action"].get(action, {"matched": 0, "total": 0, "accuracy": 0.0})
        lines.append(
            f"- {action}: {ia['matched']}/{ia['total']} -> "
            f"{ib['matched']}/{ib['total']} "
            f"(delta {ib['accuracy'] - ia['accuracy']:+.4f})"
        )

    lines.append("")
    lines.append("## Mismatch Set Delta")
    lines.append("")
    lines.append(f"- fixed: {len(fixed)}")
    lines.append(f"- regressed: {len(regressed)}")
    lines.append(f"- unchanged_mismatch: {len(unchanged)}")

    if fixed:
        lines.append("")
        lines.append("### Fixed Cases")
        for vp, ac in fixed[:50]:
            lines.append(f"- {ac}: {vp}")

    if regressed:
        lines.append("")
        lines.append("### Regressed Cases")
        for vp, ac in regressed[:50]:
            lines.append(f"- {ac}: {vp}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
