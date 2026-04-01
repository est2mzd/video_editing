#!/usr/bin/env python3
"""Build per-mp4 summary table for ver03 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/workspace/data/annotations.jsonl")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[str(obj["video_path"])] = obj
    return out


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["video_path"]): row for row in rows}


def load_eval_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {str(row["video_path"]): row for row in rows}


def score_comment(row: dict[str, Any]) -> str:
    cls = row.get("selected_class", "")
    parts: list[str] = []
    if cls == "Camera Motion Editing":
        zoom_proxy = row.get("zoom_proxy")
        if zoom_proxy is not None:
            parts.append(f"zoom_proxy={zoom_proxy:.4f}")
    if cls == "Visual Effect Editing" and row.get("selected_subclass") == "Background Change":
        ratio = row.get("bg_change_ratio")
        if ratio is not None:
            parts.append(f"bg_change_ratio={ratio:.4f}")
    if cls == "Attribute Editing" and row.get("selected_subclass") == "Color adjustment":
        hue = row.get("target_hue_score")
        if hue is not None:
            parts.append(f"target_hue_score={hue:.4f}")
    if "mean_abs_diff" in row:
        parts.append(f"mean_abs_diff={row['mean_abs_diff']:.4f}")
    return "; ".join(parts)


def main() -> int:
    args = parse_args()
    annotations = load_annotations(Path(args.annotations))
    manifest = load_manifest(Path(args.manifest))
    eval_rows = load_eval_rows(Path(args.eval_report))
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))

    out_rows: list[dict[str, Any]] = []
    for video_path, ann in annotations.items():
        row = {
            "video_path": video_path,
            "selected_class": ann.get("selected_class", ""),
            "selected_subclass": ann.get("selected_subclass", ""),
            "instruction": ann.get("instruction", ""),
            "manifest_status": "",
            "codec": "",
            "input_frames": "",
            "output_frames": "",
            "mean_abs_diff": "",
            "center_diff": "",
            "border_diff": "",
            "zoom_proxy": "",
            "bg_change_ratio": "",
            "target_hue_score": "",
            "summary_comment": "",
            "validation_status": validation.get("status", ""),
        }
        m = manifest.get(video_path, {})
        e = eval_rows.get(video_path, {})
        row["manifest_status"] = m.get("status", "")
        row["codec"] = m.get("codec", "")
        row["input_frames"] = m.get("input_frames", "")
        row["output_frames"] = m.get("output_frames", "")
        for key in [
            "mean_abs_diff",
            "center_diff",
            "border_diff",
            "zoom_proxy",
            "bg_change_ratio",
            "target_hue_score",
        ]:
            row[key] = e.get(key, "")
        row["summary_comment"] = score_comment({**ann, **e})
        out_rows.append(row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[INFO] Wrote summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
