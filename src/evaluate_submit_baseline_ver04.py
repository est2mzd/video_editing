#!/usr/bin/env python3
"""Command-level proxy evaluation for ver04."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.evaluate_submit_baseline_ver03 import (
    center_border_diff,
    edge_strength,
    estimate_zoom_score,
    hue_target_score,
    mean_abs_diff,
    pixelation_score,
    read_sample_frames,
    saturation_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/workspace/data/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parsed-commands", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def command_proxy(command: dict[str, Any], inp: np.ndarray, out: np.ndarray) -> tuple[float, str]:
    action = str(command.get("action", ""))
    value = str(command.get("value", ""))
    center_diff, border_diff = center_border_diff(inp, out)
    if action in {"zoom_in", "dolly_in"}:
        score = estimate_zoom_score(inp, out, zoom_in=True)
        return score, "zoom_proxy"
    if action == "zoom_out":
        score = estimate_zoom_score(inp, out, zoom_in=False)
        return score, "zoom_proxy"
    if action == "replace_background":
        score = border_diff / max(center_diff, 1e-6)
        return score, "bg_change_ratio"
    if action == "change_color":
        score = hue_target_score(out, value or "violet") - hue_target_score(inp, value or "violet")
        return score, "target_hue_score"
    if action == "apply_style":
        score = abs((saturation_mean(out) - saturation_mean(inp))) + abs(edge_strength(out) - edge_strength(inp)) + abs(pixelation_score(out) - pixelation_score(inp))
        return score, "style_energy"
    return mean_abs_diff(inp, out), "mean_abs_diff"


def main() -> int:
    args = parse_args()
    parsed_rows = json.loads(Path(args.parsed_commands).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    per_command: list[dict[str, Any]] = []
    per_mp4: list[dict[str, Any]] = []
    for row in parsed_rows:
        name = row["video_path"]
        input_frames, input_count = read_sample_frames(input_dir / name)
        output_frames, output_count = read_sample_frames(output_dir / name)
        if not input_frames or not output_frames:
            continue
        inp = input_frames[-1]
        out = output_frames[-1]
        edit_scores = []
        preserve_scores = []
        quality_scores = []
        for command in row["commands"]:
            score, metric_name = command_proxy(command, inp, out)
            status = "ok" if score > 0 else "weak"
            record = {
                "video_path": name,
                "command_id": command["command_id"],
                "command_type": command["type"],
                "target": command["target"],
                "action": command["action"],
                "value": command["value"],
                "metric_name": metric_name,
                "metric_value": score,
                "status": status,
                "source_text": command["source_text"],
            }
            per_command.append(record)
            if command["type"] == "edit":
                edit_scores.append(1.0 if score > 0 else 0.0)
            elif command["type"] == "preserve":
                preserve_scores.append(1.0 if mean_abs_diff(inp, out) < 15.0 else 0.0)
            elif command["type"] == "quality":
                quality_scores.append(1.0 if output_count == input_count else 0.0)

        per_mp4.append(
            {
                "video_path": name,
                "instruction": row["instruction"],
                "parsed_command_count": len(row["commands"]),
                "edit_command_count": sum(1 for c in row["commands"] if c["type"] == "edit"),
                "preserve_command_count": sum(1 for c in row["commands"] if c["type"] == "preserve"),
                "quality_command_count": sum(1 for c in row["commands"] if c["type"] == "quality"),
                "fulfilled_edit_ratio": float(np.mean(edit_scores)) if edit_scores else 1.0,
                "fulfilled_preserve_ratio": float(np.mean(preserve_scores)) if preserve_scores else 1.0,
                "fulfilled_quality_ratio": float(np.mean(quality_scores)) if quality_scores else 1.0,
                "overall_proxy_score": float(
                    0.5 * (float(np.mean(edit_scores)) if edit_scores else 1.0)
                    + 0.25 * (float(np.mean(preserve_scores)) if preserve_scores else 1.0)
                    + 0.25 * (float(np.mean(quality_scores)) if quality_scores else 1.0)
                ),
                "notes": "",
            }
        )

    Path(args.output_json).write_text(json.dumps(per_command, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_mp4[0].keys()) if per_mp4 else [])
        if per_mp4:
            writer.writeheader()
            writer.writerows(per_mp4)
    print(f"[INFO] Wrote command eval: {args.output_json}")
    print(f"[INFO] Wrote mp4 summary: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
