#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class _BranchErrorHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.messages.append(record.getMessage())


def _load_annotation_row(path: Path, index: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"index out of range: {index} / {len(rows)}")
    return rows[index]


def _build_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "2_1_add_effect",
            "action": "add_effect",
            "target": "person",
            "instruction": "Add a blue glow effect around the person.",
            "params": {
                "effect_override": "glow",
                "color_override": "blue",
            },
        },
        {
            "name": "2_2_add_object",
            "action": "add_object",
            "target": "person",
            "instruction": "Add another person next to the person.",
            "params": {},
        },
        {
            "name": "2_3_apply_style",
            "action": "apply_style",
            "target": "person",
            "instruction": "Apply oil painting style to the scene.",
            "params": {"style": "oil_painting"},
        },
        {
            "name": "2_4_change_color",
            "action": "change_color",
            "target": "shirt",
            "instruction": "Change the color of the shirt to red.",
            "params": {"color": "red"},
        },
        {
            "name": "2_5_dolly_in",
            "action": "dolly_in",
            "target": "person",
            "instruction": "Dolly in toward the person.",
            "params": {"object_end_scale": 1.2},
        },
        {
            "name": "2_6_replace_background",
            "action": "replace_background",
            "target": "person",
            "instruction": (
                "Replace background style while keeping foreground."
            ),
            "params": {
                "effect_name": "hsv_shift",
                "effect_params": {
                    "hue_shift": 8,
                    "saturation_scale": 1.15,
                    "value_scale": 0.95,
                },
            },
        },
        {
            "name": "2_7_zoom_in",
            "action": "zoom_in",
            "target": "face . person .",
            "instruction": "Zoom in on the face.",
            "params": {"zoom_factor": 1.0},
        },
    ]


def main() -> None:
    dispatch_mod = importlib.import_module("src.postprocess.dispatcher_v2")
    video_mod = importlib.import_module("src.utils.video_utility")
    run_method = getattr(dispatch_mod, "run_method")
    load_video = getattr(video_mod, "load_video")
    write_video = getattr(video_mod, "write_video")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations",
        default="/workspace/data/annotations.jsonl",
    )
    p.add_argument(
        "--video-dir",
        default="/workspace/data/videos",
    )
    p.add_argument("--row-index", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument(
        "--output-dir",
        default="/workspace/logs/test/dispatcher_v2_single_video",
    )
    args = p.parse_args()

    out_root = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dispatcher_v2_smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    err_handler = _BranchErrorHandler()
    logger.addHandler(err_handler)

    row = _load_annotation_row(Path(args.annotations), args.row_index)
    video_path = Path(args.video_dir) / row["video_path"]
    frames, fps, width, height = load_video(video_path)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    report: dict[str, Any] = {
        "video_path": str(video_path),
        "row_index": args.row_index,
        "input_frame_count": len(frames),
        "results": [],
    }

    has_error = False
    for case in _build_cases():
        name = case["name"]
        action = case["action"]
        target = case["target"]
        instruction = case["instruction"]
        params = dict(case["params"])

        base_frames = [f.copy() for f in frames]
        result_row: dict[str, Any] = {
            "name": name,
            "action": action,
            "target": target,
            "instruction": instruction,
            "success": False,
        }
        try:
            before_count = len(err_handler.messages)
            out_frames = run_method(
                action=action,
                targets=target,
                frames=base_frames,
                params=params,
                instruction=instruction,
                logger=logger,
            )
            after_count = len(err_handler.messages)
            branch_errors = err_handler.messages[before_count:after_count]

            out_path = out_root / f"{name}.mp4"
            write_video(out_path, out_frames, fps, width, height)

            result_row["success"] = len(branch_errors) == 0
            result_row["output_video"] = str(out_path)
            result_row["output_frame_count"] = len(out_frames)
            result_row["same_frame_count"] = len(out_frames) == len(frames)
            if branch_errors:
                has_error = True
                result_row["error"] = "; ".join(branch_errors)
        except Exception as e:
            has_error = True
            result_row["error"] = str(e)

        report["results"].append(result_row)

    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"report: {report_path}")
    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
