#!/usr/bin/env python3
"""Run replace_background cases from GT one by one and save videos.

Default behavior:
- target action: replace_background (all rows in GT)
- process one-third of frames per row (minimum 1 frame)
- save each output video and report.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")
JSONL_PATH = Path("/workspace/data/annotations.jsonl")
VIDEO_DIR = Path("/workspace/data/videos")
OUTPUT_ROOT = Path("/workspace/logs/test/replace_background_gt_cases")


def _load_gt(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_replace_background_rows(
    gt_rows: list[dict[str, Any]],
) -> list[int]:
    out: list[int] = []
    for i, row in enumerate(gt_rows):
        tasks = row.get("tasks", [])
        if not tasks:
            continue
        if str(tasks[0].get("action", "")).strip() == "replace_background":
            out.append(i)
    return out


def _parse_rows_arg(rows_arg: str | None, default_rows: list[int]) -> list[int]:
    if not rows_arg:
        return default_rows
    rows: list[int] = []
    for part in rows_arg.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replace_background rows from GT"
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="",
        help="Comma-separated row indices. Empty means all replace_background rows.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help=(
            "Frames per row. -1 means one-third of total frames "
            "(default), 0 means all frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Base output directory",
    )
    args = parser.parse_args()

    dispatch_mod = importlib.import_module("src.postprocess.dispatcher_v2")
    video_mod = importlib.import_module("src.utils.video_utility")
    run_method = getattr(dispatch_mod, "run_method")
    load_video = getattr(video_mod, "load_video")
    write_video = getattr(video_mod, "write_video")

    gt_rows = _load_gt(GT_PATH)
    jsonl_rows = _load_jsonl(JSONL_PATH)

    default_rows = _collect_replace_background_rows(gt_rows)
    row_indices = _parse_rows_arg(args.rows, default_rows)

    out_root = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_root / "run.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("replace_background_gt")

    report: list[dict[str, Any]] = []

    for row_idx in row_indices:
        gt_row = gt_rows[row_idx]
        jsonl_row = jsonl_rows[row_idx]
        task = gt_row["tasks"][0]

        instruction = gt_row.get("instruction", "")
        action = str(task.get("action", ""))
        target = task.get("target", "")
        params = dict(task.get("params", {}))
        video_path = VIDEO_DIR / jsonl_row.get("video_path", "")

        log.info("=" * 60)
        log.info("row=%s action=%s target=%s", row_idx, action, target)
        log.info("video=%s", video_path)
        log.info("instruction=%s", instruction[:140])

        if not video_path.exists():
            err = f"video not found: {video_path}"
            log.error(err)
            report.append(
                {
                    "row": row_idx,
                    "success": False,
                    "error": err,
                }
            )
            continue

        frames, fps, width, height = load_video(video_path)
        total_frames = len(frames)
        if args.max_frames > 0:
            use_frames = args.max_frames
        elif args.max_frames == 0:
            use_frames = total_frames
        else:
            use_frames = max(1, total_frames // 3)
        frames = frames[:use_frames]

        try:
            out_frames = run_method(
                action=action,
                targets=target,
                frames=[f.copy() for f in frames],
                params=params,
                instruction=instruction,
                logger=log,
            )
            out_name = (
                f"row{row_idx:03d}_replace_background_"
                f"{Path(video_path).name}"
            )
            out_path = out_root / out_name
            write_video(out_path, out_frames, fps, width, height)
            log.info("saved=%s", out_path)
            report.append(
                {
                    "row": row_idx,
                    "success": True,
                    "video": str(video_path),
                    "output": str(out_path),
                    "target": target,
                    "used_frames": len(frames),
                    "total_frames": total_frames,
                }
            )
        except Exception as exc:
            log.exception("FAILED row=%s: %s", row_idx, exc)
            report.append(
                {
                    "row": row_idx,
                    "success": False,
                    "video": str(video_path),
                    "target": target,
                    "error": str(exc),
                }
            )

    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in report if r.get("success"))
    log.info("Done. %s/%s succeeded. report=%s", ok, len(report), report_path)


if __name__ == "__main__":
    main()
