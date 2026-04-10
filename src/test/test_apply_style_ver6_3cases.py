#!/usr/bin/env python3
"""Run apply_style_ver6 on 3 GT apply_style rows and save output videos.

Rows tested: 5 (ukiyo-e), 17 (pixel_art), 29 (anime)
GT data    : /workspace/data/annotations_gt_task_ver10.json
Videos     : /workspace/data/videos/
Output     : /workspace/logs/test/apply_style_ver6_3cases/<timestamp>/

Usage (from workspace root):
    PYTHONPATH=/workspace:/workspace/src python \\
        src/test/test_apply_style_ver6_3cases.py [--max-frames N]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from postprocess.apply_style_ver6 import (
    apply_style_video_v6,
    extract_style_and_target,
)
from utils.video_utility import load_video

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")
JSONL_PATH = Path("/workspace/data/annotations.jsonl")
VIDEO_DIR = Path("/workspace/data/videos")
OUTPUT_ROOT = Path("/workspace/logs/test/apply_style_ver6_3cases")

# GT row indices to test (default)
DEFAULT_TEST_ROW_INDICES = [5, 17, 29]


def _load_gt(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test apply_style_ver6 on 3 GT rows"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames per video (0 = auto 1/3 of total)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Base output directory",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=",".join(str(i) for i in DEFAULT_TEST_ROW_INDICES),
        help="Comma-separated GT row indices, e.g. 5 or 5,17,29",
    )
    args = parser.parse_args()

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
    log = logging.getLogger("test_apply_style_v6")

    gt_data = _load_gt(GT_PATH)
    jsonl_data = _load_jsonl(JSONL_PATH)

    row_indices = [
        int(x.strip()) for x in args.rows.split(",") if x.strip()
    ]

    report: list[dict] = []

    for row_idx in row_indices:
        gt_row = gt_data[row_idx]
        jrow = jsonl_data[row_idx]
        task = gt_row["tasks"][0]

        instruction = gt_row.get("instruction", "")
        gt_target = task.get("target", "full_frame")
        gt_params = task.get("params", {})
        gt_style = gt_params.get("style", "")
        video_path = VIDEO_DIR / jrow["video_path"]

        log.info("=" * 60)
        log.info(f"row {row_idx}: {jrow['video_path']}")
        log.info(f"  instruction : {instruction[:80]}")
        log.info(f"  gt_target   : {gt_target}")
        log.info(f"  gt_style    : {gt_style}")

        # ── extract style and text_prompt via ver6 logic ──────────────
        style, text_prompt = extract_style_and_target(
            instruction=instruction,
            target=gt_target,
            params=gt_params,
        )
        log.info(f"  → style      : {style}")
        log.info(f"  → text_prompt: {text_prompt!r}")

        # ── load video ────────────────────────────────────────────────
        if not video_path.exists():
            log.error(f"  video not found: {video_path}")
            report.append(
                {
                    "row": row_idx,
                    "success": False,
                    "error": f"video not found: {video_path}",
                }
            )
            continue

        frames, fps, w, h = load_video(str(video_path))
        max_f = args.max_frames
        if max_f and max_f > 0:
            use_frames = max_f
        else:
            use_frames = max(1, len(frames) // 3)
        frames = frames[:use_frames]
        log.info(f"  frames loaded: {len(frames)} ({w}x{h} @ {fps:.1f}fps)")

        # ── apply style ───────────────────────────────────────────────
        out_path = out_root / f"row{row_idx:03d}_{style}_{jrow['video_path']}"
        try:
            saved = apply_style_video_v6(
                in_path=str(video_path),
                out_path=str(out_path),
                style=style,
                text_prompt=text_prompt,
                max_frames=use_frames,
            )
            log.info(f"  saved: {saved}")
            report.append(
                {
                    "row": row_idx,
                    "success": True,
                    "style": style,
                    "text_prompt": text_prompt,
                    "output": str(saved),
                    "frames_processed": len(frames),
                    "instruction": instruction,
                }
            )
        except Exception as exc:
            log.exception(f"  FAILED: {exc}")
            report.append(
                {
                    "row": row_idx,
                    "success": False,
                    "error": str(exc),
                    "style": style,
                    "text_prompt": text_prompt,
                }
            )

    # ── save report ───────────────────────────────────────────────────
    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    ok = sum(1 for r in report if r.get("success"))
    log.info(f"Done. {ok}/{len(report)} succeeded. report: {report_path}")


if __name__ == "__main__":
    main()
