from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
import logging
import os
from pathlib import Path

import yaml
from parse.instruction_parser_v3_rulebase_trial013_singlefile_kai2 import (
    build_parser as build_parser_1,
)
from parse.instruction_parser_v3_rulebase_trial020_singlefile import (
    build_parser as build_parser_2,
)
from postprocess.dispatcher_v2 import run_method
from utils.video_utility import load_video, write_video


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_adjustment_pass_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_parser(version: str):
    if version == "v1":
        return build_parser_1()
    return build_parser_2()


def _parse_actions(value: str) -> set[str]:
    actions = {
        s.strip()
        for s in str(value).split(",")
        if s.strip()
    }
    if not actions:
        return {"dolly_in", "change_color"}
    return actions


def _parse_keywords(value: str) -> list[str]:
    kws = [
        s.strip()
        for s in str(value).split(",")
        if s.strip()
    ]
    if kws:
        return kws
    return [
        "gradual",
        "gradually",
        "徐々",
        "徐々に",
        "だんだん",
        "少しずつ",
    ]


def _select_target_rows(
    rows: list[dict],
    parser,
    parser_version: str,
    max_task_num: int,
    target_actions: set[str],
    limit: int | None,
) -> list[tuple[dict, dict, list[dict]]]:
    selected: list[tuple[dict, dict, list[dict]]] = []

    for row in rows:
        instruction = row.get("instruction", "")
        parsed = parser.infer(instruction)

        if parser_version == "v1":
            action = parsed.get("action", "")
            tasks = [
                {
                    "action": action,
                    "target": parsed.get("target", []),
                    "params": parsed.get("params", {}),
                }
            ]
        else:
            tasks = parsed.get("tasks", [])
            if not tasks:
                tasks = [
                    {"action": "zoom_in", "target": ["face"], "params": {}}
                ]
            tasks = tasks[:max_task_num]

        filtered = [
            t for t in tasks
            if str(t.get("action", "")).strip() in target_actions
        ]
        if filtered:
            selected.append((row, parsed, filtered))

        if limit is not None and len(selected) >= limit:
            break

    return selected


def main() -> None:
    ap = ArgumentParser(
        description="Run adjustment-only pass for selected actions."
    )
    ap.add_argument(
        "--config",
        default="/workspace/configs/base_config_v2.yaml",
        help="Path to config yaml",
    )
    ap.add_argument(
        "--actions",
        default="dolly_in,change_color",
        help="Comma-separated actions to re-run",
    )
    ap.add_argument(
        "--dolly-end-scale",
        type=float,
        default=1.85,
        help="Override object_end_scale for dolly_in",
    )
    ap.add_argument(
        "--color-gradual-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="change_color gradual behavior",
    )
    ap.add_argument(
        "--color-gradual-keywords",
        default="gradual,gradually,徐々,徐々に,だんだん,少しずつ",
        help="Comma-separated keywords used when mode=auto",
    )
    ap.add_argument(
        "--output-prefix",
        default="adjustments",
        help="Output folder prefix under output_dir",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of selected rows",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected rows without processing video",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parser_version = str(config.get("parser_version", "v2"))
    parser = _build_parser(parser_version)
    max_task_num = int(config.get("max_task_num", 1))

    annotation = _read_jsonl(config["annotation_path"])
    target_actions = _parse_actions(args.actions)
    keywords = _parse_keywords(args.color_gradual_keywords)

    selected = _select_target_rows(
        rows=annotation,
        parser=parser,
        parser_version=parser_version,
        max_task_num=max_task_num,
        target_actions=target_actions,
        limit=args.limit,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output_dir"]) / f"{args.output_prefix}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "instruction_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(output_dir / "run_adjustment_pass_v2.log")
    logger.info(
        "selected rows=%d actions=%s",
        len(selected),
        sorted(target_actions),
    )

    if args.dry_run:
        for idx, (row, _parsed, tasks) in enumerate(selected, start=1):
            logger.info(
                "[dry-run %d/%d] %s actions=%s",
                idx,
                len(selected),
                row.get("video_path", ""),
                [t.get("action") for t in tasks],
            )
        logger.info("dry-run done: %s", output_dir)
        return

    video_dir = config["video_dir"]
    for idx, (row, parsed, tasks) in enumerate(selected, start=1):
        rel_video = row["video_path"]
        video_name = os.path.basename(rel_video)
        video_path = os.path.join(video_dir, rel_video)
        instruction = row.get("instruction", "")

        logger.info("[%d/%d] %s", idx, len(selected), rel_video)
        frames, fps, width, height = load_video(video_path)
        work_frames = frames

        task_logs: list[dict] = []
        for task_idx, task in enumerate(tasks, start=1):
            action = str(task.get("action", "")).strip()
            target = task.get("target", ["face"])
            params = dict(task.get("params", {}))
            params["video_name"] = video_name

            if action == "dolly_in":
                params["object_end_scale"] = float(args.dolly_end_scale)

            if action == "change_color":
                if args.color_gradual_mode == "always":
                    params["force_gradual"] = True
                elif args.color_gradual_mode == "never":
                    params["force_gradual"] = False
                else:
                    params["gradual_keywords"] = keywords

            logger.info(
                "[%d/%d][task %d] action=%s target=%s",
                idx,
                len(selected),
                task_idx,
                action,
                target,
            )

            work_frames = run_method(
                action=action,
                targets=target,
                frames=work_frames,
                params=params,
                instruction=instruction,
                logger=logger,
            )
            task_logs.append(
                {
                    "task_index": task_idx,
                    "action": action,
                    "target": target,
                    "params": params,
                }
            )

        out_path = output_dir / video_name
        write_video(out_path, work_frames, fps, width, height)

        analysis_log = {
            "video_path": rel_video,
            "video_name": video_name,
            "instruction": instruction,
            "selected_subclass": row.get("selected_subclass", ""),
            "parser_version": parser_version,
            "parsed_instruction": parsed,
            "task_logs": task_logs,
            "input_frame_count": len(frames),
            "output_frame_count": len(work_frames),
            "fps": fps,
            "width": width,
            "height": height,
        }
        with open(
            log_dir / f"{Path(video_name).stem}.log",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(analysis_log, f, ensure_ascii=False, indent=2)

    logger.info("Output saved to: %s", output_dir)


if __name__ == "__main__":
    main()
