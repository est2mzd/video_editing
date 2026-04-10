
from datetime import datetime
import argparse
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
    logger = logging.getLogger("run_video_editor_v2")
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


def _main() -> None:
    parser_args = argparse.ArgumentParser(description="Run video editor v2")
    parser_args.add_argument(
        "--config",
        type=str,
        default="/workspace/configs/base_config_v2.yaml",
        help="Path to YAML config file",
    )
    args = parser_args.parse_args()

    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    debug_mode = bool(config.get("debug_mode", False))
    target_subclass = str(config.get("target_subclass", "")).strip()
    video_dir = config["video_dir"]
    output_dir_top = config["output_dir"]

    parser_version = str(config.get("parser_version", "v2"))
    if parser_version == "v1":
        parser = build_parser_1()
    else:
        parser = build_parser_2()

    annotation_path = config["annotation_path"]
    annotation = _read_jsonl(annotation_path)

    if debug_mode and target_subclass:
        rows = [
            a
            for a in annotation
            if a.get("selected_subclass", "") == target_subclass
        ]
        action_type = target_subclass.replace(" ", "_").lower()
    else:
        rows = annotation
        action_type = "all"

    if not rows:
        raise RuntimeError(
            "No rows to process. Check debug_mode / target_subclass settings."
        )

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir_top) / f"{action_type}_{time_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "instruction_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(output_dir / "run_video_editor_v2.log")
    logger.info("config_path=%s", config_path)
    logger.info("start processing: rows=%d", len(rows))

    max_task_num = int(config.get("max_task_num", 1))

    for idx, row in enumerate(rows, start=1):
        rel_video = row["video_path"]
        video_path = os.path.join(video_dir, rel_video)
        instruction = row["instruction"]
        video_name = os.path.basename(rel_video)

        logger.info("[%d/%d] %s", idx, len(rows), rel_video)

        frames, fps, width, height = load_video(video_path)
        parsed_instruction = parser.infer(instruction)

        work_frames = frames
        task_logs: list[dict] = []

        if parser_version == "v1":
            action = parsed_instruction.get("action", "zoom_in")
            target = parsed_instruction.get("target", ["face"])
            params = parsed_instruction.get("params", {})
            params["video_name"] = video_name

            work_frames = run_method(
                action=action,
                targets=target,
                frames=work_frames,
                params=params,
                instruction=instruction,
                logger=logger,
            )
            task_logs.append(
                {"action": action, "target": target, "params": params}
            )
        else:
            tasks = parsed_instruction.get("tasks", [])
            if not tasks:
                tasks = [
                    {"action": "zoom_in", "target": ["face"], "params": {}}
                ]

            for task_idx, task in enumerate(tasks):
                if task_idx >= max_task_num:
                    break
                action = task.get("action", "zoom_in")
                target = task.get("target", ["face"])
                params = task.get("params", {})
                params["video_name"] = video_name

                logger.info(
                    "[%d/%d][task %d] action=%s target=%s",
                    idx,
                    len(rows),
                    task_idx + 1,
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
                        "task_index": task_idx + 1,
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
            "parsed_instruction": parsed_instruction,
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
    _main()
