from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from postprocess.dispatcher import run_method
from utils.video_utility import load_video, write_video


@dataclass
class Scenario:
    name: str
    mp4_path: str
    action: str
    target: str
    params_override: dict[str, Any]


def load_annotations_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_annotation_index(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        Path(str(row.get("video_path", ""))).name: row
        for row in rows
        if str(row.get("video_path", "")).strip()
    }


def resolve_action_plan(
    annotation_row: dict[str, Any],
    plan_map: dict[str, dict[str, Any]],
    fallback_action: str,
    fallback_target: str,
    fallback_params: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any], str]:
    instruction = str(annotation_row.get("instruction", ""))
    video_name = Path(str(annotation_row.get("video_path", ""))).name

    direct_action = annotation_row.get("action")
    direct_target = annotation_row.get("target")
    direct_params = annotation_row.get("params")
    if (
        direct_action is not None
        and direct_target is not None
        and isinstance(direct_params, dict)
    ):
        params = copy.deepcopy(direct_params)
        action = str(direct_action)
        target = str(direct_target)
        params["action"] = action
        params["target"] = target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return action, target, params, instruction

    if video_name in plan_map:
        plan = plan_map[video_name]
        action = str(plan["action"])
        target = str(plan["target"])
        params = copy.deepcopy(plan.get("params") or {})
        params["action"] = action
        params["target"] = target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return action, target, params, instruction

    if fallback_action and fallback_target:
        params = copy.deepcopy(fallback_params or {})
        params["action"] = fallback_action
        params["target"] = fallback_target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return fallback_action, fallback_target, params, instruction

    raise ValueError(
        "no direct/manual plan for video="
        f"{video_name}; add action/target/params to annotations.jsonl"
    )


def run_single_case(
    annotation_index: dict[str, dict[str, Any]],
    plan_map: dict[str, dict[str, Any]],
    video_root: Path,
    mp4_path: str,
    action: str,
    target: str,
    output_dir: Path,
    params_override: dict[str, Any] | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    logger_name: str = "gt_unit_test_02",
) -> Path:
    params_override = params_override or {}
    resolved_video = Path(mp4_path)
    if not resolved_video.exists():
        resolved_video = video_root / resolved_video.name

    ann_row = annotation_index[resolved_video.name]

    resolved_action, resolved_target, params, instruction = (
        resolve_action_plan(
            annotation_row=ann_row,
            plan_map=plan_map,
            fallback_action=action,
            fallback_target=target,
            fallback_params=params_override,
        )
    )
    params.update(params_override)

    frames, fps, width, height = load_video(resolved_video)
    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]
    if frame_stride > 1:
        frames = frames[::frame_stride]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = resolved_video.stem
    action_norm = str(resolved_action or "").strip().lower().replace(" ", "_")
    out_path = output_dir / f"{stem}__{action_norm}.mp4"

    logger = logging.getLogger(logger_name)
    out_frames = run_method(
        action=resolved_action,
        targets=resolved_target,
        frames=frames,
        params=copy.deepcopy(params),
        instruction=instruction,
        logger=logger,
    )
    write_video(out_path, out_frames, fps, width, height)
    print(
        f"[OK] video={resolved_video.name} action={resolved_action} "
        f"target={resolved_target} out={out_path}"
    )
    return out_path


def run_default_scenarios(
    scenarios: list[Scenario],
    annotation_index: dict[str, dict[str, Any]],
    plan_map: dict[str, dict[str, Any]],
    video_root: Path,
    output_dir: Path,
    max_frames: int | None,
    frame_stride: int,
    logger_name: str = "gt_unit_test_02",
) -> list[Path]:
    outputs: list[Path] = []
    for scenario in scenarios:
        print(f"\n=== scenario: {scenario.name} ===")
        out_path = run_single_case(
            annotation_index=annotation_index,
            plan_map=plan_map,
            video_root=video_root,
            mp4_path=scenario.mp4_path,
            action=scenario.action,
            target=scenario.target,
            output_dir=output_dir,
            params_override=scenario.params_override,
            max_frames=max_frames,
            frame_stride=frame_stride,
            logger_name=logger_name,
        )
        outputs.append(out_path)
    return outputs
