from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

PRIMARY_ACTION_PRIORITY = [
    "replace_background",
    "replace_object",
    "add_object",
    "increase_amount",
    "change_color",
    "remove_object",
    "edit_motion",
    "edit_expression",
    "change_camera_angle",
    "zoom_in",
    "zoom_out",
    "dolly_in",
    "orbit_camera",
    "apply_style",
    "add_effect",
    "preserve_foreground",
    "preserve_objects",
    "preserve_identity",
    "preserve_focus",
    "preserve_framing",
    "preserve_layout",
    "preserve_material_appearance",
    "align_replacement",
    "match_appearance",
    "match_lighting",
    "match_background_camera_properties",
    "match_effect_lighting",
    "match_scene_interaction",
    "stabilize_instances",
    "stabilize_edit",
    "stabilize_motion",
    "stabilize_style",
    "stabilize_effect",
    "stabilize_composite",
    "stabilize_inpaint",
    "refine_mask",
    "blend_instances",
    "inpaint_background",
    "adjust_perspective",
    "track_effect",
    "enhance_style_details",
]
PRIMARY_ACTION_RANK = {action: idx for idx, action in enumerate(PRIMARY_ACTION_PRIORITY)}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_primary_task(tasks: list[dict]) -> dict:
    if not tasks:
        return {"action": "", "target": "", "constraints": [], "params": {}}

    ranked: list[tuple[int, int, dict]] = []
    for idx, task in enumerate(tasks):
        rank = PRIMARY_ACTION_RANK.get(task.get("action", ""), 9999)
        ranked.append((rank, idx, task))

    ranked.sort(key=lambda item: (item[0], item[1]))
    primary = ranked[0][2]
    return {
        "action": primary.get("action", ""),
        "target": primary.get("target", ""),
        "constraints": primary.get("constraints", []),
        "params": primary.get("params", {}),
    }


def load_base_records(raw_path: Path, gt_path: Path) -> list[dict]:
    raw_annotations = read_jsonl(raw_path)
    gt_annotations = json.loads(gt_path.read_text(encoding="utf-8"))
    raw_by_video = {row["video_path"]: row for row in raw_annotations}

    records: list[dict] = []
    for gt_item in gt_annotations:
        raw_item = raw_by_video.get(gt_item["video_path"], {})
        records.append(
            {
                "video_path": gt_item["video_path"],
                "prediction_key": f"{gt_item['video_path']}::base",
                "class": raw_item.get("selected_class", gt_item.get("class", "")),
                "subclass": raw_item.get("selected_subclass", gt_item.get("subclass", "")),
                "instruction": raw_item.get("instruction", gt_item.get("instruction", "")),
                "gt_tasks": gt_item.get("tasks", []),
                "gt_primary": extract_primary_task(gt_item.get("tasks", [])),
            }
        )
    return records


def load_ver10_seed_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["video_path"]: {"tasks": row["tasks"]} for row in rows}


def load_grouped_unknown_records(
    grouped_paths: Iterable[Path],
    base_records: list[dict],
    instruction_keys: tuple[str, ...] = ("ver2", "ver3", "ver4"),
) -> list[dict]:
    """Generate unknown/paraphrased records for inference only.

    NOTE: This function is for evaluation/inference data preparation.
    It does not modify model logic or fit parameters.
    """
    base_by_video = {record["video_path"]: record for record in base_records}
    unknown_records: list[dict] = []

    for grouped_path in grouped_paths:
        grouped_rows = json.loads(grouped_path.read_text(encoding="utf-8"))
        for row in grouped_rows:
            video_path = row.get("video_path")
            if not video_path or video_path not in base_by_video:
                continue

            base = base_by_video[video_path]
            for key in instruction_keys:
                instruction = row.get(key)
                if not instruction:
                    continue
                unknown_records.append(
                    {
                        "video_path": video_path,
                        "prediction_key": f"{video_path}::{grouped_path.stem}:{key}",
                        "variant": key,
                        "variant_source": grouped_path.name,
                        "class": base["class"],
                        "subclass": base["subclass"],
                        "instruction": instruction,
                        "gt_tasks": base["gt_tasks"],
                        "gt_primary": base["gt_primary"],
                    }
                )

    return unknown_records
