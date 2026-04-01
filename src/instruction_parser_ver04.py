#!/usr/bin/env python3
"""Parse free-form instructions into atomic commands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParsedCommand:
    command_id: str
    type: str
    target: str
    action: str
    value: str
    source_text: str
    priority: int
    vace_prompt: str


def load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_instruction_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return parts if parts else [text.strip()]


def _make_command(
    idx: int,
    cmd_type: str,
    target: str,
    action: str,
    value: str,
    source_text: str,
    priority: int,
    vace_prompt: str,
) -> ParsedCommand:
    return ParsedCommand(
        command_id=f"cmd_{idx:02d}",
        type=cmd_type,
        target=target,
        action=action,
        value=value,
        source_text=source_text,
        priority=priority,
        vace_prompt=vace_prompt,
    )


def extract_target_color(text: str, schema: dict[str, Any]) -> str:
    lower = text.lower()
    color_map = schema.get("color_keywords", {})
    for key, value in sorted(color_map.items(), key=lambda kv: -len(kv[0])):
        if key in lower:
            return str(value)
    return ""


def parse_instruction(
    video_path: str,
    selected_class: str,
    selected_subclass: str,
    instruction: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    commands: list[ParsedCommand] = []
    idx = 0
    lower = instruction.lower()
    sentences = split_instruction_sentences(instruction)

    def add(cmd_type: str, target: str, action: str, value: str, source_text: str, priority: int, prompt: str) -> None:
        nonlocal idx
        commands.append(_make_command(idx, cmd_type, target, action, value, source_text, priority, prompt))
        idx += 1

    # Main edit command from class/subclass and text.
    color = extract_target_color(instruction, schema)
    if selected_class == "Attribute Editing" and selected_subclass == "Color adjustment":
        target = "hair" if "hair" in lower else "tie" if "tie" in lower or "necktie" in lower else "hat" if ("hat" in lower or "beanie" in lower) else "subject_region"
        add("edit", target, "change_color", color or "target_color", instruction, 40, f"Change only the {target} color to {color or 'the target color'} and preserve everything else.")
    elif selected_class == "Visual Effect Editing" and selected_subclass == "Background Change":
        add("edit", "background", "replace_background", "new_background", instruction, 20, "Replace only the background and preserve the foreground subject, text, and logos.")
    elif selected_class == "Camera Motion Editing":
        action = selected_subclass.lower().replace(" ", "_")
        add("edit", "camera", action, selected_subclass, instruction, 10, f"Apply only a {selected_subclass.lower()} camera effect while preserving scene content.")
    elif selected_class == "Camera Angle Editing":
        action = selected_subclass.lower().replace(" ", "_")
        add("edit", "camera", action, selected_subclass, instruction, 10, f"Change only the camera angle to {selected_subclass.lower()} while preserving scene content.")
    elif selected_class == "Style Editing":
        style_map = schema.get("style_keywords", {})
        style_value = style_map.get(selected_subclass.lower(), selected_subclass.lower().replace(" ", "_"))
        add("edit", "global", "apply_style", style_value, instruction, 30, f"Apply only a {selected_subclass} style while preserving composition and identity.")
    elif selected_class == "Quantity Editing" and selected_subclass == "Increase":
        add("edit", "scene_objects", "increase_quantity", "increase", instruction, 35, "Increase the number of matching objects in the scene while preserving the original composition.")
    elif selected_class == "Instance Editing" and selected_subclass == "Instance Removal":
        add("edit", "instance", "remove_instance", "remove", instruction, 35, "Remove only the specified instance and inpaint the region consistently.")
    elif selected_class == "Instance Editing" and selected_subclass == "Instance Replacement":
        target = "hat" if ("hat" in lower or "beanie" in lower) else "instance"
        add("edit", target, "replace_instance", color or "replacement", instruction, 35, "Replace only the specified instance and preserve all other content.")
    elif selected_class == "Instance Editing" and selected_subclass == "Instance Insertion":
        add("edit", "instance", "insert_instance", "insert", instruction, 35, "Insert only the requested object into the scene and preserve everything else.")
    elif selected_class == "Instance Motion Editing" and selected_subclass == "Human motion":
        motion_value = "smile" if "smile" in lower else "wave" if "wave" in lower or "raises his left hand" in lower else "human_motion"
        add("edit", "human", "change_human_motion", motion_value, instruction, 35, f"Modify only the human motion to {motion_value} and preserve identity and background.")
    else:
        add("edit", "global", "generic_edit", selected_subclass.lower().replace(" ", "_"), instruction, 50, instruction)

    # Preservation commands.
    preserve_targets = schema.get("preserve_targets", {})
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(k in s_lower for k in ["preserve", "maintain", "keep"]):
            matched = False
            for key, value in sorted(preserve_targets.items(), key=lambda kv: -len(kv[0])):
                if key in s_lower:
                    add("preserve", str(value), "keep_unchanged", "", sentence, 80, f"Preserve {value} exactly and do not modify it.")
                    matched = True
            if not matched:
                add("preserve", "other_elements", "keep_unchanged", "", sentence, 85, "Preserve all unrelated content.")

    # Quality commands.
    quality_map = schema.get("quality_keywords", {})
    for sentence in sentences:
        s_lower = sentence.lower()
        for key, value in sorted(quality_map.items(), key=lambda kv: -len(kv[0])):
            if key in s_lower:
                target = "temporal" if "temporal" in str(value) else "boundary" if "edge" in str(value) else "global_quality"
                add("quality", target, str(value), "", sentence, 90, f"Ensure {value.replace('_', ' ')}.")

    # Deduplicate exact commands.
    unique: list[ParsedCommand] = []
    seen: set[tuple[str, str, str, str]] = set()
    for cmd in commands:
        key = (cmd.type, cmd.target, cmd.action, cmd.value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cmd)

    return {
        "video_path": video_path,
        "selected_class": selected_class,
        "selected_subclass": selected_subclass,
        "instruction": instruction,
        "commands": [asdict(cmd) for cmd in unique],
    }


def parse_annotation_file(annotations_path: Path, schema_path: Path) -> list[dict[str, Any]]:
    schema = load_schema(schema_path)
    rows: list[dict[str, Any]] = []
    with annotations_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                parse_instruction(
                    video_path=str(obj["video_path"]),
                    selected_class=str(obj["selected_class"]),
                    selected_subclass=str(obj["selected_subclass"]),
                    instruction=str(obj["instruction"]),
                    schema=schema,
                )
            )
    return rows
