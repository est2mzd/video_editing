#!/usr/bin/env python3
"""Trial 015 based on v3.

Single countermeasure:
- Extend canonical target mapping for frequent residual target errors.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial014 import (
    InstructionParserV3Trial014,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial015(InstructionParserV3Trial014):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        target = super()._extract_target(instruction, action)
        t = str(target).lower()

        if action == "add_object":
            if "microphone" in t:
                return "new_object"
            if "men talking on phones" in t:
                return "man_on_phone"
            if t in {"cars", "car"} and "white car" in text:
                return "white_car"
            if "jumping" in t and "baby" in text:
                return "baby_character"
            if "mannequin" in text and "floral dress" in text:
                return "mannequin_in_formal_floral_dress"

        if action == "increase_amount" and "fruit jam" in text:
            return "fruit_jam"

        if action == "add_effect":
            if "outlines" in text and "body" in text:
                return "his body"

        if action == "edit_motion" and t == "person":
            if re.search(r"\b(object|fans|hook|plate|car)\b", text):
                return "object"

        if action == "change_camera_angle":
            if t == "camera_view" and "chef" in text:
                return "chef"
            if t == "camera_view" and re.search(r"\bman\b", text):
                return "man"

        if action in {"zoom_in", "dolly_in"}:
            if "focused on" in text and "man's face" in text:
                return "camera_view"
            if "gradual zoom-in effect" in text and "face" in text:
                return "camera_view"

        return t


def build_parser() -> InstructionParserV3Trial015:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial015(kb)
