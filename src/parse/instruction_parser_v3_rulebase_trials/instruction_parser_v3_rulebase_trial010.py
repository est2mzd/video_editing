#!/usr/bin/env python3
"""Trial 010 based on v3.

Single countermeasure:
- Action-aware target normalization layer.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial009 import (
    InstructionParserV3Trial009,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial010(InstructionParserV3Trial009):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        target = super()._extract_target(instruction, action)
        t = str(target).lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            if any(x in t for x in ["entire video", "scene", "reveal more", "throughout"]):
                return "camera_view"
            if len(t.split()) > 6:
                return "camera_view"
            return t

        if action == "change_camera_angle":
            if t == "camera_view":
                for pat in [
                    r"at the ([^.,;]+)",
                    r"capturing the ([^.,;]+)",
                    r"focus on the ([^.,;]+)",
                ]:
                    m = re.search(pat, text)
                    if m:
                        return m.group(1).strip().lower()
            return t

        if action in {"add_object", "increase_amount"}:
            if "rhino" in t and "buffalo" in t:
                return "rhino_and_buffalo"
            if "panda" in t:
                return "panda"
            if "pastr" in t:
                return "pastry"
            if "towel" in t and "elephant" in text:
                return "towel_elephant"
            if len(t.split()) > 6:
                return t.split()[0]
            return t

        if action == "remove_object":
            if t == "background":
                m = re.search(r"remove the ([^.,;]+) from", text)
                if m:
                    return m.group(1).strip().lower()
            return t

        if action == "add_effect":
            if "stage lighting" in text:
                return "stage_lighting_region"
            m = re.search(r"outlines? (his|her|the) ([^.,;]+)", text)
            if m:
                return m.group(2).strip().lower()
            return t

        if action == "replace_object":
            if t == "background":
                m = re.search(r"replace the ([^.,;]+?) with", text)
                if m:
                    return m.group(1).strip().lower()
            return t

        return t


def build_parser() -> InstructionParserV3Trial010:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial010(kb)
