#!/usr/bin/env python3
"""Trial 007 based on v3.

Single countermeasure:
- For camera-related actions, extract concrete phrase target first and only
  fallback to camera_view when phrase is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial006 import (
    InstructionParserV3Trial006,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial007(InstructionParserV3Trial006):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            for pat in [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"(?:toward|towards|on|onto|at|to) ([^.,;]+)",
                r"frame ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    tgt = m.group(1).strip().lower()
                    if tgt and "camera" not in tgt:
                        return tgt
            return "camera_view"

        if action == "change_camera_angle":
            for pat in [
                r"looking (?:up|down) at the ([^.,;]+)",
                r"at the ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    return m.group(1).strip().lower()
            return "camera_view"

        return super()._extract_target(instruction, action)


def build_parser() -> InstructionParserV3Trial007:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial007(kb)
