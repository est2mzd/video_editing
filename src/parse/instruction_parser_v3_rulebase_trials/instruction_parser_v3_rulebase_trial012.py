#!/usr/bin/env python3
"""Trial 012 based on v3.

Single countermeasure:
- Camera-action target resolver prioritizes concrete subject phrases.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial011 import (
    InstructionParserV3Trial011,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial012(InstructionParserV3Trial011):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            strong_subject_patterns = [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"focused on the ([^.,;]+)",
                r"close-up of the ([^.,;]+)",
                r"frame (?:his|her|the) ([^.,;]+)",
            ]
            for pat in strong_subject_patterns:
                m = re.search(pat, text)
                if not m:
                    continue
                cand = m.group(1).strip().lower()
                if not cand:
                    continue
                if any(k in cand for k in ["face", "profile", "mixer", "bowl", "chef", "speaker"]):
                    return cand
            return super()._extract_target(instruction, action)

        if action == "change_camera_angle":
            for pat in [
                r"looking up at the ([^.,;]+)",
                r"looking down at the ([^.,;]+)",
                r"capturing the ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip().lower()
                    if any(k in cand for k in ["man", "woman", "chef", "speaker", "hands"]):
                        return cand
            return super()._extract_target(instruction, action)

        return super()._extract_target(instruction, action)


def build_parser() -> InstructionParserV3Trial012:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial012(kb)
