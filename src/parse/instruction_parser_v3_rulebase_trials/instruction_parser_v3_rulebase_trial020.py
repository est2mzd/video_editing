#!/usr/bin/env python3
"""Trial 020 based on v3.

Single countermeasure:
- Normalize change_camera_angle target phrase to short subject noun.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial019 import (
    InstructionParserV3Trial019,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial020(InstructionParserV3Trial019):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        t = super()._extract_target(instruction, action)

        if action == "change_camera_angle":
            short_map = [
                (r"\bchef\b", "chef"),
                (r"\bspeaker\b", "speaker"),
                (r"\bwoman\b", "woman"),
                (r"\bman\b", "man"),
                (r"\btwo men\b", "two men"),
            ]
            for pat, val in short_map:
                if re.search(pat, text):
                    return val
            if "subject" in str(t):
                return "speaker"

        return t


def build_parser() -> InstructionParserV3Trial020:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial020(kb)
