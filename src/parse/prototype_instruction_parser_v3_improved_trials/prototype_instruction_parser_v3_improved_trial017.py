#!/usr/bin/env python3
"""Trial 017 based on v3.

Single countermeasure:
- Add explicit orbit_camera detection for arc/revolving shots.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial016 import (
    InstructionParserV3Trial016,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial017(InstructionParserV3Trial016):
    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()
        if re.search(r"\barc shot\b|\brevolving around\b|\borbit\b", text):
            return "orbit_camera"
        return super()._infer_action(instruction)


def build_parser() -> InstructionParserV3Trial017:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial017(kb)
