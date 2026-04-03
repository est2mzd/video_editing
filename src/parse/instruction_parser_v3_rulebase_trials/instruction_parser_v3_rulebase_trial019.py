#!/usr/bin/env python3
"""Trial 019 based on v3.

Single countermeasure:
- Normalize edit_motion target to person under human-motion context.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial018 import (
    InstructionParserV3Trial018,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial019(InstructionParserV3Trial018):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        t = super()._extract_target(instruction, action)

        if action == "edit_motion":
            if re.search(
                r"\b(man|woman|people|person|individuals|his|her)\b",
                text,
            ):
                return "person"

        return t


def build_parser() -> InstructionParserV3Trial019:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial019(kb)
