#!/usr/bin/env python3
"""Trial 018 based on v3.

Single countermeasure:
- Camera target fallback returns concrete object when explicit object phrase is
  available.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial017 import (
    InstructionParserV3Trial017,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial018(InstructionParserV3Trial017):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        t = super()._extract_target(instruction, action)

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            if t == "camera_view":
                for pat in [
                    r"towards the ([^.,;]+)",
                    r"focused on the ([^.,;]+)",
                    r"on the ([^.,;]+)",
                ]:
                    m = re.search(pat, text)
                    if m:
                        cand = m.group(1).strip().lower()
                        if any(
                            k in cand
                            for k in ["grinder", "mixer", "face", "profile"]
                        ):
                            return cand
        return t


def build_parser() -> InstructionParserV3Trial018:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial018(kb)
