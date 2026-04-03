#!/usr/bin/env python3
"""Trial 016 based on v3.

Single countermeasure:
- Resolve style-vs-add and object-vs-background replacement ambiguity.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial015 import (
    InstructionParserV3Trial015,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial016(InstructionParserV3Trial015):
    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()

        # Style transform should dominate even when additive words appear.
        if re.search(
            r"\btransform\b.*\b(video|scene|frame)\b.*"
            r"\b(style|cyberpunk|ukiyo|pixel)\b",
            text,
        ):
            return "apply_style"

        # Replacement ambiguity handling.
        m = re.search(r"\breplace\b\s+(.+?)\s+\bwith\b", text)
        if m:
            replaced = m.group(1)
            if "background" in replaced:
                return "replace_background"
            return "replace_object"

        return super()._infer_action(instruction)


def build_parser() -> InstructionParserV3Trial016:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial016(kb)
