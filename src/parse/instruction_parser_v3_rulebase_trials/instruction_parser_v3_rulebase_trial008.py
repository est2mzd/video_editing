#!/usr/bin/env python3
"""Trial 008 based on v3.

Single countermeasure:
- Refine intent-router boundaries for action disambiguation.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial007 import (
    InstructionParserV3Trial007,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial008(InstructionParserV3Trial007):
    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()

        if re.search(r"\bdolly[- ]?out\b|\bpull back\b", text):
            return "dolly_out"
        if re.search(r"\bdolly[- ]?in\b", text):
            return "dolly_in"
        if re.search(r"\bzoom[- ]?out\b", text):
            return "zoom_out"
        if re.search(r"\bzoom[- ]?in\b|\bclose[- ]?up\b", text):
            return "zoom_in"

        if re.search(r"\blow angle\b|\bhigh angle\b|\bcamera angle\b", text):
            return "change_camera_angle"

        if re.search(r"\btransform\b.*\b(style|cyberpunk|ukiyo|pixel)\b", text):
            return "apply_style"
        if re.search(r"\b(style|cyberpunk|ukiyo|pixel art|oil painting)\b", text):
            return "apply_style"

        if re.search(r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b", text):
            return "add_effect"

        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"

        if re.search(r"\breplace\b.*\bwith\b", text):
            if re.search(r"\breplace\b.*\bbackground\b", text):
                return "replace_background"
            return "replace_object"
        if re.search(r"\bbackground\b", text) and re.search(r"\breplace\b", text):
            return "replace_background"

        if re.search(r"\badd\b|\binsert\b|\bplace\b", text):
            return "add_object"
        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"
        if re.search(r"\bchange\b.*\bcolor\b|\brecolor\b", text):
            return "change_color"

        return super()._infer_action(instruction)


def build_parser() -> InstructionParserV3Trial008:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial008(kb)
