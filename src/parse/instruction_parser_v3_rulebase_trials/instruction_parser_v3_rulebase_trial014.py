#!/usr/bin/env python3
"""Trial 014 based on v3.

Single countermeasure:
- Strengthen replace_background/replace_object routing before add_effect.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial013 import (
    InstructionParserV3Trial013,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial014(InstructionParserV3Trial013):
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

        if re.search(r"\bchange\b.*\bcolor\b|\bmodify\b.*\bcolor\b|\brecolor\b", text):
            return "change_color"

        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"
        if re.search(r"\badd\b|\binsert\b|\bplace\b|\bintroduce\b", text):
            return "add_object"

        # Main change: resolve replacement class before effect/style words.
        if re.search(r"\breplace\b", text) and re.search(r"\bbackground\b", text):
            if re.search(r"\breplace\b.*\bwith\b", text):
                return "replace_background"
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"

        if re.search(r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b", text):
            return "add_effect"

        if re.search(r"\btransform\b.*\b(entire|full|whole)\b.*\b(video|scene|frame)\b", text):
            return "apply_style"
        if re.search(r"\btransform\b.*\b(style|cyberpunk|ukiyo|pixel|oil painting)\b", text):
            return "apply_style"
        if re.search(r"\b(style|cyberpunk|ukiyo|pixel art|oil painting)\b", text):
            return "apply_style"

        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"
        if re.search(r"\banimate\b|\bspin\b|\brotate\b|\bgesture\b|\bmotion\b", text):
            return "edit_motion"

        return super()._infer_action(instruction)


def build_parser() -> InstructionParserV3Trial014:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial014(kb)
