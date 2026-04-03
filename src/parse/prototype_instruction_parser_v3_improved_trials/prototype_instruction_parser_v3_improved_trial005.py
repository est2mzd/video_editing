#!/usr/bin/env python3
"""Trial 005 based on v3.

Single countermeasure:
- Action-conditioned target resolver to reduce background bias and align with
  target schema patterns (camera_view/full_frame/background/person).
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial004 import (
    InstructionParserV3Trial004,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial005(InstructionParserV3Trial004):
    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            return "camera_view"
        if action == "change_camera_angle":
            return "camera_view"
        if action == "apply_style":
            return "full_frame"
        if action == "replace_background":
            return "background"
        if action in {"edit_motion", "preserve_identity"}:
            return "person"

        if action == "change_color":
            m = re.search(r"color of the ([^.,;]+)", text)
            if m:
                return m.group(1).strip()

        if action in {"replace_object", "add_object", "increase_amount"}:
            for pat in [
                r"replace the ([^.,;]+?) with",
                r"add(?:ing)? more ([^.,;]+)",
                r"increase the amount of ([^.,;]+)",
                r"increase the number of ([^.,;]+)",
                r"place (?:a|an|the)?\\s*([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    tgt = m.group(1).strip().lower()
                    tgt = re.sub(r"\\s+", "_", tgt)
                    return tgt

        if action == "add_effect":
            m = re.search(r"effect to the ([^.,;]+)", text)
            if m:
                return m.group(1).strip().lower()

        return super()._extract_target(instruction, action)


def build_parser() -> InstructionParserV3Trial005:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial005(kb)
