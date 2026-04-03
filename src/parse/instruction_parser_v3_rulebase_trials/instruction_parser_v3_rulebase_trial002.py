#!/usr/bin/env python3
"""Trial 002 based on prototype_instruction_parser_v3_improved.py.

Countermeasure:
- Suppress over-selection of preserve_* actions when an explicit edit intent
  (zoom/dolly/add/replace/change/transform/modify) exists in the instruction.
"""

from __future__ import annotations

import re
from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import (
    InstructionParserV3,
    build_knowledge_db_v3,
)

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial002(InstructionParserV3):
    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()
        action_scores: dict[str, float] = {}

        has_edit_intent = bool(
            re.search(
                r"\\b(apply|replace|add|remove|increase|change|zoom|dolly|"
                r"transform|perform|modify|swap|insert)\\b",
                text,
            )
        )
        has_style_intent = bool(
            re.search(
                r"\\b(style|cyberpunk|ukiyo|pixel|oil painting)\\b",
                text,
            )
        )

        for action, pattern in self.action_patterns.items():
            try:
                matches = re.findall(pattern, text)
            except re.error:
                continue
            if not matches:
                continue

            score = len(matches) * (len(pattern) / 100)

            # Main fix: do not let preserve_* dominate primary edit commands.
            if action.startswith("preserve_") and has_edit_intent:
                score *= 0.40

            # Prevent style from dominating explicit object manipulation tasks.
            if (
                action == "apply_style"
                and has_edit_intent
                and not has_style_intent
            ):
                score *= 0.60

            # Keep camera edits competitive when camera cues are present.
            if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
                if re.search(r"\\b(camera|zoom|dolly|close[- ]?up)\\b", text):
                    score *= 1.25

            # Keep object edits competitive with explicit object-edit verbs.
            if action in {"add_object", "replace_object", "remove_object"}:
                if re.search(
                    r"\\b(add|replace|remove|insert|swap|increase)\\b",
                    text,
                ):
                    score *= 1.20

            action_scores[action] = score

        if not action_scores:
            return "edit_motion"

        return max(action_scores.items(), key=lambda x: x[1])[0]


def build_parser() -> InstructionParserV3Trial002:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial002(kb)
