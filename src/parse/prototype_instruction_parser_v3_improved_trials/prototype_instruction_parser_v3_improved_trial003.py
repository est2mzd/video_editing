#!/usr/bin/env python3
"""Trial 003 based on v3.

Single countermeasure:
- Add missing action categories: increase_amount and add_effect.
"""

from __future__ import annotations

from pathlib import Path

from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved_trial002 import (
    InstructionParserV3Trial002,
)
from parse.prototype_instruction_parser_v3_improved_trials.prototype_instruction_parser_v3_improved import build_knowledge_db_v3

GT_PATH = Path("/workspace/data/annotations_gt_task_ver10.json")


class InstructionParserV3Trial003(InstructionParserV3Trial002):
    def __init__(self, knowledge_db: dict):
        super().__init__(knowledge_db)
        self.action_patterns.update(
            {
                "increase_amount": (
                    r"\\bincrease the amount of\\b|"
                    r"\\bincrease the number of\\b|"
                    r"\\bmore\\b"
                ),
                "add_effect": (
                    r"\\beffect\\b|\\bglow\\b|\\bneon\\b|"
                    r"\\bpulse\\b|\\baura\\b"
                ),
            }
        )


def build_parser() -> InstructionParserV3Trial003:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3Trial003(kb)
