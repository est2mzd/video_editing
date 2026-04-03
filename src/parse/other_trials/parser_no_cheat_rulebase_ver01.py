#!/usr/bin/env python3
"""No-cheat rule-based parser (ver01).

Design constraints:
- Input: instruction text only
- Forbidden at prediction time:
    video_path, class/subclass labels, GT tasks/params
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    default_action: str = "edit_motion"
    default_target: str = "object"


ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "replace_background",
        [r"\\breplace\\b.*\\bbackground\\b", r"\\bbackground\\b"],
    ),
    (
        "replace_object",
        [r"\\breplace\\b.*\\bwith\\b", r"\\bswap\\b", r"\\bsubstitute\\b"],
    ),
    (
        "change_color",
        [
            r"\\bchange\\b.*\\bcolor\\b",
            r"\\brecolor\\b",
            r"\\b(color|colour)\\b",
        ],
    ),
    (
        "add_object",
        [
            r"\\badd\\b",
            r"\\binsert\\b",
            r"\\bplace\\b",
            r"\\bincrease the number of\\b",
        ],
    ),
    (
        "remove_object",
        [r"\\bremove\\b", r"\\bdelete\\b", r"\\berase\\b", r"\\beliminate\\b"],
    ),
    (
        "apply_style",
        [
            r"\\bstyle\\b",
            r"\\bcyberpunk\\b",
            r"\\bukiyo\\b",
            r"\\bwatercolor\\b",
            r"\\boil painting\\b",
            r"\\bpixel\\b",
        ],
    ),
    ("zoom_in", [r"\\bzoom[- ]?in\\b", r"\\bclose[- ]?up\\b"]),
    ("zoom_out", [r"\\bzoom[- ]?out\\b", r"\\bwider\\b"]),
    (
        "dolly_in",
        [
            r"\\bdolly[- ]?in\\b",
            r"\\bmove\\b.*\\btoward\\b",
            r"\\bcloser to\\b",
        ],
    ),
    (
        "change_camera_angle",
        [
            r"\\bcamera angle\\b",
            r"\\blow angle\\b",
            r"\\bhigh angle\\b",
            r"\\bperspective\\b",
        ],
    ),
    (
        "edit_motion",
        [
            r"\\bmotion\\b",
            r"\\bwave\\b",
            r"\\bwalk\\b",
            r"\\braise\\b",
            r"\\bgesture\\b",
        ],
    ),
    ("preserve_focus", [r"\\bfocus\\b", r"\\bsharp\\b"]),
    (
        "preserve_framing",
        [r"\\bframing\\b", r"\\bcentered\\b", r"\\bcomposition\\b"],
    ),
]

TARGET_PATTERNS: list[tuple[str, str]] = [
    ("man's face", r"\\bman's face\\b|\\bman\\'s face\\b"),
    ("woman's hair", r"\\bwoman\\'s hair\\b|\\bhair\\b"),
    ("background", r"\\bbackground\\b"),
    ("person", r"\\b(person|man|woman|speaker|subject|people)\\b"),
    ("face", r"\\bface\\b"),
    ("camera", r"\\bcamera\\b"),
    ("scene", r"\\bscene\\b"),
    ("object", r"\\b(object|item|thing)\\b"),
]

PREP_TARGET_PATTERNS = [
    r"(?:toward|towards|on|onto|to|of|for|around|behind)\\s+the\\s+([^.,;]+)",
    r"(?:toward|towards|on|onto|to|of|for|around|behind)\\s+([^.,;]+)",
]


class NoCheatRuleParserV01:
    """Instruction-only parser."""

    def __init__(self, cfg: RuleConfig | None = None):
        self.cfg = cfg or RuleConfig()

    def pred(self, instruction: str) -> dict:
        action = self._infer_action(instruction)
        target = self._infer_target(instruction, action)
        return {
            "tasks": [
                {
                    "action": action,
                    "target": target,
                    "constraints": [],
                    "params": {},
                }
            ]
        }

    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()
        best_action = self.cfg.default_action
        best_score = 0
        for action, patterns in ACTION_PATTERNS:
            score = sum(1 for pat in patterns if re.search(pat, text))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _infer_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        # Prefer prepositional phrase extraction for camera/object actions.
        if action in {
            "zoom_in",
            "zoom_out",
            "dolly_in",
            "replace_object",
            "add_object",
            "remove_object",
        }:
            for pat in PREP_TARGET_PATTERNS:
                m = re.search(pat, text)
                if m:
                    cand = self._sanitize_target(m.group(1))
                    if cand:
                        return cand

        for label, pat in TARGET_PATTERNS:
            if re.search(pat, text):
                return label

        return self.cfg.default_target

    @staticmethod
    def _sanitize_target(value: str) -> str:
        text = re.sub(
            r"\\b(with|while|and|ensure|maintain|keep)\\b.*$",
            "",
            value,
        ).strip()
        text = re.sub(r"[^a-z0-9\\s'_-]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        if not text:
            return ""
        if text in {"subject", "subjects"}:
            return "person"
        return text
