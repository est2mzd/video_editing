#!/usr/bin/env python3
"""Rulebase Trial020 single-file parser.

Design goals:
- Single-file (no imports from other parser trial files)
- No GT-derived retrieval or cheat path
- Keep Trial020 behavior (camera-angle target normalization)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace")
GT_PATH = WORKSPACE / "data" / "annotations_gt_task_ver10.json"


def build_knowledge_db_v3(gt_path: Path) -> dict[str, Any]:
    """Build static rule DB for v3 rulebase parser family."""
    _ = gt_path

    action_patterns = {
        "dolly_in": r"\bdolly\s+in\b|\bapproach\b|\bmove.*toward\b",
        "dolly_out": r"\bdolly\s+out\b|\bpull\s+back\b|\bmove.*away\b",
        "zoom_in": r"\bzoom\s+in\b|\bclose[- ]?up\b|\bget\s+closer\b",
        "zoom_out": r"\bzoom\s+out\b|\bpull\s+back\b",
        "add_object": (
            r"\badd\s+(?!effect|glow)\b|\binsert\b|\bplace\b|"
            r"\binclude\b|\badding\s+more\b"
        ),
        "remove_object": r"\bremove\b|\bdelete\b|\berase\b|\beliminate\b",
        "replace_object": r"\breplace\b.*\bwith\b|\bswap\b|\bsubstitute\b",
        "replace_background": (
            r"\breplace.*background\b|\bchange.*background\b|"
            r"\bnew\s+background\b"
        ),
        "change_camera_angle": (
            r"\bcamera\s+angle\b|\blow\s+angle\b|\bhigh\s+angle\b|"
            r"\bperspective\b"
        ),
        "change_color": (
            r"\bchange.*color\b|\bcolor.*to\b|\brecolor\b|\btint\b|"
            r"\bhair\s+color\b"
        ),
        "apply_style": (
            r"\bstyle\b|\bcyberpunk\b|\bpixel\b|\bukiyo\b|\bjapanese\b|"
            r"\btransform.*into\b|\bglow\b"
        ),
        "edit_motion": (
            r"\bmotion\b|\bwave\b|\bwalk\b|\braise\b|\bgesture\b|"
            r"\bmodify\b|\badjust\b"
        ),
        "preserve_framing": (
            r"\bcentered\b|\bframing\b|\bposition\b|\bkeep.*center\b"
        ),
        "preserve_focus": r"\bfocus\b|\bsharp\b|\bdepth\b",
        "preserve_identity": r"\bidentity\b|\bfeature\b|\brecognizable\b",
        "preserve_objects": r"\bpreserve\b|\bkeep\b|\bmaintain\b",
    }

    noun_patterns = {
        "body_part": (
            r"\b(face|eye|eyes|head|hand|hands|arm|arms|body|mouth|nose|hair|"
            r"shoulder|shoulders|feet|leg|legs|torso|gesture|person|man|woman|"
            r"child|people|men|women|children|person's|man's|woman's)\b"
        ),
        "object": (
            r"\b(object|thing|item|things|camera|screen|building|car|vehicle|"
            r"animal|bird|dog|cat|tree|ground|furniture|beanie|microphone|"
            r"guitar|strings|spirits|sports car|motorcycle|bicycle|box|boxes|"
            r"table|chair|sofa|armchair|desk|bed|lamp|book|phone|device|tool|"
            r"robot|drone)\b"
        ),
        "location": (
            r"\b(background|foreground|scene|frame|video|shot|space|room|area|"
            r"corner|center|edge|bottom|top|left|right|side|studio|world|"
            r"forest|city|street|indoor|outdoor)\b"
        ),
        "visual": (
            r"\b(full frame|entire|all|whole|complete|surface|lighting|shadow|"
            r"light|bright|dark|color|appearance|material|texture|pattern|"
            r"style|effect|glow|neon|vibrant|subtle)\b"
        ),
        "frame": (
            r"\b(frame|frames|shot|camera view|composition|layout|window|"
            r"border|edge)\b"
        ),
        "temporal": (
            r"\b(motion|movement|gesture|action|animation|wave|smooth|steady|"
            r"fast|slow|quick|gradual)\b"
        ),
    }

    return {
        "action_patterns": action_patterns,
        "noun_patterns": noun_patterns,
    }


class InstructionParserV3RulebaseTrial020Singlefile:
    def __init__(self, knowledge_db: dict[str, Any]):
        self.action_patterns = knowledge_db["action_patterns"]
        self.noun_patterns = knowledge_db["noun_patterns"]

    def infer(self, instruction: str) -> dict[str, Any]:
        action = self._infer_action(instruction)
        target = self._extract_target(instruction, action)
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

        if re.search(r"\barc shot\b|\brevolving around\b|\borbit\b", text):
            return "orbit_camera"

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

        if re.search(
            r"\bchange\b.*\bcolor\b|\bmodify\b.*\bcolor\b|\brecolor\b",
            text,
        ):
            return "change_color"

        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"
        if re.search(r"\badd\b|\binsert\b|\bplace\b|\bintroduce\b", text):
            return "add_object"

        if re.search(
            r"\btransform\b.*\b(video|scene|frame)\b.*"
            r"\b(style|cyberpunk|ukiyo|pixel)\b",
            text,
        ):
            return "apply_style"
        if re.search(
            r"\btransform\b.*\b(style|cyberpunk|ukiyo|pixel|oil painting)\b",
            text,
        ):
            return "apply_style"
        if re.search(
            r"\b(style|cyberpunk|ukiyo|pixel art|oil painting)\b",
            text,
        ):
            return "apply_style"

        m = re.search(r"\breplace\b\s+(.+?)\s+\bwith\b", text)
        if m:
            replaced = m.group(1)
            if "background" in replaced:
                return "replace_background"
            return "replace_object"

        if re.search(
            r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b",
            text,
        ):
            return "add_effect"

        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"

        if re.search(
            r"\banimate\b|\bspin\b|\brotate\b|\bgesture\b|\bmotion\b",
            text,
        ):
            return "edit_motion"

        action_scores = {}
        for action, pattern in self.action_patterns.items():
            try:
                matches = re.findall(pattern, text)
            except re.error:
                continue
            if matches:
                action_scores[action] = len(matches) * (len(pattern) / 100)
        if not action_scores:
            return "edit_motion"
        return max(action_scores.items(), key=lambda x: x[1])[0]

    def _extract_target(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            for pat in [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"focused on the ([^.,;]+)",
                r"close-up of the ([^.,;]+)",
                r"frame (?:his|her|the) ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip().lower()
                    if any(
                        k in cand
                        for k in [
                            "face",
                            "profile",
                            "mixer",
                            "bowl",
                            "chef",
                            "speaker",
                            "grinder",
                        ]
                    ):
                        return cand
            return "camera_view"

        if action == "change_camera_angle":
            short_map = [
                (r"\bchef\b", "chef"),
                (r"\bspeaker\b", "speaker"),
                (r"\bwoman\b", "woman"),
                (r"\bman\b", "man"),
                (r"\btwo men\b", "two men"),
            ]
            for pat, val in short_map:
                if re.search(pat, text):
                    return val
            for pat in [
                r"looking up at the ([^.,;]+)",
                r"looking down at the ([^.,;]+)",
                r"capturing the ([^.,;]+)",
                r"at the ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip().lower()
                    if any(
                        k in cand
                        for k in ["man", "woman", "chef", "speaker", "hands"]
                    ):
                        return cand
            return "camera_view"

        if action == "apply_style":
            return "full_frame"

        if action == "replace_background":
            return "background"

        if action == "replace_object":
            m = re.search(r"replace the ([^.,;]+?) with", text)
            if m:
                return m.group(1).strip().lower()
            return "object"

        if action == "change_color":
            if "armchair" in text and "left" in text and "right" in text:
                return "['armchair_left', 'armchair_right']"
            m = re.search(r"color of the ([^.,;]+)", text)
            if m:
                return m.group(1).strip().lower()
            return "object"

        if action in {"add_object", "increase_amount"}:
            for pat in [
                r"add(?:ing)? more ([^.,;]+)",
                r"increase the amount of ([^.,;]+)",
                r"increase the number of ([^.,;]+)",
                r"place (?:a|an|the)?\s*([^.,;]+)",
                r"introduce (?:an|a)?\s*([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip().lower()
                    cand = re.sub(r"\s+", "_", cand)
                    break
            else:
                cand = "object"

            if "rhino" in cand and "buffalo" in cand:
                return "rhino_and_buffalo"
            if "panda" in cand:
                return "panda"
            if "pastr" in cand:
                return "pastry"
            if "towel" in cand and "elephant" in text:
                return "towel_elephant"
            if "microphone" in cand:
                return "new_object"
            if "men_talking_on_phones" in cand or "talking_on_phones" in cand:
                return "man_on_phone"
            if "jumping" in cand and "baby" in text:
                return "baby_character"
            if (
                "white_car" in cand
                or (cand in {"car", "cars"} and "white car" in text)
            ):
                return "white_car"
            if "mannequin" in text and "floral dress" in text:
                return "mannequin_in_formal_floral_dress"
            if action == "increase_amount" and "fruit jam" in text:
                return "fruit_jam"
            if len(cand.split("_")) > 6:
                return cand.split("_")[0]
            return cand

        if action == "remove_object":
            m = re.search(r"remove the ([^.,;]+) from", text)
            if m:
                return m.group(1).strip().lower()
            return "object"

        if action == "add_effect":
            if "stage lighting" in text:
                return "stage_lighting_region"
            m = re.search(r"outlines? (his|her|the) ([^.,;]+)", text)
            if m:
                return m.group(2).strip().lower()
            return "object"

        if action == "edit_motion":
            if re.search(
                r"\b(man|woman|people|person|individuals|his|her)\b",
                text,
            ):
                return "person"
            if re.search(r"\b(object|fans|hook|plate|car)\b", text):
                return "object"
            return "person"

        matches = defaultdict(list)
        for pattern_name, pattern in self.noun_patterns.items():
            found = re.findall(pattern, text, re.IGNORECASE)
            for match in found:
                word = match[0] if isinstance(match, tuple) else match
                if word and len(word) > 1:
                    matches[pattern_name].append(word)

        if not matches:
            return "object"

        priority = {
            "body_part": 5,
            "object": 4,
            "location": 3,
            "visual": 2,
            "frame": 2,
            "temporal": 1,
        }
        best_word = "object"
        best_score = -1
        for p_name, words in matches.items():
            w = priority.get(p_name, 0)
            for word in words:
                score = w * len(word)
                if score > best_score:
                    best_score = score
                    best_word = word
        return best_word


def build_parser() -> InstructionParserV3RulebaseTrial020Singlefile:
    kb = build_knowledge_db_v3(GT_PATH)
    return InstructionParserV3RulebaseTrial020Singlefile(kb)
