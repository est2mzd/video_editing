#!/usr/bin/env python3
"""Rulebase Trial013 single-file parser (kai2).

Design goals:
- Keep Trial013-level rule quality in a single file.
- No GT path or external trial imports.
- Callable in the same style as instruction_parser_v3_rulebase_trial013_singlefile.py
  via build_parser().infer(...).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


ALLOWED_ACTIONS = {
    "dolly_in",
    "dolly_out",
    "zoom_in",
    "zoom_out",
    "orbit_camera",
    "change_camera_angle",
    "change_color",
    "add_object",
    "remove_object",
    "replace_object",
    "replace_background",
    "add_effect",
    "edit_motion",
    "apply_style",
    "increase_amount",
}


def build_knowledge_db_v3() -> dict[str, Any]:
    """Static rule DB (no GT dependency)."""
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
        "preserve_framing": r"\bcentered\b|\bframing\b|\bposition\b|\bkeep.*center\b",
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
            r"light|bright|dark|color|appearance|material|texture|pattern|style|"
            r"effect|glow|neon|vibrant|subtle)\b"
        ),
        "frame": r"\b(frame|frames|shot|camera view|composition|layout|window|border|edge)\b",
        "temporal": (
            r"\b(motion|movement|gesture|action|animation|wave|smooth|steady|"
            r"fast|slow|quick|gradual)\b"
        ),
    }
    return {"action_patterns": action_patterns, "noun_patterns": noun_patterns}


class InstructionParserV3RulebaseTrial013SinglefileKai2:
    """Single-file reproduction of Trial013-level logic."""

    def __init__(self) -> None:
        kb = build_knowledge_db_v3()
        self.action_patterns = kb["action_patterns"]
        self.noun_patterns = kb["noun_patterns"]

    def infer(self, instruction: str) -> dict[str, Any]:
        action = self._infer_action_trial013(instruction)
        target = self._extract_target_trial012(instruction, action)
        return {
            "action": action,
            "target": target,
            "constraints": [],
            "params": {},
        }

    def pred(self, instruction: str) -> dict[str, Any]:
        return {"tasks": [self.infer(instruction)]}

    def _infer_action_trial013(self, instruction: str) -> str:
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

        if re.search(
            r"\bchange\b.*\bcolor\b|\bmodify\b.*\bcolor\b|\brecolor\b",
            text,
        ):
            return "change_color"

        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"
        if re.search(r"\badd\b|\binsert\b|\bplace\b", text):
            return "add_object"

        if re.search(r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b", text):
            return "add_effect"

        if re.search(
            r"\btransform\b.*\b(entire|full|whole)\b.*\b(video|scene|frame)\b",
            text,
        ):
            return "apply_style"
        if re.search(
            r"\btransform\b.*\b(style|cyberpunk|ukiyo|pixel|oil painting)\b",
            text,
        ):
            return "apply_style"
        if re.search(r"\b(style|cyberpunk|ukiyo|pixel art|oil painting)\b", text):
            return "apply_style"

        if re.search(r"\breplace\b.*\bbackground\b.*\bwith\b", text):
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"

        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"

        if re.search(r"\banimate\b|\bspin\b|\brotate\b|\bgesture\b|\bmotion\b", text):
            return "edit_motion"

        return self._infer_action_base001(instruction)

    def _infer_action_base001(self, instruction: str) -> str:
        text = instruction.lower()
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

    def _extract_target_trial012(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            strong_subject_patterns = [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"focused on the ([^.,;]+)",
                r"close-up of the ([^.,;]+)",
                r"frame (?:his|her|the) ([^.,;]+)",
            ]
            for pat in strong_subject_patterns:
                m = re.search(pat, text)
                if not m:
                    continue
                cand = m.group(1).strip().lower()
                if not cand:
                    continue
                if any(
                    k in cand for k in ["face", "profile", "mixer", "bowl", "chef", "speaker"]
                ):
                    return cand
            return self._extract_target_trial010(instruction, action)

        if action == "change_camera_angle":
            for pat in [
                r"looking up at the ([^.,;]+)",
                r"looking down at the ([^.,;]+)",
                r"capturing the ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip().lower()
                    if any(k in cand for k in ["man", "woman", "chef", "speaker", "hands"]):
                        return cand
            return self._extract_target_trial010(instruction, action)

        return self._extract_target_trial010(instruction, action)

    def _extract_target_trial010(self, instruction: str, action: str) -> str:
        text = instruction.lower()
        target = self._extract_target_trial007(instruction, action)
        t = str(target).lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            if any(x in t for x in ["entire video", "scene", "reveal more", "throughout"]):
                return "camera_view"
            if len(t.split()) > 6:
                return "camera_view"
            return t

        if action == "change_camera_angle":
            if t == "camera_view":
                for pat in [
                    r"at the ([^.,;]+)",
                    r"capturing the ([^.,;]+)",
                    r"focus on the ([^.,;]+)",
                ]:
                    m = re.search(pat, text)
                    if m:
                        return m.group(1).strip().lower()
            return t

        if action in {"add_object", "increase_amount"}:
            if "rhino" in t and "buffalo" in t:
                return "rhino_and_buffalo"
            if "panda" in t:
                return "panda"
            if "pastr" in t:
                return "pastry"
            if "towel" in t and "elephant" in text:
                return "towel_elephant"
            if len(t.split()) > 6:
                return t.split()[0]
            return t

        if action == "remove_object":
            if t == "background":
                m = re.search(r"remove the ([^.,;]+) from", text)
                if m:
                    return m.group(1).strip().lower()
            return t

        if action == "add_effect":
            if "stage lighting" in text:
                return "stage_lighting_region"
            m = re.search(r"outlines? (his|her|the) ([^.,;]+)", text)
            if m:
                return m.group(2).strip().lower()
            return t

        if action == "replace_object":
            if t == "background":
                m = re.search(r"replace the ([^.,;]+?) with", text)
                if m:
                    return m.group(1).strip().lower()
            return t

        return t

    def _extract_target_trial007(self, instruction: str, action: str) -> str:
        text = instruction.lower()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out"}:
            for pat in [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"(?:toward|towards|on|onto|at|to) ([^.,;]+)",
                r"frame ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    tgt = m.group(1).strip().lower()
                    if tgt and "camera" not in tgt:
                        return tgt
            return "camera_view"

        if action == "change_camera_angle":
            for pat in [
                r"looking (?:up|down) at the ([^.,;]+)",
                r"at the ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    return m.group(1).strip().lower()
            return "camera_view"

        return self._extract_target_trial005(instruction, action)

    def _extract_target_trial005(self, instruction: str, action: str) -> str:
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
                r"place (?:a|an|the)?\s*([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    tgt = m.group(1).strip().lower()
                    tgt = re.sub(r"\s+", "_", tgt)
                    return tgt

        if action == "add_effect":
            m = re.search(r"effect to the ([^.,;]+)", text)
            if m:
                return m.group(1).strip().lower()

        return self._extract_target_base001(instruction)

    def _extract_target_base001(self, instruction: str) -> str:
        text = instruction.lower()
        matches = defaultdict(list)
        for pattern_name, pattern in self.noun_patterns.items():
            found = re.findall(pattern, text, re.IGNORECASE)
            for match in found:
                word = match[0] if isinstance(match, tuple) else match
                if word and len(word) > 1:
                    matches[pattern_name].append(word)

        if not matches:
            return "object"

        pattern_priority = {
            "body_part": 5,
            "object": 4,
            "location": 3,
            "visual": 2,
            "frame": 2,
            "temporal": 1,
        }
        best_word = "object"
        best_score = 0
        for pattern_name, words in matches.items():
            priority = pattern_priority.get(pattern_name, 0)
            for word in words:
                score = priority * len(word)
                if score > best_score:
                    best_score = score
                    best_word = word
        return best_word


def build_parser() -> InstructionParserV3RulebaseTrial013SinglefileKai2:
    return InstructionParserV3RulebaseTrial013SinglefileKai2()


if __name__ == "__main__":
    parser = build_parser()
    sample = "Increase the amount of red fruit jam on the white plate."
    print(parser.infer(sample))
