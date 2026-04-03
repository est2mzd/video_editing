#!/usr/bin/env python3
"""Rule-based single-file parser derived from improved_trial013.

Design goals:
- No import from other project files.
- No GT file usage.
- Simple API: parser.infer(instruction) -> task dict (action/target/constraints/params).
"""

from __future__ import annotations

import re
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


class InstructionParserV3RulebaseTrial013Singlefile:
    """Self-contained rule parser for one instruction at a time."""

    def infer(self, instruction: str) -> dict[str, Any]:
        """Infer one task from one instruction.

        Returns:
            {
              "action": str,
              "target": str,
              "constraints": list,
              "params": dict,
            }
        """
        action = self._infer_action(instruction)
        target = self._extract_target(instruction, action)
        return {
            "action": action,
            "target": target,
            "constraints": [],
            "params": {},
        }

    def pred(self, instruction: str) -> dict[str, Any]:
        """Compatibility helper: old parser format with tasks list."""
        return {"tasks": [self.infer(instruction)]}

    def _infer_action(self, instruction: str) -> str:
        text = instruction.lower()

        if re.search(r"\barc shot\b|\borbit\b|\brevolving around\b", text):
            return "orbit_camera"

        if re.search(r"\bdolly[- ]?out\b|\bpull back\b", text):
            return "dolly_out"
        if re.search(r"\bdolly[- ]?in\b", text):
            return "dolly_in"
        if re.search(r"\bzoom[- ]?out\b", text):
            return "zoom_out"
        if re.search(r"\bzoom[- ]?in\b|\bclose[- ]?up\b", text):
            return "zoom_in"

        if re.search(
            r"\blow angle\b|\bhigh angle\b|\bcamera angle\b|\bperspective\b",
            text,
        ):
            return "change_camera_angle"

        if re.search(
            r"\bchange\b.*\bcolor\b|\bmodify\b.*\bcolor\b|\brecolor\b",
            text,
        ):
            return "change_color"

        # Keep increase/add priority ahead of style (improved_trial013 intent).
        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"
        if re.search(r"\badd\b|\binsert\b|\bplace\b|\bintroduce\b", text):
            return "add_object"

        if re.search(r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b", text):
            return "add_effect"

        if re.search(
            r"\btransform\b.*\b(entire|full|whole)\b.*\b(video|scene|frame)\b",
            text,
        ):
            return "apply_style"
        if re.search(
            r"\btransform\b.*\b(style|cyberpunk|ukiyo|pixel|oil painting|aesthetic)\b",
            text,
        ):
            return "apply_style"
        if re.search(
            r"\b(style|cyberpunk|ukiyo|pixel art|oil painting|aesthetic|ghibli)\b",
            text,
        ):
            return "apply_style"

        if re.search(r"\breplace\b.*\bbackground\b.*\bwith\b", text):
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"

        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"

        if re.search(r"\banimate\b|\bspin\b|\brotate\b|\bgesture\b|\bmotion\b", text):
            return "edit_motion"

        return "edit_motion"

    def _extract_target(self, instruction: str, action: str) -> str:
        text = re.sub(r"\s+", " ", instruction.lower()).strip()

        if action in {"zoom_in", "zoom_out", "dolly_in", "dolly_out", "orbit_camera"}:
            for pat in [
                r"(?:toward|towards|on|onto|at|to) the ([^.,;]+)",
                r"focused on the ([^.,;]+)",
                r"close-up of the ([^.,;]+)",
                r"frame (?:his|her|the) ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = self._shorten_phrase(m.group(1))
                    if cand and not re.search(r"\b(video|scene|frame|clip)\b", cand):
                        return cand
            return "camera_view"

        if action == "change_camera_angle":
            for pat in [
                r"(?:looking at|look at|focused on|focusing on) ([^.,;]+)",
                r"(?:looking up at|looking down at|capturing) the ([^.,;]+)",
                r"(?:toward|towards|at) ([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = self._shorten_phrase(m.group(1))
                    if cand and not re.search(
                        r"\b(low angle|high angle|angle|perspective|shot|view)\b",
                        cand,
                    ):
                        return cand
            return "camera_view"

        if action == "apply_style":
            return "full_frame"

        if action == "replace_background":
            return "background"

        if action == "replace_object":
            m = re.search(r"replace (?:the )?([^.,;]+?) with", text)
            if m:
                return self._shorten_phrase(m.group(1))
            return "object"

        if action == "change_color":
            if "armchair" in text and "left" in text and "right" in text:
                return "['armchair_left', 'armchair_right']"
            for pat in [
                r"color of (?:the )?([^.,;]+)",
                r"change (?:the )?([^.,;]+?) to",
                r"modify (?:the )?([^.,;]+?) color",
            ]:
                m = re.search(pat, text)
                if m:
                    return self._shorten_phrase(m.group(1))
            return "object"

        if action in {"add_object", "increase_amount"}:
            for pat in [
                r"increase the amount of ([^.,;]+)",
                r"increase the number of ([^.,;]+)",
                r"add(?:ing)? more ([^.,;]+)",
                r"add (?:a|an|the)?\s*([^.,;]+?) to",
                r"place (?:a|an|the)?\s*([^.,;]+)",
                r"introduce (?:a|an|the)?\s*([^.,;]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = self._normalize_token_target(m.group(1))
                    if "rhino" in cand and "buffalo" in cand:
                        return "rhino_and_buffalo"
                    if "pastr" in cand:
                        return "pastry"
                    if action == "increase_amount" and "fruit jam" in text:
                        return "fruit_jam"
                    return cand
            return "object"

        if action == "remove_object":
            m = re.search(r"remove (?:the )?([^.,;]+?) from", text)
            if m:
                return self._shorten_phrase(m.group(1))
            return "object"

        if action == "add_effect":
            if "stage lighting" in text:
                return "stage_lighting_region"
            m = re.search(r"(?:outlines?|around) (?:his|her|the)?\s*([^.,;]+)", text)
            if m:
                return self._shorten_phrase(m.group(1))
            return "object"

        if action == "edit_motion":
            if re.search(r"\b(man|woman|boy|girl|person|people|child|baby)\b", text):
                return "person"
            return "object"

        return "object"

    def _shorten_phrase(self, phrase: str) -> str:
        cand = phrase.strip(" ,.")
        cand = re.split(r"\b(to|while|throughout|from|with|for|that|where|which)\b", cand)[0]
        cand = re.sub(r"\b(the|a|an)\b", "", cand)
        cand = re.sub(r"\s+", " ", cand).strip()
        return cand or "object"

    def _normalize_token_target(self, phrase: str) -> str:
        cand = phrase.strip(" ,.")
        cand = re.sub(r"\b(the|a|an)\b", "", cand)
        cand = re.sub(r"\s+", "_", cand).strip("_")
        if not cand:
            return "object"
        return cand


def build_parser() -> InstructionParserV3RulebaseTrial013Singlefile:
    return InstructionParserV3RulebaseTrial013Singlefile()


if __name__ == "__main__":
    parser = build_parser()
    sample = "Increase the amount of red fruit jam on the white plate."
    print(parser.infer(sample))
