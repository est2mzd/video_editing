#!/usr/bin/env python3
"""Mixed parser trial 002.

Design constraints:
- Do not import prototype_instruction_parser_v3_singlefile.py.
- Do not use per-video GT lookup.
- Predict from instruction text only.

Mix modes:
- llm_main: LLM prediction is primary, rule-based guardrails as fallback.
- rule_main: Rule-based prediction is primary.
    LLM is used only for low-confidence cases.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


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


class LLMConfig:
    def __init__(self) -> None:
        self.model_name = os.environ.get(
            "LLM_MODEL_NAME",
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.max_new_tokens = int(os.environ.get("LLM_MAX_NEW_TOKENS", "96"))
        self.batch_size = int(os.environ.get("LLM_BATCH_SIZE", "8"))
        self.mix_mode = os.environ.get("MIX_MODE", "llm_main").strip().lower()


class InstructionParserV3MixedTrial002:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._tok = None
        self._model = None
        self._ready = False
        self._cache: dict[str, dict[str, str]] = {}

    def pred(self, instruction: str) -> dict[str, Any]:
        pred = self._pred_one(instruction)
        return {
            "tasks": [
                {
                    "action": pred["action"],
                    "target": pred["target"],
                    "constraints": [],
                    "params": {},
                }
            ]
        }

    def pred_batch(
        self,
        instructions: list[str],
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        if not instructions:
            return []

        bs = max(1, int(batch_size or self.cfg.batch_size))
        out: list[dict[str, Any]] = [
            {
                "tasks": [
                    {
                        "action": "edit_motion",
                        "target": "object",
                        "constraints": [],
                        "params": {},
                    }
                ]
            }
            for _ in instructions
        ]

        pending_idx: list[int] = []
        prompts: list[str] = []
        for i, inst in enumerate(instructions):
            cached = self._cache.get(inst)
            if cached is not None:
                out[i] = {
                    "tasks": [
                        {
                            "action": cached["action"],
                            "target": cached["target"],
                            "constraints": [],
                            "params": {},
                        }
                    ]
                }
                continue
            pending_idx.append(i)
            prompts.append(self._build_prompt(inst))

        if pending_idx and self._ensure_llm():
            for start in range(0, len(prompts), bs):
                p_chunk = prompts[start:start + bs]
                idx_chunk = pending_idx[start:start + bs]
                inst_chunk = [instructions[i] for i in idx_chunk]
                try:
                    inputs = self._tok(
                        p_chunk,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    gen = self._model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                    )
                    texts = self._tok.batch_decode(
                        gen,
                        skip_special_tokens=True,
                    )
                    for i2, inst, text in zip(idx_chunk, inst_chunk, texts):
                        parsed = self._extract_json(text)
                        norm = self._normalize_prediction(parsed, inst)
                        self._cache[inst] = norm
                        out[i2] = {
                            "tasks": [
                                {
                                    "action": norm["action"],
                                    "target": norm["target"],
                                    "constraints": [],
                                    "params": {},
                                }
                            ]
                        }
                except Exception:
                    for i2, inst in zip(idx_chunk, inst_chunk):
                        norm = self._heuristic_prediction(inst)
                        self._cache[inst] = norm
                        out[i2] = {
                            "tasks": [
                                {
                                    "action": norm["action"],
                                    "target": norm["target"],
                                    "constraints": [],
                                    "params": {},
                                }
                            ]
                        }

        for i, inst in enumerate(instructions):
            if inst not in self._cache:
                norm = self._heuristic_prediction(inst)
                self._cache[inst] = norm
                out[i] = {
                    "tasks": [
                        {
                            "action": norm["action"],
                            "target": norm["target"],
                            "constraints": [],
                            "params": {},
                        }
                    ]
                }

        return out

    def _pred_one(self, instruction: str) -> dict[str, str]:
        if instruction in self._cache:
            return self._cache[instruction]

        if self._ensure_llm():
            try:
                prompt = self._build_prompt(instruction)
                inputs = self._tok(prompt, return_tensors="pt")
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=False,
                )
                text = self._tok.decode(gen[0], skip_special_tokens=True)
                parsed = self._extract_json(text)
                norm = self._normalize_prediction(parsed, instruction)
                self._cache[instruction] = norm
                return norm
            except Exception:
                pass

        norm = self._heuristic_prediction(instruction)
        self._cache[instruction] = norm
        return norm

    def _ensure_llm(self) -> bool:
        if self._ready:
            return True
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return False
        try:
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
            )
            self._ready = True
            return True
        except Exception:
            return False

    def _build_prompt(self, instruction: str) -> str:
        lines = [
            "You are an instruction-to-edit-task parser.",
            "Output JSON only with keys action,target.",
            (
                "Allowed actions: dolly_in,dolly_out,zoom_in,zoom_out,"
                "orbit_camera,change_camera_angle,change_color,add_object,"
                "remove_object,replace_object,replace_background,add_effect,"
                "edit_motion,apply_style,increase_amount."
            ),
            "Rules:",
            (
                "- If whole visual style is edited: "
                "action=apply_style, target=full_frame."
            ),
            (
                "- If camera movement/angle/zoom: "
                "camera action + target=camera_view."
            ),
            (
                "- If object color is changed: "
                "action=change_color with object target."
            ),
            "- If glow/fire/particle/light is added: action=add_effect.",
            "- If add more / increase number/count: action=increase_amount.",
            "- If add/insert/place object: action=add_object.",
            "- Keep target short and specific.",
            f"Instruction: {instruction}",
            "JSON:",
        ]
        return "\\n".join(lines)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _normalize_prediction(
        self,
        pred: dict[str, Any] | None,
        instruction: str,
    ) -> dict[str, str]:
        locked_action = self._locked_action(instruction)
        if locked_action is not None:
            return {
                "action": locked_action,
                "target": self._heuristic_target(
                    instruction,
                    locked_action,
                ),
            }

        if not pred:
            return self._heuristic_prediction(instruction)

        llm_action = str(pred.get("action", "")).strip().lower()
        llm_target = str(pred.get("target", "")).strip().lower()
        llm_action = self._sanitize_action(llm_action, instruction)
        llm_target = self._sanitize_target(llm_target)

        rule_action = self._rulebase_action(instruction)
        rule_target = self._rulebase_target(instruction, rule_action)
        rule_high = self._is_rule_high_confidence(instruction)

        if self.cfg.mix_mode == "rule_main":
            action = rule_action
            # Let LLM rescue low-confidence generic edits only.
            if (
                action == "edit_motion"
                and llm_action in ALLOWED_ACTIONS
                and llm_action != "edit_motion"
                and not rule_high
            ):
                action = llm_action
            target = rule_target
            if (
                target in {"object", "camera_view"}
                and llm_target
                and llm_target not in {"object", "camera_view"}
            ):
                target = llm_target
            return {"action": action, "target": target}

        # llm_main (default)
        action = llm_action
        if action not in ALLOWED_ACTIONS:
            action = rule_action
        if rule_high and action != rule_action:
            action = rule_action
        target = llm_target
        if not target or target in {"object", "camera_view"}:
            target = self._rulebase_target(instruction, action)
        if not target:
            target = rule_target
        return {"action": action, "target": target}

    def _sanitize_action(self, action: str, instruction: str) -> str:
        text = instruction.lower()
        if action not in ALLOWED_ACTIONS:
            return ""
        if (
            action == "apply_style"
            and "style" not in text
            and "cyberpunk" not in text
            and "transform" not in text
        ):
            return ""
        if action == "replace_background" and "background" not in text:
            return ""
        return action

    def _sanitize_target(self, target: str) -> str:
        target = re.sub(r"\s+", " ", target).strip()
        target = re.sub(r"[^a-z0-9 _\-\[\]',]", "", target)
        return target[:96]

    def _is_rule_high_confidence(self, instruction: str) -> bool:
        text = instruction.lower()
        return bool(
            re.search(
                r"\bdolly[- ]?(in|out)\b|\bzoom[- ]?(in|out)\b|"
                r"\bchange\b.*\bcolor\b|\brecolor\b|"
                r"\breplace\b.*\bwith\b|\bremove\b|\berase\b|\bdelete\b|"
                r"\bincrease the (number|amount) of\b|\badd more\b",
                text,
            )
        )

    def _rulebase_action(self, instruction: str) -> str:
        text = instruction.lower()
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
        if re.search(r"\bincrease the amount of\b", text):
            return "increase_amount"
        if re.search(r"\bincrease the number of\b", text):
            return "add_object"
        if re.search(r"\badd\b|\binsert\b|\bplace\b", text):
            return "add_object"
        if re.search(
            r"\beffect\b|\bglow\b|\baura\b|\bflame\b|\bpulse\b",
            text,
        ):
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
        if re.search(
            r"\b(style|cyberpunk|ukiyo|pixel art|oil painting)\b",
            text,
        ):
            return "apply_style"
        if re.search(r"\breplace\b.*\bbackground\b.*\bwith\b", text):
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"
        if re.search(r"\bremove\b|\bdelete\b|\berase\b", text):
            return "remove_object"
        if re.search(
            r"\banimate\b|\bspin\b|\brotate\b|\bgesture\b|\bmotion\b",
            text,
        ):
            return "edit_motion"
        return self._heuristic_action(instruction)

    def _rulebase_target(self, instruction: str, action: str) -> str:
        return self._heuristic_target(instruction, action)

    def _heuristic_prediction(self, instruction: str) -> dict[str, str]:
        action = self._heuristic_action(instruction)
        target = self._heuristic_target(instruction, action)
        return {"action": action, "target": target}

    def _locked_action(self, instruction: str) -> str | None:
        text = instruction.lower()
        # apply_style: style keywords override replace_background lock
        if re.search(
            r"\bcyberpunk\b|\bneon sign\b|\bvintage\b|\banime\b"
            r"|\bwatercolor\b|\boil.?paint\b|\bsketch\b",
            text,
        ):
            return "apply_style"
        # replace_background: only when background is the direct object of
        # replace (i.e. within ~8 words after 'replace')
        if re.search(
            r"\breplace\s+(?:\w+\s+){0,8}background\b", text
        ):
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"
        if re.search(
            r"\bincrease\b.*\b(number|count|amount)\b|\badd more\b",
            text,
        ):
            return "add_object"
        if re.search(r"\bdolly.?in\b", text):
            return "dolly_in"
        if re.search(r"\bdolly.?out\b", text):
            return "dolly_out"
        if re.search(r"\bzoom in\b|\bzoom-in\b", text):
            return "zoom_in"
        if re.search(r"\bzoom out\b", text):
            return "zoom_out"
        if re.search(r"\blow angle\b|\bhigh angle\b|\bcamera angle\b", text):
            return "change_camera_angle"
        if re.search(r"\bchange\b.*\bcolor\b|\brecolor\b", text):
            return "change_color"
        if re.search(r"\bremove\b|\berase\b|\bdelete\b", text):
            return "remove_object"
        return None

    def _heuristic_action(self, instruction: str) -> str:
        text = instruction.lower()
        if re.search(r"\bcyberpunk\b|\bstyle\b|\bstylize\b|\bcomic\b", text):
            return "apply_style"
        if re.search(r"\bzoom in\b|\bzoom-in\b|\bclose-up\b", text):
            return "zoom_in"
        if re.search(r"\bzoom out\b|\bwide shot\b|\bwider\b", text):
            return "zoom_out"
        if re.search(r"\bdolly.?in\b", text):
            return "dolly_in"
        if re.search(r"\bdolly.?out\b", text):
            return "dolly_out"
        if re.search(r"\borbit\b", text):
            return "orbit_camera"
        if re.search(
            r"\blow angle\b|\bhigh angle\b|\bcamera angle\b|\bperspective\b",
            text,
        ):
            return "change_camera_angle"
        if re.search(r"\breplace\b.*\bbackground\b", text):
            return "replace_background"
        if re.search(r"\bremove\b|\berase\b|\bdelete\b", text):
            return "remove_object"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"
        if re.search(r"\bchange\b.*\bcolor\b|\brecolor\b", text):
            return "change_color"
        if re.search(
            r"\bincrease\b.*\b(number|count|amount)\b|\badd more\b",
            text,
        ):
            return "add_object"
        if re.search(
            r"\bglow\b|\bflame\b|\bparticle\b|\bfire\b|\beffect\b",
            text,
        ):
            return "add_effect"
        if re.search(r"\badd\b|\binsert\b|\bplace\b", text):
            return "add_object"
        return "edit_motion"

    def _heuristic_target(self, instruction: str, action: str) -> str:
        text = re.sub(r"\s+", " ", instruction.lower()).strip()

        if action == "edit_motion":
            motion_verbs = (
                r"raise|tilt|nod|turn|rotate|move|smile|look|speak|walk|"
                r"jump|wave|blink|lean|point|gesture"
            )
            for pat in [
                rf"(?:so that|so) (?:the )?([a-z0-9'\- ]+?) "
                rf"(?:{motion_verbs})(?:s|ing)?\b",
                r"(?:animate|modify|adjust|make) (?:the )?"
                r"([a-z0-9'\- ]+?) (?:to|so that)",
                rf"(?:the )?([a-z0-9'\- ]+?) "
                rf"(?:{motion_verbs})(?:s|ing)?\b",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip(" ,.")
                    cand = re.sub(
                        r"\b(in the foreground|in the background|"
                        r"throughout the entire video|"
                        r"throughout the video)\b",
                        "",
                        cand,
                    )
                    cand = re.sub(r"\b(the|a|an)\b", "", cand)
                    cand = re.sub(
                        r"\b(video|scene|frame|clip|shot)\b",
                        "",
                        cand,
                    )
                    cand = re.sub(r"\s+", "_", cand).strip("_")
                    if cand:
                        return cand
            return "object"

        if action in {"add_object", "increase_amount"}:
            for pat in [
                r"increase the number of ([a-z0-9'\- ]+)",
                r"increase the amount of ([a-z0-9'\- ]+)",
                r"add more ([a-z0-9'\- ]+)",
                r"add a second ([a-z0-9'\- ]+)",
                r"add ([a-z0-9'\- ]+) to",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip(" ,.")
                    cand = re.sub(r"\b(the|a|an)\b", "", cand)
                    cand = re.sub(r"\s+", "_", cand).strip("_")
                    if cand:
                        return cand

        if action in {
            "zoom_in",
            "zoom_out",
            "dolly_in",
            "dolly_out",
            "orbit_camera",
        }:
            m = re.search(
                r"(?:toward|towards|on|at) ([a-z0-9'\- ]+)",
                text,
            )
            if m:
                cand = m.group(1).strip(" ,.")
                if not re.search(
                    r"\b(entire video|scene|frame|clip)\b",
                    cand,
                ):
                    return cand
            return "camera_view"

        if action == "apply_style":
            return "full_frame"

        if action == "replace_background":
            return "background"

        if action == "change_camera_angle":
            for pat in [
                r"(?:looking at|look at|focused on|focusing on) "
                r"([a-z0-9'\- ]+)",
                r"(?:toward|towards|at) ([a-z0-9'\- ]+)",
            ]:
                m = re.search(pat, text)
                if m:
                    cand = m.group(1).strip(" ,.")
                    cand = re.split(
                        r"\b(to|while|throughout|from|with|for|that)\b",
                        cand,
                    )[0].strip(" ,.")
                    if re.search(
                        r"\b(low angle|high angle|angle|perspective|"
                        r"shot|view)\b",
                        cand,
                    ):
                        continue
                    if re.search(
                        r"\b(entire video|scene|frame|clip)\b",
                        cand,
                    ):
                        return "camera_view"
                    cand = re.sub(r"\b(the|a|an)\b", "", cand)
                    cand = re.sub(r"\s+", " ", cand).strip()
                    if cand:
                        return cand
            return "camera_view"

        for pat in [
            r"color of ([a-z0-9'\- ]+)",
            r"change ([a-z0-9'\- ]+) to",
            r"add [a-z0-9'\- ]+ to ([a-z0-9'\- ]+)",
            r"remove ([a-z0-9'\- ,and]+) from",
            r"replace ([a-z0-9'\- ]+) with",
        ]:
            m = re.search(pat, text)
            if m:
                candidate = m.group(1).strip(" ,.")
                candidate = re.sub(
                    r"\bthroughout the entire video\b",
                    "",
                    candidate,
                )
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if candidate:
                    return candidate

        if action == "add_effect":
            m = re.search(r"to ([a-z0-9'\- ]+)", text)
            if m:
                return m.group(1).strip(" ,.")

        return "object"


def build_parser() -> InstructionParserV3MixedTrial002:
    return InstructionParserV3MixedTrial002(LLMConfig())


def main() -> None:
    parser = build_parser()
    samples = [
        "Transform the entire scene into Cyberpunk style with neon lights.",
        "Change the shirt color to deep blue.",
        "Apply a smooth zoom-in effect on the speaker.",
    ]
    print(
        json.dumps(
            [parser.pred(s) for s in samples],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
