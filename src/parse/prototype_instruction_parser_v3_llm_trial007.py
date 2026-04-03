#!/usr/bin/env python3
"""LLM trial 007 parser (LLM-first, no singlefile import).

Design constraints:
- Do not import prototype_instruction_parser_v3_singlefile.py.
- Do not use per-video GT lookup.
- Predict from instruction text only.
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


class InstructionParserV3LLMTrial007:
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

        action = str(pred.get("action", "")).strip().lower()
        target = str(pred.get("target", "")).strip().lower()

        if action not in ALLOWED_ACTIONS:
            action = self._heuristic_action(instruction)

        target = re.sub(r"\s+", " ", target).strip()
        target = re.sub(r"[^a-z0-9 _\-\[\]',]", "", target)
        target = target[:96]

        if not target:
            target = self._heuristic_target(instruction, action)

        text = instruction.lower()
        if (
            action == "apply_style"
            and "style" not in text
            and "cyberpunk" not in text
        ):
            action = self._heuristic_action(instruction)
        if action == "replace_background" and "background" not in text:
            action = self._heuristic_action(instruction)

        return {"action": action, "target": target}

    def _heuristic_prediction(self, instruction: str) -> dict[str, str]:
        action = self._heuristic_action(instruction)
        target = self._heuristic_target(instruction, action)
        return {"action": action, "target": target}

    def _locked_action(self, instruction: str) -> str | None:
        text = instruction.lower()
        if re.search(r"\breplace\b.*\bbackground\b", text):
            return "replace_background"
        if re.search(r"\breplace\b.*\bwith\b", text):
            return "replace_object"
        if re.search(
            r"\bincrease\b.*\b(number|count|amount)\b|\badd more\b",
            text,
        ):
            return "add_object"
        if re.search(r"\bdolly in\b", text):
            return "dolly_in"
        if re.search(r"\bdolly out\b", text):
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
        if re.search(r"\bdolly in\b", text):
            return "dolly_in"
        if re.search(r"\bdolly out\b", text):
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
            m = re.search(r"(?:at|toward|on|to) ([a-z0-9'\- ]+)", text)
            if m:
                return m.group(1).strip()
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


def build_parser() -> InstructionParserV3LLMTrial007:
    return InstructionParserV3LLMTrial007(LLMConfig())


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
