#!/usr/bin/env python3
"""LLM trial 002 parser.

Strategy:
- Keep v3 single-file rule parser as base for stability.
- Run selective LLM refinement only for low-confidence instructions.
- If LLM output is invalid, keep rule result.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

WORKSPACE = Path("/workspace")
BASE_PARSER_PATH = (
    WORKSPACE
    / "src"
    / "parse"
    / "prototype_instruction_parser_v3_singlefile.py"
)


class LLMConfig:
    def __init__(self) -> None:
        self.enabled = True
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.max_new_tokens = 96
        self.temperature = 0.0
        self.batch_size = int(os.environ.get("LLM_BATCH_SIZE", "1"))


class InstructionParserV3LLMTrial002:
    def __init__(self, base_parser: Any, cfg: LLMConfig):
        self.base_parser = base_parser
        self.cfg = cfg
        self._tok = None
        self._model = None
        self._cache: dict[str, dict[str, str]] = {}
        self._llm_ready = False

    def pred(self, instruction: str) -> dict[str, Any]:
        base = self.base_parser.pred(instruction)
        base_task = base["tasks"][0]

        if not self._is_low_confidence(instruction, base_task):
            return base

        refined = self._refine_with_llm(instruction, base_task)
        if not refined:
            return base

        return {
            "tasks": [
                {
                    "action": refined.get(
                        "action",
                        base_task.get("action", "edit_motion"),
                    ),
                    "target": refined.get(
                        "target",
                        base_task.get("target", "object"),
                    ),
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

        bs = int(batch_size or self.cfg.batch_size or 1)
        base_preds = [self.base_parser.pred(inst) for inst in instructions]
        refined_out: list[dict[str, str] | None] = [None] * len(instructions)

        low_indices = []
        low_insts = []
        low_tasks = []
        for i, (inst, base_pred) in enumerate(zip(instructions, base_preds)):
            base_task = base_pred["tasks"][0]
            if self._is_low_confidence(inst, base_task):
                low_indices.append(i)
                low_insts.append(inst)
                low_tasks.append(base_task)

        if low_insts:
            refined_list = self._refine_many_with_llm(low_insts, low_tasks, bs)
            for idx, refined in zip(low_indices, refined_list):
                refined_out[idx] = refined

        out = []
        for i, base_pred in enumerate(base_preds):
            base_task = base_pred["tasks"][0]
            refined = refined_out[i]
            if not refined:
                out.append(base_pred)
                continue
            out.append(
                {
                    "tasks": [
                        {
                            "action": refined.get(
                                "action",
                                base_task.get("action", "edit_motion"),
                            ),
                            "target": refined.get(
                                "target",
                                base_task.get("target", "object"),
                            ),
                            "constraints": [],
                            "params": {},
                        }
                    ]
                }
            )
        return out

    def _is_low_confidence(
        self,
        instruction: str,
        task: dict[str, Any],
    ) -> bool:
        text = instruction.lower().strip()
        action = str(task.get("action", ""))
        target = str(task.get("target", ""))

        ambiguous_markers = [
            "transform",
            "while",
            "without changing",
            "instead of",
            "keep",
            "preserve",
            "as is",
        ]
        has_ambiguous_phrase = any(k in text for k in ambiguous_markers)

        generic_target = target in {
            "object",
            "camera_view",
            "person",
            "full_frame",
        }
        uncertain_action = action in {
            "edit_motion",
            "add_object",
            "apply_style",
        }

        return has_ambiguous_phrase and (generic_target or uncertain_action)

    def _ensure_llm(self) -> bool:
        if not self.cfg.enabled:
            return False
        if self._llm_ready:
            return True
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return False

        try:
            model_name = os.environ.get("LLM_MODEL_NAME", self.cfg.model_name)
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            self._llm_ready = True
            return True
        except Exception:
            return False

    def _refine_with_llm(
        self,
        instruction: str,
        base_task: dict[str, Any],
    ) -> dict[str, str] | None:
        if instruction in self._cache:
            return self._cache[instruction]

        if not self._ensure_llm():
            return None

        prompt = self._build_prompt(instruction, base_task)
        try:
            inputs = self._tok(prompt, return_tensors="pt")
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=False,
            )
            text = self._tok.decode(out[0], skip_special_tokens=True)
            parsed = self._extract_json(text)
            sanitized = self._sanitize_prediction(
                parsed,
                instruction,
                base_task,
            )
            if sanitized:
                self._cache[instruction] = sanitized
            return sanitized
        except Exception:
            return None

    def _refine_many_with_llm(
        self,
        instructions: list[str],
        base_tasks: list[dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, str] | None]:
        out: list[dict[str, str] | None] = [None] * len(instructions)
        if not instructions:
            return out

        # Use cache first to avoid repeated generations.
        pending_indices = []
        prompts = []
        pending_tasks = []
        for i, (inst, base_task) in enumerate(zip(instructions, base_tasks)):
            if inst in self._cache:
                out[i] = self._cache[inst]
                continue
            pending_indices.append(i)
            prompts.append(self._build_prompt(inst, base_task))
            pending_tasks.append(base_task)

        if not pending_indices:
            return out

        if not self._ensure_llm():
            return out

        bs = max(1, int(batch_size))
        for start in range(0, len(prompts), bs):
            prompt_chunk = prompts[start:start + bs]
            idx_chunk = pending_indices[start:start + bs]
            task_chunk = pending_tasks[start:start + bs]
            try:
                inputs = self._tok(
                    prompt_chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.max_new_tokens),
                    do_sample=False,
                )
                texts = self._tok.batch_decode(gen, skip_special_tokens=True)
                inst_chunk = [instructions[j] for j in idx_chunk]
                for text, out_i, inst_i, base_task_i in zip(
                    texts,
                    idx_chunk,
                    inst_chunk,
                    task_chunk,
                ):
                    parsed = self._extract_json(text)
                    sanitized = self._sanitize_prediction(
                        parsed,
                        inst_i,
                        base_task_i,
                    )
                    if sanitized:
                        self._cache[inst_i] = sanitized
                    out[out_i] = sanitized
            except Exception:
                continue

        return out

    def _build_prompt(
        self,
        instruction: str,
        base_task: dict[str, Any],
    ) -> str:
        actions = [
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
        ]
        return (
            "You convert one instruction into one task. Return JSON only.\\n"
            'Format: {"action": string, "target": string}\\n'
            f"Allowed actions: {actions}\\n"
            f"Instruction: {instruction}\\n"
            f"Rule guess action: {base_task.get('action', 'edit_motion')}\\n"
            f"Rule guess target: {base_task.get('target', 'object')}\\n"
            "Output JSON only."
        )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _sanitize_prediction(
        self,
        pred: dict[str, Any] | None,
        instruction: str,
        base_task: dict[str, Any],
    ) -> dict[str, str] | None:
        if not pred:
            return None

        action = str(pred.get("action", "")).strip().lower()
        target = str(pred.get("target", "")).strip().lower()

        allowed_actions = {
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

        if action not in allowed_actions:
            action = str(base_task.get("action", "edit_motion"))

        if not target or len(target) > 80:
            target = str(base_task.get("target", "object"))

        target = re.sub(r"\s+", "_", target)
        target = re.sub(r"[^a-z0-9_\-\[\]',]", "", target)
        target = target.strip("_")

        if not target:
            target = str(base_task.get("target", "object"))

        # Keep changes conservative and avoid unstable action flips.
        text = instruction.lower()
        if action != base_task.get("action"):
            if action == "change_color" and "color" not in text:
                action = str(base_task.get("action", "edit_motion"))
            if action == "replace_background" and "background" not in text:
                action = str(base_task.get("action", "edit_motion"))

        return {"action": action, "target": target}


def _load_base_parser() -> Any:
    spec = importlib.util.spec_from_file_location(
        "base_singlefile",
        BASE_PARSER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base parser: {BASE_PARSER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_parser()


def build_parser() -> InstructionParserV3LLMTrial002:
    base_parser = _load_base_parser()
    cfg = LLMConfig()
    return InstructionParserV3LLMTrial002(base_parser, cfg)


def main() -> None:
    parser = build_parser()
    samples = [
        (
            "Transform the scene into cyberpunk style while keeping "
            "the woman centered"
        ),
        "Replace the background with a beach at sunset",
        "Add more pandas near the tree",
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
