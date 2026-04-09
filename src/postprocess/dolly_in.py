import logging
import json
import os
import re
import sys
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

if "/workspace/src" not in sys.path:
    sys.path.append("/workspace/src")
if "/workspace/src/utils" not in sys.path:
    sys.path.append("/workspace/src/utils")

from postprocess.camera_ops import compose_scaled_mask_foreground
from postprocess.detectors import detect_all_boxes, detect_primary_box, get_sam_mask_from_box
#from utils.video_utility import load_video, write_video, show_before_after


class LLMConfig:
    def __init__(self) -> None:
        """Initialize LLM runtime settings from environment variables."""
        self.model_name = os.environ.get(
            "LLM_MODEL_NAME",
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.max_new_tokens = int(os.environ.get("LLM_MAX_NEW_TOKENS", "48"))


class DollyInTargetExtractor:
    """LLM-first extractor for dolly/zoom target prompt."""

    ADJECTIVES = {
        "smooth", "detailed", "subtle", "steady", "slow", "gradual",
        "bright", "dark", "vibrant", "metallic", "modern", "classic",
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple",
        "orange", "silver", "gold", "golden", "gray", "grey",
        "small", "large", "big", "tiny", "huge", "deep", "light",
    }

    HUMAN_WORDS = {
        "person", "people", "man", "woman", "guy", "girl", "speaker", "human", "face", "head"
    }

    PHRASE_MAP = {
        "manual coffee grinder": "hand coffee grinder",
        "coffee mill": "coffee grinder",
        "mobile phone": "phone",
        "cell phone": "phone",
        "dining table": "table",
    }

    TOKEN_MAP = {
        "manual": "hand",
        "cellphone": "phone",
        "automobile": "car",
        "vehicle": "car",
        "bicycle": "bike",
        "motorbike": "motorcycle",
        "television": "tv",
        "couch": "sofa",
    }

    def __init__(self, cfg: LLMConfig | None = None):
        """Create extractor state, model cache, and prompt cache."""
        self.cfg = cfg or LLMConfig()
        self._tok = None
        self._model = None
        self._ready = False
        self._cache: dict[str, str] = {}

    def _ensure_llm(self) -> bool:
        """Lazily load tokenizer/model and return whether inference is available."""
        if self._ready:
            return True
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return False
        try:
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
            self._ready = True
            return True
        except Exception:
            return False

    def _build_prompt(self, instruction: str) -> str:
        """Build instruction-to-target prompt for the causal LLM."""
        lines = [
            "You are an instruction parser for camera motion.",
            "Task: find the target of dolly_in or zoom_in.",
            "Output JSON only: {\"is_person\": true/false, \"target\": \"core noun\"}",
            "If target is a person/human/face or a person's name, set is_person=true and target=person.",
            "If target is not person, return only the object noun phrase.",
            "Use GroundingDINO-friendly names when possible (example: manual coffee grinder -> hand coffee grinder).",
            f"Instruction: {instruction}",
            "JSON:",
        ]
        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from raw LLM output text."""
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _apply_term_map(self, text: str) -> str:
        """Apply phrase and token replacement dictionaries for DINO-friendly terms."""
        t = re.sub(r"\s+", " ", str(text).strip().lower())
        if not t:
            return ""

        for src, dst in sorted(self.PHRASE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            t = re.sub(rf"\b{re.escape(src)}\b", dst, t)

        words = t.split(" ")
        words = [self.TOKEN_MAP.get(w, w) for w in words]
        return re.sub(r"\s+", " ", " ".join(words)).strip()

    def _noun_core(self, text: str) -> str:
        """Reduce a phrase to its core noun phrase after mapping and stopword drop."""
        t = self._apply_term_map(text)
        words = t.split(" ")
        out: list[str] = []
        for w in words:
            if w in {"the", "a", "an"}:
                continue
            if w in self.ADJECTIVES:
                continue
            out.append(w)
        if not out:
            out = words
        return " ".join(out).strip()

    def _looks_like_person(self, instruction: str, target: str) -> bool:
        """Heuristically detect whether target refers to a human/person entity."""
        inst_l = instruction.lower()
        tgt_l = str(target).lower().strip()

        if any(w in tgt_l.split() for w in self.HUMAN_WORDS):
            return True
        if re.search(r"\b(person|people|man|woman|speaker|human|face|head)\b", inst_l):
            return True
        if re.search(r"(?:toward|towards|on|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", instruction):
            return True
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", str(target).strip()):
            return True

        return False

    def _contextual_override(self, instruction: str, base_target: str) -> str:
        """Promote mapped multi-word terms when the instruction explicitly contains them."""
        inst_l = instruction.lower()
        base = self._apply_term_map(base_target)

        for src, dst in self.PHRASE_MAP.items():
            src_core = self._noun_core(src)
            if src in inst_l and base in {src, src_core, self._noun_core(dst), dst}:
                return dst

        return base

    def _normalize_target(self, instruction: str, raw_target: str) -> str:
        """Normalize raw target text into a short, detection-friendly noun phrase."""
        cleaned = re.sub(r"[^A-Za-z0-9\s\-']", " ", str(raw_target)).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return "object"

        base = self._noun_core(cleaned)
        base = self._contextual_override(instruction, base)
        return base if base else "object"

    def _heuristic_fallback(self, instruction: str) -> str:
        """Fallback parser when LLM fails, using regex target extraction rules."""
        text = instruction
        text_l = text.lower()

        if re.search(r"\b(person|people|man|woman|guy|girl|speaker|human|face|head)\b", text_l):
            return "person . "
        if re.search(r"(?:toward|towards|on|at)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text):
            return "person . "

        m = re.search(
            r"(?:dolly[\s_-]*in|zoom[\s_-]*in|move camera).*?"
            r"(?:toward|towards|on|at)\s+(?:the\s+)?([A-Za-z][A-Za-z\s\-']{1,100})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            chunk = m.group(1)
            chunk = re.split(
                r"\b(to|while|with|during|throughout|and|that|which|for)\b",
                chunk,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            return f"{self._normalize_target(instruction, chunk)} . "

        return "object . "

    def _finalize_prompt(self, instruction: str, prompt: str) -> str:
        """Finalize output format as '<target> . ' with final normalization pass."""
        p = str(prompt).strip()
        if not p:
            return "object . "

        if p.lower() == "person .":
            return "person . "

        core = p.rstrip(" .")
        core = self._normalize_target(instruction, core)
        return f"{core} . "

    def extract_target(self, instruction: str) -> str:
        """Infer dolly/zoom target from instruction text and return DINO prompt string."""
        if instruction in self._cache:
            return self._cache[instruction]

        result: str | None = None
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
                if parsed:
                    is_person = bool(parsed.get("is_person", False))
                    target = str(parsed.get("target", "")).strip()
                    if is_person or self._looks_like_person(instruction, target):
                        result = "person . "
                    elif target and target.lower() not in {"", "null", "none"}:
                        result = f"{self._normalize_target(instruction, target)} . "
            except Exception:
                result = None

        if result is None:
            result = self._heuristic_fallback(instruction)

        result = self._finalize_prompt(instruction, result)
        self._cache[instruction] = result
        return result


_EXTRACTOR = DollyInTargetExtractor(LLMConfig())


def extract_dolly_in_target(instruction: str) -> str:
    """Extract GroundingDINO text prompt for dolly-in from instruction."""
    return _EXTRACTOR.extract_target(instruction)


def normalize_dolly_target_prompt(prompt: str) -> str:
    """Normalize manual target prompt for GroundingDINO usage."""
    p = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if not p:
        p = "object"

    p = p.replace("man's face", "man face")
    p = p.replace("woman's face", "woman face")
    p = p.replace("mans face", "man face")
    p = p.replace("womans face", "woman face")

    if p.endswith("."):
        p = p[:-1].strip()
    if not p:
        p = "object"
    return f"{p} . "


def run_dolly_in_with_instruction(
    frames: list[np.ndarray],
    instruction: str,
    object_end_scale: float,
    logger: logging.Logger,
    target_prompt_override: str | None = None,
) -> list[np.ndarray]:
    """Run dolly-in pipeline using module-level target extraction."""
    if target_prompt_override and str(target_prompt_override).strip():
        target_prompt = normalize_dolly_target_prompt(target_prompt_override)
    else:
        target_prompt = normalize_dolly_target_prompt(
            extract_dolly_in_target(instruction)
        )

    return stable_object_zoom_v2(
        frames,
        target_prompt=target_prompt,
        object_end_scale=object_end_scale,
        logger=logger,
    )


def _box_area(b: tuple[float, float, float, float]) -> float:
    """Return area of an xyxy box in pixel units."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two xyxy boxes for temporal box matching."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _select_best_box(
    frame_bgr: np.ndarray,
    target_prompt: str,
    prev_box: tuple[float, float, float, float] | None,
    logger: logging.Logger,
) -> tuple[float, float, float, float] | None:
    """Pick the most plausible target box using size prior + temporal IoU."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(frame_rgb, text_prompt=target_prompt, logger=logger)

    if not boxes:
        return detect_primary_box(frame_bgr, text_prompt=target_prompt, logger=logger)

    best_box = None
    best_score = -1e9
    for b in boxes:
        area_ratio = _box_area(b) / float(max(1, h * w))

        score = 0.0
        if 0.001 <= area_ratio <= 0.20:
            score += 2.0
        elif area_ratio > 0.35:
            score -= 3.0

        score -= area_ratio * 3.0

        if prev_box is not None:
            score += _box_iou(prev_box, b) * 4.0

        if score > best_score:
            best_score = score
            best_box = b

    return best_box


def _refine_mask(mask: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Keep the connected component nearest the selected box center."""
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 2:
        return mask_u8

    cx = int((box[0] + box[2]) / 2.0)
    cy = int((box[1] + box[3]) / 2.0)
    cx = int(np.clip(cx, 0, mask_u8.shape[1] - 1))
    cy = int(np.clip(cy, 0, mask_u8.shape[0] - 1))

    center_label = labels[cy, cx]
    if center_label > 0:
        return (labels == center_label).astype(np.uint8)

    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(comp_areas)) + 1
    return (labels == k).astype(np.uint8)


def stable_object_zoom_v2(
    frames: list[np.ndarray],
    target_prompt: str,
    object_end_scale: float,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Apply object-only dolly-in using per-frame DINO+SAM masks and compositing.

    Steps:
    1. Build linear scale schedule from 1.0 to object_end_scale.
    2. Detect/select a stable target box per frame.
    3. Generate and refine SAM mask from the box.
    4. Reject overly large masks, fallback to previous valid mask.
    5. Scale only foreground via compose_scaled_mask_foreground.
    """
    if not frames:
        return frames

    object_end_scale = float(np.clip(object_end_scale, 1.0, 3.0))
    scales = np.linspace(1.0, object_end_scale, len(frames))
    out: list[np.ndarray] = []

    prev_mask: np.ndarray | None = None
    prev_box: tuple[float, float, float, float] | None = None

    h, w = frames[0].shape[:2]
    frame_area = float(max(1, h * w))

    for i, frame in enumerate(tqdm(frames, desc="stable_object_zoom_in")):
        box = _select_best_box(frame, target_prompt, prev_box, logger)
        curr_mask: np.ndarray | None = None

        if box is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_mask = get_sam_mask_from_box(
                frame_rgb,
                [box[0], box[1], box[2], box[3]],
                logger=logger,
            ).astype(np.uint8)

            curr_mask = _refine_mask(raw_mask, box)
            area_ratio = float((curr_mask > 0).sum()) / frame_area

            if area_ratio > 0.25:
                curr_mask = None

        if curr_mask is None:
            if prev_mask is None:
                out.append(frame.copy())
                continue
            curr_mask = prev_mask
        else:
            prev_mask = curr_mask
            prev_box = box

        out.append(compose_scaled_mask_foreground(frame, curr_mask, float(scales[i])))

    return out
