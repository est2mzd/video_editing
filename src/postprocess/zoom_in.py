from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import sys

import cv2
import numpy as np
from tqdm.auto import tqdm

if "/workspace/src" not in sys.path:
    sys.path.append("/workspace/src")
if "/workspace/src/utils" not in sys.path:
    sys.path.append("/workspace/src/utils")

from postprocess.detectors import (
    detect_all_boxes,
    detect_primary_box,
    get_sam_mask_from_box,
)


@dataclass
class ZoomInstruction:
    target_object: str


@dataclass
class ZoomInConfig:
    instruction: str
    target_prompt_override: str | None = None
    end_scale: float = 1.85
    center_smooth_alpha: float = 0.80
    box_smooth_alpha: float = 0.70
    safe_margin_ratio: float = 0.15


def parse_zoom_instruction_rulebase(instruction: str) -> ZoomInstruction:
    """Extract zoom target noun phrase from instruction."""
    t = re.sub(r"\s+", " ", instruction.strip().lower())

    m = re.search(
        r"(?:focus(?:ing)?\s+(?:closer\s+)?on|zoom\s+in\s+on)\s+"
        r"(?:the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9\-\s]{0,60}?)"
        r"(?:[,.]|\s+while\b|\s+and\b|$)",
        t,
    )
    if m:
        obj = m.group(1).strip()
        words = [w for w in obj.split() if w]
        if words:
            return ZoomInstruction(target_object=" ".join(words[-4:]))

    for kw in [
        "man's face",
        "woman's face",
        "mans face",
        "womans face",
        "face",
        "man",
        "woman",
        "person",
        "subject",
    ]:
        if kw in t:
            return ZoomInstruction(target_object=kw)

    return ZoomInstruction(target_object="person")


def normalize_zoom_target_prompt(prompt: str) -> str:
    """Normalize target prompt into GroundingDINO-friendly format."""
    obj = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if not obj:
        obj = "person"

    obj = obj.replace("man's face", "man face")
    obj = obj.replace("woman's face", "woman face")
    obj = obj.replace("mans face", "man face")
    obj = obj.replace("womans face", "woman face")

    if obj.endswith("."):
        obj = obj[:-1].strip()
    if not obj:
        obj = "person"
    return f"{obj} ."


def build_target_prompt(instruction: str, override: str | None = None) -> str:
    if override:
        return normalize_zoom_target_prompt(override)
    parsed = parse_zoom_instruction_rulebase(instruction)
    return normalize_zoom_target_prompt(parsed.target_object)


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _select_best_box(
    frame_bgr: np.ndarray,
    target_prompt: str,
    prev_box: tuple[float, float, float, float] | None,
    logger: logging.Logger,
) -> tuple[float, float, float, float] | None:
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(
        frame_rgb,
        text_prompt=target_prompt,
        logger=logger,
    )

    if not boxes:
        return detect_primary_box(
            frame_bgr,
            text_prompt=target_prompt,
            logger=logger,
        )

    best_box = None
    best_score = -1e9
    for b in boxes:
        area_ratio = _box_area(b) / float(max(1, h * w))
        score = 0.0

        if 0.0008 <= area_ratio <= 0.25:
            score += 2.0
        elif area_ratio > 0.40:
            score -= 3.5

        score -= area_ratio * 3.0

        if prev_box is not None:
            score += _box_iou(prev_box, b) * 4.0

        if score > best_score:
            best_score = score
            best_box = b

    return best_box


def _refine_mask(
    mask: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
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


def _smooth_box(
    prev_box: tuple[float, float, float, float] | None,
    curr_box: tuple[float, float, float, float] | None,
    alpha: float,
) -> tuple[float, float, float, float] | None:
    if curr_box is None:
        return prev_box
    if prev_box is None:
        return curr_box

    a = float(np.clip(alpha, 0.0, 1.0))
    return tuple(a * p + (1.0 - a) * c for p, c in zip(prev_box, curr_box))


def _smooth_center(
    prev_center: tuple[float, float] | None,
    curr_center: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    if prev_center is None:
        return curr_center

    a = float(np.clip(alpha, 0.0, 1.0))
    x = a * prev_center[0] + (1.0 - a) * curr_center[0]
    y = a * prev_center[1] + (1.0 - a) * curr_center[1]
    return (x, y)


def _max_safe_scale(
    w: int,
    h: int,
    box: tuple[float, float, float, float],
    margin_ratio: float,
) -> float:
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    pad = 1.0 + float(max(0.0, margin_ratio))
    sx = w / (bw * pad)
    sy = h / (bh * pad)
    return max(1.0, min(sx, sy))


def _crop_resize_centered(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    scale: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    s = float(max(1.0, scale))

    crop_w = max(2, int(round(w / s)))
    crop_h = max(2, int(round(h / s)))

    cx, cy = center_xy
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))

    x1 = int(np.clip(x1, 0, max(0, w - crop_w)))
    y1 = int(np.clip(y1, 0, max(0, h - crop_h)))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def run_stable_zoom_in_gdino_sam_cv2(
    frames: list[np.ndarray],
    cfg: ZoomInConfig,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    target_prompt = build_target_prompt(
        cfg.instruction,
        cfg.target_prompt_override,
    )
    logger.info("target_prompt: %s", target_prompt)

    h, w = frames[0].shape[:2]
    out: list[np.ndarray] = []
    scales = np.linspace(
        1.0,
        float(np.clip(cfg.end_scale, 1.0, 3.0)),
        len(frames),
    )

    prev_box = None
    prev_center = None

    for i, frame in enumerate(
        tqdm(frames, desc="stable_zoom_in_gdino_sam_cv2")
    ):
        det_box = _select_best_box(frame, target_prompt, prev_box, logger)
        box = _smooth_box(prev_box, det_box, cfg.box_smooth_alpha)

        if box is None:
            if prev_center is None:
                out.append(
                    _crop_resize_centered(
                        frame,
                        (w / 2.0, h / 2.0),
                        float(scales[i]),
                    )
                )
            else:
                out.append(
                    _crop_resize_centered(
                        frame,
                        prev_center,
                        float(scales[i]),
                    )
                )
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_mask = get_sam_mask_from_box(
            frame_rgb,
            [box[0], box[1], box[2], box[3]],
            logger=logger,
        ).astype(np.uint8)
        mask = _refine_mask(raw_mask, box)

        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx = float(np.median(xs))
            cy = float(np.median(ys))
        else:
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)

        center = _smooth_center(prev_center, (cx, cy), cfg.center_smooth_alpha)

        desired_scale = float(scales[i])
        safe_scale = _max_safe_scale(w, h, box, cfg.safe_margin_ratio)
        applied_scale = min(desired_scale, safe_scale)

        out_frame = _crop_resize_centered(frame, center, applied_scale)
        out.append(out_frame)

        prev_box = box
        prev_center = center

    return out


def run_zoom_in_with_instruction(
    frames: list[np.ndarray],
    instruction: str,
    logger: logging.Logger,
    target_prompt_override: str | None = None,
    end_scale: float = 1.85,
    center_smooth_alpha: float = 0.80,
    box_smooth_alpha: float = 0.70,
    safe_margin_ratio: float = 0.15,
) -> list[np.ndarray]:
    cfg = ZoomInConfig(
        instruction=instruction,
        target_prompt_override=target_prompt_override,
        end_scale=end_scale,
        center_smooth_alpha=center_smooth_alpha,
        box_smooth_alpha=box_smooth_alpha,
        safe_margin_ratio=safe_margin_ratio,
    )
    return run_stable_zoom_in_gdino_sam_cv2(frames, cfg, logger)


def zoom_in(
    frames: list[np.ndarray],
    w: int,
    h: int,
    zoom_target: str = "face . person .",
    zoom_factor: float = 1.0,
) -> list[np.ndarray]:
    """Backward-compatible wrapper for old call sites.

    Uses the notebook-derived stable implementation and treats zoom_target as
    explicit override. w/h are unused because frame size comes from frames.
    """
    del w, h
    end_scale = float(np.clip(max(1.0, zoom_factor), 1.0, 3.0))
    logger = logging.getLogger("zoom_in")
    return run_zoom_in_with_instruction(
        frames,
        instruction="",
        logger=logger,
        target_prompt_override=zoom_target,
        end_scale=end_scale,
    )
