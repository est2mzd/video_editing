from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import re
import sys

import cv2
import numpy as np
from tqdm.auto import tqdm

sys.path.append("/workspace/src")

from postprocess.detectors import detect_primary_box, get_sam_mask_from_box
from utils.video_utility import load_video, write_video, show_before_after


@dataclass
class AddObjectInstruction:
    target_object: str


@dataclass
class AddObjectConfig:
    input_video: str
    output_video: str
    instruction: str
    target_prompt_override: str | None = None
    offset_x_ratio: float = 0.28
    offset_y_ratio: float = 0.0
    alpha: float = 0.95
    box_smooth_alpha: float = 0.65
    center_smooth_alpha: float = 0.70
    mask_ema_alpha: float = 0.75
    mask_threshold: float = 0.50
    min_mask_pixels: int = 48


def parse_add_object_instruction_rulebase(instruction: str) -> AddObjectInstruction | None:
    """Extract target noun phrase for add_object from instruction text."""
    text = re.sub(r"\s+", " ", instruction.strip().lower())

    m = re.search(
        r"(?:add|insert|place|put|duplicate|copy|clone|create)\s+"
        r"(?:another\s+|one\s+more\s+|a\s+|an\s+|the\s+)?"
        r"([a-z0-9][a-z0-9\-\s]{0,50}?)(?:\s+(?:next\s+to|beside|near|on|at|in|to)\b|[,.]|$)",
        text,
    )
    if m:
        obj = m.group(1).strip()
        obj = re.sub(r"^(?:new|extra|same)\s+", "", obj).strip()
        words = [w for w in obj.split() if w]
        if words:
            return AddObjectInstruction(target_object=" ".join(words[-3:]))

    m = re.search(r"(?:add|insert|place|put|duplicate|copy|clone|create)\s+(.+)$", text)
    if m:
        obj = m.group(1)
        obj = re.split(r"\b(?:next\s+to|beside|near|on|at|in|to|while|with|and)\b", obj, maxsplit=1)[0]
        obj = re.sub(r"^(?:another\s+|one\s+more\s+|a\s+|an\s+|the\s+)", "", obj).strip(" .,")
        words = [w for w in obj.split() if w]
        if words:
            return AddObjectInstruction(target_object=" ".join(words[-3:]))

    return None


def parse_add_object_instruction(instruction: str) -> AddObjectInstruction:
    parsed = parse_add_object_instruction_rulebase(instruction)
    if parsed is not None and parsed.target_object:
        return parsed
    return AddObjectInstruction(target_object="object")


def build_target_prompt(instruction: str, override: str | None = None) -> str:
    if override:
        return override
    parsed = parse_add_object_instruction(instruction)
    return f"{parsed.target_object} ."


def largest_connected_component(mask_u8: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    if int(mask_u8.sum()) == 0:
        return mask_u8
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 2:
        return mask_u8
    k = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == k).astype(np.uint8)


def clean_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Simple morphology cleanup for SAM noise."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return largest_connected_component((out > 0).astype(np.uint8))


def smooth_box(
    prev_box: tuple[float, float, float, float] | None,
    curr_box: tuple[float, float, float, float] | None,
    alpha: float,
) -> tuple[float, float, float, float] | None:
    """EMA smoothing for detected boxes to reduce jitter."""
    if curr_box is None:
        return prev_box
    if prev_box is None:
        return curr_box
    a = float(np.clip(alpha, 0.0, 1.0))
    return tuple(a * p + (1.0 - a) * c for p, c in zip(prev_box, curr_box))


def smooth_center(prev_center: tuple[int, int] | None, curr_center: tuple[int, int], alpha: float) -> tuple[int, int]:
    """EMA smoothing for paste center coordinates."""
    if prev_center is None:
        return curr_center
    a = float(np.clip(alpha, 0.0, 1.0))
    x = int(a * prev_center[0] + (1.0 - a) * curr_center[0])
    y = int(a * prev_center[1] + (1.0 - a) * curr_center[1])
    return (x, y)


def temporal_smooth_mask(
    curr_mask_u8: np.ndarray,
    prev_soft_mask: np.ndarray | None,
    ema_alpha: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal EMA on mask logits to suppress frame-to-frame flicker."""
    curr = (curr_mask_u8 > 0).astype(np.float32)
    if prev_soft_mask is None:
        soft = curr
    else:
        a = float(np.clip(ema_alpha, 0.0, 1.0))
        soft = a * prev_soft_mask + (1.0 - a) * curr
    stable = (soft >= float(threshold)).astype(np.uint8)
    return stable, soft


def extract_object_patch(
    frame_bgr: np.ndarray,
    box: tuple[float, float, float, float],
    mask_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Extract object RGB patch + alpha patch + tight bbox from frame and mask."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 2, x1))
    y1 = max(0, min(h - 2, y1))
    x2 = max(x1 + 1, min(w - 1, x2))
    y2 = max(y1 + 1, min(h - 1, y2))

    crop = frame_bgr[y1:y2, x1:x2].copy()
    m = mask_u8[y1:y2, x1:x2].copy()
    m = largest_connected_component(m)

    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        alpha = np.ones(crop.shape[:2], dtype=np.float32)
        return crop, alpha, (x1, y1, x2, y2)

    tx1, ty1, tx2, ty2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    obj = crop[ty1:ty2, tx1:tx2].copy()
    alpha = (m[ty1:ty2, tx1:tx2] > 0).astype(np.float32)

    return obj, alpha, (x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2)


def alpha_paste(
    dst_bgr: np.ndarray,
    src_bgr: np.ndarray,
    src_alpha: np.ndarray,
    center_xy: tuple[int, int],
    alpha_gain: float,
) -> np.ndarray:
    """Alpha-blend source patch into destination image at center_xy."""
    out = dst_bgr.copy()
    h, w = out.shape[:2]
    sh, sw = src_bgr.shape[:2]

    cx, cy = center_xy
    x1 = int(cx - sw // 2)
    y1 = int(cy - sh // 2)
    x2 = x1 + sw
    y2 = y1 + sh

    rx1 = max(0, x1)
    ry1 = max(0, y1)
    rx2 = min(w, x2)
    ry2 = min(h, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return out

    sx1 = rx1 - x1
    sy1 = ry1 - y1
    sx2 = sx1 + (rx2 - rx1)
    sy2 = sy1 + (ry2 - ry1)

    src_roi = src_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    a_roi = np.clip(src_alpha[sy1:sy2, sx1:sx2].astype(np.float32) * float(alpha_gain), 0.0, 1.0)
    dst_roi = out[ry1:ry2, rx1:rx2].astype(np.float32)

    blended = (1.0 - a_roi[..., None]) * dst_roi + a_roi[..., None] * src_roi
    out[ry1:ry2, rx1:rx2] = blended.astype(np.uint8)
    return out


def run_add_object_gdino_sam_cv2(frames: list[np.ndarray], cfg: AddObjectConfig, logger: logging.Logger) -> list[np.ndarray]:
    """Run add_object by duplicating detected object per frame using DINO+SAM+OpenCV."""
    out_frames: list[np.ndarray] = []

    target_prompt = build_target_prompt(cfg.instruction, cfg.target_prompt_override)
    logger.info("target_prompt: %s", target_prompt)

    prev_box: tuple[float, float, float, float] | None = None
    prev_center: tuple[int, int] | None = None
    prev_soft_mask: np.ndarray | None = None
    prev_obj_patch: np.ndarray | None = None
    prev_alpha_patch: np.ndarray | None = None

    for frame in tqdm(frames, desc="add_object_gdino_sam_cv2"):
        det_box = detect_primary_box(frame, text_prompt=target_prompt, logger=logger)
        box = smooth_box(prev_box, det_box, alpha=cfg.box_smooth_alpha)

        if box is None:
            out_frames.append(frame.copy())
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_mask = get_sam_mask_from_box(frame_rgb, [box[0], box[1], box[2], box[3]], logger=logger).astype(np.uint8)
        raw_mask = clean_mask(raw_mask)

        stable_mask, prev_soft_mask = temporal_smooth_mask(
            raw_mask,
            prev_soft_mask,
            ema_alpha=cfg.mask_ema_alpha,
            threshold=cfg.mask_threshold,
        )

        use_prev_patch = int(stable_mask.sum()) < int(cfg.min_mask_pixels)
        if use_prev_patch and prev_obj_patch is not None and prev_alpha_patch is not None:
            obj_patch = prev_obj_patch
            alpha_patch = prev_alpha_patch
            x1, y1, x2, y2 = [int(v) for v in box]
            tight_box = (x1, y1, x2, y2)
        else:
            obj_patch, alpha_patch, tight_box = extract_object_patch(frame, box, stable_mask)
            prev_obj_patch = obj_patch
            prev_alpha_patch = alpha_patch

        x1, y1, x2, y2 = tight_box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        raw_center = (
            int((x1 + x2) / 2.0 + cfg.offset_x_ratio * bw),
            int((y1 + y2) / 2.0 + cfg.offset_y_ratio * bh),
        )
        center = smooth_center(prev_center, raw_center, alpha=cfg.center_smooth_alpha)

        out = alpha_paste(frame, obj_patch, alpha_patch, center, alpha_gain=cfg.alpha)
        out_frames.append(out)

        prev_box = box
        prev_center = center

    return out_frames


def run_add_object_video(cfg: AddObjectConfig, logger: logging.Logger | None = None) -> dict:
    """Convenience wrapper: load video, run pipeline, and write output."""
    if logger is None:
        logger = logging.getLogger("add_object")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(h)

    parsed = parse_add_object_instruction(cfg.instruction)
    target_prompt = build_target_prompt(cfg.instruction, cfg.target_prompt_override)

    frames, fps, width, height = load_video(cfg.input_video)
    out_frames = run_add_object_gdino_sam_cv2(frames, cfg, logger)
    Path(cfg.output_video).parent.mkdir(parents=True, exist_ok=True)
    write_video(cfg.output_video, out_frames, fps, width, height)

    return {
        "input_video": cfg.input_video,
        "output_video": cfg.output_video,
        "instruction": cfg.instruction,
        "target_object": parsed.target_object,
        "target_prompt": target_prompt,
        "frames": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add object with GroundingDINO + SAM + OpenCV")
    p.add_argument("--input-video", required=True)
    p.add_argument("--output-video", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--target-prompt-override", default=None)
    p.add_argument("--offset-x-ratio", type=float, default=0.28)
    p.add_argument("--offset-y-ratio", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--box-smooth-alpha", type=float, default=0.65)
    p.add_argument("--center-smooth-alpha", type=float, default=0.70)
    p.add_argument("--mask-ema-alpha", type=float, default=0.75)
    p.add_argument("--mask-threshold", type=float, default=0.50)
    p.add_argument("--min-mask-pixels", type=int, default=48)
    p.add_argument("--show", action="store_true", help="Show before/after preview")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = AddObjectConfig(
        input_video=args.input_video,
        output_video=args.output_video,
        instruction=args.instruction,
        target_prompt_override=args.target_prompt_override,
        offset_x_ratio=args.offset_x_ratio,
        offset_y_ratio=args.offset_y_ratio,
        alpha=args.alpha,
        box_smooth_alpha=args.box_smooth_alpha,
        center_smooth_alpha=args.center_smooth_alpha,
        mask_ema_alpha=args.mask_ema_alpha,
        mask_threshold=args.mask_threshold,
        min_mask_pixels=args.min_mask_pixels,
    )
    result = run_add_object_video(cfg)
    for k, v in result.items():
        print(f"{k}: {v}")
    if args.show:
        show_before_after(cfg.input_video, cfg.output_video, width=480)


if __name__ == "__main__":
    main()
