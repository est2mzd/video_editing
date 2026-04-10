from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import logging
import re
import sys
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm


sys.path.append("/workspace/src")


from postprocess.detectors import (  # noqa: E402
    detect_all_boxes,
    detect_primary_box,
    get_sam_mask_from_box,
)
from utils.video_utility import load_video, show_before_after, write_video  # noqa: E402


@dataclass
class AddEffectInstruction:
    target_object: str
    grounding_target: str
    how: str
    color: str | None = None


@dataclass
class AddEffectConfig:
    input_video: str
    output_video: str
    instruction: str
    target_prompt_override: str | None = None
    effect_override: str | None = None
    color_override: str | None = None
    intensity: float = 0.8
    glow_blur_ksize: int = 31
    brightness_alpha: float = 1.08
    brightness_beta: float = 10.0
    blur_ksize: int = 9
    background_dim: float = 0.92


def parse_add_effect_instruction(
    instruction: str,
) -> AddEffectInstruction:
    """Extract target object, cv2-friendly effect type, and color."""
    text = re.sub(r"\s+", " ", instruction.strip().lower())

    if "stage lighting" in text or "lighting decoration" in text:
        return AddEffectInstruction(
            target_object="stage lighting region",
            grounding_target="upper scene",
            how="stage_lighting",
            color="warm",
        )

    target = "object"
    if "his body" in text or "outlines his body" in text:
        target = "person body"
    elif "strings of the bass guitar" in text:
        target = "strings of the bass guitar"
    elif "bass guitar" in text:
        target = "bass guitar"
    elif "basketball player" in text:
        target = "basketball player"
    else:
        match = re.search(
            r"(?:apply|add|enhance)\s+(?:a\s+|an\s+|the\s+)?(.+?)"
            r"\s+(?:to|on|around)\s+(?:the\s+)?(.+?)(?:[,.]|\s+that\b|$)",
            text,
        )
        if match:
            target = match.group(2).strip()
        if "basketball player" in target:
            target = "basketball player"

    how = "glow"
    if "blur" in text or "blurry" in text:
        how = "blur"
    elif (
        "bright" in text
        or "brightness" in text
        or "illuminate" in text
    ):
        how = "brightness"
    elif (
        "glow" in text
        or "aura" in text
        or "flames" in text
        or "electric" in text
    ):
        how = "glow"

    color = None
    match = re.search(
        r"(red|orange|yellow|green|blue|purple|violet|pink|gold|silver|"
        r"beige|warm|cool|mint)\b",
        text,
    )
    if match:
        color = match.group(1)

    grounding_target = build_grounding_target(target)
    return AddEffectInstruction(
        target_object=target,
        grounding_target=grounding_target,
        how=how,
        color=color,
    )


def build_grounding_target(target: str) -> str:
    """Build a concise detector-friendly noun phrase."""
    text = target.lower().strip()
    if "strings of the bass guitar" in text:
        return "bass guitar strings"
    known = [
        "bass guitar strings",
        "bass guitar",
        "guitar strings",
        "basketball player",
        "player",
        "person body",
        "body",
        "upper scene",
        "stage lighting region",
    ]
    for phrase in known:
        if phrase in text:
            return phrase

    text = re.sub(r"\b(the|a|an|his|her|their)\b", " ", text)
    text = re.sub(
        r"\b(that|which|who|throughout|entire|video|sequence)\b.*$",
        "",
        text,
    ).strip()
    tokens = re.findall(r"[a-z]+", text)
    if len(tokens) >= 2:
        return " ".join(tokens[-2:])
    if len(tokens) == 1:
        return tokens[0]
    return "object"


def normalize_dino_target_phrase(
    base_target: str,
    instruction: str,
) -> str:
    """Use a more specific sports-person prompt in crowded scenes."""
    base = (base_target or "").lower().strip()
    text = (instruction or "").lower().strip()
    if "basketball player" in base or "basketball player" in text:
        return "basketball player"
    return base_target


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


def select_best_box(
    frame_bgr: np.ndarray,
    target_prompt: str,
    prev_box: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(frame_rgb, text_prompt=target_prompt)
    if not boxes:
        return detect_primary_box(frame_bgr, text_prompt=target_prompt)

    height, width = frame_bgr.shape[:2]
    best_box = None
    best_score = -1e9
    for box in boxes:
        area_ratio = _box_area(box) / float(max(1, height * width))
        score = -area_ratio * 2.0
        if 0.001 <= area_ratio <= 0.25:
            score += 2.0
        if prev_box is not None:
            score += 4.0 * _box_iou(prev_box, box)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box


def refine_mask(
    mask: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return mask_u8

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    if n_labels <= 2:
        return mask_u8

    cx = int(np.clip((box[0] + box[2]) / 2.0, 0, mask_u8.shape[1] - 1))
    cy = int(np.clip((box[1] + box[3]) / 2.0, 0, mask_u8.shape[0] - 1))
    label = labels[cy, cx]
    if label > 0:
        return (labels == label).astype(np.uint8)

    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == idx).astype(np.uint8)


def color_to_bgr(name: str | None) -> tuple[int, int, int]:
    lut = {
        "red": (40, 40, 255),
        "orange": (0, 140, 255),
        "yellow": (0, 220, 255),
        "green": (60, 220, 60),
        "blue": (255, 80, 40),
        "purple": (180, 60, 200),
        "violet": (220, 80, 180),
        "pink": (180, 120, 255),
        "gold": (40, 200, 255),
        "silver": (190, 190, 190),
        "beige": (180, 210, 230),
        "warm": (70, 180, 255),
        "cool": (255, 180, 80),
        "mint": (170, 255, 210),
    }
    return lut.get((name or "").lower(), (0, 180, 255))


def apply_brightness_to_mask(
    frame: np.ndarray,
    mask_u8: np.ndarray,
    alpha: float = 1.08,
    beta: float = 10.0,
) -> np.ndarray:
    out = frame.copy()
    boosted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    mask3 = np.repeat((mask_u8 > 0)[..., None], 3, axis=2)
    out[mask3] = boosted[mask3]
    return out


def apply_blur_to_mask(
    frame: np.ndarray,
    mask_u8: np.ndarray,
    ksize: int = 9,
) -> np.ndarray:
    out = frame.copy()
    kernel = max(3, int(ksize) | 1)
    blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    mask3 = np.repeat((mask_u8 > 0)[..., None], 3, axis=2)
    out[mask3] = blurred[mask3]
    return out


def apply_glow_to_mask(
    frame: np.ndarray,
    mask_u8: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float = 0.8,
    blur_ksize: int = 31,
    background_dim: float = 1.0,
) -> np.ndarray:
    mask_f = (mask_u8 > 0).astype(np.float32)
    kernel = max(3, int(blur_ksize) | 1)
    glow = cv2.GaussianBlur(mask_f, (kernel, kernel), 0)
    glow = glow / max(1e-6, float(glow.max()))

    out = frame.astype(np.float32).copy()
    out *= float(background_dim)
    glow_rgb = np.zeros_like(out, dtype=np.float32)
    glow_rgb[:, :] = np.array(color_bgr, dtype=np.float32)
    out = np.clip(
        out + glow[..., None] * glow_rgb * float(intensity),
        0,
        255,
    )

    edge = cv2.GaussianBlur(mask_f, (9, 9), 0)
    edge = np.clip(edge * 1.2, 0.0, 1.0)
    out = (
        out * (1.0 - edge[..., None] * 0.25)
        + frame.astype(np.float32) * (edge[..., None] * 0.25)
    )
    return out.astype(np.uint8)


def apply_stage_lighting(
    frame: np.ndarray,
    intensity: float = 0.6,
) -> np.ndarray:
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width / 2.0
    cy = height * 0.08
    rx = width * 0.45
    ry = height * 0.55
    beam = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 3.0)
    color = np.array((80, 180, 255), dtype=np.float32)
    out = frame.astype(np.float32).copy()
    out = np.clip(out + beam[..., None] * color * float(intensity), 0, 255)
    return out.astype(np.uint8)


def run_add_effect_cv2(
    frames: list[np.ndarray],
    cfg: AddEffectConfig,
) -> list[np.ndarray]:
    parsed = parse_add_effect_instruction(cfg.instruction)
    normalized_target = normalize_dino_target_phrase(
        parsed.grounding_target,
        cfg.instruction,
    )
    target_prompt = cfg.target_prompt_override or f"{normalized_target} ."
    effect_name = cfg.effect_override or parsed.how
    color_name = cfg.color_override or parsed.color

    out_frames: list[np.ndarray] = []
    prev_box = None

    for frame in tqdm(frames, desc="add_effect_cv2"):
        if effect_name == "stage_lighting":
            out_frames.append(
                apply_stage_lighting(frame, intensity=cfg.intensity)
            )
            continue

        box = select_best_box(frame, target_prompt, prev_box)
        if box is None:
            out_frames.append(frame.copy())
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_mask = get_sam_mask_from_box(
            frame_rgb,
            [box[0], box[1], box[2], box[3]],
        ).astype(np.uint8)
        mask = refine_mask(raw_mask, box)

        if effect_name == "brightness":
            edited = apply_brightness_to_mask(
                frame,
                mask,
                alpha=cfg.brightness_alpha,
                beta=cfg.brightness_beta,
            )
        elif effect_name == "blur":
            edited = apply_blur_to_mask(
                frame,
                mask,
                ksize=cfg.blur_ksize,
            )
        else:
            edited = apply_glow_to_mask(
                frame,
                mask,
                color_bgr=color_to_bgr(color_name),
                intensity=cfg.intensity,
                blur_ksize=cfg.glow_blur_ksize,
                background_dim=cfg.background_dim,
            )

        out_frames.append(edited)
        prev_box = box

    return out_frames


def run_add_effect_video(
    cfg: AddEffectConfig,
) -> dict[str, Any]:
    frames, fps, width, height = load_video(cfg.input_video)
    out_frames = run_add_effect_cv2(frames, cfg)
    Path(cfg.output_video).parent.mkdir(parents=True, exist_ok=True)
    write_video(cfg.output_video, out_frames, fps, width, height)

    parsed = parse_add_effect_instruction(cfg.instruction)
    normalized_target = normalize_dino_target_phrase(
        parsed.grounding_target,
        cfg.instruction,
    )
    target_prompt = cfg.target_prompt_override or f"{normalized_target} ."

    return {
        "input_video": cfg.input_video,
        "output_video": cfg.output_video,
        "instruction": cfg.instruction,
        "parsed_target": parsed.target_object,
        "grounding_target": normalized_target,
        "target_prompt": target_prompt,
        "effect_how": cfg.effect_override or parsed.how,
        "effect_color": cfg.color_override or parsed.color,
        "frames": len(out_frames),
        "fps": fps,
        "width": width,
        "height": height,
    }


def extract_add_effect_cases(
    annotations_path: str,
) -> list[dict[str, Any]]:
    with open(annotations_path) as f:
        ann_data = json.load(f)

    rows: list[dict[str, Any]] = []
    for item in ann_data:
        tasks = item.get("tasks", [])
        if not tasks:
            continue
        task0 = tasks[0]
        if task0.get("action") != "add_effect":
            continue

        instruction = item.get("instruction", "")
        parsed = parse_add_effect_instruction(instruction)
        normalized_target = normalize_dino_target_phrase(
            parsed.grounding_target,
            instruction,
        )
        rows.append(
            {
                "video_path": item.get("video_path", ""),
                "instruction": instruction,
                "task_target": task0.get("target", ""),
                "effect_type": task0.get("params", {}).get(
                    "effect_type",
                    "",
                ),
                "parsed_target": parsed.target_object,
                "grounding_target": normalized_target,
                "how": parsed.how,
                "color": parsed.color,
            }
        )
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cv2-based add_effect pipeline",
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--target-prompt-override", default=None)
    parser.add_argument("--effect-override", default=None)
    parser.add_argument("--color-override", default=None)
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--glow-blur-ksize", type=int, default=31)
    parser.add_argument("--brightness-alpha", type=float, default=1.08)
    parser.add_argument("--brightness-beta", type=float, default=10.0)
    parser.add_argument("--blur-ksize", type=int, default=9)
    parser.add_argument("--background-dim", type=float, default=0.92)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = AddEffectConfig(
        input_video=args.input_video,
        output_video=args.output_video,
        instruction=args.instruction,
        target_prompt_override=args.target_prompt_override,
        effect_override=args.effect_override,
        color_override=args.color_override,
        intensity=args.intensity,
        glow_blur_ksize=args.glow_blur_ksize,
        brightness_alpha=args.brightness_alpha,
        brightness_beta=args.brightness_beta,
        blur_ksize=args.blur_ksize,
        background_dim=args.background_dim,
    )
    result = run_add_effect_video(cfg)
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.show:
        show_before_after(cfg.input_video, cfg.output_video, width=480)


if __name__ == "__main__":
    main()
