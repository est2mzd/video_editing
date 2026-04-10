from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import sys
import re

import cv2
import numpy as np
from tqdm.auto import tqdm

# Project imports
sys.path.append('/workspace/src')
sys.path.append('/workspace/src/utils')

from postprocess.detectors import detect_all_boxes, get_sam_mask_from_box
from video_utility import load_video, write_video, show_before_after


# Simple debug switch:
# When True, force background color to DEBUG_BG_COLOR_BGR regardless of effect settings.
DEBUG_MODE_ = False
DEBUG_BG_COLOR_BGR: tuple[int, int, int] = (255, 255, 255)

# Global effect intensity controls (1.0 = keep current behavior).
# You can tune each method without editing dispatcher/default params.
#
# How to read each value:
# - 1.0: keep the original inferred/default intensity
# - >1.0: strengthen that effect family
# - 0.0: neutralize that effect family (almost no visible change)
#
# GLOBAL_COLOR_TINT_INTENSITY
# - Scales color_tint strength.
# - Example: 0.5 -> lighter tint overlay, 1.5 -> stronger color overlay.
#
# GLOBAL_HSV_SHIFT_INTENSITY
# - Scales hue/saturation/value shifts around the neutral point.
# - Example: 0.5 -> smaller hue/sat/value shift, 1.5 -> stronger tone shift.
#
# GLOBAL_BRIGHTNESS_CONTRAST_INTENSITY
# - Scales brightness/contrast deltas from neutral values.
# - Example: 0.5 -> milder exposure/contrast change, 1.5 -> punchier look.
#
# GLOBAL_GAUSSIAN_BLUR_INTENSITY
# - Scales blur kernel size/sigma.
# - Example: 0.5 -> weaker blur, 2.0 -> much softer background.
#
# GLOBAL_CLAHE_INTENSITY
# - Scales CLAHE clip_limit.
# - Example: 0.5 -> softer local contrast, 1.5 -> stronger local contrast.
GLOBAL_COLOR_TINT_INTENSITY = 1.5
GLOBAL_HSV_SHIFT_INTENSITY = 1.5
GLOBAL_BRIGHTNESS_CONTRAST_INTENSITY = 1.5
GLOBAL_GAUSSIAN_BLUR_INTENSITY = 2.0
GLOBAL_CLAHE_INTENSITY = 1.5


def infer_foreground_prompt_by_yolo(
    frames: list[np.ndarray],
    weights_path: str = "/workspace/weights/yolo/yolov8m.pt",
) -> str:
    """Infer a stable foreground prompt from YOLO detections.

    Priority:
    1. Return "person ." when person is detected.
    2. Otherwise return the highest-confidence class as "<class> .".
    3. Fallback to "person ." on failure.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(weights_path)
        stats: dict[str, float] = {}
        max_scan = min(3, len(frames))
        for frame in frames[:max_scan]:
            results = model(frame, verbose=False)
            if not results:
                continue
            boxes = getattr(results[0], "boxes", None)
            if boxes is None or boxes.cls is None:
                continue

            names = results[0].names
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
            if boxes.conf is not None:
                confs = boxes.conf.detach().cpu().numpy().tolist()
            else:
                confs = [1.0] * len(cls_ids)

            for cls_id, conf in zip(cls_ids, confs):
                cls_name = str(names.get(cls_id, "")).strip().lower()
                if not cls_name:
                    continue
                stats[cls_name] = stats.get(cls_name, 0.0) + float(conf)

        if not stats:
            return "person ."
        if "person" in stats:
            return "person ."

        best_class = max(stats.items(), key=lambda kv: kv[1])[0]
        return f"{best_class} ."
    except Exception:
        return "person ."


@dataclass
class ReplaceBackgroundConfig:
    """Configuration for background color/style editing pipeline.

    Attributes:
        input_video: Input video path.
        output_video: Output video path.
        foreground_prompt: Text prompt used by GroundingDINO for foreground detection.
        dino_box_threshold: Box confidence threshold for GroundingDINO.
        dino_text_threshold: Text confidence threshold for GroundingDINO.
        max_frames: Optional cap for quick validation runs.
        effect_name: Registered cv2 effect key.
        effect_params: Parameters passed to the selected effect.
    """

    input_video: str
    output_video: str
    foreground_prompt: str = 'person .'
    dino_box_threshold: float = 0.30
    dino_text_threshold: float = 0.25
    max_frames: int | None = None

    # Generic cv2 effect interface
    effect_name: str = 'color_tint'
    effect_params: dict = field(default_factory=lambda: {
        'tint_bgr': (30, 200, 255),
        'strength': 0.35,
    })


def _box_area(box: tuple[float, float, float, float]) -> float:
    """Compute area of an xyxy box.

    Args:
        box: Bounding box as (x1, y1, x2, y2).

    Returns:
        Non-negative box area in pixel units.
    """
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU (Intersection over Union) of two xyxy boxes.

    Args:
        a: First bounding box in (x1, y1, x2, y2).
        b: Second bounding box in (x1, y1, x2, y2).

    Returns:
        IoU value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _box_area(a) + _box_area(b) - inter
    return inter / max(ua, 1e-6)


def _select_stable_box(
    boxes: list[tuple[float, float, float, float]],
    frame_shape: tuple[int, int, int],
    prev_box: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Pick the most stable detection by area and temporal continuity.

    Args:
        boxes: Candidate detections for current frame.
        frame_shape: Current frame shape (H, W, C).
        prev_box: Selected box from previous frame, if available.

    Returns:
        The best box candidate or None when no valid box exists.
    """
    if not boxes:
        return None

    h, w = frame_shape[:2]
    frame_area = float(h * w)
    scored: list[tuple[float, tuple[float, float, float, float]]] = []
    for b in boxes:
        area_ratio = _box_area(b) / max(frame_area, 1.0)
        if area_ratio < 0.01 or area_ratio > 0.85:
            continue
        score = area_ratio
        if prev_box is not None:
            score += 2.0 * _box_iou(b, prev_box)
        scored.append((score, b))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component to reduce SAM noise.

    Args:
        mask_u8: Binary-like uint8 mask.

    Returns:
        Refined uint8 mask containing only the largest component.
    """
    if mask_u8.max() == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def estimate_background_mask_gdino_sam(
    frame_bgr: np.ndarray,
    foreground_prompt: str,
    prev_box: tuple[float, float, float, float] | None = None,
    prev_fg_mask: np.ndarray | None = None,
    dino_box_threshold: float = 0.30,
    dino_text_threshold: float = 0.25,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None, np.ndarray | None]:
    """Estimate background mask by detecting foreground then taking inverse.

    Args:
        frame_bgr: Input frame in BGR.
        foreground_prompt: Prompt text for foreground detection.
        prev_box: Previous frame selected box for temporal fallback.
        prev_fg_mask: Previous foreground mask for temporal fallback.
        dino_box_threshold: GroundingDINO box threshold.
        dino_text_threshold: GroundingDINO text threshold.

    Returns:
        background_mask_u8: uint8 mask in {0, 1} where 1 means background.
        selected_box: Selected foreground box (or previous box fallback).
        fg_mask_u8: Foreground mask in {0, 1}, or None when unavailable.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(
        frame_rgb,
        text_prompt=foreground_prompt,
        box_threshold=dino_box_threshold,
        text_threshold=dino_text_threshold,
    )

    selected = _select_stable_box(boxes, frame_bgr.shape, prev_box)
    if selected is None and prev_box is not None:
        selected = prev_box

    fg_mask = None
    if selected is not None:
        fg_mask = get_sam_mask_from_box(frame_rgb, selected).astype(np.uint8)
        fg_mask = _largest_component(fg_mask)

        # Reject obviously wrong masks and fallback to previous
        fg_ratio = float(fg_mask.mean())
        if fg_ratio < 0.003 or fg_ratio > 0.85:
            fg_mask = None

    if fg_mask is None and prev_fg_mask is not None:
        fg_mask = prev_fg_mask.copy()

    if fg_mask is None:
        # Full background fallback when nothing is detected
        bg_mask = np.ones(frame_bgr.shape[:2], dtype=np.uint8)
        return bg_mask, selected, None

    bg_mask = (1 - fg_mask).astype(np.uint8)
    return bg_mask, selected, fg_mask


def effect_hsv_shift(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Shift hue/saturation/value in HSV space.

    Args:
        frame_bgr: Input frame in BGR.
        params: Dict with hue_shift, saturation_scale, value_scale.

    Returns:
        Color-adjusted BGR frame.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = float(params.get('hue_shift', 0.0))
    sat_scale = float(params.get('saturation_scale', 1.0))
    val_scale = float(params.get('value_scale', 1.0))

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def effect_brightness_contrast(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Apply linear brightness/contrast adjustment.

    Args:
        frame_bgr: Input frame in BGR.
        params: Dict with contrast (alpha) and brightness (beta).

    Returns:
        Color-adjusted BGR frame.
    """
    alpha = float(params.get('contrast', 1.0))
    beta = float(params.get('brightness', 0.0))
    return cv2.convertScaleAbs(frame_bgr, alpha=alpha, beta=beta)


def effect_color_tint(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Blend a solid BGR tint color to the frame.

    Args:
        frame_bgr: Input frame in BGR.
        params: Dict with tint_bgr and strength in [0, 1].

    Returns:
        Tinted BGR frame.
    """
    bgr = params.get('tint_bgr', (180, 80, 40))
    strength = float(params.get('strength', 0.25))
    strength = float(np.clip(strength, 0.0, 1.0))

    tint = np.zeros_like(frame_bgr)
    tint[..., 0] = int(bgr[0])
    tint[..., 1] = int(bgr[1])
    tint[..., 2] = int(bgr[2])
    return cv2.addWeighted(frame_bgr, 1.0 - strength, tint, strength, 0.0)


def effect_clahe(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Enhance local contrast in LAB color space using CLAHE.

    Args:
        frame_bgr: Input frame in BGR.
        params: Dict with clip_limit and tile_grid_size.

    Returns:
        Locally contrast-enhanced BGR frame.
    """
    clip_limit = float(params.get('clip_limit', 2.0))
    tile_grid = int(params.get('tile_grid_size', 8))
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def effect_gaussian_blur(frame_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Blur image by Gaussian filter.

    Args:
        frame_bgr: Input frame in BGR.
        params: Dict with ksize and sigma.

    Returns:
        Blurred BGR frame.
    """
    ksize = int(params.get('ksize', 11))
    if ksize % 2 == 0:
        ksize += 1
    sigma = float(params.get('sigma', 0.0))
    return cv2.GaussianBlur(frame_bgr, (ksize, ksize), sigmaX=sigma)


def _apply_global_effect_intensity(effect_name: str, params: dict) -> dict:
    """Apply file-level global intensity knobs to effect parameters.

    Each intensity knob scales the method-specific parameters around a neutral point.
    """
    out = dict(params)

    if effect_name == 'color_tint':
        strength = float(out.get('strength', 0.25))
        strength *= float(max(0.0, GLOBAL_COLOR_TINT_INTENSITY))
        out['strength'] = float(np.clip(strength, 0.0, 1.0))
        return out

    if effect_name == 'hsv_shift':
        k = float(max(0.0, GLOBAL_HSV_SHIFT_INTENSITY))
        out['hue_shift'] = float(out.get('hue_shift', 0.0)) * k
        sat = 1.0 + (float(out.get('saturation_scale', 1.0)) - 1.0) * k
        val = 1.0 + (float(out.get('value_scale', 1.0)) - 1.0) * k
        out['saturation_scale'] = float(np.clip(sat, 0.0, 5.0))
        out['value_scale'] = float(np.clip(val, 0.0, 5.0))
        return out

    if effect_name == 'brightness_contrast':
        k = float(max(0.0, GLOBAL_BRIGHTNESS_CONTRAST_INTENSITY))
        contrast = 1.0 + (float(out.get('contrast', 1.0)) - 1.0) * k
        brightness = float(out.get('brightness', 0.0)) * k
        out['contrast'] = float(np.clip(contrast, 0.0, 5.0))
        out['brightness'] = float(np.clip(brightness, -255.0, 255.0))
        return out

    if effect_name == 'gaussian_blur':
        k = float(max(0.0, GLOBAL_GAUSSIAN_BLUR_INTENSITY))
        base_ksize = float(out.get('ksize', 11))
        ksize = int(max(1, round(base_ksize * k)))
        if ksize % 2 == 0:
            ksize += 1
        out['ksize'] = ksize
        out['sigma'] = float(max(0.0, float(out.get('sigma', 0.0)) * k))
        return out

    if effect_name == 'clahe':
        k = float(max(0.0, GLOBAL_CLAHE_INTENSITY))
        out['clip_limit'] = float(max(0.01, float(out.get('clip_limit', 2.0)) * k))
        return out

    return out


EFFECT_REGISTRY: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    'hsv_shift': effect_hsv_shift,
    'brightness_contrast': effect_brightness_contrast,
    'color_tint': effect_color_tint,
    'clahe': effect_clahe,
    'gaussian_blur': effect_gaussian_blur,
}


def apply_background_effect(
    frame_bgr: np.ndarray,
    background_mask_u8: np.ndarray,
    effect_name: str,
    effect_params: dict,
    custom_effect_fn: Callable[[np.ndarray, dict], np.ndarray] | None = None,
) -> np.ndarray:
    """Generic IF for applying cv2 effects only on background region.

    Args:
        frame_bgr: Input frame in BGR.
        background_mask_u8: uint8 mask in {0, 1}, where 1 means background.
        effect_name: Key in EFFECT_REGISTRY.
        effect_params: Parameters passed to effect function.
        custom_effect_fn: Optional custom callable to override registry.

    Returns:
        Output BGR frame where only background region is modified.
    """
    if custom_effect_fn is not None:
        effected = custom_effect_fn(frame_bgr, effect_params)
    else:
        if effect_name not in EFFECT_REGISTRY:
            raise ValueError(f'Unknown effect_name: {effect_name}. Available: {list(EFFECT_REGISTRY)}')
        effected = EFFECT_REGISTRY[effect_name](frame_bgr, effect_params)

    mask3 = background_mask_u8[..., None].astype(np.float32)
    out = frame_bgr.astype(np.float32) * (1.0 - mask3) + effected.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def run_replace_background_color_pipeline(cfg: ReplaceBackgroundConfig) -> dict:
    """Run full video pipeline: GDINO+SAM mask + background-only cv2 effect.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Summary dict with video metadata and effect settings.
    """
    frames, fps, width, height = load_video(cfg.input_video)
    if cfg.max_frames is not None:
        frames = frames[: cfg.max_frames]

    out_frames: list[np.ndarray] = []
    prev_box: tuple[float, float, float, float] | None = None
    prev_fg_mask: np.ndarray | None = None

    effect_name = cfg.effect_name
    effect_params = dict(cfg.effect_params)
    if DEBUG_MODE_:
        effect_name = 'color_tint'
        effect_params = {
            'tint_bgr': DEBUG_BG_COLOR_BGR,
            'strength': 1.0,
        }

    # Apply global per-method intensity knobs once per run.
    effect_params = _apply_global_effect_intensity(effect_name, effect_params)

    for idx, frame in enumerate(tqdm(frames, desc='replace_background_gdino_sam')):
        bg_mask, prev_box, prev_fg_mask = estimate_background_mask_gdino_sam(
            frame_bgr=frame,
            foreground_prompt=cfg.foreground_prompt,
            prev_box=prev_box,
            prev_fg_mask=prev_fg_mask,
            dino_box_threshold=cfg.dino_box_threshold,
            dino_text_threshold=cfg.dino_text_threshold,
        )
        edited = apply_background_effect(
            frame_bgr=frame,
            background_mask_u8=bg_mask,
            effect_name=effect_name,
            effect_params=effect_params,
        )
        out_frames.append(edited)

    write_video(cfg.output_video, out_frames, fps, width, height)
    summary = {
        'input_video': cfg.input_video,
        'output_video': cfg.output_video,
        'fps': fps,
        'width': width,
        'height': height,
        'num_frames': len(out_frames),
        'effect_name': effect_name,
        'effect_params': effect_params,
        'debug_mode': bool(DEBUG_MODE_),
    }
    if DEBUG_MODE_:
        summary['debug_color_bgr'] = list(DEBUG_BG_COLOR_BGR)
    return summary


# Presets: color variations with descriptions

EFFECT_PRESETS: dict[str, dict] = {
    'cool_neon_night': {
        'description': '青紫寄りの夜景トーン。ネオン感を少し強調。',
        'effect_name': 'hsv_shift',
        'effect_params': {
            'hue_shift': 12,
            'saturation_scale': 1.25,
            'value_scale': 0.92,
        },
    },
    'warm_cinematic': {
        'description': '暖色寄りで映画風。少し暗めで落ち着いた雰囲気。',
        'effect_name': 'color_tint',
        'effect_params': {
            'tint_bgr': (60, 120, 200),
            'strength': 0.22,
        },
    },
    'bright_clean_commercial': {
        'description': '明るくクリーンな商用映像寄り。見通しを良くする。',
        'effect_name': 'brightness_contrast',
        'effect_params': {
            'contrast': 1.08,
            'brightness': 8,
        },
    },
    'soft_pastel': {
        'description': '低コントラストで柔らかいパステル調。',
        'effect_name': 'hsv_shift',
        'effect_params': {
            'hue_shift': -6,
            'saturation_scale': 0.82,
            'value_scale': 1.05,
        },
    },
    'deep_contrast_drama': {
        'description': 'コントラストを上げた重厚なトーン。',
        'effect_name': 'brightness_contrast',
        'effect_params': {
            'contrast': 1.22,
            'brightness': -10,
        },
    },
    'background_defocus': {
        'description': '背景のみ軽くぼかして被写体を目立たせる。',
        'effect_name': 'gaussian_blur',
        'effect_params': {
            'ksize': 17,
            'sigma': 0.0,
        },
    },
    'green_translucent_light': {
        'description': '薄い半透明の緑を背景に乗せる。',
        'effect_name': 'color_tint',
        'effect_params': {
            'tint_bgr': (0, 200, 0),
            'strength': 0.15,
        },
    },
    'green_translucent_medium': {
        'description': '中程度の半透明の緑を背景に乗せる。',
        'effect_name': 'color_tint',
        'effect_params': {
            'tint_bgr': (0, 200, 0),
            'strength': 0.30,
        },
    },
    'pink_translucent_light': {
        'description': '薄い半透明のピンクを背景に乗せる。',
        'effect_name': 'color_tint',
        'effect_params': {
            'tint_bgr': (180, 120, 200),
            'strength': 0.15,
        },
    },
    'pink_translucent_medium': {
        'description': '中程度の半透明のピンクを背景に乗せる。',
        'effect_name': 'color_tint',
        'effect_params': {
            'tint_bgr': (180, 120, 200),
            'strength': 0.30,
        },
    },
}


_COLOR_TINT_BRG_MAP: dict[str, tuple[int, int, int]] = {
    "green": (0, 200, 0),
    "pink": (180, 120, 200),
    "blue": (220, 120, 40),
    "red": (40, 40, 220),
    "orange": (40, 130, 220),
    "purple": (170, 80, 170),
}


def _extract_color_tint_params_from_instruction(instruction: str) -> dict | None:
    """Extract color tint parameters from instruction text.

    Returns None when no color keyword is detected.
    """
    text = str(instruction or "").lower()
    if not text:
        return None

    strength = 0.22
    if re.search(r"\b(slight|subtle|light|薄い|うっすら)\b", text):
        strength = 0.15
    elif re.search(r"\b(strong|deep|vivid|濃い|強い)\b", text):
        strength = 0.35

    for color, bgr in _COLOR_TINT_BRG_MAP.items():
        if re.search(rf"\b{re.escape(color)}\b", text):
            return {
                "tint_bgr": bgr,
                "strength": strength,
            }
    return None


def infer_background_effect_from_instruction(
    instruction: str,
) -> tuple[str, dict] | None:
    """Infer (effect_name, effect_params) from free-form instruction.

    Rule priority:
    1. Blur/defocus instructions -> gaussian_blur
    2. Explicit color words -> color_tint
    3. Brightness/contrast words -> brightness_contrast
    4. Style tone words -> named preset mapping
    """
    text = str(instruction or "").lower().strip()
    if not text:
        return None

    if re.search(r"\b(blur|blurry|defocus|bokeh|ぼかし|ぼかす)\b", text):
        return "gaussian_blur", {"ksize": 17, "sigma": 0.0}

    tint_params = _extract_color_tint_params_from_instruction(text)
    if tint_params is not None:
        return "color_tint", tint_params

    if re.search(r"\b(bright|brighter|dark|darker|contrast|明る|暗く|コントラスト)\b", text):
        contrast = 1.0
        brightness = 0
        if re.search(r"\b(bright|brighter|明る)\b", text):
            contrast = 1.08
            brightness = 10
        elif re.search(r"\b(dark|darker|暗く)\b", text):
            contrast = 1.10
            brightness = -12
        if re.search(r"\b(contrast|コントラスト)\b", text):
            contrast = max(contrast, 1.15)
        return "brightness_contrast", {
            "contrast": contrast,
            "brightness": brightness,
        }

    style_to_preset = [
        (r"\b(cyberpunk|neon|night)\b", "cool_neon_night"),
        (r"\b(warm|cinematic|movie)\b", "warm_cinematic"),
        (r"\b(clean|commercial|bright and clean)\b", "bright_clean_commercial"),
        (r"\b(pastel|soft)\b", "soft_pastel"),
        (r"\b(drama|dramatic|deep contrast)\b", "deep_contrast_drama"),
    ]
    for pattern, preset in style_to_preset:
        if re.search(pattern, text):
            return get_effect_preset(preset)

    return None


def list_effect_presets() -> None:
    """Print available presets with short descriptions and parameters."""
    print('Available background effect presets:')
    for name, spec in EFFECT_PRESETS.items():
        print(f"- {name}: {spec['description']}")
        print(f"    effect_name={spec['effect_name']}, effect_params={spec['effect_params']}")


def get_effect_preset(name: str) -> tuple[str, dict]:
    """Resolve preset name to (effect_name, effect_params)."""
    if name not in EFFECT_PRESETS:
        raise ValueError(f'Unknown preset: {name}. Available: {list(EFFECT_PRESETS)}')
    spec = EFFECT_PRESETS[name]
    return spec['effect_name'], dict(spec['effect_params'])


def test():
    # Execution example (preset selection)
    input_video = "/workspace/data/videos/DaUJkmBvTKM_2_0to150.mp4"
    out_dir = Path('/workspace/logs/submit/replace_background_00')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ここを変えるだけで色バリエーションを選択できます
    # 選択肢: cool_neon_night, warm_cinematic, bright_clean_commercial, soft_pastel,
    #         deep_contrast_drama, background_defocus,
    #         green_translucent_light, green_translucent_medium,
    #         pink_translucent_light, pink_translucent_medium
    selected_preset = 'green_translucent_light'
    effect_name, effect_params = get_effect_preset(selected_preset)

    output_video = (out_dir / f'DaUJkmBvTKM_2_0to150__{selected_preset}.mp4').as_posix()

    cfg = ReplaceBackgroundConfig(
        input_video=input_video,
        output_video=output_video,
        foreground_prompt='man .',
        effect_name=effect_name,
        effect_params=effect_params,
        # 初回検証用。全尺で回す場合は None にしてください
        max_frames=10 # None,
    )

    result = run_replace_background_color_pipeline(cfg)
    print('selected_preset:', selected_preset)
    print('description:', EFFECT_PRESETS[selected_preset]['description'])
    print(result)
    show_before_after(result['input_video'], result['output_video'], width=560)


if __name__ == '__main__':
    list_effect_presets()
    test() 