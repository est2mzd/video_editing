"""apply_style_ver6 – foreground-aware full-frame style transfer.

Key differences from ver5
--------------------------
ver5: splits fg / bg → stylizes each separately → composites them.
ver6: applies style to the **full frame** (single diffusion pass), and uses
      the foreground mask (person / animal) only to guide temporal
      stabilization.  This removes fg/bg seam artefacts while keeping
      motion-aware regeneration for moving subjects.

Instruction-aware entry points
-------------------------------
``extract_style_and_target(instruction, target, params)``
    Derives (style, foreground_text_prompt) from the parsed results
    (target field + params dict) or directly from free-form instruction
    text as a fallback.

``apply_style_video_v6(in_path, out_path, instruction, target, params)``
    Full pipeline: parse → detect fg → stylize full-frame with fg-aware
    temporal stabilization.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from postprocess.progress import Path  # noqa: F811 – workspace patched Path
from postprocess import apply_style_common as common

# ── re-export helpers from common ───────────────────────────────────────────
STYLE_ALIASES = common.STYLE_ALIASES
APPLY_STYLES = common.APPLY_STYLES
blend_regenerated = common.blend_regenerated
build_style_prompt = common.build_style_prompt
calc_flow_farneback = common.calc_flow_farneback
detect_breakdown_mask = common.detect_breakdown_mask
get_img2img_pipe = common.get_img2img_pipe
normalize_style_alias = common.normalize_style_alias
run_img2img = common.run_img2img
sample_flow = common.sample_flow
warp_with_flow = common.warp_with_flow

# ============================================================
# Settings  (identical to ver5)
# ============================================================
NUM_STEPS = 10
STRENGTH = 0.35
GUIDANCE = 5.5
MASK_PHOTO_THRESHOLD = 30
MASK_FB_THRESHOLD = 2.0
MIN_REGEN_RATIO = 0.05
TEMPORAL_BLEND = 0.25

# ── foreground mask detection threshold boost for known-moving regions ──────
# When fg_mask says a pixel is foreground, we raise the regen priority so that
# even small flow-breakdown areas inside the fg get regenerated.
FG_REGEN_BOOST = 0.5   # additional regen_ratio contribution per fg pixel ratio

# ── foreground keyword tables ───────────────────────────────────────────────
_PERSON_KEYWORDS: set[str] = {
    "person", "people", "man", "woman", "child", "human", "character",
    "figure", "athlete", "player", "performer", "pedestrian",
    # Japanese
    "人", "人物", "男性", "女性", "子供",
}

_ANIMAL_KEYWORDS: set[str] = {
    "animal", "dog", "cat", "horse", "bird", "rabbit", "cow", "sheep",
    "lion", "tiger", "bear", "elephant", "monkey", "fox", "deer", "wolf",
    # Japanese
    "動物", "犬", "猫", "馬", "鳥", "ウサギ",
}

_SPECIFIC_ANIMALS: tuple[str, ...] = (
    "dog", "cat", "horse", "bird", "rabbit", "cow", "sheep",
    "lion", "tiger", "bear", "elephant", "monkey", "fox", "deer", "wolf",
)

_DEFAULT_FG_PROMPT = "person . animal ."

_PIPE = None
_LOCK = threading.Lock()


# ============================================================
# Style helpers (same as ver5)
# ============================================================

def normalize_style(style: str) -> str:
    return normalize_style_alias(
        style, aliases=dict(STYLE_ALIASES), fallback="lower"
    )


def get_prompt(style: str) -> str:
    canonical = normalize_style(style)
    return build_style_prompt(canonical)


def get_pipe():
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                _PIPE = get_img2img_pipe(adapter_weight=0.8)
    return _PIPE


def _stylize_img2img(frame_bgr: np.ndarray, prompt: str) -> np.ndarray:
    return run_img2img(
        get_pipe(),
        frame_bgr=frame_bgr,
        prompt=prompt,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
    )


def _calc_flow_farneback(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return calc_flow_farneback(src, dst)


def _warp_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return warp_with_flow(image, flow)


def _detect_breakdown_mask(
    frame_t: np.ndarray,
    frame_tp1: np.ndarray,
    flow_fwd: np.ndarray,
) -> np.ndarray:
    return detect_breakdown_mask(
        frame_t,
        frame_tp1,
        flow_fwd,
        photo_threshold=MASK_PHOTO_THRESHOLD,
        fb_threshold=MASK_FB_THRESHOLD,
    )


def _blend_regenerated(
    warped: np.ndarray,
    regenerated: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return blend_regenerated(warped, regenerated, mask)


# ============================================================
# Foreground target → GroundingDINO text prompt
# ============================================================

def _contains_word(keyword: str, text: str) -> bool:
    """Return True when *keyword* appears as a whole word in *text*."""
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


def target_to_text_prompt(target: str | None) -> str | None:
    """Convert parsed ``target`` field to a GroundingDINO text prompt.

    Returns ``None`` when no foreground detection is needed (full-frame mode).

    Rules
    -----
    * ``None`` / empty / ``"full_frame"`` → ``None`` (full-frame mode,
      no fg mask).
    * Contains person keyword       → include ``"person"``
    * Contains specific animal name → include that animal
    * Contains generic ``"animal"``  → include ``"animal"``
    * Default fallback for unrecognised targets → use target string directly.
    """
    if not target or target.strip().lower() == "":
        return None

    # Even in full-frame styling, we still try to detect people/animals so
    # foreground and background can be stylized separately per frame.
    if target.strip().lower() == "full_frame":
        return _DEFAULT_FG_PROMPT

    lower = target.lower()
    parts: list[str] = []

    has_person = any(_contains_word(kw, lower) for kw in _PERSON_KEYWORDS)
    if has_person:
        parts.append("person")

    for animal in _SPECIFIC_ANIMALS:
        if _contains_word(animal, lower):
            parts.append(animal)
            break

    if not parts and any(_contains_word(kw, lower) for kw in _ANIMAL_KEYWORDS):
        parts.append("animal")

    if not parts:
        # Use target as-is for GroundingDINO (e.g. "car", "hat", ...)
        return f"{target.strip()} ."

    return " . ".join(parts) + " ."


def _detect_foreground_from_instruction(instruction: str) -> str:
    """Scan instruction text to guess a foreground text prompt.

    Falls back to the default ``"person . animal ."`` prompt when no
    recognisable keyword is found.  Whole-word matching is used to avoid
    substring false positives (e.g. 'cat' inside 'application').
    """
    lower = instruction.lower()
    parts: list[str] = []

    if any(_contains_word(kw, lower) for kw in _PERSON_KEYWORDS):
        parts.append("person")
    for animal in _SPECIFIC_ANIMALS:
        if _contains_word(animal, lower):
            parts.append(animal)
            break
    if not parts and any(_contains_word(kw, lower) for kw in _ANIMAL_KEYWORDS):
        parts.append("animal")

    return (" . ".join(parts) + " .") if parts else _DEFAULT_FG_PROMPT


def _extract_style_from_instruction(instruction: str) -> str | None:
    """Scan instruction text for a style keyword.  Returns canonical name or
    ``None`` when none is found."""
    lower = instruction.lower()
    # Check APPLY_STYLES list first (exact or hyphen-free match)
    for s in APPLY_STYLES:
        s_norm = s.replace("_", " ").replace("-", " ")
        if s in lower or s_norm in lower:
            return s
    # Check alias groups
    for canonical, aliases in STYLE_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                return canonical
    return None


def extract_style_and_target(
    instruction: str,
    target: str | None = None,
    params: dict | None = None,
) -> tuple[str, str | None]:
    """Derive (style, foreground_text_prompt) from parsed task fields.

    Priority
    --------
    style
        1. ``params["style"]`` or ``params["style_name"]``
        2. keyword scan of ``instruction``
        3. default ``"oil_painting"``

    foreground_text_prompt
        1. ``target`` field via :func:`target_to_text_prompt`
        2. keyword scan of ``instruction``
        3. default ``_DEFAULT_FG_PROMPT`` = ``"person . animal ."``

    Parameters
    ----------
    instruction:
        Free-form instruction text.
    target:
        Parsed ``target`` field (e.g. ``"full_frame"``, ``"person"``,
        ``"horse"``).  ``None`` if not available.
    params:
        Parsed ``params`` dict that may contain ``"style"`` / ``"style_name"``.
        ``None`` if not available.

    Returns
    -------
    (style, text_prompt)
        ``text_prompt`` is ``None`` when the style should be applied to the
        full frame without any foreground mask (target == ``"full_frame"``).
    """
    # --- style ---
    style: str = "oil_painting"
    if params:
        if "style" in params:
            style = str(params["style"])
        elif "style_name" in params:
            style = str(params["style_name"])
    if style == "oil_painting":   # might still be the default
        from_instr = _extract_style_from_instruction(instruction)
        if from_instr:
            style = from_instr

    # --- foreground text prompt ---
    text_prompt: str | None
    if target is not None:
        text_prompt = target_to_text_prompt(target)
    else:
        text_prompt = _detect_foreground_from_instruction(instruction)

    return style, text_prompt


# ============================================================
# Foreground mask builder (GroundingDINO + SAM)
# ============================================================

def build_foreground_mask(
    frame_bgr: np.ndarray,
    text_prompt: str = "person . animal .",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    dilate_iter: int = 1,
    close_iter: int = 1,
) -> np.ndarray:
    """Return a binary mask (0/255) for detected foreground objects.

    Uses GroundingDINO for bounding boxes and SAM for pixel-level masks.
    Returns a black mask (all zeros) when nothing is detected.
    """
    from postprocess.detectors import detect_all_boxes, get_sam_mask_from_box

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(
        frame_rgb,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    h, w = frame_bgr.shape[:2]
    fg_mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        sam_mask = get_sam_mask_from_box(frame_rgb, box)
        if sam_mask is None:
            continue
        fg_mask = np.maximum(fg_mask, (sam_mask > 0).astype(np.uint8) * 255)

    if close_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter
        )
    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=dilate_iter)

    return fg_mask


def split_foreground_background(
    frame_bgr: np.ndarray,
    fg_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one frame into foreground/background images by mask."""
    mask3 = (fg_mask > 0)[:, :, None]
    foreground = np.where(mask3, frame_bgr, 0).astype(np.uint8)
    background = np.where(~mask3, frame_bgr, 0).astype(np.uint8)
    return foreground, background


def apply_style_foreground_background_frame(
    frame_bgr: np.ndarray,
    style: str,
    text_prompt: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stylize foreground/background separately on a single frame.

    Returns:
        merged frame, foreground mask
    """
    fg_mask = build_foreground_mask(frame_bgr, text_prompt=text_prompt)
    fg, bg = split_foreground_background(frame_bgr, fg_mask)

    prompt = get_prompt(style)
    stylized_fg = _stylize_img2img(fg, prompt)
    stylized_bg = _stylize_img2img(bg, prompt)

    alpha = cv2.GaussianBlur((fg_mask.astype(np.float32) / 255.0), (0, 0), 1.2)
    alpha3 = alpha[:, :, None]
    merged = (
        stylized_fg.astype(np.float32) * alpha3
        + stylized_bg.astype(np.float32) * (1.0 - alpha3)
    )
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    return merged, fg_mask


# ============================================================
# Core frame-sequence stylizer (ver6 – full-frame + fg-aware temporal)
# ============================================================

def apply_style_frames_v6(
    frames: list[np.ndarray],
    style: str,
    text_prompt: str | None = "person . animal .",
    mask_refresh_every: int = 5,
) -> list[np.ndarray]:
    """Apply style transfer to a frame sequence.

    Current v6 behavior (updated):
    - If ``text_prompt`` is provided, each frame is split into foreground /
      background by GroundingDINO + SAM.
    - Foreground and background are stylized separately on every frame.
    - Two stylized outputs are alpha-composited to the final frame.
    - If ``text_prompt`` is None, fallback to full-frame stylization per frame.

    Parameters
    ----------
    frames:
        Input frames in BGR order.
    style:
        Style name (will be normalised via ``normalize_style``).
    text_prompt:
        GroundingDINO query for foreground detection.  Pass ``None`` to skip
        fg detection (equivalent to full-frame-only mode).
    mask_refresh_every:
        Kept for API compatibility. Not used in per-frame split mode.
    """
    if not frames:
        return []

    outputs: list[np.ndarray] = []
    prompt = get_prompt(style)

    for frame in tqdm(frames, desc=f"apply_style_v6 [{style}]"):
        if text_prompt:
            merged, _fg_mask = apply_style_foreground_background_frame(
                frame,
                style=style,
                text_prompt=text_prompt,
            )
            outputs.append(merged)
        else:
            outputs.append(_stylize_img2img(frame, prompt))

    return outputs


# ============================================================
# Single-frame helper (for testing)
# ============================================================

def apply_style_frame_v6(
    frame_bgr: np.ndarray,
    style: str,
) -> np.ndarray:
    """Stylize a single frame (full frame, no temporal context)."""
    prompt = get_prompt(style)
    return _stylize_img2img(frame_bgr, prompt)


# ============================================================
# Video-level entry points
# ============================================================

def apply_style_video_v6(
    in_path: str | Path,
    out_path: str | Path,
    style: str = "oil_painting",
    text_prompt: str | None = "person . animal .",
    max_frames: int | None = None,
    mask_refresh_every: int = 5,
) -> Path:
    """Process a video file with fg-aware full-frame style transfer.

    Parameters
    ----------
    in_path / out_path:
        Input and output video paths.
    style:
        Style name passed to :func:`apply_style_frames_v6`.
    text_prompt:
        GroundingDINO query for foreground detection.  ``None`` → full-frame
        only (no fg mask).
    max_frames:
        Limit number of processed frames (useful for debugging).
    mask_refresh_every:
        Foreground mask refresh interval (frames).
    """
    from utils.video_utility import load_video, write_video as _write_vid

    frames, fps, width, height = load_video(in_path)
    if max_frames is not None:
        frames = frames[:max_frames]

    out_frames = apply_style_frames_v6(
        frames,
        style=style,
        text_prompt=text_prompt,
        mask_refresh_every=mask_refresh_every,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_vid(out_path, out_frames, fps, width, height)
    return out_path


def apply_style_video_from_instruction(
    in_path: str | Path,
    out_path: str | Path,
    instruction: str,
    target: str | None = None,
    params: dict | None = None,
    max_frames: int | None = None,
    mask_refresh_every: int = 5,
) -> Path:
    """High-level entry point: parse instruction → fg target + style → process.

    Parameters
    ----------
    instruction:
        Free-form instruction string (e.g. ``"Make it look like an anime."``).
    target:
        Pre-parsed target field.  ``None`` falls back to instruction scanning.
    params:
        Pre-parsed params dict (may contain ``"style"``).  ``None`` falls back
        to instruction scanning.

    Example
    -------
    >>> apply_style_video_from_instruction(
    ...     "input.mp4", "output.mp4",
    ...     instruction="Apply an oil painting style",
    ...     target="person",
    ...     params={"style": "oil_painting"},
    ... )
    """
    style, text_prompt = extract_style_and_target(
        instruction=instruction,
        target=target,
        params=params,
    )
    return apply_style_video_v6(
        in_path=in_path,
        out_path=out_path,
        style=style,
        text_prompt=text_prompt,
        max_frames=max_frames,
        mask_refresh_every=mask_refresh_every,
    )


# ============================================================
# Quick self-test / smoke-test
# ============================================================

def _test() -> None:
    """Manual smoke test – update paths as needed."""
    WORKSPACE = Path("/workspace")
    VIDEO_DIR = WORKSPACE / "data" / "videos"
    OUT_DIR = WORKSPACE / "logs" / "notebook" / "task_03_apply_style_v6"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    VIDEO_NAME = "94msufYZzaQ_26_0to273.mp4"
    INSTRUCTION = "Apply an oil painting style to the scene."
    TARGET = "person"
    PARAMS = {"style": "oil_painting"}
    MAX_FRAMES = 10

    in_path = VIDEO_DIR / VIDEO_NAME
    out_path = OUT_DIR / f"v6_test_{VIDEO_NAME}"

    if not in_path.exists():
        raise FileNotFoundError(f"input video not found: {in_path}")

    style, text_prompt = extract_style_and_target(INSTRUCTION, TARGET, PARAMS)
    print(f"style      : {style}")
    print(f"text_prompt: {text_prompt}")

    saved = apply_style_video_v6(
        in_path=in_path,
        out_path=out_path,
        style=style,
        text_prompt=text_prompt,
        max_frames=MAX_FRAMES,
    )
    print(f"saved      : {saved}")


if __name__ == "__main__":
    _test()
