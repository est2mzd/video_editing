from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color_utils import extract_target_color, target_color_bgr
from .mask_ops import estimate_foreground_mask
from .progress import iter_frames_with_progress


def change_background_color(
    frames: list[np.ndarray], instruction: str
) -> list[np.ndarray]:
    """Replace background with instruction-derived solid color.

    Tools: GrabCut mask estimation + OpenCV/Numpy compositing.
    Steps:
    1. Parse target color from instruction text.
    2. Estimate foreground mask for each frame.
    3. Fill background with selected BGR color.
    4. Composite original foreground over colored background.
    """
    color = target_color_bgr(extract_target_color(instruction))
    out: list[np.ndarray] = []
    params_for_progress = {"action": "change_color"}
    for frame in iter_frames_with_progress(
        frames,
        params_for_progress,
        "change_color",
        "change_background_color",
    ):
        mask = estimate_foreground_mask(frame)
        bg = np.full_like(frame, color)
        out.append(np.where(mask[:, :, None] > 0, frame, bg).astype(np.uint8))
    return out


def replace_background(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
) -> list[np.ndarray]:
    """Replace background by blur or color while preserving foreground.

    Tools: GrabCut mask estimation + OpenCV blur/compositing.
    Steps:
    1. Estimate foreground mask for each frame.
    2. Build new background (Gaussian blur or solid color).
    3. Composite preserved foreground with replaced background.
    """
    blur_background = bool(params.get("blur_background", True))
    color = target_color_bgr(extract_target_color(instruction))
    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames,
        params,
        "replace_background",
        "replace_background",
    ):
        mask = estimate_foreground_mask(frame)
        if blur_background:
            bg = cv2.GaussianBlur(frame, (21, 21), 0)
        else:
            bg = np.full_like(frame, color)
        out.append(np.where(mask[:, :, None] > 0, frame, bg).astype(np.uint8))
    return out


def inpaint(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Inpaint dark/empty regions as object-removal fallback.

    Tools: OpenCV threshold, morphology, and Telea inpainting.
    Steps:
    1. Create rough hole mask from low-intensity pixels.
    2. Dilate mask to cover borders.
    3. Apply Telea inpaint where mask is non-empty.
    """
    radius = int(params.get("inpaint_radius", 5))
    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames, params, "remove_object", "inpaint"
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        out.append(
            cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)
            if mask.sum() > 0
            else frame
        )
    return out
