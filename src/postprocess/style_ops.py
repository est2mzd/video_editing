from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from .apply_style_ver5 import apply_style_frames
from .progress import iter_frames_with_progress


def stylize(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Apply style-transfer pipeline to all frames.

    Tools: apply_style_ver5 backend (diffusion/style pipeline).
    Steps: resolve style label and delegate batch processing.
    """
    style = str(params.get("style", "anime"))
    return apply_style_frames(frames, style)


def blur_or_brightness(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Apply blur effect or brightness adjustment frame-wise.

    Tools: OpenCV GaussianBlur / convertScaleAbs.
    """
    mode = str(params.get("mode", "blur"))
    if mode == "brightness":
        alpha = float(params.get("alpha", 1.05))
        beta = float(params.get("beta", 8.0))
        out: list[np.ndarray] = []
        for frame in iter_frames_with_progress(
            frames,
            params,
            "add_effect",
            "blur_or_brightness",
        ):
            out.append(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta))
        return out

    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames,
        params,
        "add_effect",
        "blur_or_brightness",
    ):
        out.append(cv2.GaussianBlur(frame, (5, 5), 0))
    return out


def sharpness(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Sharpen frames with configurable convolution kernel strength.

    Tools: OpenCV filter2D.
    """
    strength = float(params.get("strength", 0.5))
    kernel = (
        np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
            dtype=np.float32,
        )
        * strength
    )
    kernel[1, 1] = 1.0 + 8.0 * strength
    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames,
        params,
        "enhance_style_details",
        "sharpness",
    ):
        out.append(cv2.filter2D(frame, -1, kernel))
    return out


def histogram_match(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Equalize luminance channel for simple appearance normalization.

    Tools: OpenCV YUV conversion + equalizeHist.
    """
    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames,
        params,
        "match_appearance",
        "histogram_match",
    ):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        out.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))
    return out


def identity(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """No-op passthrough used as safe fallback operation.

    Tools: none.
    """
    return frames
