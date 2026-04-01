from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .add_object_service import add_object_frames
from .background_ops import (
    change_background_color,
    inpaint,
    replace_background,
)
from .camera_ops import (
    horizontal_shift,
    perspective_warp,
    stable_zoom_in,
    zoom_out,
)
from .style_ops import (
    blur_or_brightness,
    histogram_match,
    identity,
    sharpness,
    stylize,
)


def run_method(
    method: str,
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    action = str(params.get("action", params.get("_action", "")))
    params.setdefault("instruction", instruction)

    if action == "add_object":
        return add_object_frames(frames, params, instruction, logger)

    passthrough_actions = {
        "replace_object",
        "edit_motion",
        "edit_expression",
        "align_replacement",
        "refine_mask",
        "blend_instances",
        "track_effect",
        "stabilize_object",
    }
    if action in passthrough_actions:
        return frames

    if method == "crop_resize":
        return stable_zoom_in(frames, params, logger)
    if method == "resize_pad":
        return zoom_out(frames, params)
    if method == "progressive_crop_resize":
        return stable_zoom_in(frames, params, logger)
    if method == "progressive_resize_pad":
        return zoom_out(frames, params)
    if method == "perspective_warp":
        return perspective_warp(frames, params)
    if method == "horizontal_shift":
        return horizontal_shift(frames, params)
    if method == "hsv_retarget":
        return change_background_color(frames, instruction)
    if method == "segment_and_replace":
        return replace_background(frames, params, instruction)
    if method == "opencv_blur":
        return replace_background(
            frames,
            {"blur_background": True},
            instruction,
        )
    if method == "inpaint":
        return inpaint(frames, params)
    if method == "stylize":
        return stylize(frames, params)
    if method == "blur_or_brightness":
        return blur_or_brightness(frames, params)
    if method == "sharpness":
        return sharpness(frames, params)
    if method == "histogram_match":
        return histogram_match(frames, params)
    return identity(frames, params)
