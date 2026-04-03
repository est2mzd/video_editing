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
    stable_object_zoom_in,
    zoom_out,
)
from .style_ops import (
    blur_or_brightness,
    stylize,
    # histogram_match,
    # identity,
    # sharpness,
)


def _normalize_target_text(targets: Any) -> str:
    """Normalize task targets into a single prompt-friendly string.

    Tools: Python type checks and string joins.
    Steps:
    1. Accept list/tuple/set or scalar target payload.
    2. Trim empty values and stringify entries.
    3. Return comma-joined text used by DINO/SAM prompts.
    """
    if targets is None:
        return ""
    if isinstance(targets, (list, tuple, set)):
        parts = [str(t).strip() for t in targets if str(t).strip()]
        return ", ".join(parts)
    return str(targets).strip()


def _resolve_action(action: str, params: dict[str, Any]) -> str:
    """Resolve action from schema-first keys, then notebook-style fallbacks."""
    direct = str(action or "").strip()
    if direct:
        return direct

    for key in ("action", "_action"):
        value = str(params.get(key, "")).strip()
        if value:
            return value

    method = str(params.get("method", "")).strip().lower()
    method_to_action = {
        "stylize": "apply_style",
        "object_zoom_in": "dolly_in",
        "stable_zoom_in": "zoom_in",
        "zoom_out": "zoom_out",
        "perspective_warp": "change_camera_angle",
        "horizontal_shift": "orbit_camera",
        "blur_or_brightness": "add_effect",
        "replace_background": "replace_background",
        "change_background_color": "change_color",
        "inpaint": "remove_object",
    }
    return method_to_action.get(method, "")


def run_method(
    action: str,
    targets: Any,
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
    # method: str | None = None,
) -> list[np.ndarray]:
    """Dispatch postprocess by action and apply target-aware params.

    Tools routed here include:
    - GroundingDINO/SAM/OpenCV camera/object ops
    - GrabCut/OpenCV background ops
    - Style transfer backends
    - Add-object versioned pipelines (ver1-ver9)

    Steps:
    1. Normalize incoming action and targets.
    2. Inject action/target hints into params for downstream modules.
    3. Route primarily by action from annotation schema.
    4. Use method-based fallback for backward compatibility.
    """
    resolved_action = _resolve_action(action, params)
    target_text = _normalize_target_text(
        targets
        if targets is not None
        else params.get("target", params.get("targets"))
    )

    params["action"] = resolved_action
    if target_text:
        params["target"] = target_text
    if "constraints" not in params and "_constraints" in params:
        params["constraints"] = params.get("_constraints")
    params.setdefault("instruction", instruction)

    if resolved_action in {"add_object", "increase_amount"}:
        return add_object_frames(frames, params, instruction, logger)

    # 準備できていない関数はpassthroughして後で実装する
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
    if resolved_action in passthrough_actions:
        return frames

    # 準備できた関数
    if resolved_action == "apply_style":
        return stylize(frames, params)
    if resolved_action == "replace_background":
        return replace_background(frames, params, instruction)
    if resolved_action == "change_color":
        return change_background_color(frames, instruction)
    if resolved_action == "zoom_in":
        return stable_zoom_in(frames, params, logger)
    if resolved_action == "dolly_in":
        return stable_object_zoom_in(frames, params, instruction, logger)
    if resolved_action == "zoom_out":
        return zoom_out(frames, params)
    if resolved_action == "change_camera_angle":
        return perspective_warp(frames, params)
    if resolved_action == "orbit_camera":
        return horizontal_shift(frames, params)
    if resolved_action == "add_effect":
        return blur_or_brightness(frames, params)
    if resolved_action == "remove_object":
        return inpaint(frames, params)

    return frames
