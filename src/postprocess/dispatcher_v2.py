from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from utils.video_utility import load_video, write_video
from .add_effect import AddEffectConfig, run_add_effect_cv2
from .add_object import AddObjectConfig, run_add_object_gdino_sam_cv2
from .apply_style_ver5 import apply_style_video_foreground_background
from .change_color import (
    parse_color_change_instruction,
    run_change_color_gradual_pipeline,
)
from .dolly_in import run_dolly_in_with_instruction
from .replace_background import (
    ReplaceBackgroundConfig,
    infer_foreground_prompt_by_yolo,
    run_replace_background_color_pipeline,
)
from .zoom_in import run_zoom_in_with_instruction
from .background_ops import inpaint
from .camera_ops import (
    horizontal_shift,
    perspective_warp,
    zoom_out,
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


def _target_to_prompt(
    target_text: str,
    default_prompt: str = "person .",
) -> str:
    target_text = str(target_text or "").strip()
    if not target_text:
        return default_prompt
    if target_text.endswith("."):
        return target_text
    return f"{target_text} ."


def _run_apply_style(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
) -> list[np.ndarray]:
    style = str(params.get("style", params.get("style_name", "oil_painting")))
    # apply_style はモジュール既定の person foreground を使う。
    text_prompt = _target_to_prompt(str(params.get("target", "person")))

    with tempfile.TemporaryDirectory(prefix="dispatch_style_") as tmp_dir:
        tmp = Path(tmp_dir)
        in_path = tmp / "input.mp4"
        out_path = tmp / "output.mp4"
        h, w = frames[0].shape[:2]
        write_video(in_path, frames, 30.0, w, h)
        apply_style_video_foreground_background(
            in_path=in_path,
            out_path=out_path,
            style=style,
            text_prompt=text_prompt,
            max_frames=None,
        )
        out_frames, _fps, _w, _h = load_video(out_path)
    return out_frames


def _run_add_effect(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
) -> list[np.ndarray]:
    cfg = AddEffectConfig(
        input_video="",
        output_video="",
        instruction=instruction,
        # Use add_effect.py parser-derived target from instruction.
        target_prompt_override=None,
        effect_override=params.get("effect_override"),
        color_override=params.get("color_override"),
    )
    return run_add_effect_cv2(frames, cfg)


def _run_add_object(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    cfg = AddObjectConfig(
        input_video="",
        output_video="",
        instruction=instruction,
        # Use add_object.py parser-derived target from instruction.
        target_prompt_override=None,
    )
    return run_add_object_gdino_sam_cv2(frames, cfg, logger)


def _run_change_color(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
) -> list[np.ndarray]:
    # change_color.py の instruction parser を使って target を決める。
    parsed = parse_color_change_instruction(instruction)
    target_prompt = _target_to_prompt(
        str(parsed.target_object or "object"),
        default_prompt="object .",
    )
    target_color = str(params.get("color", parsed.to_color or "blue"))
    force_gradual = params.get("force_gradual")
    if isinstance(force_gradual, bool):
        gradual = force_gradual
    else:
        inst = str(instruction or "").lower()
        keywords = params.get("gradual_keywords")
        if isinstance(keywords, (list, tuple, set)):
            gradual_keywords = [
                str(k).strip().lower()
                for k in keywords
                if str(k).strip()
            ]
        else:
            gradual_keywords = [
                "gradual",
                "gradually",
                "徐々",
                "徐々に",
                "だんだん",
                "少しずつ",
            ]
        gradual = any(kw in inst for kw in gradual_keywords)

    with tempfile.TemporaryDirectory(prefix="dispatch_color_") as tmp_dir:
        tmp = Path(tmp_dir)
        in_path = tmp / "input.mp4"
        out_path = tmp / "output.mp4"
        h, w = frames[0].shape[:2]
        write_video(in_path, frames, 30.0, w, h)
        run_change_color_gradual_pipeline(
            input_video=str(in_path),
            output_video=str(out_path),
            target_prompt=target_prompt,
            target_color=target_color,
            max_frames=None,
            gradual=gradual,
        )
        out_frames, _fps, _w, _h = load_video(out_path)
    return out_frames


def _run_replace_background(
    frames: list[np.ndarray],
    params: dict[str, Any],
) -> list[np.ndarray]:
    fg_prompt = infer_foreground_prompt_by_yolo(frames)
    effect_name = str(params.get("effect_name", "hsv_shift"))
    default_effect_params = {
        "hue_shift": 8,
        "saturation_scale": 1.15,
        "value_scale": 0.95,
    }
    effect_params = params.get("effect_params", default_effect_params)
    if not isinstance(effect_params, dict):
        effect_params = default_effect_params

    with tempfile.TemporaryDirectory(prefix="dispatch_bg_") as tmp_dir:
        tmp = Path(tmp_dir)
        in_path = tmp / "input.mp4"
        out_path = tmp / "output.mp4"
        h, w = frames[0].shape[:2]
        write_video(in_path, frames, 30.0, w, h)
        cfg = ReplaceBackgroundConfig(
            input_video=str(in_path),
            output_video=str(out_path),
            foreground_prompt=fg_prompt,
            effect_name=effect_name,
            effect_params=effect_params,
            max_frames=None,
        )
        run_replace_background_color_pipeline(cfg)
        out_frames, _fps, _w, _h = load_video(out_path)
    return out_frames


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

    if not frames:
        return frames

    if resolved_action in {"add_object", "increase_amount"}:
        return _run_add_object(frames, params, instruction, logger)

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
        print("--------------- Passthrough ---------------")
        return frames

    # 準備できた関数
    try:
        if resolved_action == "apply_style":
            logger.info(
                "dispatch apply_style via "
                "apply_style_video_foreground_background"
            )
            return _run_apply_style(frames, params, instruction)

        if resolved_action == "replace_background":
            logger.info(
                "dispatch replace_background via "
                "run_replace_background_color_pipeline"
            )
            return _run_replace_background(frames, params)

        if resolved_action == "change_color":
            logger.info(
                "dispatch change_color via "
                "run_change_color_gradual_pipeline"
            )
            return _run_change_color(frames, params, instruction)

        if resolved_action == "zoom_in":
            logger.info("dispatch zoom_in via run_zoom_in_with_instruction")
            end_scale = float(
                params.get(
                    "end_scale",
                    params.get("zoom_factor", 1.85),
                )
            )
            return run_zoom_in_with_instruction(
                frames,
                instruction=instruction,
                logger=logger,
                target_prompt_override=params.get("target_prompt_override"),
                end_scale=end_scale,
                center_smooth_alpha=float(
                    params.get("center_smooth_alpha", 0.80)
                ),
                box_smooth_alpha=float(params.get("box_smooth_alpha", 0.70)),
                safe_margin_ratio=float(params.get("safe_margin_ratio", 0.15)),
            )

        if resolved_action == "dolly_in":
            logger.info("dispatch dolly_in via run_dolly_in_with_instruction")
            object_end_scale = float(
                params.get(
                    "object_end_scale",
                    params.get("zoom_factor", 1.85),
                )
            )
            return run_dolly_in_with_instruction(
                frames,
                instruction=instruction,
                object_end_scale=object_end_scale,
                logger=logger,
                target_prompt_override=params.get("target_prompt_override"),
            )

        if resolved_action == "add_effect":
            logger.info("dispatch add_effect via run_add_effect_cv2")
            return _run_add_effect(frames, params, instruction)

        # Not in 2.1-2.7: keep existing dispatcher behavior.
        if resolved_action == "zoom_out":
            return zoom_out(frames, params)

        if resolved_action == "change_camera_angle":
            return perspective_warp(frames, params)

        if resolved_action == "orbit_camera":
            return horizontal_shift(frames, params)

        if resolved_action == "remove_object":
            return inpaint(frames, params)
    except Exception as e:
        logger.exception(
            "dispatcher_v2 failed on action=%s: %s",
            resolved_action,
            e,
        )
        return frames

    return frames
