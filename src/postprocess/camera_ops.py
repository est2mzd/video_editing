from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .detectors import (
    build_detection_prompts,
    detect_primary_box,
    get_sam_mask_from_box,
    resolve_target_union_box,
)
from .mask_ops import inpaint_masked_background
from .progress import iter_frames_with_progress


def compose_scaled_mask_foreground(
    frame: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale only masked foreground while preserving background continuity.

    Tools: OpenCV affine warp + inpainting.
    Steps:
    1. Compute foreground center/top from binary mask.
    2. Warp masked object with scale transform.
    3. Inpaint original foreground hole in background.
    4. Paste scaled foreground back with warped mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame.copy()

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    top_y = float(ys.min())

    obj_only = np.where(mask_u8[:, :, None] > 0, frame, 0).astype(np.uint8)
    affine = cv2.getRotationMatrix2D((center_x, center_y), 0.0, float(scale))

    scaled_top_y = center_y + float(scale) * (top_y - center_y)
    if scaled_top_y < 0.0:
        affine[1, 2] += -scaled_top_y

    h, w = frame.shape[:2]
    scaled_obj = cv2.warpAffine(
        obj_only,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    scaled_mask = cv2.warpAffine(
        mask_u8,
        affine,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    result = inpaint_masked_background(frame, mask_u8)
    paste_region = scaled_mask > 0
    result[paste_region] = scaled_obj[paste_region]
    return result


def stable_object_zoom_in(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Object-centric zoom using DINO+SAM masks and OpenCV compositing.

    Tools: GroundingDINO, SAM, OpenCV.
    Steps:
    1. Per frame, resolve target union box with DINO prompts.
    2. Convert box to mask via SAM.
    3. Build temporal scale schedule for dolly-in style zoom.
    4. Apply masked foreground scaling and background restoration.
    """
    if not frames:
        return frames

    zoom_end_scale = params.get("end_scale")
    if zoom_end_scale is None:
        object_end_scale = float(params.get("max_scale", 1.3))
    else:
        zoom_end_scale = float(zoom_end_scale)
        if zoom_end_scale > 1.0:
            object_end_scale = zoom_end_scale
        else:
            object_end_scale = 1.0 / max(zoom_end_scale, 1e-4)
    object_end_scale = float(np.clip(object_end_scale, 1.0, 3.0))

    scales = np.linspace(1.0, object_end_scale, len(frames))
    out: list[np.ndarray] = []
    prev_mask: np.ndarray | None = None

    for i, frame in enumerate(
        iter_frames_with_progress(frames, params, "dolly_in", "object_zoom_in")
    ):
        box = resolve_target_union_box(frame, params, instruction, logger)
        curr_mask: np.ndarray | None = None
        if box is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_mask = get_sam_mask_from_box(
                frame_rgb,
                [box[0], box[1], box[2], box[3]],
                logger=logger,
            ).astype(np.uint8)
            if int((curr_mask > 0).sum()) == 0:
                curr_mask = None

        if curr_mask is None:
            if prev_mask is None:
                out.append(frame.copy())
                continue
            curr_mask = prev_mask
        else:
            prev_mask = curr_mask

        out.append(
            compose_scaled_mask_foreground(frame, curr_mask, float(scales[i]))
        )
    return out


def stable_zoom_in(
    frames: list[np.ndarray],
    params: dict[str, Any],
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Stable zoom-in camera effect with optional object-centric branch.

    Tools: GroundingDINO + OpenCV crop/resize, optionally SAM branch.
    Steps:
    1. If action is dolly_in, delegate to object-mask zoom pipeline.
    2. Detect primary box once from the first frame (DINO).
    3. Compute per-frame zoom scale curve.
    4. Crop around center and resize back to original resolution.
    """
    if not frames:
        return frames

    default_target = "face"
    target = params.get("target", default_target)
    instruction = str(params.get("instruction", ""))

    if isinstance(target, str):
        target_text = target
    elif isinstance(target, (list, tuple, set)):
        target_text = " ".join(
            str(t).strip() for t in target if str(t).strip()
        )
    else:
        target_text = str(target).strip()

    h, w = frames[0].shape[:2]
    box: tuple[float, float, float, float] | None = None
    for prompt in build_detection_prompts(target_text, instruction):
        box = detect_primary_box(frames[0], text_prompt=prompt, logger=logger)
        if box is not None:
            break

    if box is None:
        union_box = resolve_target_union_box(
            frames[0],
            params,
            instruction,
            logger,
        )
        if union_box is not None:
            box = tuple(float(v) for v in union_box)

    if box is None:
        box = (w * 0.3, h * 0.2, w * 0.7, h * 0.8)

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bbox_long = max(1.0, x2 - x1, y2 - y1)

    zoom_factor = float(params.get("zoom_factor", 0.0))
    if zoom_factor <= 0:
        max_scale = float(params.get("max_scale", 1.3))
        zoom_factor = 1.0 / max(0.1, max_scale)

    end_scale = params.get("end_scale")
    if end_scale is None:
        end_scale = (bbox_long / max(w, h)) / max(zoom_factor, 1e-4)
    end_scale = float(np.clip(end_scale, 0.12, 1.0))

    n = len(frames)
    scales = np.linspace(1.0, end_scale, n)
    out: list[np.ndarray] = []
    for i, frame in enumerate(
        iter_frames_with_progress(frames, params, "zoom_in", "stable_zoom_in")
    ):
        scale = float(scales[i])
        crop_w = max(2, int(w * scale))
        crop_h = max(2, int(h * scale))

        xx1 = int(cx - crop_w / 2)
        yy1 = int(cy - crop_h / 2)
        xx2 = xx1 + crop_w
        yy2 = yy1 + crop_h

        xx1 = max(0, min(w - 2, xx1))
        yy1 = max(0, min(h - 2, yy1))
        xx2 = max(xx1 + 2, min(w, xx2))
        yy2 = max(yy1 + 2, min(h, yy2))

        cropped = frame[yy1:yy2, xx1:xx2]
        out.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def zoom_out(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Create zoom-out effect by progressive downscale + center padding.

    Tools: OpenCV resize and canvas composition.
    Steps:
    1. Build linear scale schedule from 1.0 to min_scale.
    2. Resize each frame to scaled size.
    3. Paste resized frame at canvas center.
    """
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    min_scale = float(params.get("min_scale", params.get("end_scale", 0.8)))
    min_scale = float(np.clip(min_scale, 0.2, 1.0))
    n = len(frames)
    scales = np.linspace(1.0, min_scale, n)
    out: list[np.ndarray] = []
    for i, frame in enumerate(
        iter_frames_with_progress(frames, params, "zoom_out", "zoom_out")
    ):
        scale = float(scales[i])
        sw = max(2, int(w * scale))
        sh = max(2, int(h * scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(frame)
        ox = (w - sw) // 2
        oy = (h - sh) // 2
        canvas[oy:oy + sh, ox:ox + sw] = small
        out.append(canvas)
    return out


def perspective_warp(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Apply synthetic camera angle change with perspective transform.

    Tools: OpenCV perspective transform.
    Steps:
    1. Derive source and destination corner points from strength.
    2. Compute perspective matrix.
    3. Warp each frame with the matrix.
    """
    strength = float(params.get("strength", 0.07))
    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames,
        params,
        "adjust_perspective",
        "perspective_warp",
    ):
        h, w = frame.shape[:2]
        s = strength
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w * s, h * s],
            [w * (1 - s), 0],
            [w * (1 - s * 0.5), h],
            [w * s * 0.5, h],
        ])
        m = cv2.getPerspectiveTransform(src, dst)
        out.append(cv2.warpPerspective(frame, m, (w, h)))
    return out


def horizontal_shift(
    frames: list[np.ndarray], params: dict[str, Any]
) -> list[np.ndarray]:
    """Simulate simple orbit/pan by progressive horizontal translation.

    Tools: OpenCV affine warp.
    Steps:
    1. Compute frame-wise shift from progress ratio.
    2. Build affine translation matrix.
    3. Warp each frame with translated x position.
    """
    if not frames:
        return frames
    max_ratio = float(params.get("max_shift_ratio", 0.1))
    n = len(frames)
    out: list[np.ndarray] = []
    for i, frame in enumerate(
        iter_frames_with_progress(
            frames,
            params,
            "orbit_camera",
            "horizontal_shift",
        )
    ):
        h, w = frame.shape[:2]
        _ = h
        progress = i / max(n - 1, 1)
        shift = int(w * max_ratio * progress)
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        out.append(cv2.warpAffine(frame, m, (w, h)))
    return out
