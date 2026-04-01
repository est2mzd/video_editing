from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .detectors import detect_all_boxes, get_sam_mask_from_box
from .mask_ops import (
    build_fg_mask_from_boxes,
    clip_box,
    derive_dynamic_box_from_masks,
    expand_box,
    fuse_masks_adaptive,
    mask_area,
    mask_iou,
    mask_to_box,
    refine_mask,
    temporal_stabilize_mask,
    warp_mask_with_flow,
)
from .model_registry import make_xmem_processor
from .progress import iter_frames_with_progress
from .trackers import (
    estimate_optical_flow,
    track_mask_with_xmem_or_ostrack,
    xmem_predict_mask,
)


def resolve_add_object_prompts(
    params: dict[str, Any], instruction: str
) -> tuple[str, str]:
    raw_target = str(params.get("target", instruction or "person"))
    if raw_target in ("", "new_object", "object"):
        target_prompt = "person . animal . object ."
    else:
        target_prompt = raw_target.replace("_", " ") + " ."
    fg_prompt = "person . face . animal . car . object ."
    return target_prompt, fg_prompt


def compose_shifted_add_object(
    frame: np.ndarray,
    target_box: tuple[int, int, int, int],
    target_mask: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    h, w = frame.shape[:2]
    tx1, ty1, tx2, ty2 = target_box

    obj_w = tx2 - tx1
    shift_x = obj_w // 2
    if tx2 + shift_x > w:
        shift_x = -shift_x

    dst_x1 = max(0, tx1 + shift_x)
    dst_y1 = ty1
    dst_x2 = min(w, tx2 + shift_x)
    dst_y2 = ty2
    paste_w = dst_x2 - dst_x1
    paste_h = dst_y2 - dst_y1
    if paste_w <= 0 or paste_h <= 0:
        return frame.copy()

    src_off_x = dst_x1 - (tx1 + shift_x)
    src_x1 = tx1 + src_off_x
    src_y1 = ty1
    src_x2 = src_x1 + paste_w
    src_y2 = src_y1 + paste_h

    result = frame.copy()
    add_patch = frame[src_y1:src_y2, src_x1:src_x2].copy()
    add_patch_mask = target_mask[src_y1:src_y2, src_x1:src_x2]
    fg_at_dst = fg_mask[dst_y1:dst_y2, dst_x1:dst_x2]

    composite = (add_patch_mask > 0) & (fg_at_dst == 0)
    dst_region = result[dst_y1:dst_y2, dst_x1:dst_x2].copy()
    dst_region[composite] = add_patch[composite]
    result[dst_y1:dst_y2, dst_x1:dst_x2] = dst_region
    return result


def add_object_frames_ver1(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(
            f"add_object ver1: '{target_prompt}' not detected, passthrough"
        )
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(w, tx2), min(h, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return frames

    target_box = (tx1, ty1, tx2, ty2)
    target_mask = get_sam_mask_from_box(
        frame0_rgb, [tx1, ty1, tx2, ty2], logger=logger
    )

    fg_boxes = detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    for box in fg_boxes:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        m = get_sam_mask_from_box(
            frame0_rgb, [bx1, by1, bx2, by2], logger=logger
        )
        fg_mask = np.maximum(fg_mask, m)

    out: list[np.ndarray] = []
    for frame in iter_frames_with_progress(
        frames, params, "add_object", "add_object_ver1"
    ):
        out.append(
            compose_shifted_add_object(frame, target_box, target_mask, fg_mask)
        )
    return out


def add_object_frames_ver2(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)

    out: list[np.ndarray] = []
    prev_target_box: tuple[int, int, int, int] | None = None
    prev_target_mask: np.ndarray | None = None
    prev_fg_mask: np.ndarray | None = None

    for frame_idx, frame in enumerate(
        iter_frames_with_progress(frames, params, "add_object", "add_object_ver2")
    ):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_boxes = detect_all_boxes(frame_rgb, target_prompt, logger=logger)
        if target_boxes:
            tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)
            if tx2 > tx1 and ty2 > ty1:
                target_box = (tx1, ty1, tx2, ty2)
                target_mask = get_sam_mask_from_box(
                    frame_rgb, [tx1, ty1, tx2, ty2], logger=logger
                )
                prev_target_box = target_box
                prev_target_mask = target_mask
            else:
                target_box = prev_target_box
                target_mask = prev_target_mask
        else:
            target_box = prev_target_box
            target_mask = prev_target_mask

        if target_box is None or target_mask is None:
            if frame_idx == 0:
                logger.warning(
                    f"add_object: '{target_prompt}' not detected, passthrough"
                )
            out.append(frame.copy())
            continue

        tx1, ty1, tx2, ty2 = target_box

        fg_boxes = detect_all_boxes(frame_rgb, fg_prompt, logger=logger)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        if fg_boxes:
            for box in fg_boxes:
                bx1, by1, bx2, by2 = [int(c) for c in box]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                m = get_sam_mask_from_box(
                    frame_rgb, [bx1, by1, bx2, by2], logger=logger
                )
                fg_mask = np.maximum(fg_mask, m)
            prev_fg_mask = fg_mask
        elif prev_fg_mask is not None:
            fg_mask = prev_fg_mask

        out.append(
            compose_shifted_add_object(
                frame,
                (tx1, ty1, tx2, ty2),
                target_mask,
                fg_mask,
            )
        )

    return out


def add_object_frames_ver3(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(
            f"add_object ver3: '{target_prompt}' not detected, passthrough"
        )
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(w, tx2), min(h, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return frames
    target_box = (tx1, ty1, tx2, ty2)

    fg_boxes0 = detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        fixed_fg_boxes.append((bx1, by1, bx2, by2))

    out: list[np.ndarray] = []
    prev_target_mask: np.ndarray | None = None
    prev_fg_mask: np.ndarray | None = None

    for frame in iter_frames_with_progress(
        frames, params, "add_object", "add_object_ver3"
    ):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_mask = get_sam_mask_from_box(
            frame_rgb,
            [target_box[0], target_box[1], target_box[2], target_box[3]],
            logger=logger,
        )
        if target_mask is None and prev_target_mask is not None:
            target_mask = prev_target_mask
        prev_target_mask = target_mask

        fg_mask = np.zeros((h, w), dtype=np.uint8)
        if fixed_fg_boxes:
            for bx1, by1, bx2, by2 in fixed_fg_boxes:
                m = get_sam_mask_from_box(
                    frame_rgb, [bx1, by1, bx2, by2], logger=logger
                )
                fg_mask = np.maximum(fg_mask, m)
            prev_fg_mask = fg_mask
        elif prev_fg_mask is not None:
            fg_mask = prev_fg_mask

        out.append(
            compose_shifted_add_object(
                frame,
                target_box,
                target_mask,
                fg_mask,
            )
        )

    return out


def add_object_frames_ver4(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(
            f"add_object ver4: '{target_prompt}' not detected, passthrough"
        )
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(w, tx2), min(h, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return frames
    target_box = (tx1, ty1, tx2, ty2)

    fg_boxes0 = detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        fixed_fg_boxes.append((bx1, by1, bx2, by2))

    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb,
        [target_box[0], target_box[1], target_box[2], target_box[3]],
        logger=logger,
    )
    prev_fg_mask = np.zeros((h, w), dtype=np.uint8)
    for bx1, by1, bx2, by2 in fixed_fg_boxes:
        m = get_sam_mask_from_box(frame0_rgb, [bx1, by1, bx2, by2], logger=logger)
        prev_fg_mask = np.maximum(prev_fg_mask, m)

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))

    out: list[np.ndarray] = []
    out.append(
        compose_shifted_add_object(
            frames[0],
            target_box,
            prev_target_mask,
            prev_fg_mask,
        )
    )

    for i in iter_frames_with_progress(
        range(1, len(frames)), params, "add_object", "add_object_ver4"
    ):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        tracked_target = track_mask_with_xmem_or_ostrack(
            prev_target_mask,
            prev_frame,
            curr_frame,
            logger=logger,
        )
        tracked_fg = track_mask_with_xmem_or_ostrack(
            prev_fg_mask,
            prev_frame,
            curr_frame,
            logger=logger,
        )

        flow = estimate_optical_flow(prev_frame, curr_frame, logger=logger)

        stable_target = temporal_stabilize_mask(
            prev_target_mask,
            tracked_target,
            flow,
            smooth_alpha,
        )
        stable_fg = temporal_stabilize_mask(
            prev_fg_mask,
            tracked_fg,
            flow,
            smooth_alpha,
        )

        out.append(
            compose_shifted_add_object(
                curr_frame,
                target_box,
                stable_target,
                stable_fg,
            )
        )

        prev_target_mask = stable_target
        prev_fg_mask = stable_fg

    return out


def add_object_frames_ver5(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(
            f"add_object ver5: '{target_prompt}' not detected, passthrough"
        )
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(w, tx2), min(h, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return frames
    target_box = (tx1, ty1, tx2, ty2)

    fg_boxes0 = detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        fixed_fg_boxes.append((bx1, by1, bx2, by2))

    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb,
        [target_box[0], target_box[1], target_box[2], target_box[3]],
        logger=logger,
    )
    prev_fg_mask = np.zeros((h, w), dtype=np.uint8)
    for bx1, by1, bx2, by2 in fixed_fg_boxes:
        m = get_sam_mask_from_box(frame0_rgb, [bx1, by1, bx2, by2], logger=logger)
        prev_fg_mask = np.maximum(prev_fg_mask, m)

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))
    sam_blend_alpha = float(np.clip(params.get("sam_blend_alpha", 0.6), 0.0, 1.0))

    out: list[np.ndarray] = []
    out.append(
        compose_shifted_add_object(
            frames[0],
            target_box,
            prev_target_mask,
            prev_fg_mask,
        )
    )

    for i in iter_frames_with_progress(
        range(1, len(frames)), params, "add_object", "add_object_ver5"
    ):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        sam_target = get_sam_mask_from_box(
            curr_rgb,
            [target_box[0], target_box[1], target_box[2], target_box[3]],
            logger=logger,
        )
        sam_fg = np.zeros((h, w), dtype=np.uint8)
        for bx1, by1, bx2, by2 in fixed_fg_boxes:
            m = get_sam_mask_from_box(curr_rgb, [bx1, by1, bx2, by2], logger=logger)
            sam_fg = np.maximum(sam_fg, m)

        tracked_target = track_mask_with_xmem_or_ostrack(
            prev_target_mask,
            prev_frame,
            curr_frame,
            logger=logger,
        )
        tracked_fg = track_mask_with_xmem_or_ostrack(
            prev_fg_mask,
            prev_frame,
            curr_frame,
            logger=logger,
        )

        flow = estimate_optical_flow(prev_frame, curr_frame, logger=logger)

        stable_target = temporal_stabilize_mask(
            prev_target_mask,
            tracked_target,
            flow,
            smooth_alpha,
        )
        stable_fg = temporal_stabilize_mask(
            prev_fg_mask,
            tracked_fg,
            flow,
            smooth_alpha,
        )

        fused_target = (
            sam_blend_alpha * sam_target.astype(np.float32)
            + (1.0 - sam_blend_alpha) * stable_target.astype(np.float32)
        )
        fused_fg = (
            sam_blend_alpha * sam_fg.astype(np.float32)
            + (1.0 - sam_blend_alpha) * stable_fg.astype(np.float32)
        )

        curr_target_mask = (fused_target > 0.5).astype(np.uint8)
        curr_fg_mask = (fused_fg > 0.5).astype(np.uint8)

        out.append(
            compose_shifted_add_object(
                curr_frame,
                target_box,
                curr_target_mask,
                curr_fg_mask,
            )
        )

        prev_target_mask = curr_target_mask
        prev_fg_mask = curr_fg_mask

    return out


def add_object_frames_ver6(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = resolve_add_object_prompts(params, instruction)
    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(f"add_object ver6: '{target_prompt}' not detected, passthrough")
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    target_box_init = clip_box((tx1, ty1, tx2, ty2), w, h)
    if (
        target_box_init[2] <= target_box_init[0]
        or target_box_init[3] <= target_box_init[1]
    ):
        return frames

    fg_boxes0 = detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        cbox = clip_box((bx1, by1, bx2, by2), w, h)
        if cbox[2] > cbox[0] and cbox[3] > cbox[1]:
            fixed_fg_boxes.append(cbox)

    prev_target_box = target_box_init
    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb,
        [prev_target_box[0], prev_target_box[1], prev_target_box[2], prev_target_box[3]],
        logger=logger,
    ).astype(np.uint8)
    prev_target_mask = refine_mask(prev_target_mask)

    prev_fg_mask = build_fg_mask_from_boxes(
        frame0_rgb, fixed_fg_boxes, logger=logger
    )
    prev_fg_mask = refine_mask(prev_fg_mask)

    xmem_target_processor = make_xmem_processor(
        frames[0], prev_target_mask, params, logger=logger
    )
    xmem_fg_processor = make_xmem_processor(
        frames[0], prev_fg_mask, params, logger=logger
    )

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))
    sam_blend_alpha = float(np.clip(params.get("sam_blend_alpha", 0.6), 0.0, 1.0))
    target_expand_scale = float(params.get("target_expand_scale", 1.18))
    fg_expand_scale = float(params.get("fg_expand_scale", 1.10))

    out: list[np.ndarray] = []
    out.append(
        compose_shifted_add_object(
            frames[0],
            prev_target_box,
            prev_target_mask,
            prev_fg_mask,
        )
    )

    for i in iter_frames_with_progress(
        range(1, len(frames)), params, "add_object", "add_object_ver6"
    ):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        flow = estimate_optical_flow(prev_frame, curr_frame, logger=logger)
        warped_target = warp_mask_with_flow(prev_target_mask, flow)
        warped_fg = warp_mask_with_flow(prev_fg_mask, flow)

        xmem_target_mask = xmem_predict_mask(
            xmem_target_processor, curr_frame, logger=logger
        )
        xmem_fg_mask = xmem_predict_mask(
            xmem_fg_processor, curr_frame, logger=logger
        )

        if xmem_target_mask is None:
            xmem_target_mask = warped_target
        if xmem_fg_mask is None:
            xmem_fg_mask = warped_fg

        stable_target = temporal_stabilize_mask(
            prev_target_mask,
            xmem_target_mask,
            flow,
            smooth_alpha,
        )
        stable_fg = temporal_stabilize_mask(
            prev_fg_mask,
            xmem_fg_mask,
            flow,
            smooth_alpha,
        )

        target_box_dyn = derive_dynamic_box_from_masks(
            warped_target | stable_target,
            prev_target_box,
            w,
            h,
            expand_scale=target_expand_scale,
        )

        sam_target = get_sam_mask_from_box(
            curr_rgb,
            [target_box_dyn[0], target_box_dyn[1], target_box_dyn[2], target_box_dyn[3]],
            logger=logger,
        ).astype(np.uint8)
        sam_target = refine_mask(sam_target)

        fg_box_dyn = derive_dynamic_box_from_masks(
            warped_fg | stable_fg,
            mask_to_box(prev_fg_mask, (0, 0, w, h)),
            w,
            h,
            expand_scale=fg_expand_scale,
        )

        sam_fg_dyn = get_sam_mask_from_box(
            curr_rgb,
            [fg_box_dyn[0], fg_box_dyn[1], fg_box_dyn[2], fg_box_dyn[3]],
            logger=logger,
        ).astype(np.uint8)
        sam_fg_fixed = build_fg_mask_from_boxes(
            curr_rgb, fixed_fg_boxes, logger=logger
        )
        sam_fg = np.maximum(sam_fg_dyn, sam_fg_fixed)
        sam_fg = refine_mask(sam_fg)

        curr_target_mask = fuse_masks_adaptive(
            sam_mask=sam_target,
            stable_mask=stable_target,
            prev_mask=prev_target_mask,
            sam_blend_alpha=sam_blend_alpha,
        )
        curr_fg_mask = fuse_masks_adaptive(
            sam_mask=sam_fg,
            stable_mask=stable_fg,
            prev_mask=prev_fg_mask,
            sam_blend_alpha=sam_blend_alpha,
        )

        curr_fg_mask = np.where(curr_target_mask > 0, 0, curr_fg_mask).astype(
            np.uint8
        )

        curr_target_box = mask_to_box(curr_target_mask, target_box_dyn)
        curr_target_box = expand_box(
            curr_target_box, w, h, scale=1.05, min_margin=4
        )

        out.append(
            compose_shifted_add_object(
                curr_frame,
                curr_target_box,
                curr_target_mask,
                curr_fg_mask,
            )
        )

        prev_target_box = curr_target_box
        prev_target_mask = curr_target_mask
        prev_fg_mask = curr_fg_mask

    return out


def add_object_frames_ver7(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    frames_v6 = add_object_frames_ver6(frames, params, instruction, logger)
    shift_ratio = float(params.get("shift_ratio", 0.5))
    out: list[np.ndarray] = []

    for orig, v6 in zip(frames, frames_v6):
        diff = (v6.astype(int) - orig.astype(int)) != 0
        mask = diff.any(axis=2).astype("uint8")

        ys, xs = (mask > 0).nonzero()
        if len(xs) == 0:
            out.append(orig)
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        h = y2 - y1
        w = x2 - x1

        roi = orig[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].copy()

        dx = int(w * shift_ratio)
        new_x1 = max(0, min(orig.shape[1] - w, x1 + dx))
        new_y1 = y1

        canvas = orig.copy()
        target = canvas[new_y1 : new_y1 + h, new_x1 : new_x1 + w]
        mask_bool = roi_mask > 0
        target[mask_bool] = roi[mask_bool]
        canvas[new_y1 : new_y1 + h, new_x1 : new_x1 + w] = target
        out.append(canvas)

    return out


def add_object_frames_ver8(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    frame_h, frame_w = frames[0].shape[:2]
    target_prompt, _fg_prompt = resolve_add_object_prompts(params, instruction)
    shift_ratio = float(params.get("shift_ratio", 0.5))

    out: list[np.ndarray] = []
    prev_mask: np.ndarray | None = None
    prev_box: tuple[int, int, int, int] = (0, 0, frame_w, frame_h)

    for frame_idx, frame in enumerate(
        iter_frames_with_progress(
            frames,
            params,
            "add_object",
            "add_object_ver8",
        )
    ):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidate_boxes = detect_all_boxes(frame_rgb, target_prompt, logger=logger)

        candidate_masks: list[np.ndarray] = []
        for box in candidate_boxes:
            tx1, ty1, tx2, ty2 = [int(c) for c in box]
            clipped_box = clip_box((tx1, ty1, tx2, ty2), frame_w, frame_h)
            if (
                clipped_box[2] <= clipped_box[0]
                or clipped_box[3] <= clipped_box[1]
            ):
                continue
            mask = get_sam_mask_from_box(
                frame_rgb,
                [
                    clipped_box[0],
                    clipped_box[1],
                    clipped_box[2],
                    clipped_box[3],
                ],
                logger=logger,
            ).astype(np.uint8)
            mask = refine_mask(mask)
            if mask_area(mask) > 0:
                candidate_masks.append(mask)

        selected_mask: np.ndarray | None = None
        if candidate_masks:
            if prev_mask is None:
                selected_mask = max(candidate_masks, key=mask_area)
            else:
                selected_mask = max(
                    candidate_masks,
                    key=lambda mask: mask_iou(prev_mask, mask),
                )
        elif prev_mask is not None:
            selected_mask = prev_mask.copy()

        if selected_mask is None:
            if frame_idx == 0:
                logger.warning(
                    f"add_object ver8: '{target_prompt}' not detected, passthrough"
                )
            out.append(frame.copy())
            continue

        selected_mask = (selected_mask > 0).astype(np.uint8)
        selected_box = mask_to_box(selected_mask, prev_box)
        selected_box = clip_box(selected_box, frame_w, frame_h)

        x1, y1, x2, y2 = selected_box
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            out.append(frame.copy())
            prev_mask = selected_mask
            prev_box = selected_box
            continue

        shift_x = int(round(box_w * shift_ratio))
        dst_x1 = x1 + shift_x
        dst_y1 = y1
        dst_x2 = dst_x1 + box_w
        dst_y2 = dst_y1 + box_h

        if dst_x1 < 0:
            dst_x1 = 0
            dst_x2 = box_w
        if dst_x2 > frame_w:
            dst_x2 = frame_w
            dst_x1 = max(0, dst_x2 - box_w)
        if dst_y1 < 0:
            dst_y1 = 0
            dst_y2 = box_h
        if dst_y2 > frame_h:
            dst_y2 = frame_h
            dst_y1 = max(0, dst_y2 - box_h)

        src_off_x = dst_x1 - (x1 + shift_x)
        src_off_y = dst_y1 - y1
        src_x1 = x1 + src_off_x
        src_y1 = y1 + src_off_y
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        src_x1 = max(0, min(frame_w - 1, src_x1))
        src_y1 = max(0, min(frame_h - 1, src_y1))
        src_x2 = max(src_x1 + 1, min(frame_w, src_x2))
        src_y2 = max(src_y1 + 1, min(frame_h, src_y2))
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        result = frame.copy()
        src_patch = frame[src_y1:src_y2, src_x1:src_x2].copy()
        src_mask = selected_mask[src_y1:src_y2, src_x1:src_x2] > 0
        dst_region = result[dst_y1:dst_y2, dst_x1:dst_x2].copy()
        dst_region[src_mask] = src_patch[src_mask]
        result[dst_y1:dst_y2, dst_x1:dst_x2] = dst_region
        out.append(result)

        prev_mask = selected_mask
        prev_box = selected_box

    return out


def add_object_frames_ver9(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if not frames:
        return frames

    frame_h, frame_w = frames[0].shape[:2]
    target_prompt, _fg_prompt = resolve_add_object_prompts(params, instruction)
    shift_ratio = float(params.get("shift_ratio", 0.5))
    ema_prev_weight = float(params.get("ema_prev_weight", 0.7))
    ema_curr_weight = float(params.get("ema_curr_weight", 0.3))

    out: list[np.ndarray] = []
    prev_mask: np.ndarray | None = None
    prev_center: tuple[float, float] | None = None
    prev_box: tuple[int, int, int, int] = (0, 0, frame_w, frame_h)

    for frame_idx, frame in enumerate(
        iter_frames_with_progress(
            frames,
            params,
            "add_object",
            "add_object_ver9",
        )
    ):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidate_boxes = detect_all_boxes(frame_rgb, target_prompt, logger=logger)

        candidate_masks: list[np.ndarray] = []
        for box in candidate_boxes:
            tx1, ty1, tx2, ty2 = [int(c) for c in box]
            clipped_box = clip_box((tx1, ty1, tx2, ty2), frame_w, frame_h)
            if (
                clipped_box[2] <= clipped_box[0]
                or clipped_box[3] <= clipped_box[1]
            ):
                continue
            mask = get_sam_mask_from_box(
                frame_rgb,
                [
                    clipped_box[0],
                    clipped_box[1],
                    clipped_box[2],
                    clipped_box[3],
                ],
                logger=logger,
            ).astype(np.uint8)
            mask = refine_mask(mask)
            if mask_area(mask) > 0:
                candidate_masks.append(mask)

        selected_mask: np.ndarray | None = None
        if candidate_masks:
            if prev_mask is None:
                selected_mask = max(candidate_masks, key=mask_area)
            else:
                selected_mask = max(
                    candidate_masks,
                    key=lambda mask: mask_iou(prev_mask, mask),
                )
        elif prev_mask is not None:
            selected_mask = prev_mask.copy()

        if selected_mask is None:
            if frame_idx == 0:
                logger.warning(
                    f"add_object ver9: '{target_prompt}' not detected, passthrough"
                )
            out.append(frame.copy())
            continue

        selected_mask = (selected_mask > 0).astype(np.uint8)
        selected_box = mask_to_box(selected_mask, prev_box)
        selected_box = clip_box(selected_box, frame_w, frame_h)

        ys, xs = np.where(selected_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            out.append(frame.copy())
            prev_mask = selected_mask
            prev_box = selected_box
            continue

        raw_cx = float(xs.mean())
        raw_cy = float(ys.mean())
        if prev_center is None:
            smooth_cx = raw_cx
            smooth_cy = raw_cy
        else:
            smooth_cx = ema_prev_weight * prev_center[0] + ema_curr_weight * raw_cx
            smooth_cy = ema_prev_weight * prev_center[1] + ema_curr_weight * raw_cy

        x1, y1, x2, y2 = selected_box
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            out.append(frame.copy())
            prev_mask = selected_mask
            prev_box = selected_box
            prev_center = (smooth_cx, smooth_cy)
            continue

        shift_x = float(box_w) * shift_ratio
        new_cx = smooth_cx + shift_x
        new_cy = smooth_cy

        dst_x1 = int(round(new_cx - box_w / 2.0))
        dst_y1 = int(round(new_cy - box_h / 2.0))
        dst_x2 = dst_x1 + box_w
        dst_y2 = dst_y1 + box_h

        if dst_x1 < 0:
            dst_x1 = 0
            dst_x2 = box_w
        if dst_x2 > frame_w:
            dst_x2 = frame_w
            dst_x1 = max(0, dst_x2 - box_w)
        if dst_y1 < 0:
            dst_y1 = 0
            dst_y2 = box_h
        if dst_y2 > frame_h:
            dst_y2 = frame_h
            dst_y1 = max(0, dst_y2 - box_h)

        src_x1 = x1
        src_y1 = y1
        src_x2 = x2
        src_y2 = y2

        copy_w = min(src_x2 - src_x1, dst_x2 - dst_x1)
        copy_h = min(src_y2 - src_y1, dst_y2 - dst_y1)
        if copy_w <= 0 or copy_h <= 0:
            out.append(frame.copy())
            prev_mask = selected_mask
            prev_box = selected_box
            prev_center = (smooth_cx, smooth_cy)
            continue

        src_x2 = src_x1 + copy_w
        src_y2 = src_y1 + copy_h
        dst_x2 = dst_x1 + copy_w
        dst_y2 = dst_y1 + copy_h

        result = frame.copy()
        src_patch = frame[src_y1:src_y2, src_x1:src_x2].copy()
        src_mask = selected_mask[src_y1:src_y2, src_x1:src_x2] > 0
        dst_region = result[dst_y1:dst_y2, dst_x1:dst_x2].copy()
        dst_region[src_mask] = src_patch[src_mask]
        result[dst_y1:dst_y2, dst_x1:dst_x2] = dst_region
        out.append(result)

        prev_mask = selected_mask
        prev_box = selected_box
        prev_center = (smooth_cx, smooth_cy)

    return out
