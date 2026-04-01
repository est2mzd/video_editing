from __future__ import annotations

import cv2
import numpy as np


def estimate_foreground_mask(frame: np.ndarray) -> np.ndarray:
    """Estimate coarse foreground mask with GrabCut rectangle prior.

    Tools: OpenCV GrabCut.
    Steps: initialize center rectangle, run GrabCut, convert GC labels
    to binary foreground mask (fallback: all-foreground on error).
    """
    h, w = frame.shape[:2]
    mask_gc = np.zeros((h, w), dtype=np.uint8)
    rect = (w // 8, h // 8, w * 3 // 4, h * 3 // 4)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame, mask_gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_gc == 0) | (mask_gc == 2), 0, 1).astype(np.uint8)
    except Exception:
        mask = np.ones((h, w), dtype=np.uint8)
    return mask


def mask_area(mask: np.ndarray) -> int:
    """Return foreground pixel count of a binary-like mask.

    Tools: NumPy boolean counting.
    """
    return int((mask > 0).sum())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks.

    Tools: NumPy logical operations.
    Steps: build boolean masks, compute intersection/union, handle zero-union.
    """
    a_bin = a > 0
    b_bin = b > 0
    inter = int((a_bin & b_bin).sum())
    union = int((a_bin | b_bin).sum())
    if union == 0:
        return 0.0
    return inter / union


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a mask.

    Tools: OpenCV connectedComponentsWithStats.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return mask_u8
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def refine_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Morphologically clean mask and keep dominant component.

    Tools: OpenCV morphology (open/close) + connected components.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return keep_largest_component(mask_u8).astype(np.uint8)


def mask_to_box(
    mask: np.ndarray,
    fallback_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Convert foreground mask to tight xyxy bounding box.

    Tools: NumPy coordinate extraction.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return fallback_box
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return fallback_box
    return (x1, y1, x2, y2)


def clip_box(
    box: tuple[int, int, int, int], w: int, h: int
) -> tuple[int, int, int, int]:
    """Clamp xyxy box to image bounds while preserving positive size.

    Tools: pure arithmetic clipping.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return (x1, y1, x2, y2)


def expand_box(
    box: tuple[int, int, int, int],
    w: int,
    h: int,
    scale: float = 1.15,
    min_margin: int = 6,
) -> tuple[int, int, int, int]:
    """Expand box around center with scale and minimum margin constraints.

    Tools: geometric box transform.
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = max(bw * scale, bw + 2 * min_margin)
    new_h = max(bh * scale, bh + 2 * min_margin)
    nx1 = int(round(cx - new_w / 2.0))
    ny1 = int(round(cy - new_h / 2.0))
    nx2 = int(round(cx + new_w / 2.0))
    ny2 = int(round(cy + new_h / 2.0))
    return clip_box((nx1, ny1, nx2, ny2), w, h)


def fuse_masks_adaptive(
    sam_mask: np.ndarray,
    stable_mask: np.ndarray,
    prev_mask: np.ndarray,
    sam_blend_alpha: float = 0.6,
    iou_switch: float = 0.35,
    area_ratio_limit: float = 2.5,
) -> np.ndarray:
    """Fuse SAM mask and temporal-stable mask with adaptive trust rules.

    Tools: NumPy blending + morphology refinement.
    Steps:
    1. Validate empty-mask edge cases.
    2. Compare IoU and area ratio against previous mask.
    3. Choose/blend masks based on consistency.
    4. Refine final mask morphologically.
    """
    sam_mask = (sam_mask > 0).astype(np.uint8)
    stable_mask = (stable_mask > 0).astype(np.uint8)
    prev_mask = (prev_mask > 0).astype(np.uint8)

    sam_area = mask_area(sam_mask)
    stable_area = mask_area(stable_mask)
    prev_area = max(1, mask_area(prev_mask))

    if sam_area == 0 and stable_area == 0:
        return prev_mask.copy()
    if sam_area == 0:
        return refine_mask(stable_mask)
    if stable_area == 0:
        return refine_mask(sam_mask)

    iou = mask_iou(sam_mask, stable_mask)
    sam_ratio = sam_area / prev_area
    stable_ratio = stable_area / prev_area

    if iou < iou_switch:
        sam_dist = abs(np.log(max(sam_ratio, 1e-6)))
        stable_dist = abs(np.log(max(stable_ratio, 1e-6)))
        chosen = sam_mask if sam_dist <= stable_dist else stable_mask
        return refine_mask(chosen)

    if sam_ratio > area_ratio_limit or sam_ratio < 1.0 / area_ratio_limit:
        fused = stable_mask
    else:
        fused = (
            sam_blend_alpha * sam_mask.astype(np.float32)
            + (1.0 - sam_blend_alpha) * stable_mask.astype(np.float32)
        )
        fused = (fused > 0.5).astype(np.uint8)

    return refine_mask(fused)


def build_fg_mask_from_boxes(
    frame_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    logger=None,
) -> np.ndarray:
    """Aggregate foreground mask by running SAM on multiple boxes.

    Tools: SAM via detectors.get_sam_mask_from_box.
    """
    from .detectors import get_sam_mask_from_box

    h, w = frame_rgb.shape[:2]
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    for bx1, by1, bx2, by2 in boxes:
        if bx2 <= bx1 or by2 <= by1:
            continue
        m = get_sam_mask_from_box(
            frame_rgb, [bx1, by1, bx2, by2], logger=logger
        )
        fg_mask = np.maximum(fg_mask, m.astype(np.uint8))
    return fg_mask


def derive_dynamic_box_from_masks(
    warped_prev_mask: np.ndarray,
    fallback_box: tuple[int, int, int, int],
    w: int,
    h: int,
    expand_scale: float = 1.15,
) -> tuple[int, int, int, int]:
    """Derive next-frame search box from propagated mask geometry.

    Tools: mask-to-box conversion + box expansion.
    """
    raw_box = mask_to_box(warped_prev_mask, fallback_box)
    return expand_box(raw_box, w, h, scale=expand_scale)


def inpaint_masked_background(
    frame: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Inpaint masked region to reconstruct plausible background texture.

    Tools: OpenCV dilation + Telea inpainting.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return frame.copy()

    ys, xs = np.where(mask_u8 > 0)
    box_size = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    dilate_size = max(3, min(15, box_size // 20 * 2 + 1))
    radius = max(3, min(9, box_size // 25))
    kernel = np.ones((dilate_size, dilate_size), dtype=np.uint8)
    hole_mask = cv2.dilate(mask_u8, kernel, iterations=1) * 255
    return cv2.inpaint(frame, hole_mask, radius, cv2.INPAINT_TELEA)


def warp_mask_with_flow(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp previous-frame mask into current frame using dense optical flow.

    Tools: OpenCV remap.
    """
    h, w = mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0.5).astype(np.uint8)


def temporal_stabilize_mask(
    prev_mask: np.ndarray,
    tracked_mask: np.ndarray,
    flow: np.ndarray,
    smooth_alpha: float,
) -> np.ndarray:
    """Stabilize tracked mask by blending with flow-warped previous mask.

    Tools: flow-based warp + NumPy weighted blending.
    """
    warped_prev = warp_mask_with_flow(prev_mask, flow)
    blended = (
        smooth_alpha * tracked_mask.astype(np.float32)
        + (1.0 - smooth_alpha) * warped_prev.astype(np.float32)
    )
    return (blended > 0.5).astype(np.uint8)
