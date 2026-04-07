import logging
import sys

import cv2
import numpy as np
from tqdm.auto import tqdm

if "/workspace/src" not in sys.path:
    sys.path.append("/workspace/src")
if "/workspace/src/utils" not in sys.path:
    sys.path.append("/workspace/src/utils")

from postprocess.camera_ops import compose_scaled_mask_foreground
from postprocess.detectors import detect_all_boxes, detect_primary_box, get_sam_mask_from_box
#from utils.video_utility import load_video, write_video, show_before_after


def _box_area(b: tuple[float, float, float, float]) -> float:
    """Return area of an xyxy box in pixel units."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two xyxy boxes for temporal box matching."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _select_best_box(
    frame_bgr: np.ndarray,
    target_prompt: str,
    prev_box: tuple[float, float, float, float] | None,
    logger: logging.Logger,
) -> tuple[float, float, float, float] | None:
    """Pick the most plausible target box using size prior + temporal IoU."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(frame_rgb, text_prompt=target_prompt, logger=logger)

    if not boxes:
        return detect_primary_box(frame_bgr, text_prompt=target_prompt, logger=logger)

    best_box = None
    best_score = -1e9
    for b in boxes:
        area_ratio = _box_area(b) / float(max(1, h * w))

        score = 0.0
        if 0.001 <= area_ratio <= 0.20:
            score += 2.0
        elif area_ratio > 0.35:
            score -= 3.0

        score -= area_ratio * 3.0

        if prev_box is not None:
            score += _box_iou(prev_box, b) * 4.0

        if score > best_score:
            best_score = score
            best_box = b

    return best_box


def _refine_mask(mask: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Keep the connected component nearest the selected box center."""
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 2:
        return mask_u8

    cx = int((box[0] + box[2]) / 2.0)
    cy = int((box[1] + box[3]) / 2.0)
    cx = int(np.clip(cx, 0, mask_u8.shape[1] - 1))
    cy = int(np.clip(cy, 0, mask_u8.shape[0] - 1))

    center_label = labels[cy, cx]
    if center_label > 0:
        return (labels == center_label).astype(np.uint8)

    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(comp_areas)) + 1
    return (labels == k).astype(np.uint8)


def stable_object_zoom_v2(
    frames: list[np.ndarray],
    target_prompt: str,
    object_end_scale: float,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Apply object-only dolly-in using per-frame DINO+SAM masks and compositing.

    Steps:
    1. Build linear scale schedule from 1.0 to object_end_scale.
    2. Detect/select a stable target box per frame.
    3. Generate and refine SAM mask from the box.
    4. Reject overly large masks, fallback to previous valid mask.
    5. Scale only foreground via compose_scaled_mask_foreground.
    """
    if not frames:
        return frames

    object_end_scale = float(np.clip(object_end_scale, 1.0, 3.0))
    scales = np.linspace(1.0, object_end_scale, len(frames))
    out: list[np.ndarray] = []

    prev_mask: np.ndarray | None = None
    prev_box: tuple[float, float, float, float] | None = None

    h, w = frames[0].shape[:2]
    frame_area = float(max(1, h * w))

    for i, frame in enumerate(tqdm(frames, desc="stable_object_zoom_in")):
        box = _select_best_box(frame, target_prompt, prev_box, logger)
        curr_mask: np.ndarray | None = None

        if box is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_mask = get_sam_mask_from_box(
                frame_rgb,
                [box[0], box[1], box[2], box[3]],
                logger=logger,
            ).astype(np.uint8)

            curr_mask = _refine_mask(raw_mask, box)
            area_ratio = float((curr_mask > 0).sum()) / frame_area

            if area_ratio > 0.25:
                curr_mask = None

        if curr_mask is None:
            if prev_mask is None:
                out.append(frame.copy())
                continue
            curr_mask = prev_mask
        else:
            prev_mask = curr_mask
            prev_box = box

        out.append(compose_scaled_mask_foreground(frame, curr_mask, float(scales[i])))

    return out
