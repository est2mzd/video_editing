from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from .. import model_registry as registry


def get_sam_mask_from_box(
    frame_rgb: np.ndarray,
    box_xyxy: list[float] | tuple[float, float, float, float],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Predict foreground mask inside a bbox with SAM.

    Tools: SAM (segment-anything) via model_registry.SAM_PREDICTOR.
    Steps:
    1. Ensure SAM predictor is loaded.
    2. Clamp bbox to frame bounds.
    3. Run SAM box prompt inference and return binary mask.
    4. Return zero mask on failure.
    """
    if not registry.load_sam_predictor(logger=logger):
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)

    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        h, w = frame_rgb.shape[:2]
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(x1 + 1.0, min(float(w), x2))
        y2 = max(y1 + 1.0, min(float(h), y2))

        registry.SAM_PREDICTOR.set_image(frame_rgb)
        masks, _scores, _ = registry.SAM_PREDICTOR.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False,
        )
        return masks[0].astype(np.uint8)
    except Exception as exc:
        if logger is not None:
            logger.debug(f"SAM infer failed: {exc}")
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)


def detect_all_boxes(
    frame_rgb: np.ndarray,
    text_prompt: str,
    logger: logging.Logger | None = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """Detect all candidate boxes from a text prompt with GroundingDINO.

    Tools: GroundingDINO + its preprocess transforms.
    Steps:
    1. Ensure GroundingDINO model and transforms are loaded.
    2. Apply resize/normalize transform to RGB frame.
    3. Run text-conditioned box prediction.
    4. Convert normalized cxcywh boxes to pixel xyxy boxes.
    """
    if not registry.load_grounding_dino_model(logger=logger):
        return []
    try:
        import torch
        from groundingdino.util.inference import predict

        transform = registry.GROUNDING_DINO_TRANSFORMS.Compose([
            registry.GROUNDING_DINO_TRANSFORMS.RandomResize(
                [800], max_size=1333
            ),
            registry.GROUNDING_DINO_TRANSFORMS.ToTensor(),
            registry.GROUNDING_DINO_TRANSFORMS.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])
        image_t = transform(frame_rgb, None)[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        boxes, _logits, _phrases = predict(
            model=registry.GROUNDING_DINO_MODEL,
            image=image_t,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        h, w = frame_rgb.shape[:2]
        if boxes is None or len(boxes) == 0:
            return []

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h

        out = []
        for i in range(len(boxes)):
            out.append(
                (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))
            )
        return out
    except Exception as exc:
        if logger is not None:
            logger.debug(f"GroundingDINO infer failed: {exc}")
        return []


def detect_primary_box(
    frame: np.ndarray,
    text_prompt: str,
    logger: logging.Logger | None = None,
) -> tuple[float, float, float, float] | None:
    """Get the first detected box for zoom/camera operations.

    Tools: OpenCV (BGR->RGB) + GroundingDINO via detect_all_boxes.
    Steps:
    1. Convert frame to RGB.
    2. Detect boxes from prompt.
    3. Return first box or None when no detection exists.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(frame_rgb, text_prompt, logger=logger)
    if not boxes:
        return None
    return boxes[0]


def split_target_keywords(text: str) -> list[str]:
    """Tokenize target text into prompt-friendly keywords.

    Tools: regex tokenization only.
    Steps:
    1. Lowercase and remove non-alphanumeric separators.
    2. Split by commas/whitespace.
    3. Keep non-empty tokens with minimum length.
    """
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9_\s,]", " ", t)
    tokens = [x.strip() for x in re.split(r"[,\s]+", t) if x.strip()]
    normalized: list[str] = []
    token_aliases = {
        "mas": "man",  # tolerate typo like "mas's face"
    }
    for token in tokens:
        token = token_aliases.get(token, token)
        if len(token) >= 3:
            normalized.append(token)
    return normalized


def build_detection_prompts(
    target: str,
    instruction: str,
) -> list[str]:
    """Build prompt candidates from target/instruction for robust detection."""
    prompts: list[str] = []

    raw_target = (target or "").strip()
    if raw_target:
        prompts.append(raw_target)

    keys = split_target_keywords(target) + split_target_keywords(instruction)
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            prompts.append(key.replace("_", " ") + " .")
            seen.add(key)

    if not prompts:
        prompts = ["face .", "person .", "object ."]
    return prompts


def resolve_target_union_box(
    frame: np.ndarray,
    params: dict,
    instruction: str,
    logger: logging.Logger | None = None,
) -> tuple[int, int, int, int] | None:
    """Build a union target box by probing multiple prompt keywords.

    Tools: OpenCV + GroundingDINO via detect_all_boxes.
    Steps:
    1. Extract keywords from task target/instruction.
    2. Query DINO per keyword and collect all candidate boxes.
    3. Compute union xyxy box in image coordinates.
    4. Return fallback center box if no detection exists.
    """
    h, w = frame.shape[:2]
    target = str(params.get("target", ""))
    keys = split_target_keywords(target) + split_target_keywords(instruction)
    if not keys:
        keys = ["person", "face", "object"]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    all_boxes: list[tuple[float, float, float, float]] = []
    for key in keys:
        prompt = key.replace("_", " ") + " ."
        all_boxes.extend(detect_all_boxes(frame_rgb, prompt, logger=logger))

    if not all_boxes:
        fallback = (w * 0.3, h * 0.2, w * 0.7, h * 0.8)
        all_boxes = [fallback]

    x1 = max(0, int(min(box[0] for box in all_boxes)))
    y1 = max(0, int(min(box[1] for box in all_boxes)))
    x2 = min(w, int(max(box[2] for box in all_boxes)))
    y2 = min(h, int(max(box[3] for box in all_boxes)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
