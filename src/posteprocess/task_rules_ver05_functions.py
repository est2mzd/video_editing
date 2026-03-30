from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable


GROUNDING_DINO_MODEL: Any = None
GROUNDING_DINO_TRANSFORMS: Any = None
SAM_PREDICTOR: Any = None
RAFT_MODEL: Any = None
RAFT_DEVICE: str | None = None


def _resolve_video_name_for_progress(params: dict[str, Any]) -> str:
    raw = (
        params.get("video_name")
        or params.get("input_video_name")
        or params.get("mp4_name")
        or params.get("video_path")
        or params.get("_video_path")
        or ""
    )
    text = str(raw)
    if not text:
        return "unknown.mp4"
    return Path(text).name


def _iter_frames_with_progress(
    iterable,
    params: dict[str, Any],
    action_hint: str,
    stage: str,
):
    action = str(params.get("action", params.get("_action", action_hint)))
    video_name = _resolve_video_name_for_progress(params)
    return tqdm(
        iterable,
        desc=f"{action}:{stage} [{video_name}]",
        unit="frame",
        leave=False,
        dynamic_ncols=True,
    )


def extract_target_color(instruction: str) -> str | None:
    names = [
        "navy blue", "violet", "purple", "red", "blue", "green",
        "yellow", "orange", "pink", "black", "white", "silver",
    ]
    text = (instruction or "").lower()
    for name in names:
        if name in text:
            return name
    return None


def target_color_bgr(name: str | None) -> tuple[int, int, int]:
    lut = {
        "red": (40, 40, 220),
        "orange": (0, 128, 255),
        "yellow": (40, 220, 220),
        "green": (60, 180, 60),
        "blue": (220, 80, 40),
        "navy blue": (160, 60, 20),
        "violet": (200, 80, 180),
        "purple": (180, 60, 180),
        "pink": (180, 120, 255),
        "black": (16, 16, 16),
        "white": (240, 240, 240),
        "silver": (180, 180, 180),
    }
    return lut.get((name or "").lower(), (0, 128, 255))


def estimate_foreground_mask(frame: np.ndarray) -> np.ndarray:
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


def load_grounding_dino_model(logger: logging.Logger | None = None) -> bool:
    global GROUNDING_DINO_MODEL, GROUNDING_DINO_TRANSFORMS
    if GROUNDING_DINO_MODEL is not None and GROUNDING_DINO_TRANSFORMS is not None:
        return True
    try:
        import torch
        from groundingdino.util.inference import load_model
        from groundingdino.datasets import transforms as transforms_mod

        config_path = Path("/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        ckpt_path = Path("/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        if not config_path.exists() or not ckpt_path.exists():
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        GROUNDING_DINO_MODEL = load_model(str(config_path), str(ckpt_path), device=device)
        GROUNDING_DINO_TRANSFORMS = transforms_mod
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"GroundingDINO load failed, fallback to center crop: {exc}")
        return False


def load_sam_predictor(
    logger: logging.Logger | None = None,
) -> bool:
    """Load SAM vit_h predictor (segment-anything)."""
    global SAM_PREDICTOR
    if SAM_PREDICTOR is not None:
        return True
    try:
        import torch
        _SAM_DIR = "/workspace/third_party/segment-anything"
        if _SAM_DIR not in sys.path:
            sys.path.insert(0, _SAM_DIR)
        from segment_anything import sam_model_registry, SamPredictor

        ckpt = Path("/workspace/weights/sam_vit_h_4b8939.pth")
        if not ckpt.exists():
            return False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(device=device)
        SAM_PREDICTOR = SamPredictor(sam)
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"SAM load failed: {exc}")
        return False


def get_sam_mask_from_box(
    frame_rgb: np.ndarray,
    box_xyxy: list[float],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Return binary (h,w) mask via SAM prediction from a GroundingDINO bbox.
    Falls back to filled rectangle when SAM is unavailable.
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = (
        int(box_xyxy[0]), int(box_xyxy[1]),
        int(box_xyxy[2]), int(box_xyxy[3]),
    )
    if not load_sam_predictor(logger=logger):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return mask
    try:
        SAM_PREDICTOR.set_image(frame_rgb)
        input_box = np.array([x1, y1, x2, y2], dtype=float)
        masks, _, _ = SAM_PREDICTOR.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks[0].astype(np.uint8)
    except Exception as exc:
        if logger is not None:
            logger.debug(f"SAM predict failed: {exc}")
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return mask


def _detect_all_boxes(
    frame_rgb: np.ndarray,
    text_prompt: str,
    logger: logging.Logger | None = None,
) -> list[list[float]]:
    """Return ALL GroundingDINO bbox results [[x1,y1,x2,y2], ...] in pixels."""
    if not load_grounding_dino_model(logger=logger):
        return []
    try:
        import torch
        from PIL import Image
        from groundingdino.util.inference import predict

        transform = GROUNDING_DINO_TRANSFORMS.Compose([
            GROUNDING_DINO_TRANSFORMS.RandomResize([800], max_size=1333),
            GROUNDING_DINO_TRANSFORMS.ToTensor(),
            GROUNDING_DINO_TRANSFORMS.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])
        h, w = frame_rgb.shape[:2]
        tensor = transform(Image.fromarray(frame_rgb), None)[0]
        boxes, _logits, _phrases = predict(
            model=GROUNDING_DINO_MODEL,
            image=tensor,
            caption=text_prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if len(boxes) == 0:
            return []
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]
        xyxy = torch.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
        )
        xyxy = (xyxy * torch.tensor([w, h, w, h])).cpu().numpy()
        return [list(row) for row in xyxy]
    except Exception as exc:
        if logger is not None:
            logger.debug(f"_detect_all_boxes failed: {exc}")
        return []


def detect_primary_box(frame: np.ndarray, text_prompt: str, logger: logging.Logger | None = None) -> tuple[float, float, float, float] | None:
    if not load_grounding_dino_model(logger=logger):
        return None
    try:
        import torch
        from PIL import Image
        from groundingdino.util.inference import predict

        transform = GROUNDING_DINO_TRANSFORMS.Compose([
            GROUNDING_DINO_TRANSFORMS.RandomResize([800], max_size=1333),
            GROUNDING_DINO_TRANSFORMS.ToTensor(),
            GROUNDING_DINO_TRANSFORMS.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(frame_rgb), None)[0]
        boxes, logits, phrases = predict(
            model=GROUNDING_DINO_MODEL,
            image=tensor,
            caption=text_prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if len(boxes) == 0:
            return None

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        h, w = frame.shape[:2]
        xyxy = (xyxy * torch.tensor([w, h, w, h])).cpu().numpy()

        chosen = xyxy[0]
        for idx, phrase in enumerate(phrases):
            p = str(phrase).lower()
            if "face" in p or "person" in p:
                chosen = xyxy[idx]
                break
        return float(chosen[0]), float(chosen[1]), float(chosen[2]), float(chosen[3])
    except Exception as exc:
        if logger is not None:
            logger.debug(f"GroundingDINO inference failed, fallback to center crop: {exc}")
        return None


def stable_zoom_in(frames: list[np.ndarray], params: dict[str, Any], logger: logging.Logger, text_prompt: str = "face . person .") -> list[np.ndarray]:
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    box = detect_primary_box(frames[0], text_prompt=text_prompt, logger=logger)
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
    for i, frame in enumerate(_iter_frames_with_progress(frames, params, "zoom_in", "stable_zoom_in")):
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


def zoom_out(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    min_scale = float(params.get("min_scale", params.get("end_scale", 0.8)))
    min_scale = float(np.clip(min_scale, 0.2, 1.0))
    n = len(frames)
    scales = np.linspace(1.0, min_scale, n)
    out: list[np.ndarray] = []
    for i, frame in enumerate(_iter_frames_with_progress(frames, params, "zoom_out", "zoom_out")):
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


def perspective_warp(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    strength = float(params.get("strength", 0.07))
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(frames, params, "adjust_perspective", "perspective_warp"):
        h, w = frame.shape[:2]
        s = strength
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * s, h * s], [w * (1 - s), 0], [w * (1 - s * 0.5), h], [w * s * 0.5, h]])
        m = cv2.getPerspectiveTransform(src, dst)
        out.append(cv2.warpPerspective(frame, m, (w, h)))
    return out


def horizontal_shift(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    if not frames:
        return frames
    max_ratio = float(params.get("max_shift_ratio", 0.1))
    n = len(frames)
    out: list[np.ndarray] = []
    for i, frame in enumerate(_iter_frames_with_progress(frames, params, "orbit_camera", "horizontal_shift")):
        h, w = frame.shape[:2]
        progress = i / max(n - 1, 1)
        shift = int(w * max_ratio * progress)
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        out.append(cv2.warpAffine(frame, m, (w, h)))
    return out


def change_background_color(frames: list[np.ndarray], instruction: str) -> list[np.ndarray]:
    color = target_color_bgr(extract_target_color(instruction))
    out: list[np.ndarray] = []
    params_for_progress = {"action": "change_color"}
    for frame in _iter_frames_with_progress(frames, params_for_progress, "change_color", "change_background_color"):
        mask = estimate_foreground_mask(frame)
        bg = np.full_like(frame, color)
        out.append(np.where(mask[:, :, None] > 0, frame, bg).astype(np.uint8))
    return out


def replace_background(frames: list[np.ndarray], params: dict[str, Any], instruction: str) -> list[np.ndarray]:
    blur_background = bool(params.get("blur_background", True))
    color = target_color_bgr(extract_target_color(instruction))
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(frames, params, "replace_background", "replace_background"):
        mask = estimate_foreground_mask(frame)
        if blur_background:
            bg = cv2.GaussianBlur(frame, (21, 21), 0)
        else:
            bg = np.full_like(frame, color)
        out.append(np.where(mask[:, :, None] > 0, frame, bg).astype(np.uint8))
    return out


def inpaint(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    radius = int(params.get("inpaint_radius", 5))
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(frames, params, "remove_object", "inpaint"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        out.append(cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA) if mask.sum() > 0 else frame)
    return out


def stylize(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    blend = float(params.get("blend", 0.3))
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(frames, params, "apply_style", "stylize"):
        smooth = cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)
        out.append(cv2.addWeighted(frame, 1.0 - blend, smooth, blend, 0))
    return out


def blur_or_brightness(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    mode = str(params.get("mode", "blur"))
    if mode == "brightness":
        alpha = float(params.get("alpha", 1.05))
        beta = float(params.get("beta", 8.0))
        out: list[np.ndarray] = []
        for frame in _iter_frames_with_progress(
            frames, params, "add_effect", "blur_or_brightness"
        ):
            out.append(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta))
        return out

    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(
        frames, params, "add_effect", "blur_or_brightness"
    ):
        out.append(cv2.GaussianBlur(frame, (5, 5), 0))
    return out


def sharpness(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    strength = float(params.get("strength", 0.5))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) * strength
    kernel[1, 1] = 1.0 + 8.0 * strength
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(
        frames, params, "enhance_style_details", "sharpness"
    ):
        out.append(cv2.filter2D(frame, -1, kernel))
    return out


def histogram_match(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in _iter_frames_with_progress(frames, params, "match_appearance", "histogram_match"):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        out.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))
    return out


def identity(frames: list[np.ndarray], params: dict[str, Any]) -> list[np.ndarray]:
    return frames


def _resolve_add_object_prompts(
    params: dict[str, Any],
    instruction: str,
) -> tuple[str, str]:
    raw_target = str(params.get("target", instruction or "person"))
    if raw_target in ("", "new_object", "object"):
        target_prompt = "person . animal . object ."
    else:
        target_prompt = raw_target.replace("_", " ") + " ."
    fg_prompt = "person . face . animal . car . object ."
    return target_prompt, fg_prompt


def _compose_shifted_add_object(
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


def load_raft_model(logger: logging.Logger | None = None) -> bool:
    global RAFT_MODEL, RAFT_DEVICE
    if RAFT_MODEL is not None and RAFT_DEVICE is not None:
        return True
    try:
        import torch
        from types import SimpleNamespace

        raft_root = Path("/workspace/third_party/RAFT")
        model_path = raft_root / "models" / "raft-things.pth"
        if not raft_root.exists() or not model_path.exists():
            return False

        if str(raft_root) not in sys.path:
            sys.path.insert(0, str(raft_root))

        from raft.raft import RAFT

        args = SimpleNamespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RAFT(args)

        state = torch.load(str(model_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            # Handle DataParallel checkpoints.
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        RAFT_MODEL = model
        RAFT_DEVICE = device
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"RAFT load failed: {exc}")
        return False


def _estimate_optical_flow(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Estimate dense flow(prev->curr). Prefer RAFT; fallback to Farneback."""
    try:
        import torch
        if load_raft_model(logger=logger):
            raft_root = "/workspace/third_party/RAFT"
            if raft_root not in sys.path:
                sys.path.insert(0, raft_root)
            from raft.utils.utils import InputPadder

            prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            image1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).float()[None].to(RAFT_DEVICE)
            image2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).float()[None].to(RAFT_DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                _flow_low, flow_up = RAFT_MODEL(image1, image2, iters=20, test_mode=True)

            flow = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            return flow
    except Exception as exc:
        if logger is not None:
            logger.debug(f"RAFT flow failed, fallback to Farneback: {exc}")

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def _warp_mask_with_flow(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp prev mask to current frame using flow(prev->curr)."""
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


def _track_mask_with_xmem_or_ostrack(
    prev_mask: np.ndarray,
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Tracking stage hook. Prefer XMem/OSTrack when available; fallback to prev mask."""
    xmem_dir = Path("/workspace/third_party/XMem")
    ostrack_dir = Path("/workspace/third_party/OSTrack")
    if xmem_dir.exists() or ostrack_dir.exists():
        # Integration point: call XMem / OSTrack tracker implementation here.
        return prev_mask
    if logger is not None:
        logger.debug("XMem/OSTrack not found; using fallback tracker")
    return prev_mask


def _temporal_stabilize_mask(
    prev_mask: np.ndarray,
    tracked_mask: np.ndarray,
    flow: np.ndarray,
    smooth_alpha: float,
) -> np.ndarray:
    warped_prev = _warp_mask_with_flow(prev_mask, flow)
    blended = smooth_alpha * tracked_mask.astype(np.float32) + (1.0 - smooth_alpha) * warped_prev.astype(np.float32)
    return (blended > 0.5).astype(np.uint8)


def add_object_frames_ver1(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver1 pipeline:
    1) detect target/foreground bbox(es) only on first frame via GroundingDINO
    2) predict SAM masks only on first frame
    3) reuse those fixed masks for all remaining frames
    4) compose shifted target while avoiding foreground overlap
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
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

    fg_boxes = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
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
    for frame in _iter_frames_with_progress(frames, params, "add_object", "add_object_ver1"):
        out.append(_compose_shifted_add_object(frame, target_box, target_mask, fg_mask))
    return out


def add_object_frames_ver2(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver2 pipeline:
    1) detect target bbox on each frame via GroundingDINO
    2) predict target SAM mask on each frame
    3) detect foreground bbox(es) and predict foreground SAM masks on each frame
    4) compose shifted target with per-frame masks (with previous-frame fallback)
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)

    out: list[np.ndarray] = []
    prev_target_box: tuple[int, int, int, int] | None = None
    prev_target_mask: np.ndarray | None = None
    prev_fg_mask: np.ndarray | None = None

    for frame_idx, frame in enumerate(_iter_frames_with_progress(frames, params, "add_object", "add_object_ver2")):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ============================================================
        # Step 1: GroundingDINO + SAM で target をマスク化 (全フレーム)
        # ============================================================
        target_boxes = _detect_all_boxes(
            frame_rgb, target_prompt, logger=logger
        )
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

        # ============================================================
        # Step 2: GroundingDINO + SAM で前衛全体をマスク化 (全フレーム)
        # ============================================================
        fg_boxes = _detect_all_boxes(
            frame_rgb, fg_prompt, logger=logger
        )
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

        result = _compose_shifted_add_object(
            frame,
            (tx1, ty1, tx2, ty2),
            target_mask,
            fg_mask,
        )
        out.append(result)

    return out


def add_object_frames_ver3(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver3 pipeline:
    1) detect target/foreground bbox(es) once from first frame via GroundingDINO
    2) keep bbox(es) fixed for all frames
    3) predict SAM masks on every frame using fixed bbox(es)
    4) compose shifted target with per-frame SAM masks
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    # Fixed target bbox from initial original frame
    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
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

    # Fixed foreground bboxes from initial original frame
    fg_boxes0 = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
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

    for frame in _iter_frames_with_progress(frames, params, "add_object", "add_object_ver3"):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Per-frame SAM using fixed target bbox
        target_mask = get_sam_mask_from_box(
            frame_rgb,
            [target_box[0], target_box[1], target_box[2], target_box[3]],
            logger=logger,
        )
        if target_mask is None and prev_target_mask is not None:
            target_mask = prev_target_mask
        prev_target_mask = target_mask

        # Per-frame SAM using fixed foreground bboxes
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

        result = _compose_shifted_add_object(
            frame,
            target_box,
            target_mask,
            fg_mask,
        )
        out.append(result)

    return out


def add_object_frames_ver4(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver4 pipeline:
    1) detect target/foreground bbox(es) on first frame via GroundingDINO
    2) build initial SAM masks on first frame
    3) propagate masks with tracker hook (XMem/OSTrack) per frame
    4) estimate optical flow with RAFT (fallback: Farneback)
    5) temporally stabilize masks via flow warp + smoothing
    6) compose shifted target using stabilized masks
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
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

    fg_boxes0 = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        fixed_fg_boxes.append((bx1, by1, bx2, by2))

    # ① initial masks
    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb, [target_box[0], target_box[1], target_box[2], target_box[3]], logger=logger
    )
    prev_fg_mask = np.zeros((h, w), dtype=np.uint8)
    for bx1, by1, bx2, by2 in fixed_fg_boxes:
        m = get_sam_mask_from_box(frame0_rgb, [bx1, by1, bx2, by2], logger=logger)
        prev_fg_mask = np.maximum(prev_fg_mask, m)

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))

    out: list[np.ndarray] = []
    # frame0 edit
    out.append(
        _compose_shifted_add_object(frames[0], target_box, prev_target_mask, prev_fg_mask)
    )

    for i in _iter_frames_with_progress(range(1, len(frames)), params, "add_object", "add_object_ver4"):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        # ② mask tracking (XMem/OSTrack hook)
        tracked_target = _track_mask_with_xmem_or_ostrack(prev_target_mask, prev_frame, curr_frame, logger=logger)
        tracked_fg = _track_mask_with_xmem_or_ostrack(prev_fg_mask, prev_frame, curr_frame, logger=logger)

        # ③ optical flow (RAFT)
        flow = _estimate_optical_flow(prev_frame, curr_frame, logger=logger)

        # ⑤ temporal correction (flow-warp + smoothing)
        stable_target = _temporal_stabilize_mask(prev_target_mask, tracked_target, flow, smooth_alpha)
        stable_fg = _temporal_stabilize_mask(prev_fg_mask, tracked_fg, flow, smooth_alpha)

        # ④ edit processing (OpenCV compose)
        edited = _compose_shifted_add_object(curr_frame, target_box, stable_target, stable_fg)
        out.append(edited)

        prev_target_mask = stable_target
        prev_fg_mask = stable_fg

    return out


def add_object_frames_ver5(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver5 pipeline:
    1) detect fixed target/foreground bbox(es) from first frame via GroundingDINO
    2) predict per-frame SAM masks using fixed bbox(es)
    3) propagate previous masks via tracker hook + RAFT-based temporal stabilization
    4) fuse SAM masks and stabilized masks
    5) compose shifted target with fused masks
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)

    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
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

    fg_boxes0 = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
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
        _compose_shifted_add_object(frames[0], target_box, prev_target_mask, prev_fg_mask)
    )

    for i in _iter_frames_with_progress(range(1, len(frames)), params, "add_object", "add_object_ver5"):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        # Per-frame SAM masks from fixed bbox(es).
        sam_target = get_sam_mask_from_box(
            curr_rgb,
            [target_box[0], target_box[1], target_box[2], target_box[3]],
            logger=logger,
        )
        sam_fg = np.zeros((h, w), dtype=np.uint8)
        for bx1, by1, bx2, by2 in fixed_fg_boxes:
            m = get_sam_mask_from_box(curr_rgb, [bx1, by1, bx2, by2], logger=logger)
            sam_fg = np.maximum(sam_fg, m)

        tracked_target = _track_mask_with_xmem_or_ostrack(
            prev_target_mask, prev_frame, curr_frame, logger=logger
        )
        tracked_fg = _track_mask_with_xmem_or_ostrack(
            prev_fg_mask, prev_frame, curr_frame, logger=logger
        )

        flow = _estimate_optical_flow(prev_frame, curr_frame, logger=logger)

        stable_target = _temporal_stabilize_mask(
            prev_target_mask, tracked_target, flow, smooth_alpha
        )
        stable_fg = _temporal_stabilize_mask(
            prev_fg_mask, tracked_fg, flow, smooth_alpha
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

        edited = _compose_shifted_add_object(
            curr_frame,
            target_box,
            curr_target_mask,
            curr_fg_mask,
        )
        out.append(edited)

        prev_target_mask = curr_target_mask
        prev_fg_mask = curr_fg_mask

    return out


def add_object_frames(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Compatibility wrapper for add_object; choose version with params['add_object_version'].
    - ver1: first-frame masks only
    - ver2: per-frame masks (default)
    - ver3: fixed initial-frame bbox + per-frame SAM masks
    - ver4: fixed initial-frame bbox + tracking + RAFT + temporal stabilization
    - ver5: ver4 + per-frame SAM fusion
    """
    version = str(params.get("add_object_version", "ver2")).lower()
    if version in {"ver1", "1", "first", "first_frame"}:
        return add_object_frames_ver1(frames, params, instruction, logger)
    if version in {"ver3", "3", "fixed_bbox"}:
        return add_object_frames_ver3(frames, params, instruction, logger)
    if version in {"ver4", "4", "tracked"}:
        return add_object_frames_ver4(frames, params, instruction, logger)
    if version in {"ver5", "5", "tracked_sam_fusion", "hybrid"}:
        return add_object_frames_ver5(frames, params, instruction, logger)
    return add_object_frames_ver2(frames, params, instruction, logger)


def run_method(
    method: str,
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    # NOTE: action/constraints are carried in params by caller-side rule assembly.
    action = str(params.get("action", params.get("_action", "")))
    raw_constraints = params.get("constraints", params.get("_constraints", []))
    constraints: set[str] = set()
    if isinstance(raw_constraints, list):
        constraints = {str(c) for c in raw_constraints}

    # TODO branches for unsupported actions (currently routed to identity in rules).
    if action == "add_object":
        # GroundingDINO+SAM で target マスク化 →
        # 前衛マスク化 → 50%ずらして合成
        return add_object_frames(frames, params, instruction, logger)
    if action == "replace_object":
        print(f"{action} ---> pass throught")
        return frames
    if action == "edit_motion":
        print(f"{action} ---> pass throught")
        return frames
    if action == "edit_expression":
        print(f"{action} ---> pass throught")
        return frames
    if action == "align_replacement":
        print(f"{action} ---> pass throught")
        return frames
    if action == "refine_mask":
        print(f"{action} ---> pass throught")
        return frames
    if action == "blend_instances":
        print(f"{action} ---> pass throught")
        return frames
    if action == "track_effect":
        print(f"{action} ---> pass throught")
        return frames
    if action == "stabilize_object":
        print(f"{action} ---> pass throught")
        return frames

    # TODO branches for unsupported/constraint-only requirements.
    if "unchanged" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "unchanged_identity" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "natural_motion" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "temporal_consistency" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "no_flicker" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "clean_edges" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "no_artifacts" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "no_color_bleeding" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "smooth_motion" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "continuous_motion" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "no_jitter" in constraints:
        print(f"{action} ---> pass throught")
        return frames
    if "no_distortion" in constraints:
        print(f"{action} ---> pass throught")
        return frames

    if method == "crop_resize":
        # Tool: stable_zoom_in (GroundingDINO + OpenCV)
        # action: zoom_in
        # constraints (typical): smooth_motion
        return stable_zoom_in(frames, params, logger)
    if method == "resize_pad":
        # Tool: zoom_out (OpenCV resize/pad)
        # action: zoom_out
        # constraints (typical): smooth_motion
        return zoom_out(frames, params)
    if method == "progressive_crop_resize":
        # Tool: stable_zoom_in (GroundingDINO + OpenCV)
        # action: dolly_in
        # constraints (typical): smooth_motion
        return stable_zoom_in(frames, params, logger)
    if method == "progressive_resize_pad":
        # Tool: zoom_out (OpenCV resize/pad)
        # action: dolly_out
        # constraints (typical): smooth_motion
        return zoom_out(frames, params)
    if method == "perspective_warp":
        # Tool: perspective_warp (OpenCV perspective transform)
        # action: change_camera_angle, adjust_perspective
        # constraints (typical): keep perspective natural / no_distortion
        return perspective_warp(frames, params)
    if method == "horizontal_shift":
        # Tool: horizontal_shift (OpenCV affine transform)
        # action: orbit_camera
        # constraints (typical): smooth_motion
        return horizontal_shift(frames, params)
    if method == "hsv_retarget":
        # Tool: change_background_color (GrabCut + OpenCV color fill)
        # action: change_color
        # constraints (typical): no_color_bleeding
        return change_background_color(frames, instruction)
    if method == "segment_and_replace":
        # Tool: replace_background (GrabCut + OpenCV blur/fill)
        # action: replace_background
        # constraints (typical): clean_edges, no_flicker, temporal_consistency
        return replace_background(frames, params, instruction)
    if method == "opencv_blur":
        # Tool: replace_background with blur_background=True
        # action: replace_background (fallback)
        # constraints (typical): clean_edges, no_flicker
        return replace_background(frames, {"blur_background": True}, instruction)
    if method == "inpaint":
        # Tool: inpaint (OpenCV inpainting)
        # action: remove_object, inpaint_background
        # constraints (typical): no_artifacts, temporal_consistency
        return inpaint(frames, params)
    if method == "stylize":
        # Tool: stylize (OpenCV bilateral filter blend)
        # action: apply_style
        # constraints (typical): temporal_consistency
        return stylize(frames, params)
    if method == "blur_or_brightness":
        # Tool: blur_or_brightness (OpenCV blur/brightness)
        # action: add_effect
        # constraints (typical): no_flicker
        return blur_or_brightness(frames, params)
    if method == "sharpness":
        # Tool: sharpness (OpenCV filter2D sharpen kernel)
        # action: enhance_style_details
        # constraints (typical): temporal_consistency
        return sharpness(frames, params)
    if method == "histogram_match":
        # Tool: histogram_match (OpenCV histogram equalization)
        # action: match_appearance, match_lighting
        # constraints (typical): unchanged, appearance_match
        return histogram_match(frames, params)
    # Tool: identity (no-op fallback)
    # action: preserve_*, stabilize_*, add_object, replace_object, edit_motion,
    #         edit_expression, refine_mask, blend_instances, track_effect, and
    #         undefined actions via _default rule.
    # constraints (typical): unchanged, unchanged_identity,
    #         temporal_consistency, natural_motion, no_flicker, clean_edges
    return identity(frames, params)
