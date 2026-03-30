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
XMEM_NETWORK: Any = None
XMEM_DEVICE: str | None = None
XMEM_IMAGE_TO_TORCH: Any = None
XMEMInferenceCore: Any = None


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


def _find_xmem_model_path(params: dict[str, Any] | None = None) -> Path | None:
    candidates: list[Path] = []
    if params is not None:
        explicit = params.get("xmem_model_path")
        if explicit:
            candidates.append(Path(str(explicit)))
    candidates.extend([
        Path("/workspace/third_party/XMem/saves/XMem.pth"),
        Path("/workspace/weights/XMem.pth"),
        Path("/workspace/weights/xmem.pth"),
        Path("/workspace/models/XMem.pth"),
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


def load_xmem_model(
    params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    global XMEM_NETWORK, XMEM_DEVICE, XMEM_IMAGE_TO_TORCH, XMEMInferenceCore
    if XMEM_NETWORK is not None and XMEM_DEVICE is not None and XMEM_IMAGE_TO_TORCH is not None and XMEMInferenceCore is not None:
        return True
    try:
        import torch

        xmem_root = Path("/workspace/third_party/XMem")
        model_path = _find_xmem_model_path(params=params)
        if not xmem_root.exists() or model_path is None:
            if logger is not None:
                logger.debug("XMem repo or weight file not found")
            return False

        if str(xmem_root) not in sys.path:
            sys.path.insert(0, str(xmem_root))

        from model.network import XMem as XMemNetwork
        from inference.inference_core import InferenceCore
        from inference.interact.interactive_utils import image_to_torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {
            "model": str(model_path),
            "top_k": int((params or {}).get("xmem_top_k", 30)),
            "mem_every": int((params or {}).get("xmem_mem_every", 5)),
            "deep_update_every": int((params or {}).get("xmem_deep_update_every", -1)),
            "enable_long_term": bool((params or {}).get("xmem_enable_long_term", True)),
            "enable_long_term_count_usage": bool((params or {}).get("xmem_enable_long_term_count_usage", True)),
            "max_mid_term_frames": int((params or {}).get("xmem_max_mid_term_frames", 10)),
            "min_mid_term_frames": int((params or {}).get("xmem_min_mid_term_frames", 5)),
            "num_prototypes": int((params or {}).get("xmem_num_prototypes", 128)),
            "max_long_term_elements": int((params or {}).get("xmem_max_long_term_elements", 10000)),
            "single_object": True,
        }
        network = XMemNetwork(config, model_path=str(model_path), map_location=device)
        network.to(device)
        network.eval()

        XMEM_NETWORK = network
        XMEM_DEVICE = device
        XMEM_IMAGE_TO_TORCH = image_to_torch
        XMEMInferenceCore = InferenceCore
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem load failed: {exc}")
        return False


def _make_xmem_processor(
    first_bgr: np.ndarray,
    first_mask: np.ndarray,
    params: dict[str, Any],
    logger: logging.Logger | None = None,
):
    if not load_xmem_model(params=params, logger=logger):
        return None
    try:
        import torch

        config = {
            "top_k": int(params.get("xmem_top_k", 30)),
            "mem_every": int(params.get("xmem_mem_every", 5)),
            "deep_update_every": int(params.get("xmem_deep_update_every", -1)),
            "enable_long_term": bool(params.get("xmem_enable_long_term", True)),
            "enable_long_term_count_usage": bool(params.get("xmem_enable_long_term_count_usage", True)),
            "max_mid_term_frames": int(params.get("xmem_max_mid_term_frames", 10)),
            "min_mid_term_frames": int(params.get("xmem_min_mid_term_frames", 5)),
            "num_prototypes": int(params.get("xmem_num_prototypes", 128)),
            "max_long_term_elements": int(params.get("xmem_max_long_term_elements", 10000)),
        }
        processor = XMEMInferenceCore(XMEM_NETWORK, config)
        processor.set_all_labels([1])

        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        image_t, _ = XMEM_IMAGE_TO_TORCH(first_rgb, device=XMEM_DEVICE)
        mask_t = torch.from_numpy((first_mask > 0).astype(np.float32)).unsqueeze(0).to(XMEM_DEVICE)
        _ = processor.step(image_t, mask_t)
        return processor
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem processor init failed: {exc}")
        return None


def _xmem_predict_mask(
    processor,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray | None:
    if processor is None:
        return None
    try:
        import torch

        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        image_t, _ = XMEM_IMAGE_TO_TORCH(curr_rgb, device=XMEM_DEVICE)
        with torch.no_grad():
            prob = processor.step(image_t)
        if prob is None:
            return None
        if hasattr(prob, "detach"):
            prob_t = prob.detach()
            if prob_t.ndim != 3 or prob_t.shape[0] < 2:
                return None
            mask = torch.argmax(prob_t, dim=0).cpu().numpy().astype(np.uint8)
            return (mask == 1).astype(np.uint8)
        return None
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem predict failed: {exc}")
        return None


def _track_mask_with_xmem_or_ostrack(
    prev_mask: np.ndarray,
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Compatibility wrapper. Real XMem tracking is handled in ver6."""
    xmem_dir = Path("/workspace/third_party/XMem")
    ostrack_dir = Path("/workspace/third_party/OSTrack")
    if xmem_dir.exists() or ostrack_dir.exists():
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



def _mask_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    inter = int((a_bin & b_bin).sum())
    union = int((a_bin | b_bin).sum())
    if union == 0:
        return 0.0
    return inter / union


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def _refine_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _keep_largest_component(mask_u8).astype(np.uint8)


def _mask_to_box(mask: np.ndarray, fallback_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
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


def _clip_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return (x1, y1, x2, y2)


def _expand_box(
    box: tuple[int, int, int, int],
    w: int,
    h: int,
    scale: float = 1.15,
    min_margin: int = 6,
) -> tuple[int, int, int, int]:
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
    return _clip_box((nx1, ny1, nx2, ny2), w, h)


def _fuse_masks_adaptive(
    sam_mask: np.ndarray,
    stable_mask: np.ndarray,
    prev_mask: np.ndarray,
    sam_blend_alpha: float = 0.6,
    iou_switch: float = 0.35,
    area_ratio_limit: float = 2.5,
) -> np.ndarray:
    sam_mask = (sam_mask > 0).astype(np.uint8)
    stable_mask = (stable_mask > 0).astype(np.uint8)
    prev_mask = (prev_mask > 0).astype(np.uint8)

    sam_area = _mask_area(sam_mask)
    stable_area = _mask_area(stable_mask)
    prev_area = max(1, _mask_area(prev_mask))

    if sam_area == 0 and stable_area == 0:
        return prev_mask.copy()
    if sam_area == 0:
        return _refine_mask(stable_mask)
    if stable_area == 0:
        return _refine_mask(sam_mask)

    iou = _mask_iou(sam_mask, stable_mask)
    sam_ratio = sam_area / prev_area
    stable_ratio = stable_area / prev_area

    if iou < iou_switch:
        sam_dist = abs(np.log(max(sam_ratio, 1e-6)))
        stable_dist = abs(np.log(max(stable_ratio, 1e-6)))
        chosen = sam_mask if sam_dist <= stable_dist else stable_mask
        return _refine_mask(chosen)

    if sam_ratio > area_ratio_limit or sam_ratio < 1.0 / area_ratio_limit:
        fused = stable_mask
    else:
        fused = (
            sam_blend_alpha * sam_mask.astype(np.float32)
            + (1.0 - sam_blend_alpha) * stable_mask.astype(np.float32)
        )
        fused = (fused > 0.5).astype(np.uint8)

    return _refine_mask(fused)


def _build_fg_mask_from_boxes(
    frame_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    logger: logging.Logger | None = None,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    for bx1, by1, bx2, by2 in boxes:
        if bx2 <= bx1 or by2 <= by1:
            continue
        m = get_sam_mask_from_box(frame_rgb, [bx1, by1, bx2, by2], logger=logger)
        fg_mask = np.maximum(fg_mask, m.astype(np.uint8))
    return fg_mask


def _derive_dynamic_box_from_masks(
    warped_prev_mask: np.ndarray,
    fallback_box: tuple[int, int, int, int],
    w: int,
    h: int,
    expand_scale: float = 1.15,
) -> tuple[int, int, int, int]:
    raw_box = _mask_to_box(warped_prev_mask, fallback_box)
    return _expand_box(raw_box, w, h, scale=expand_scale)


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



def add_object_frames_ver6(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """add_object ver6 pipeline:
    1) detect initial target/foreground bbox(es) on first frame via GroundingDINO
    2) initialize SAM masks on first frame
    3) initialize XMem processors from the first-frame masks
    4) for each new frame, run XMem + RAFT flow + dynamic SAM
    5) adaptively fuse SAM and temporally stable XMem masks
    6) refine masks and compose shifted object
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)
    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(f"add_object ver6: '{target_prompt}' not detected, passthrough")
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    target_box_init = _clip_box((tx1, ty1, tx2, ty2), w, h)
    if target_box_init[2] <= target_box_init[0] or target_box_init[3] <= target_box_init[1]:
        return frames

    fg_boxes0 = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        cbox = _clip_box((bx1, by1, bx2, by2), w, h)
        if cbox[2] > cbox[0] and cbox[3] > cbox[1]:
            fixed_fg_boxes.append(cbox)

    prev_target_box = target_box_init
    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb,
        [prev_target_box[0], prev_target_box[1], prev_target_box[2], prev_target_box[3]],
        logger=logger,
    ).astype(np.uint8)
    prev_target_mask = _refine_mask(prev_target_mask)

    prev_fg_mask = _build_fg_mask_from_boxes(frame0_rgb, fixed_fg_boxes, logger=logger)
    prev_fg_mask = _refine_mask(prev_fg_mask)

    xmem_target_processor = _make_xmem_processor(frames[0], prev_target_mask, params, logger=logger)
    xmem_fg_processor = _make_xmem_processor(frames[0], prev_fg_mask, params, logger=logger)

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))
    sam_blend_alpha = float(np.clip(params.get("sam_blend_alpha", 0.6), 0.0, 1.0))
    target_expand_scale = float(params.get("target_expand_scale", 1.18))
    fg_expand_scale = float(params.get("fg_expand_scale", 1.10))

    out: list[np.ndarray] = []
    out.append(
        _compose_shifted_add_object(
            frames[0],
            prev_target_box,
            prev_target_mask,
            prev_fg_mask,
        )
    )

    for i in _iter_frames_with_progress(range(1, len(frames)), params, "add_object", "add_object_ver6"):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        flow = _estimate_optical_flow(prev_frame, curr_frame, logger=logger)
        warped_target = _warp_mask_with_flow(prev_target_mask, flow)
        warped_fg = _warp_mask_with_flow(prev_fg_mask, flow)

        xmem_target_mask = _xmem_predict_mask(xmem_target_processor, curr_frame, logger=logger)
        xmem_fg_mask = _xmem_predict_mask(xmem_fg_processor, curr_frame, logger=logger)

        if xmem_target_mask is None:
            xmem_target_mask = warped_target
        if xmem_fg_mask is None:
            xmem_fg_mask = warped_fg

        stable_target = _temporal_stabilize_mask(
            prev_target_mask, xmem_target_mask, flow, smooth_alpha
        )
        stable_fg = _temporal_stabilize_mask(
            prev_fg_mask, xmem_fg_mask, flow, smooth_alpha
        )

        target_box_dyn = _derive_dynamic_box_from_masks(
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
        sam_target = _refine_mask(sam_target)

        fg_box_dyn = _derive_dynamic_box_from_masks(
            warped_fg | stable_fg,
            _mask_to_box(prev_fg_mask, (0, 0, w, h)),
            w,
            h,
            expand_scale=fg_expand_scale,
        )

        sam_fg_dyn = get_sam_mask_from_box(
            curr_rgb,
            [fg_box_dyn[0], fg_box_dyn[1], fg_box_dyn[2], fg_box_dyn[3]],
            logger=logger,
        ).astype(np.uint8)
        sam_fg_fixed = _build_fg_mask_from_boxes(curr_rgb, fixed_fg_boxes, logger=logger)
        sam_fg = np.maximum(sam_fg_dyn, sam_fg_fixed)
        sam_fg = _refine_mask(sam_fg)

        curr_target_mask = _fuse_masks_adaptive(
            sam_mask=sam_target,
            stable_mask=stable_target,
            prev_mask=prev_target_mask,
            sam_blend_alpha=sam_blend_alpha,
        )
        curr_fg_mask = _fuse_masks_adaptive(
            sam_mask=sam_fg,
            stable_mask=stable_fg,
            prev_mask=prev_fg_mask,
            sam_blend_alpha=sam_blend_alpha,
        )

        curr_fg_mask = np.where(curr_target_mask > 0, 0, curr_fg_mask).astype(np.uint8)

        curr_target_box = _mask_to_box(curr_target_mask, target_box_dyn)
        curr_target_box = _expand_box(curr_target_box, w, h, scale=1.05, min_margin=4)

        edited = _compose_shifted_add_object(
            curr_frame,
            curr_target_box,
            curr_target_mask,
            curr_fg_mask,
        )
        out.append(edited)

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
    """add_object ver7 pipeline:
    1) detect initial target/foreground bbox(es) on first frame via GroundingDINO
    2) initialize SAM masks on first frame
    3) initialize one multi-label XMem processor from first-frame target/foreground masks
    4) for each new frame, run XMem + RAFT flow + dynamic SAM
    5) adaptively fuse SAM and temporally stable XMem masks for target/foreground
    6) preserve ver1-style foreground priority during composition
    """
    if not frames:
        return frames

    def _make_xmem_processor_multilabel(
        first_bgr: np.ndarray,
        first_target_mask: np.ndarray,
        first_fg_mask: np.ndarray,
    ):
        if not load_xmem_model(params=params, logger=logger):
            return None
        try:
            import torch

            config = {
                "top_k": int(params.get("xmem_top_k", 30)),
                "mem_every": int(params.get("xmem_mem_every", 5)),
                "deep_update_every": int(params.get("xmem_deep_update_every", -1)),
                "enable_long_term": bool(params.get("xmem_enable_long_term", True)),
                "enable_long_term_count_usage": bool(params.get("xmem_enable_long_term_count_usage", True)),
                "max_mid_term_frames": int(params.get("xmem_max_mid_term_frames", 10)),
                "min_mid_term_frames": int(params.get("xmem_min_mid_term_frames", 5)),
                "num_prototypes": int(params.get("xmem_num_prototypes", 128)),
                "max_long_term_elements": int(params.get("xmem_max_long_term_elements", 10000)),
            }
            processor = XMEMInferenceCore(XMEM_NETWORK, config)
            processor.set_all_labels([1, 2])

            first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
            image_t, _ = XMEM_IMAGE_TO_TORCH(first_rgb, device=XMEM_DEVICE)

            target_bin = (first_target_mask > 0).astype(np.float32)
            fg_bin = (first_fg_mask > 0).astype(np.float32)
            # foreground has priority over target
            target_bin[fg_bin > 0] = 0.0

            mask_t = torch.from_numpy(
                np.stack([target_bin, fg_bin], axis=0)
            ).to(XMEM_DEVICE)
            _ = processor.step(image_t, mask_t)
            return processor
        except Exception as exc:
            logger.debug(f"XMem multi processor init failed: {exc}")
            return None

    def _xmem_predict_multilabel(processor, curr_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if processor is None:
            return None, None
        try:
            import torch

            curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            image_t, _ = XMEM_IMAGE_TO_TORCH(curr_rgb, device=XMEM_DEVICE)
            with torch.no_grad():
                prob = processor.step(image_t)
            if prob is None or not hasattr(prob, "detach"):
                return None, None

            prob_t = prob.detach()
            if prob_t.ndim != 3 or prob_t.shape[0] < 3:
                return None, None

            label_map = torch.argmax(prob_t, dim=0).cpu().numpy().astype(np.uint8)
            target_mask = (label_map == 1).astype(np.uint8)
            fg_mask = (label_map == 2).astype(np.uint8)
            return target_mask, fg_mask
        except Exception as exc:
            logger.debug(f"XMem multi predict failed: {exc}")
            return None, None

    h, w = frames[0].shape[:2]
    target_prompt, fg_prompt = _resolve_add_object_prompts(params, instruction)
    frame0_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    target_boxes = _detect_all_boxes(frame0_rgb, target_prompt, logger=logger)
    if not target_boxes:
        logger.warning(f"add_object ver7: '{target_prompt}' not detected, passthrough")
        return frames

    tx1, ty1, tx2, ty2 = [int(c) for c in target_boxes[0]]
    target_box_init = _clip_box((tx1, ty1, tx2, ty2), w, h)
    if target_box_init[2] <= target_box_init[0] or target_box_init[3] <= target_box_init[1]:
        return frames

    fg_boxes0 = _detect_all_boxes(frame0_rgb, fg_prompt, logger=logger)
    fixed_fg_boxes: list[tuple[int, int, int, int]] = []
    for box in fg_boxes0:
        bx1, by1, bx2, by2 = [int(c) for c in box]
        cbox = _clip_box((bx1, by1, bx2, by2), w, h)
        if cbox[2] > cbox[0] and cbox[3] > cbox[1]:
            fixed_fg_boxes.append(cbox)

    prev_target_box = target_box_init
    prev_target_mask = get_sam_mask_from_box(
        frame0_rgb,
        [prev_target_box[0], prev_target_box[1], prev_target_box[2], prev_target_box[3]],
        logger=logger,
    ).astype(np.uint8)
    prev_target_mask = _refine_mask(prev_target_mask)

    prev_fg_mask = _build_fg_mask_from_boxes(frame0_rgb, fixed_fg_boxes, logger=logger)
    prev_fg_mask = _refine_mask(prev_fg_mask)

    # foreground has priority over target, consistent with ver1 composition
    prev_target_mask = np.where(prev_fg_mask > 0, 0, prev_target_mask).astype(np.uint8)

    xmem_processor = _make_xmem_processor_multilabel(
        frames[0], prev_target_mask, prev_fg_mask
    )

    smooth_alpha = float(params.get("temporal_smooth_alpha", 0.7))
    sam_blend_alpha = float(np.clip(params.get("sam_blend_alpha", 0.6), 0.0, 1.0))
    target_expand_scale = float(params.get("target_expand_scale", 1.18))
    fg_expand_scale = float(params.get("fg_expand_scale", 1.10))

    out: list[np.ndarray] = []
    out.append(
        _compose_shifted_add_object(
            frames[0],
            prev_target_box,
            prev_target_mask,
            prev_fg_mask,
        )
    )

    for i in _iter_frames_with_progress(range(1, len(frames)), params, "add_object", "add_object_ver7"):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        flow = _estimate_optical_flow(prev_frame, curr_frame, logger=logger)
        warped_target = _warp_mask_with_flow(prev_target_mask, flow)
        warped_fg = _warp_mask_with_flow(prev_fg_mask, flow)

        xmem_target_mask, xmem_fg_mask = _xmem_predict_multilabel(
            xmem_processor, curr_frame
        )
        if xmem_target_mask is None:
            xmem_target_mask = warped_target
        if xmem_fg_mask is None:
            xmem_fg_mask = warped_fg

        stable_target = _temporal_stabilize_mask(
            prev_target_mask, xmem_target_mask, flow, smooth_alpha
        )
        stable_fg = _temporal_stabilize_mask(
            prev_fg_mask, xmem_fg_mask, flow, smooth_alpha
        )

        target_box_dyn = _derive_dynamic_box_from_masks(
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
        sam_target = _refine_mask(sam_target)

        fg_box_dyn = _derive_dynamic_box_from_masks(
            warped_fg | stable_fg,
            _mask_to_box(prev_fg_mask, (0, 0, w, h)),
            w,
            h,
            expand_scale=fg_expand_scale,
        )
        sam_fg_dyn = get_sam_mask_from_box(
            curr_rgb,
            [fg_box_dyn[0], fg_box_dyn[1], fg_box_dyn[2], fg_box_dyn[3]],
            logger=logger,
        ).astype(np.uint8)
        sam_fg_fixed = _build_fg_mask_from_boxes(curr_rgb, fixed_fg_boxes, logger=logger)
        sam_fg = np.maximum(sam_fg_dyn, sam_fg_fixed)
        sam_fg = _refine_mask(sam_fg)

        curr_target_mask = _fuse_masks_adaptive(
            sam_mask=sam_target,
            stable_mask=stable_target,
            prev_mask=prev_target_mask,
            sam_blend_alpha=sam_blend_alpha,
        )
        curr_fg_mask = _fuse_masks_adaptive(
            sam_mask=sam_fg,
            stable_mask=stable_fg,
            prev_mask=prev_fg_mask,
            sam_blend_alpha=sam_blend_alpha,
        )

        # keep ver1-style priority: foreground occluder wins over target
        curr_target_mask = np.where(curr_fg_mask > 0, 0, curr_target_mask).astype(np.uint8)

        curr_target_box = _mask_to_box(curr_target_mask, target_box_dyn)
        curr_target_box = _expand_box(curr_target_box, w, h, scale=1.05, min_margin=4)

        edited = _compose_shifted_add_object(
            curr_frame,
            curr_target_box,
            curr_target_mask,
            curr_fg_mask,
        )
        out.append(edited)

        prev_target_box = curr_target_box
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
    - ver6: ver5 + real XMem tracking + dynamic SAM box + adaptive fusion
    - ver7: ver6 + one multi-label XMem processor for target/foreground with ver1-style foreground priority
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
    if version in {"ver6", "6", "xmem", "xmem_hybrid", "tracked_sam_xmem"}:
        return add_object_frames_ver6(frames, params, instruction, logger)
    if version in {"ver7", "7", "xmem_multilabel", "tracked_sam_xmem_multilabel", "foreground_priority"}:
        return add_object_frames_ver7(frames, params, instruction, logger)
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
