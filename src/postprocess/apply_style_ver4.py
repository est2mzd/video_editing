import threading

import cv2
import numpy as np
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image

# ============================================
# Settings
# ============================================
NUM_STEPS = 25
GUIDANCE = 7.0
STRENGTH = 0.5
MASK_PHOTO_THRESHOLD = 18
MASK_FB_THRESHOLD = 1.5
MIN_REGEN_RATIO = 0.01

# ============================================
# Style
# ============================================
APPLY_STYLES = [
    "ukiyo-e",
    "ghibli",
    "pixel_art",
    "anime",
    "cyberpunk",
    "watercolor",
    "oil_painting",
    "american_comic",
]

_STYLE_ALIASES = {
    "cyberpunk": ["cyberpunk", "cyber", "neon"],
    "pixel_art": ["pixel", "pixelart", "8bit"],
    "american_comic": ["comic", "americancomic", "cartooncomic"],
    "anime": ["anime", "japanimation"],
    "ghibli": ["ghibli", "miyazaki", "studio ghibli"],
    "watercolor": ["watercolor", "watercolour"],
    "oil_painting": ["oil", "oilpainting", "oil_painting", "oil paint"],
    "ukiyo-e": ["ukiyoe", "ukiyo", "japaneseprint"],
}

_PIPE = None
_LOCK = threading.Lock()


def normalize_style(style: str) -> str:
    s = style.lower().replace("-", "").replace("_", "").strip()
    for canonical, aliases in _STYLE_ALIASES.items():
        for alias in aliases:
            if alias in s:
                return canonical
    return style.lower()


def get_prompt(style: str) -> str:
    canonical = normalize_style(style)
    return f"apply style of {canonical.replace('_', ' ')}"


def get_pipe() -> StableDiffusionImg2ImgPipeline:
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    with _LOCK:
        if _PIPE is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
            ).to(device)

            pipe.load_lora_weights(
                "/workspace/weights/lora",
                weight_name="anime.safetensors",
            )
            pipe.set_adapters(["default_0"], adapter_weights=[0.8])

            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.enable_attention_slicing()

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("xformers not available, skip")

            # Keep generating all regions; caller handles blending by mask.
            pipe.safety_checker = None
            pipe.requires_safety_checker = False

            _PIPE = pipe

    return _PIPE


def _stylize_img2img(frame_bgr: np.ndarray, prompt: str) -> np.ndarray:
    pipe = get_pipe()
    image = Image.fromarray(frame_bgr[:, :, ::-1])

    out = pipe(
        prompt=prompt,
        image=image,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
    ).images[0]

    return np.array(out)[:, :, ::-1]


def _calc_flow_farneback(
    src_bgr: np.ndarray,
    dst_bgr: np.ndarray,
) -> np.ndarray:
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        src_gray,
        dst_gray,
        None,
        0.5,
        3,
        21,
        5,
        7,
        1.5,
        0,
    )


def _warp_with_flow(image_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + flow[:, :, 0]).astype(np.float32)
    map_y = (yy + flow[:, :, 1]).astype(np.float32)
    return cv2.remap(
        image_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _sample_flow(
    flow: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    sampled_x = cv2.remap(
        flow[:, :, 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_y = cv2.remap(
        flow[:, :, 1],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.stack([sampled_x, sampled_y], axis=-1)


def _detect_breakdown_mask(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fwd: np.ndarray,
) -> np.ndarray:
    h, w = frame_t_bgr.shape[:2]

    warped_gray_t = cv2.cvtColor(
        _warp_with_flow(frame_t_bgr, flow_fwd),
        cv2.COLOR_BGR2GRAY,
    )
    gray_tp1 = cv2.cvtColor(frame_tp1_bgr, cv2.COLOR_BGR2GRAY)
    photo_error = cv2.absdiff(warped_gray_t, gray_tp1)
    mask_photo = (photo_error > MASK_PHOTO_THRESHOLD).astype(np.uint8)

    flow_bwd = _calc_flow_farneback(frame_tp1_bgr, frame_t_bgr)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + flow_fwd[:, :, 0]).astype(np.float32)
    map_y = (yy + flow_fwd[:, :, 1]).astype(np.float32)
    sampled_bwd = _sample_flow(flow_bwd, map_x, map_y)
    fb_error = np.linalg.norm(flow_fwd + sampled_bwd, axis=2)
    mask_fb = (fb_error > MASK_FB_THRESHOLD).astype(np.uint8)

    valid = (
        (map_x >= 0)
        & (map_x < w)
        & (map_y >= 0)
        & (map_y < h)
    ).astype(np.uint8)

    mask = ((mask_photo | mask_fb) & valid).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _blend_regenerated(
    warped_bgr: np.ndarray,
    regenerated_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha3 = alpha[:, :, None]
    blended = regenerated_bgr.astype(np.float32) * alpha3
    blended += warped_bgr.astype(np.float32) * (1.0 - alpha3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_style_frame(frame, style):
    prompt = get_prompt(style)
    return _stylize_img2img(frame, prompt)


def apply_style_frames(frames, style):
    if not frames:
        return []

    prompt = get_prompt(style)
    outputs = []

    # frame0 -> LoRA -> stylized_0
    stylized_t = _stylize_img2img(frames[0], prompt)
    outputs.append(stylized_t)

    for idx in range(len(frames) - 1):
        frame_t = frames[idx]
        frame_tp1 = frames[idx + 1]

        # flow(t->t+1)
        flow_fwd = _calc_flow_farneback(frame_t, frame_tp1)

        # warp(stylized_t)
        warped = _warp_with_flow(stylized_t, flow_fwd)

        # マスクで破綻領域検出
        mask = _detect_breakdown_mask(frame_t, frame_tp1, flow_fwd)
        regen_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if regen_ratio >= MIN_REGEN_RATIO:
            # そこだけ diffusion 再生成（全体生成後、mask 部分のみ採用）
            regenerated = _stylize_img2img(frame_tp1, prompt)
            stylized_tp1 = _blend_regenerated(warped, regenerated, mask)
        else:
            stylized_tp1 = warped

        outputs.append(stylized_tp1)
        stylized_t = stylized_tp1

    return outputs
