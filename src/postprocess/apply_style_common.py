from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image

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

STYLE_ALIASES = {
    "cyberpunk": ["cyberpunk", "cyber", "neon"],
    "pixel_art": ["pixel", "pixelart", "8bit"],
    "american_comic": ["comic", "americancomic", "cartooncomic"],
    "anime": ["anime", "japanimation"],
    "ghibli": ["ghibli", "miyazaki", "studio ghibli"],
    "watercolor": ["watercolor", "watercolour"],
    "oil_painting": [
        "oil",
        "oilpainting",
        "oil_painting",
        "oil paint",
        "oil_paint",
    ],
    "ukiyo-e": ["ukiyoe", "ukiyo", "japaneseprint"],
}

_PIPE: StableDiffusionImg2ImgPipeline | None = None
_PIPE_LOCK = threading.Lock()


def normalize_style_alias(
    style: str,
    aliases: dict[str, list[str]] | None = None,
    fallback: str = "lower",
) -> str:
    """Normalize free-form style text using alias dictionary."""
    alias_map = STYLE_ALIASES if aliases is None else aliases
    lowered = style.lower().replace("-", "").replace("_", "").strip()

    for canonical, values in alias_map.items():
        for alias in values:
            if alias in lowered:
                return canonical

    if fallback == "cleaned":
        return lowered
    if fallback == "stripped":
        return style.strip()
    return style.strip().lower()


def build_style_prompt(style: str) -> str:
    """Build canonical img2img prompt text for one style."""
    return f"apply style of {style.replace('_', ' ')}"


def get_img2img_pipe(
    adapter_weight: float = 0.8,
) -> StableDiffusionImg2ImgPipeline:
    """Get or initialize shared Stable Diffusion img2img pipeline."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    with _PIPE_LOCK:
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
            pipe.set_adapters(["default_0"], adapter_weights=[adapter_weight])

            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.enable_attention_slicing()

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            _PIPE = pipe

    return _PIPE


def run_img2img(
    pipe: StableDiffusionImg2ImgPipeline,
    frame_bgr: np.ndarray,
    prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int | None = None,
) -> np.ndarray:
    """Run one img2img inference from BGR frame to BGR frame."""
    image = Image.fromarray(frame_bgr[:, :, ::-1])
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": image,
        "strength": strength,
        "guidance_scale": guidance_scale,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps

    out = pipe(**kwargs).images[0]
    return np.array(out)[:, :, ::-1]


def run_img2img_batch(
    pipe: StableDiffusionImg2ImgPipeline,
    frames_bgr: list[np.ndarray],
    prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
) -> list[np.ndarray]:
    """Run batched img2img inference from BGR frames."""
    images = [Image.fromarray(frame[:, :, ::-1]) for frame in frames_bgr]
    outputs = pipe(
        prompt=[prompt] * len(images),
        image=images,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images
    return [np.array(out)[:, :, ::-1] for out in outputs]


def calc_flow_farneback(
    src_bgr: np.ndarray,
    dst_bgr: np.ndarray,
) -> np.ndarray:
    """Compute dense Farneback optical flow (src -> dst)."""
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


def warp_with_flow(image_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image pixels according to dense flow."""
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


def sample_flow(
    flow: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Sample flow vectors at remapped coordinates."""
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


def detect_breakdown_mask(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fwd: np.ndarray,
    photo_threshold: float,
    fb_threshold: float,
) -> np.ndarray:
    """Detect unstable temporal regions using photo and FB consistency."""
    h, w = frame_t_bgr.shape[:2]
    warped_gray_t = cv2.cvtColor(
        warp_with_flow(frame_t_bgr, flow_fwd),
        cv2.COLOR_BGR2GRAY,
    )
    gray_tp1 = cv2.cvtColor(frame_tp1_bgr, cv2.COLOR_BGR2GRAY)
    photo_error = cv2.absdiff(warped_gray_t, gray_tp1)
    mask_photo = (photo_error > photo_threshold).astype(np.uint8)

    flow_bwd = calc_flow_farneback(frame_tp1_bgr, frame_t_bgr)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + flow_fwd[:, :, 0]).astype(np.float32)
    map_y = (yy + flow_fwd[:, :, 1]).astype(np.float32)
    sampled_bwd = sample_flow(flow_bwd, map_x, map_y)
    fb_error = np.linalg.norm(flow_fwd + sampled_bwd, axis=2)
    mask_fb = (fb_error > fb_threshold).astype(np.uint8)

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


def blend_regenerated(
    warped_bgr: np.ndarray,
    regenerated_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Feather-blend regenerated regions into warped frame."""
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha3 = alpha[:, :, None]
    blended = regenerated_bgr.astype(np.float32) * alpha3
    blended += warped_bgr.astype(np.float32) * (1.0 - alpha3)
    return np.clip(blended, 0, 255).astype(np.uint8)
