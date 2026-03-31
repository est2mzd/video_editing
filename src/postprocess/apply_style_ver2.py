import threading

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# test_lora.py と同じスタイル定義
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
    "oil_painting": ["oil", "oilpainting", "oil_painting", "oil paint", "oil_paint"],
    "ukiyo-e": ["ukiyoe", "ukiyo", "japaneseprint"],
}

_PIPE = None
_PIPE_LOCK = threading.Lock()


def normalize_style(style: str) -> str:
    s = style.lower().replace("-", "").replace("_", "").strip()

    for canonical, aliases in _STYLE_ALIASES.items():
        for alias in aliases:
            if alias in s:
                return canonical

    return style.strip().lower()


def build_style_prompts(styles=None):
    style_list = APPLY_STYLES if styles is None else styles
    return {
        style: f"apply style of {style.replace('_', ' ')}"
        for style in style_list
    }


def get_style_prompt(style: str) -> str:
    canonical = normalize_style(style)
    return f"apply style of {canonical.replace('_', ' ')}"


def _get_pipe() -> StableDiffusionImg2ImgPipeline:
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
            )
            pipe = pipe.to(device)
            pipe.load_lora_weights(
                "/workspace/weights/lora",
                weight_name="anime.safetensors",
            )
            pipe.set_adapters(["default_0"], adapter_weights=[0.8])
            _PIPE = pipe

    return _PIPE


def stylize(frame: np.ndarray, prompt: str) -> np.ndarray:
    pipe = _get_pipe()

    # OpenCV(BGR)前提の配列を RGB に変換して PIL へ渡す。
    rgb_frame = frame[:, :, ::-1]
    image = Image.fromarray(rgb_frame)

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.5,
        guidance_scale=7.5,
    ).images[0]

    # 推論結果を BGR で返す。
    result_rgb = np.array(result)
    return result_rgb[:, :, ::-1]


def apply_style_frame(frame, style):
    prompt = get_style_prompt(style)
    return stylize(frame, prompt)


def apply_style_frames(frames, style):
    prompt = get_style_prompt(style)
    return [stylize(frame, prompt) for frame in frames]
