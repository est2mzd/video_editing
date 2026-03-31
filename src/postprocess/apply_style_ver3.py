import threading
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# ============================================
# 設定
# ============================================
BATCH_SIZE = 1
NUM_STEPS = 25
GUIDANCE = 7.0
STRENGTH = 0.5

# ============================================
# style
# ============================================
APPLY_STYLES = [
    "ukiyo-e", "ghibli", "pixel_art", "anime",
    "cyberpunk", "watercolor", "oil_painting", "american_comic",
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


# ============================================
# style normalize
# ============================================
def normalize_style(style: str) -> str:
    s = style.lower().replace("-", "").replace("_", "").strip()
    for canonical, aliases in _STYLE_ALIASES.items():
        for alias in aliases:
            if alias in s:
                return canonical
    return style.lower()


def get_prompt(style: str) -> str:
    style = normalize_style(style)
    return f"apply style of {style.replace('_', ' ')}"


# ============================================
# pipeline
# ============================================
def get_pipe():
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

            # LoRA
            pipe.load_lora_weights(
                "/workspace/weights/lora",
                weight_name="anime.safetensors",
            )
            pipe.set_adapters(["default_0"], adapter_weights=[0.8])

            # 高速化
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.enable_attention_slicing()
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, skip")

            # safety無効
            pipe.safety_checker = None
            pipe.requires_safety_checker = False

            _PIPE = pipe

    return _PIPE


# ============================================
# batch stylize
# ============================================
def stylize_batch(frames, prompt, batch_size=BATCH_SIZE):
    pipe = get_pipe()
    results = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        images = [
            Image.fromarray(f[:, :, ::-1]) for f in batch
        ]

        outputs = pipe(
            prompt=[prompt] * len(images),
            image=images,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=NUM_STEPS,
        ).images

        for out in outputs:
            arr = np.array(out)[:, :, ::-1]
            results.append(arr)

    return results


# ============================================
# public API
# ============================================
def apply_style_frames(frames, style):
    prompt = get_prompt(style)
    return stylize_batch(frames, prompt)