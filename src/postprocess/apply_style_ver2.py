import threading

import numpy as np

from .apply_style_common import (
    APPLY_STYLES,
    STYLE_ALIASES,
    build_style_prompt,
    get_img2img_pipe,
    normalize_style_alias,
    run_img2img,
)

_STYLE_ALIASES = dict(STYLE_ALIASES)

_PIPE = None
_PIPE_LOCK = threading.Lock()


def normalize_style(style: str) -> str:
    """Normalize free-form style text to canonical style key.

    Tools:
    - Python string cleanup and alias matching table.

    Steps:
    1. Lowercase and strip separators.
    2. Resolve aliases to canonical names.
    3. Return cleaned fallback when unresolved.
    """
    return normalize_style_alias(
        style,
        aliases=_STYLE_ALIASES,
        fallback="lower",
    )


def build_style_prompts(styles=None):
    """Build prompt templates for all supported styles.

    Tools:
    - Python dictionary comprehension.

    Steps:
    1. Select default or provided style list.
    2. Generate text prompt per style for img2img.
    3. Return style-to-prompt mapping.
    """
    style_list = APPLY_STYLES if styles is None else styles
    return {
        style: f"apply style of {style.replace('_', ' ')}"
        for style in style_list
    }


def get_style_prompt(style: str) -> str:
    """Create a single diffusion prompt from user style input.

    Tools:
    - normalize_style and formatted prompt string.

    Steps:
    1. Normalize the style identifier.
    2. Expand underscores for human-readable prompt text.
    3. Return prompt consumed by Stable Diffusion.
    """
    canonical = normalize_style(style)
    return build_style_prompt(canonical)


def _get_pipe():
    """Lazily initialize and cache the Stable Diffusion img2img pipeline.

    Tools:
    - Diffusers StableDiffusionImg2ImgPipeline.
    - Torch device/dtype selection.
    - LoRA loading for anime style weights.

    Steps:
    1. Reuse existing global pipeline if available.
    2. Initialize model on CUDA/CPU with suitable dtype.
    3. Load LoRA weights and adapter strength.
    4. Cache and return singleton pipeline.
    """
    global _PIPE
    if _PIPE is None:
        with _PIPE_LOCK:
            if _PIPE is None:
                _PIPE = get_img2img_pipe(adapter_weight=0.8)
    return _PIPE


def stylize(frame: np.ndarray, prompt: str) -> np.ndarray:
    """Stylize one BGR frame using Stable Diffusion img2img.

    Tools:
    - Diffusers img2img inference.
    - PIL for NumPy image conversion.

    Steps:
    1. Convert OpenCV BGR frame to RGB PIL image.
    2. Run img2img with fixed strength and guidance.
    3. Convert generated image back to BGR NumPy array.
    """
    pipe = _get_pipe()
    return run_img2img(
        pipe,
        frame_bgr=frame,
        prompt=prompt,
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=None,
    )


def apply_style_frame(frame, style):
    """Apply one style to one frame through diffusion prompt routing.

    Tools:
    - get_style_prompt and stylize.

    Steps:
    1. Build style prompt from input token.
    2. Execute img2img inference for a single frame.
    3. Return stylized frame.
    """
    prompt = get_style_prompt(style)
    return stylize(frame, prompt)


def apply_style_frames(frames, style):
    """Apply diffusion style transfer to all frames with shared prompt.

    Tools:
    - Stable Diffusion img2img per frame.

    Steps:
    1. Build one canonical prompt for the sequence.
    2. Iterate through frames in order.
    3. Stylize each frame and collect outputs.
    """
    prompt = get_style_prompt(style)
    return [stylize(frame, prompt) for frame in frames]
