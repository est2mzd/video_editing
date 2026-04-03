import threading

from .apply_style_common import (
    STYLE_ALIASES,
    build_style_prompt,
    get_img2img_pipe,
    normalize_style_alias,
    run_img2img_batch,
)

# ============================================
# 設定
# ============================================
BATCH_SIZE = 1
NUM_STEPS = 25
GUIDANCE = 7.0
STRENGTH = 0.5

_STYLE_ALIASES = dict(STYLE_ALIASES)

_PIPE = None
_LOCK = threading.Lock()


# ============================================
# style normalize
# ============================================
def normalize_style(style: str) -> str:
    """Normalize style text to canonical token used by prompt builder.

    Tools:
    - Python alias-table lookup after string cleanup.

    Steps:
    1. Lowercase and remove '-'/'_' separators.
    2. Match aliases against canonical style names.
    3. Return lowercase fallback when unmatched.
    """
    return normalize_style_alias(
        style,
        aliases=_STYLE_ALIASES,
        fallback="lower",
    )


def get_prompt(style: str) -> str:
    """Generate img2img prompt text from a style token.

    Tools:
    - normalize_style and formatted prompt template.

    Steps:
    1. Normalize incoming style alias.
    2. Convert style key to readable phrase.
    3. Return deterministic text prompt.
    """
    style = normalize_style(style)
    return build_style_prompt(style)


# ============================================
# pipeline
# ============================================
def get_pipe():
    """Create and cache a tuned Stable Diffusion img2img pipeline.

    Tools:
    - Diffusers StableDiffusionImg2ImgPipeline + Euler scheduler.
    - Torch device selection and optional xformers.
    - LoRA adapter loading.

    Steps:
    1. Return cached pipeline if already initialized.
    2. Load base model on CUDA/CPU with dtype policy.
    3. Attach LoRA weights and scheduler optimizations.
    4. Disable safety checker and store global singleton.
    """
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                _PIPE = get_img2img_pipe(adapter_weight=0.8)
    return _PIPE


# ============================================
# batch stylize
# ============================================
def stylize_batch(frames, prompt, batch_size=BATCH_SIZE):
    """Stylize frames in mini-batches for faster diffusion inference.

    Tools:
    - Diffusers batched img2img execution.
    - PIL conversion from OpenCV BGR frames.

    Steps:
    1. Slice sequence into configurable mini-batches.
    2. Convert each batch frame to RGB PIL image.
    3. Run batched img2img with shared prompt.
    4. Convert outputs back to BGR NumPy arrays.
    """
    pipe = get_pipe()
    results = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        results.extend(
            run_img2img_batch(
                pipe,
                frames_bgr=batch,
                prompt=prompt,
                strength=STRENGTH,
                guidance_scale=GUIDANCE,
                num_inference_steps=NUM_STEPS,
            )
        )

    return results


# ============================================
# public API
# ============================================
def apply_style_frames(frames, style):
    """Public API: stylize full frame sequence with one style prompt.

    Tools:
    - get_prompt and stylize_batch.

    Steps:
    1. Normalize style and build diffusion prompt.
    2. Run batched stylization over all frames.
    3. Return stylized sequence.
    """
    prompt = get_prompt(style)
    return stylize_batch(frames, prompt)
