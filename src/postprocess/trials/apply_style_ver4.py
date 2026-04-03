import threading

import numpy as np

from .apply_style_common import (
    STYLE_ALIASES,
    blend_regenerated,
    build_style_prompt,
    calc_flow_farneback,
    detect_breakdown_mask,
    get_img2img_pipe,
    normalize_style_alias,
    run_img2img,
    sample_flow,
    warp_with_flow,
)

# ============================================
# Settings
# ============================================
NUM_STEPS = 25
GUIDANCE = 7.0
STRENGTH = 0.5
MASK_PHOTO_THRESHOLD = 18
MASK_FB_THRESHOLD = 1.5
MIN_REGEN_RATIO = 0.01

_STYLE_ALIASES = dict(STYLE_ALIASES)

_PIPE = None
_LOCK = threading.Lock()


def normalize_style(style: str) -> str:
    """Normalize input style text into canonical style key.

    Tools:
    - String cleanup and alias dictionary matching.

    Steps:
    1. Lowercase and remove separators.
    2. Resolve aliases to canonical names.
    3. Return lowercase fallback when no alias matches.
    """
    return normalize_style_alias(
        style,
        aliases=_STYLE_ALIASES,
        fallback="lower",
    )


def get_prompt(style: str) -> str:
    """Build diffusion prompt from canonicalized style token.

    Tools:
    - normalize_style and prompt template formatting.

    Steps:
    1. Normalize style input.
    2. Convert style token into readable phrase.
    3. Return prompt for img2img inference.
    """
    canonical = normalize_style(style)
    return build_style_prompt(canonical)


def get_pipe():
    """Initialize and cache Stable Diffusion img2img with LoRA tuning.

    Tools:
    - Diffusers StableDiffusionImg2ImgPipeline.
    - EulerAncestralDiscreteScheduler.
    - Torch device selection and optional xformers acceleration.

    Steps:
    1. Reuse global pipeline when available.
    2. Load base model and move to selected device.
    3. Load LoRA adapters and configure scheduler/runtime options.
    4. Disable safety checker and cache singleton pipeline.
    """
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                _PIPE = get_img2img_pipe(adapter_weight=0.8)
    return _PIPE


def _stylize_img2img(frame_bgr: np.ndarray, prompt: str) -> np.ndarray:
    """Run one-frame img2img stylization and return BGR output.

    Tools:
    - Diffusers pipeline inference.
    - PIL for OpenCV BGR <-> RGB conversion.

    Steps:
    1. Convert BGR frame to RGB PIL image.
    2. Run img2img with configured strength/guidance/steps.
    3. Convert generated RGB output back to BGR array.
    """
    return run_img2img(
        get_pipe(),
        frame_bgr=frame_bgr,
        prompt=prompt,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
    )


def _calc_flow_farneback(
    src_bgr: np.ndarray,
    dst_bgr: np.ndarray,
) -> np.ndarray:
    """Estimate dense optical flow between two frames.

    Tools:
    - OpenCV Farneback optical flow.

    Steps:
    1. Convert source and destination frames to grayscale.
    2. Compute dense forward motion field.
    3. Return flow vectors in pixel coordinates.
    """
    return calc_flow_farneback(src_bgr, dst_bgr)


def _warp_with_flow(image_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp an image according to a dense flow field.

    Tools:
    - NumPy grid generation and OpenCV remap.

    Steps:
    1. Build x/y sampling maps from flow offsets.
    2. Remap pixels with linear interpolation.
    3. Reflect borders to avoid empty seams.
    """
    return warp_with_flow(image_bgr, flow)


def _sample_flow(
    flow: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Sample flow vectors at remapped coordinates.

    Tools:
    - OpenCV remap on flow channels.

    Steps:
    1. Bilinearly sample x/y flow channels.
    2. Fill out-of-range values with zeros.
    3. Stack sampled channels into vector flow.
    """
    return sample_flow(flow, map_x, map_y)


def _detect_breakdown_mask(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fwd: np.ndarray,
) -> np.ndarray:
    """Detect temporal breakdown regions that need regeneration.

    Tools:
    - OpenCV warping, photometric error, Farneback flow, morphology.
    - Forward-backward consistency check.

    Steps:
    1. Warp frame t toward t+1 and compute photometric residual.
    2. Compute backward flow and forward-backward inconsistency.
    3. Combine masks with validity constraint.
    4. Refine mask using open+dilate morphology.
    """
    return detect_breakdown_mask(
        frame_t_bgr,
        frame_tp1_bgr,
        flow_fwd,
        photo_threshold=MASK_PHOTO_THRESHOLD,
        fb_threshold=MASK_FB_THRESHOLD,
    )


def _blend_regenerated(
    warped_bgr: np.ndarray,
    regenerated_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Feather-blend regenerated regions into flow-warped stylization.

    Tools:
    - OpenCV Gaussian blur and NumPy alpha compositing.

    Steps:
    1. Convert binary mask to soft alpha map.
    2. Blend regenerated pixels where confidence is low.
    3. Keep warped pixels elsewhere for temporal continuity.
    """
    return blend_regenerated(warped_bgr, regenerated_bgr, mask)


def apply_style_frame(frame, style):
    """Stylize a single frame without temporal propagation.

    Tools:
    - Prompt generation and one-shot img2img inference.

    Steps:
    1. Build style prompt from input.
    2. Run single-frame diffusion stylization.
    3. Return BGR stylized frame.
    """
    prompt = get_prompt(style)
    return _stylize_img2img(frame, prompt)


def apply_style_frames(frames, style):
    """Stylize video with flow warp + selective diffusion regeneration.

    Tools:
    - Stable Diffusion img2img for style transfer.
    - OpenCV Farneback flow for temporal propagation.
    - Masked blending for artifact repair.

    Steps:
    1. Stylize first frame via diffusion.
    2. Propagate stylization by optical-flow warping.
    3. Detect unstable regions with photometric and FB checks.
    4. Regenerate only unstable areas and blend into propagated frame.
    """
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
