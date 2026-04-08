from pathlib import Path
import threading

import cv2
import numpy as np
from tqdm import tqdm

from postprocess.progress import Path
from postprocess import apply_style_common as common

STYLE_ALIASES = common.STYLE_ALIASES
blend_regenerated = common.blend_regenerated
build_style_prompt = common.build_style_prompt
calc_flow_farneback = common.calc_flow_farneback
detect_breakdown_mask = common.detect_breakdown_mask
get_img2img_pipe = common.get_img2img_pipe
normalize_style_alias = common.normalize_style_alias
run_img2img = common.run_img2img
sample_flow = common.sample_flow
warp_with_flow = common.warp_with_flow

# ============================================
# Settings
# ============================================
NUM_STEPS = 10
STRENGTH = 0.35   # ← 下げる（重要）
GUIDANCE = 5.5    # ← 下げる
MASK_PHOTO_THRESHOLD = 30
MASK_FB_THRESHOLD = 2.0
MIN_REGEN_RATIO = 0.05

# ============================================
# Temporal stabilization（追加）
# ============================================
TEMPORAL_BLEND = 0.25  # 前フレームの影響度（0.2〜0.3推奨）

_STYLE_ALIASES = dict(STYLE_ALIASES)

_PIPE = None
_LOCK = threading.Lock()


def normalize_style(style: str) -> str:
    """Normalize style aliases into canonical style names.

    Tools:
    - Python string normalization and alias dictionary lookup.

    Steps:
    1. Lowercase and remove separators from input.
    2. Match against alias groups.
    3. Return canonical token or lowercase fallback.
    """
    return normalize_style_alias(
        style,
        aliases=_STYLE_ALIASES,
        fallback="lower",
    )


def get_prompt(style: str) -> str:
    """Create diffusion prompt text from style input.

    Tools:
    - normalize_style and template formatting.

    Steps:
    1. Normalize style token.
    2. Convert token to readable phrase.
    3. Return prompt for img2img inference.
    """
    canonical = normalize_style(style)
    return build_style_prompt(canonical)


def get_pipe():
    """Build and cache Stable Diffusion img2img pipeline with LoRA.

    Tools:
    - Diffusers StableDiffusionImg2ImgPipeline.
    - EulerAncestralDiscreteScheduler and attention optimizations.
    - Torch device/dtype policy and optional xformers.

    Steps:
    1. Return existing pipeline when already initialized.
    2. Load model on target device and apply LoRA adapters.
    3. Configure scheduler and memory optimizations.
    4. Disable safety checker and cache singleton pipeline.
    """
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                _PIPE = get_img2img_pipe(adapter_weight=0.8)
    return _PIPE


def _stylize_img2img(frame_bgr: np.ndarray, prompt: str) -> np.ndarray:
    """Stylize one frame by diffusion img2img and return BGR result.

    Tools:
    - PIL conversion and Diffusers pipeline inference.

    Steps:
    1. Convert OpenCV BGR frame to RGB PIL image.
    2. Run img2img with tuned strength/guidance/steps.
    3. Convert generated frame back to BGR NumPy array.
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
    """Compute dense optical flow from source frame to destination frame.

    Tools:
    - OpenCV Farneback optical flow estimator.

    Steps:
    1. Convert both frames to grayscale.
    2. Estimate dense motion vectors.
    3. Return flow field for remapping.
    """
    return calc_flow_farneback(src_bgr, dst_bgr)


def _warp_with_flow(image_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image by flow to temporally propagate previous stylization.

    Tools:
    - NumPy meshgrid and OpenCV remap.

    Steps:
    1. Build remap coordinates from flow offsets.
    2. Resample image using linear interpolation.
    3. Reflect borders to avoid black edges.
    """
    return warp_with_flow(image_bgr, flow)


def _sample_flow(
    flow: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Sample backward flow at forward-warped coordinates.

    Tools:
    - OpenCV remap for channel-wise flow sampling.

    Steps:
    1. Sample x and y flow components independently.
    2. Use constant-zero fill for invalid coordinates.
    3. Recombine components into 2-channel flow array.
    """
    return sample_flow(flow, map_x, map_y)


def _detect_breakdown_mask(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fwd: np.ndarray,
) -> np.ndarray:
    """Find unstable regions where flow propagation breaks down.

    Tools:
    - OpenCV warping, photometric residuals, Farneback FB check.
    - Morphology operations for mask cleanup.

    Steps:
    1. Compare warped previous frame against current frame.
    2. Compute forward-backward consistency error.
    3. Combine errors with valid-coordinate mask.
    4. Denoise and expand mask for robust blending.
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
    """Blend regenerated details into warped stylized frame with soft alpha.

    Tools:
    - OpenCV Gaussian blur and NumPy alpha compositing.

    Steps:
    1. Normalize mask into soft alpha map.
    2. Composite regenerated content on masked regions.
    3. Keep warped content elsewhere for continuity.
    """
    return blend_regenerated(warped_bgr, regenerated_bgr, mask)


def apply_style_frame(frame, style):
    """Apply one-shot style transfer to a single frame.

    Tools:
    - Prompt construction + diffusion img2img.

    Steps:
    1. Build style prompt.
    2. Run single-frame diffusion stylization.
    3. Return stylized BGR frame.
    """
    prompt = get_prompt(style)
    return _stylize_img2img(frame, prompt)


def apply_style_frames(frames, style):
    """Apply temporally stabilized style transfer over a frame sequence.

    Tools:
    - Stable Diffusion img2img with LoRA.
    - OpenCV Farneback optical flow propagation.
    - Temporal input blending and masked regeneration.
    - tqdm progress reporting.

    Steps:
    1. Stylize the first frame with diffusion.
    2. Warp previous stylized frame to current time using optical flow.
    3. Detect unstable regions and decide regeneration by mask ratio.
    4. Blend temporally mixed diffusion result into unstable regions.
    """
    if not frames:
        return []

    prompt = get_prompt(style)
    outputs = []

    # frame0 -> LoRA -> stylized_0
    stylized_t = _stylize_img2img(frames[0], prompt)
    outputs.append(stylized_t)

    # tqdm にする
    for idx in tqdm(range(len(frames) - 1), desc=f"Applying style {style}"):
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
            # --------------------------------------------
            # 【重要】Temporal consistency 対策
            # --------------------------------------------
            # 背景意図：
            # diffusion(img2img)は毎フレーム独立生成のため、
            # 同一人物でも「別の絵」を描いてしまい flicker が発生する。
            #
            # 対策：
            # 前フレーム（stylized_t）を入力に混ぜることで、
            # diffusionに「過去情報」を与え、連続性を強制する。
            #
            # 技術的意味：
            # ・frame_tp1（現在の構造）
            # ・stylized_t（過去のスタイル）
            # を合成し、両方を満たす方向に収束させる
            #
            # 効果：
            # ・顔の形崩れ防止
            # ・色の安定化
            # ・筆タッチの一貫性
            # --------------------------------------------

            blended_input = cv2.addWeighted(
                frame_tp1, 1.0 - TEMPORAL_BLEND,
                stylized_t, TEMPORAL_BLEND,
                0,
            )

            regenerated = _stylize_img2img(blended_input, prompt)

            stylized_tp1 = _blend_regenerated(warped, regenerated, mask)
        else:
            stylized_tp1 = warped

        outputs.append(stylized_tp1)
        stylized_t = stylized_tp1

    return outputs

#------------------- Add for foreground/background separation -------------------#
from postprocess.detectors import detect_all_boxes, get_sam_mask_from_box
from postprocess.apply_style_ver5 import apply_style_frame
from utils.video_utility import load_video, write_video


def build_foreground_mask_dino_sam(
    frame_bgr: np.ndarray,
    text_prompt: str = 'person .',
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    dilate_iter: int = 1,
    close_iter: int = 1,
) -> np.ndarray:
    """GroundingDINO + SAM で前景マスク(0/255)を作る。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(
        frame_rgb,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    h, w = frame_bgr.shape[:2]
    fg_mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        sam_mask = get_sam_mask_from_box(frame_rgb, box)
        if sam_mask is None:
            continue
        fg_mask = np.maximum(fg_mask, (sam_mask > 0).astype(np.uint8) * 255)

    if close_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=dilate_iter)

    return fg_mask


def split_foreground_background(
    frame_bgr: np.ndarray,
    fg_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """前景画像と背景画像に分割する。"""
    mask3 = (fg_mask > 0)[:, :, None]
    foreground = np.where(mask3, frame_bgr, 0).astype(np.uint8)
    background = np.where(~mask3, frame_bgr, 0).astype(np.uint8)
    return foreground, background


def apply_style_foreground_background(
    frame_bgr: np.ndarray,
    style: str = 'oil_painting',
    text_prompt: str = 'person .',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1) DINO+SAM で前景マスク生成
    2) 前景/背景に分割
    3) 前景・背景それぞれに apply_style を実行
    4) 合成して最終フレームを返す
    """
    fg_mask = build_foreground_mask_dino_sam(frame_bgr, text_prompt=text_prompt)
    fg, bg = split_foreground_background(frame_bgr, fg_mask)

    stylized_fg = apply_style_frame(fg, style)
    stylized_bg = apply_style_frame(bg, style)

    alpha = cv2.GaussianBlur((fg_mask.astype(np.float32) / 255.0), (0, 0), 1.2)
    alpha3 = alpha[:, :, None]
    merged = (stylized_fg.astype(np.float32) * alpha3 + stylized_bg.astype(np.float32) * (1.0 - alpha3))
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    return merged, fg_mask, fg, bg, stylized_fg


def apply_style_video_foreground_background(
    in_path: str | Path,
    out_path: str | Path,
    style: str = 'oil_painting',
    text_prompt: str = 'person .',
    max_frames: int | None = None,
) -> Path:
    frames, fps, width, height = load_video(in_path)

    if max_frames is not None:
        frames = frames[:max_frames]

    out_frames: list[np.ndarray] = []
    for frame in tqdm(frames, desc=f'fg/bg apply_style ({style})'):
        merged, _mask, _fg, _bg, _stylized_fg = apply_style_foreground_background(
            frame,
            style=style,
            text_prompt=text_prompt,
        )
        out_frames.append(merged)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out_path, out_frames, fps, width, height)
    return out_path


def test():
    # 動画実行
    WORKSPACE = Path('/workspace')
    VIDEO_DIR = WORKSPACE / 'data' / 'videos'
    OUT_DIR = WORKSPACE / 'logs' / 'notebook' / 'task_03_apply_style_fg_bg'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    VIDEO_NAME = '94msufYZzaQ_26_0to273.mp4'
    STYLE = 'oil_painting'
    TARGET_PROMPT = 'person .'
    MAX_FRAMES = 10   # None  # デバッグ時は 24 などに変更

    in_path = VIDEO_DIR / VIDEO_NAME
    out_path = OUT_DIR / f'processed_fg_bg_{STYLE}_{VIDEO_NAME}'

    if not in_path.exists():
        raise FileNotFoundError(f'input video not found: {in_path}')

    saved_path = apply_style_video_foreground_background(
        in_path=in_path,
        out_path=out_path,
        style=STYLE,
        text_prompt=TARGET_PROMPT,
        max_frames=MAX_FRAMES,
    )

    print('input :', in_path)
    print('style :', STYLE)
    print('prompt:', TARGET_PROMPT)
    print('out   :', saved_path)

if __name__ == '__main__':
    test()
