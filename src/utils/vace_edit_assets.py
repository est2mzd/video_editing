"""
追加する理由

今の notebook ver3 は prompt しか渡していない。
そのため、VACE にとっては「どこを変えるか」が不明なままになる。

そこで、

入力動画から
instruction に応じた
mask動画 (src_mask.mp4)
を作る専用モジュールを追加する。

さらに、

VACE に渡す prompt を、mask前提の指示文に整形する
追加系 instruction では src_ref_images が必要なら早めに止める

ここまでを1ファイルにまとめる。
"""

# src/utils/vace_edit_assets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.submit_baseline_ver03 import (
    detect_primary_face,
    estimate_subject_mask,
    mask_from_face_region,
)


@dataclass
class VaceEditAssets:
    prompt: str
    src_mask: Path | None
    src_ref_images: list[str] | None
    edit_mode: str


_DEFAULT_MASK_CFG = {
    "body_center_y_scale": 1.8,
    "body_radius_x_scale": 1.4,
    "body_radius_y_scale": 2.6,
    "grabcut_pad_x": 1.2,
    "grabcut_pad_top": 0.8,
    "grabcut_pad_bottom": 3.6,
    "grabcut_iter": 2,
    "center_subject_prior": 0.42,
    "subject_blur_sigma": 11,
}


def _normalize_edit_mode(instruction: str, edit_mode: str) -> str:
    mode = (edit_mode or "auto").strip().lower()
    if mode != "auto":
        return mode

    text = instruction.lower()

    if any(k in text for k in ["background", "sky", "sunset", "cityscape", "scene behind", "replace the background"]):
        return "background"

    if any(k in text for k in ["face", "mouth", "eyes", "cheek", "forehead"]):
        return "face"

    if "hair" in text:
        return "hair"
    if "tie" in text or "necktie" in text:
        return "tie"
    if "hat" in text or "beanie" in text:
        return "hat"

    if any(k in text for k in ["add ", "insert ", "more ", "extra ", "another "]):
        return "subject_nearby"

    return "subject"


def _requires_ref_images(instruction: str, edit_mode: str) -> bool:
    text = instruction.lower()
    mode = edit_mode.lower()
    if mode == "subject_nearby" and any(k in text for k in ["animal", "rhino", "rhinoceros", "dog", "cat", "horse"]):
        return True
    return False


def build_vace_prompt(instruction: str, edit_mode: str) -> str:
    mode = edit_mode.lower()

    if mode == "background":
        scope = "Modify only the masked background region."
        preserve = "Keep the presenter, face, body, text, logos, and camera motion unchanged."
    elif mode in {"face", "hair", "tie", "hat"}:
        scope = "Modify only the masked local foreground region."
        preserve = "Keep identity, background, text, logos, and all unmasked regions unchanged."
    elif mode == "subject_nearby":
        scope = "Modify only the masked region near the subject."
        preserve = "Keep the subject identity, background structure, text, and logos unchanged."
    else:
        scope = "Modify only the masked subject region."
        preserve = "Keep the background, text, logos, and all unmasked regions unchanged."

    return (
        "Edit the source video according to the instruction.\n"
        f"{scope}\n"
        f"{preserve}\n"
        "Maintain temporal consistency across all frames.\n"
        "Do not change framing, video length, or aspect ratio.\n\n"
        f"Instruction:\n{instruction}\n"
    )


def _subject_mask_from_frame(
    frame_bgr: np.ndarray,
    prev_face_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    face_box = detect_primary_face(frame_bgr)
    if face_box is None:
        face_box = prev_face_box
    subject_mask = estimate_subject_mask(frame_bgr, face_box, _DEFAULT_MASK_CFG)
    return subject_mask, face_box


def _build_mask_for_mode(
    frame_bgr: np.ndarray,
    edit_mode: str,
    prev_face_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    subject_mask, face_box = _subject_mask_from_frame(frame_bgr, prev_face_box)
    mode = edit_mode.lower()

    if mode == "background":
        mask = 255 - subject_mask

    elif mode in {"face", "hair", "tie", "hat"}:
        if face_box is not None:
            region = "face" if mode == "face" else mode
            mask = mask_from_face_region(frame_bgr, face_box, region)
        else:
            mask = subject_mask

    elif mode == "subject_nearby":
        kernel = np.ones((31, 31), np.uint8)
        dilated = cv2.dilate(subject_mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(dilated, (0, 0), 9)

    elif mode == "subject":
        mask = subject_mask

    else:
        raise ValueError(f"Unsupported edit_mode: {edit_mode}")

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask, face_box


def build_mask_video(
    video_path: Path,
    output_mask_path: Path,
    instruction: str,
    edit_mode: str,
) -> Path:
    video_path = Path(video_path).resolve()
    output_mask_path = Path(output_mask_path).resolve()
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mask_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create mask video: {output_mask_path}")

    prev_face_box = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            mask, prev_face_box = _build_mask_for_mode(
                frame_bgr=frame_bgr,
                edit_mode=edit_mode,
                prev_face_box=prev_face_box,
            )
            mask_rgb = np.repeat(mask[..., None], 3, axis=2)
            writer.write(mask_rgb)
    finally:
        cap.release()
        writer.release()

    return output_mask_path


def prepare_vace_edit_assets(
    video_path: Path,
    work_dir: Path,
    instruction: str,
    edit_mode: str = "auto",
    src_ref_images: Sequence[str] | None = None,
) -> VaceEditAssets:
    mode = _normalize_edit_mode(instruction, edit_mode)

    ref_images = [str(Path(p).resolve()) for p in (src_ref_images or [])]
    if _requires_ref_images(instruction, mode) and not ref_images:
        raise ValueError(
            "This instruction looks like instance insertion. "
            "Provide EX_SRC_REF_IMAGES or route this case to an insertion pipeline."
        )

    mask_path = build_mask_video(
        video_path=Path(video_path),
        output_mask_path=Path(work_dir) / "src_mask.mp4",
        instruction=instruction,
        edit_mode=mode,
    )

    prompt = build_vace_prompt(instruction, mode)
    return VaceEditAssets(
        prompt=prompt,
        src_mask=mask_path,
        src_ref_images=ref_images or None,
        edit_mode=mode,
    )