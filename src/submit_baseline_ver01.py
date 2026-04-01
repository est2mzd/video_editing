#!/usr/bin/env python3
"""Rule-based baseline submission generator for GIVE challenge."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class AnnotationRecord:
    video_path: str
    selected_class: str
    selected_subclass: str
    instruction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a rule-based GIVE submission zip."
    )
    parser.add_argument(
        "--annotations",
        default="/workspace/data/annotations.jsonl",
    )
    parser.add_argument(
        "--video-dir",
        default="/workspace/data/videos",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/data/submission_ver01_videos",
    )
    parser.add_argument(
        "--output-zip",
        default="/workspace/data/submission_ver01.zip",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N records. 0 means all.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--codec",
        default=os.environ.get("VIDEO_CODEC", "auto"),
        help="auto, h264_nvenc, libx264",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> list[AnnotationRecord]:
    rows: list[AnnotationRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                AnnotationRecord(
                    video_path=str(obj["video_path"]),
                    selected_class=str(obj["selected_class"]),
                    selected_subclass=str(obj["selected_subclass"]),
                    instruction=str(obj["instruction"]),
                )
            )
    return rows


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def hsv_color_bounds(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    bounds = {
        "red": ((0, 70, 40), (12, 255, 255)),
        "orange": ((8, 70, 40), (25, 255, 255)),
        "yellow": ((20, 60, 40), (38, 255, 255)),
        "green": ((35, 40, 30), (90, 255, 255)),
        "blue": ((90, 40, 30), (130, 255, 255)),
        "navy blue": ((100, 40, 20), (125, 255, 180)),
        "violet": ((125, 40, 30), (155, 255, 255)),
        "purple": ((125, 40, 30), (155, 255, 255)),
        "pink": ((150, 30, 40), (179, 255, 255)),
        "black": ((0, 0, 0), (179, 255, 55)),
        "white": ((0, 0, 180), (179, 70, 255)),
        "silver": ((0, 0, 80), (179, 50, 230)),
    }
    value = bounds.get(name)
    if value is None:
        return None
    low, high = value
    return np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)


def extract_target_color(instruction: str) -> str | None:
    ordered = [
        "navy blue",
        "violet",
        "purple",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "pink",
        "black",
        "white",
        "silver",
    ]
    lower = instruction.lower()
    for color in ordered:
        if color in lower:
            return color
    return None


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_f = alpha.astype(np.float32)[..., None] / 255.0
    out = base.astype(np.float32) * (1.0 - alpha_f) + overlay.astype(np.float32) * alpha_f
    return np.clip(out, 0, 255).astype(np.uint8)


def build_center_weight(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ry = max(height * 0.42, 1.0)
    rx = max(width * 0.34, 1.0)
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    weight = np.clip(1.0 - dist, 0.0, 1.0)
    return (weight * 255.0).astype(np.uint8)


def apply_zoom(frame: np.ndarray, progress: float, zoom_in: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    max_scale = 1.24
    min_scale = 0.84
    scale = 1.0 + (max_scale - 1.0) * progress if zoom_in else 1.0 - (1.0 - min_scale) * progress
    src = frame
    if scale >= 1.0:
        crop_w = max(2, int(round(w / scale)))
        crop_h = max(2, int(round(h / scale)))
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        src = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    else:
        canvas = np.zeros_like(frame)
        scaled_w = max(2, int(round(w * scale)))
        scaled_h = max(2, int(round(h * scale)))
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        x0 = (w - scaled_w) // 2
        y0 = (h - scaled_h) // 2
        canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = resized
        src = canvas
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_perspective_tilt(frame: np.ndarray, progress: float, low_angle: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    margin = max(6, int(round(h * 0.06 * (0.35 + 0.65 * progress))))
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    if low_angle:
        dst = np.float32([[margin, 0], [w - 1 - margin, 0], [0, h - 1], [w - 1, h - 1]])
    else:
        dst = np.float32([[0, 0], [w - 1, 0], [margin, h - 1], [w - 1 - margin, h - 1]])
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def apply_color_retarget(frame: np.ndarray, target_color: str | None) -> np.ndarray:
    if not target_color:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_weight = build_center_weight(frame.shape[0], frame.shape[1])

    if target_color in {"violet", "purple"}:
        overlay = frame.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * 1.38 + 48, 0, 255)
        overlay[..., 2] = np.clip(overlay[..., 2] * 1.18 + 26, 0, 255)
        strong_alpha = np.clip(center_weight.astype(np.float32) * 1.18, 0, 255).astype(np.uint8)
        return alpha_blend(frame, overlay, strong_alpha)

    bounds = hsv_color_bounds(target_color)
    if bounds is None:
        return frame
    low, high = bounds
    overlay_hsv = hsv.copy()
    hue_map = {
        "red": 0,
        "orange": 15,
        "yellow": 28,
        "green": 60,
        "blue": 108,
        "navy blue": 112,
        "violet": 138,
        "purple": 145,
        "pink": 165,
        "black": 0,
        "white": 0,
        "silver": 0,
    }
    overlay_hsv[..., 0] = hue_map.get(target_color, overlay_hsv[..., 0])
    if target_color == "black":
        overlay_hsv[..., 1] = np.clip(overlay_hsv[..., 1] * 0.3, 0, 255)
        overlay_hsv[..., 2] = np.clip(overlay_hsv[..., 2] * 0.35, 0, 255)
    elif target_color in {"white", "silver"}:
        overlay_hsv[..., 1] = np.clip(overlay_hsv[..., 1] * 0.2, 0, 255)
        overlay_hsv[..., 2] = np.clip(overlay_hsv[..., 2] * 1.2 + 10, 0, 255)
    else:
        overlay_hsv[..., 1] = np.clip(np.maximum(overlay_hsv[..., 1], 120), 0, 255)
        overlay_hsv[..., 2] = np.clip(overlay_hsv[..., 2] * 1.06 + 8, 0, 255)
    overlay = cv2.cvtColor(overlay_hsv, cv2.COLOR_HSV2BGR)

    base_mask = cv2.inRange(hsv, low, high)
    if cv2.countNonZero(base_mask) == 0:
        return alpha_blend(frame, overlay, (center_weight * 0.45).astype(np.uint8))

    mask = cv2.GaussianBlur(base_mask, (0, 0), 9)
    return alpha_blend(frame, overlay, mask)


def filter_pixel(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    block = max(6, min(h, w) // 60)
    small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    palette = (small // 48) * 48
    return cv2.resize(palette, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_watercolor(frame: np.ndarray) -> np.ndarray:
    smoothed = cv2.bilateralFilter(frame, 9, 80, 80)
    edges = cv2.Canny(smoothed, 70, 140)
    edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    mix = cv2.addWeighted(smoothed, 0.82, edges, 0.18, 0)
    return cv2.stylization(mix, sigma_s=45, sigma_r=0.35)


def filter_oil_painting(frame: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "oilPainting"):
        smoothed = cv2.xphoto.oilPainting(frame, 6, 1)
    else:
        smoothed = cv2.bilateralFilter(frame, 11, 90, 90)
        smoothed = cv2.medianBlur(smoothed, 5)
    return cv2.detailEnhance(smoothed, sigma_s=8, sigma_r=0.18)


def filter_anime(frame: np.ndarray) -> np.ndarray:
    color = cv2.bilateralFilter(frame, 9, 120, 120)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        7,
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    boosted = cv2.convertScaleAbs(color, alpha=1.15, beta=8)
    return cv2.bitwise_and(boosted, edges)


def filter_cyberpunk(frame: np.ndarray) -> np.ndarray:
    blue = np.full_like(frame, (255, 120, 20))
    pink = np.full_like(frame, (80, 30, 255))
    grad = np.linspace(0.0, 1.0, frame.shape[1], dtype=np.float32)[None, :, None]
    overlay = (blue.astype(np.float32) * (1.0 - grad) + pink.astype(np.float32) * grad).astype(np.uint8)
    base = cv2.convertScaleAbs(frame, alpha=1.18, beta=-8)
    mixed = cv2.addWeighted(base, 0.68, overlay, 0.32, 0)
    glow = cv2.GaussianBlur(mixed, (0, 0), 10)
    return cv2.addWeighted(mixed, 0.82, glow, 0.18, 0)


def filter_comic(frame: np.ndarray) -> np.ndarray:
    color = cv2.bilateralFilter(frame, 7, 110, 110)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_bgr = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    halftone = ((gray // 48) * 48)
    halftone = cv2.cvtColor(halftone, cv2.COLOR_GRAY2BGR)
    mixed = cv2.addWeighted(color, 0.7, halftone, 0.3, 0)
    return cv2.bitwise_and(mixed, edges_bgr)


def filter_ukiyoe(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.75 + 25, 0, 255)
    hsv[..., 2] = np.clip((hsv[..., 2] // 36) * 36 + 18, 0, 255)
    flat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), 70, 150)
    edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(flat, 0.86, edges, 0.14, 0)


def filter_ghibli(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.82 + 12, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.08 + 6, 0, 255)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.GaussianBlur(out, (0, 0), 0.8)


def filter_background_change(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (0, 0), 15)
    overlay = np.zeros_like(frame)
    overlay[:] = (60, 75, 90)
    for y in range(h):
        ratio = y / max(h - 1, 1)
        overlay[y, :, 0] = 90 + int(20 * ratio)
        overlay[y, :, 1] = 90 + int(35 * ratio)
        overlay[y, :, 2] = 105 + int(50 * ratio)
    bg = cv2.addWeighted(blurred, 0.45, overlay, 0.55, 0)
    center = build_center_weight(h, w)
    fg_alpha = cv2.GaussianBlur(center, (0, 0), 35)
    bg_alpha = (255 - fg_alpha)
    return alpha_blend(frame, bg, bg_alpha)


def filter_decoration_effect(frame: np.ndarray, progress: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 120, 220)
    glow = cv2.GaussianBlur(edges, (0, 0), 6 + 4 * progress)
    glow_bgr = np.zeros_like(frame)
    glow_bgr[..., 0] = np.clip(glow * 1.1, 0, 255).astype(np.uint8)
    glow_bgr[..., 1] = np.clip(glow * 0.75, 0, 255).astype(np.uint8)
    glow_bgr[..., 2] = np.clip(glow * 1.45, 0, 255).astype(np.uint8)
    boosted = cv2.addWeighted(frame, 0.78, glow_bgr, 0.52, 0)
    return cv2.addWeighted(boosted, 0.9, cv2.GaussianBlur(glow_bgr, (0, 0), 10), 0.2, 0)


def feather_mask(mask: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    return cv2.GaussianBlur(mask, (0, 0), sigma)


def overlay_microphone(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cx = int(w * 0.5)
    base_y = int(h * 0.76)
    stand_top = int(h * 0.57)
    cv2.ellipse(overlay, (cx, int(h * 0.5)), (int(w * 0.025), int(h * 0.065)), 0, 0, 360, (45, 45, 45), -1)
    cv2.rectangle(overlay, (cx - int(w * 0.005), stand_top), (cx + int(w * 0.005), base_y), (60, 60, 60), -1)
    cv2.ellipse(overlay, (cx, base_y), (int(w * 0.045), int(h * 0.012)), 0, 0, 360, (35, 35, 35), -1)
    shadow = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(shadow, (cx, base_y + int(h * 0.012)), (int(w * 0.06), int(h * 0.018)), 0, 0, 360, 180, -1)
    with_shadow = alpha_blend(frame, overlay, feather_mask(np.where(overlay != frame, 220, 0).max(axis=2).astype(np.uint8), 5))
    shadow_overlay = np.zeros_like(frame)
    return alpha_blend(with_shadow, shadow_overlay, feather_mask(shadow, 12))


def duplicate_patch(frame: np.ndarray, src_rect: tuple[int, int, int, int], dst_center: tuple[int, int], scale: float) -> np.ndarray:
    x0, y0, x1, y1 = src_rect
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return frame
    ph, pw = patch.shape[:2]
    new_w = max(8, int(pw * scale))
    new_h = max(8, int(ph * scale))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cx, cy = dst_center
    dx0 = max(0, cx - new_w // 2)
    dy0 = max(0, cy - new_h // 2)
    dx1 = min(frame.shape[1], dx0 + new_w)
    dy1 = min(frame.shape[0], dy0 + new_h)
    resized = resized[:dy1 - dy0, :dx1 - dx0]
    out = frame.copy()
    roi = out[dy0:dy1, dx0:dx1]
    mask = np.zeros((resized.shape[0], resized.shape[1]), dtype=np.uint8)
    cv2.ellipse(mask, (resized.shape[1] // 2, resized.shape[0] // 2), (max(4, resized.shape[1] // 2 - 2), max(4, resized.shape[0] // 2 - 2)), 0, 0, 360, 255, -1)
    out[dy0:dy1, dx0:dx1] = alpha_blend(roi, resized, feather_mask(mask, 7))
    return out


def apply_quantity_increase(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    src_rect = (
        int(w * 0.38),
        int(h * 0.42),
        int(w * 0.62),
        int(h * 0.72),
    )
    out = duplicate_patch(frame, src_rect, (int(w * 0.22), int(h * 0.68)), 0.55)
    out = duplicate_patch(out, src_rect, (int(w * 0.78), int(h * 0.66)), 0.5)
    return out


def apply_instance_replacement(frame: np.ndarray, instruction: str) -> np.ndarray:
    target_color = extract_target_color(instruction)
    if target_color:
        return apply_color_retarget(frame, target_color)
    h, w = frame.shape[:2]
    center = build_center_weight(h, w)
    boosted = cv2.detailEnhance(frame, sigma_s=6, sigma_r=0.15)
    return alpha_blend(frame, boosted, np.clip(center * 0.6, 0, 255).astype(np.uint8))


def apply_style(frame: np.ndarray, subclass: str) -> np.ndarray:
    if subclass == "Pixel":
        return filter_pixel(frame)
    if subclass == "Watercolor":
        return filter_watercolor(frame)
    if subclass == "Oil painting":
        return filter_oil_painting(frame)
    if subclass == "Anime":
        return filter_anime(frame)
    if subclass == "Cyberpunk":
        return filter_cyberpunk(frame)
    if subclass == "American comic style":
        return filter_comic(frame)
    if subclass == "Ukiyo-e":
        return filter_ukiyoe(frame)
    if subclass == "Ghibli":
        return filter_ghibli(frame)
    return frame


def apply_edit(
    frame: np.ndarray,
    record: AnnotationRecord,
    index: int,
    total_frames: int,
) -> np.ndarray:
    progress = 0.0 if total_frames <= 1 else index / (total_frames - 1)
    edited = frame
    cls = record.selected_class
    sub = record.selected_subclass

    if cls == "Camera Motion Editing":
        if sub in {"Zoom in", "Dolly in"}:
            edited = apply_zoom(edited, progress, zoom_in=True)
        elif sub == "Zoom out":
            edited = apply_zoom(edited, progress, zoom_in=False)
        elif sub == "Arc shot":
            sway = math.sin(progress * math.pi) * 0.08
            h, w = edited.shape[:2]
            shift = int(round(w * sway))
            mat = np.float32([[1, 0, shift], [0, 1, 0]])
            edited = cv2.warpAffine(
                edited,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            edited = apply_zoom(edited, progress * 0.5, zoom_in=True)
    elif cls == "Camera Angle Editing":
        if sub == "Low angle":
            edited = apply_perspective_tilt(edited, progress, low_angle=True)
        elif sub == "High angle":
            edited = apply_perspective_tilt(edited, progress, low_angle=False)
    elif cls == "Style Editing":
        edited = apply_style(edited, sub)
    elif cls == "Visual Effect Editing":
        if sub == "Background Change":
            edited = filter_background_change(edited)
        elif sub == "Decoration effect":
            edited = filter_decoration_effect(edited, progress)
    elif cls == "Attribute Editing" and sub == "Color adjustment":
        target_color = extract_target_color(record.instruction)
        edited = apply_color_retarget(edited, target_color)
    elif cls == "Quantity Editing" and sub == "Increase":
        edited = apply_quantity_increase(edited)
    elif cls == "Instance Editing":
        if sub == "Instance Replacement":
            edited = apply_instance_replacement(edited, record.instruction)
        elif sub == "Instance Insertion" and "microphone" in record.instruction.lower():
            edited = overlay_microphone(edited)

    return edited


def count_frames_ffprobe(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return int(text) if text.isdigit() else None


def probe_resolution(path: Path) -> tuple[int, int] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    if "x" not in value:
        return None
    width, height = value.split("x", 1)
    return int(width), int(height)


def iter_records(records: list[AnnotationRecord], start_index: int, limit: int) -> Iterable[tuple[int, AnnotationRecord]]:
    sliced = records[start_index:]
    if limit > 0:
        sliced = sliced[:limit]
    for idx, record in enumerate(sliced, start=start_index):
        yield idx, record


def choose_codec(codec_arg: str) -> str:
    if codec_arg != "auto":
        return codec_arg
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode == 0 and "h264_nvenc" in probe.stdout:
        return "h264_nvenc"
    return "libx264"


def open_ffmpeg_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> subprocess.Popen:
    base_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.06f}",
        "-i",
        "-",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if codec == "h264_nvenc":
        codec_args = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "0",
        ]
    else:
        codec_args = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
        ]
    cmd = base_cmd + codec_args + [str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError(f"Failed to open ffmpeg stdin for {output_path}")
    return proc


def encode_video_frames(
    frames: list[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> str:
    writer = open_ffmpeg_writer(output_path, width, height, fps, codec)
    try:
        for frame in frames:
            writer.stdin.write(frame.tobytes())
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="ignore")
        rc = writer.wait()
        if rc != 0:
            raise RuntimeError(stderr[-1200:])
        return codec
    except Exception:
        try:
            writer.kill()
        except Exception:
            pass
        raise


def process_video(
    record: AnnotationRecord,
    input_path: Path,
    output_path: Path,
    codec: str,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {input_path}")

    edited_frames: list[np.ndarray] = []
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        edited = apply_edit(frame, record, written, frame_count)
        if edited.shape[:2] != (height, width):
            edited = cv2.resize(edited, (width, height), interpolation=cv2.INTER_LINEAR)
        edited_frames.append(edited)
        written += 1

    cap.release()

    if written != frame_count:
        raise RuntimeError(
            f"Frame read/write mismatch for {input_path.name}: expected={frame_count} written={written}"
        )

    used_codec = codec
    try:
        used_codec = encode_video_frames(edited_frames, output_path, width, height, fps, codec)
    except Exception as exc:
        if codec != "libx264":
            print(f"[WARN] GPU encode failed for {output_path.name}, retry with libx264: {exc}")
            used_codec = encode_video_frames(edited_frames, output_path, width, height, fps, "libx264")
        else:
            raise RuntimeError(f"ffmpeg encode failed for {output_path.name}: {exc}")

    output_frames = count_frames_ffprobe(output_path)
    output_resolution = probe_resolution(output_path)
    result = {
        "input_frames": frame_count,
        "output_frames": output_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "codec": used_codec,
        "status": "ok",
    }
    if output_frames != frame_count:
        raise RuntimeError(
            f"Output frame mismatch after ffmpeg for {output_path.name}: input={frame_count} output={output_frames}"
        )
    if output_resolution != (width, height):
        raise RuntimeError(
            f"Output resolution mismatch for {output_path.name}: input={(width, height)} output={output_resolution}"
        )
    return result


def validate_outputs(
    records: list[AnnotationRecord],
    video_dir: Path,
    output_dir: Path,
    start_index: int,
    limit: int,
) -> dict[str, object]:
    expected_records = [record for _, record in iter_records(records, start_index, limit)]
    expected_names = [record.video_path for record in expected_records]
    actual_names = sorted(path.name for path in output_dir.glob("*.mp4"))
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    mismatched_frames: list[dict[str, object]] = []
    for name in expected_names:
        output_path = output_dir / name
        input_path = video_dir / name
        if not output_path.exists():
            continue
        input_frames = count_frames_ffprobe(input_path)
        output_frames = count_frames_ffprobe(output_path)
        if input_frames != output_frames:
            mismatched_frames.append(
                {
                    "video_path": name,
                    "input_frames": input_frames,
                    "output_frames": output_frames,
                }
            )
    return {
        "expected_count": len(expected_names),
        "actual_count": len(actual_names),
        "missing": missing,
        "extra": extra,
        "frame_mismatches": mismatched_frames,
        "status": "ok" if not missing and not extra and not mismatched_frames else "error",
    }


def make_zip(output_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.glob("*.mp4")):
            zf.write(path, arcname=path.name)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    annotations_path = Path(args.annotations)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_zip = Path(args.output_zip)
    manifest_path = output_dir / "manifest_ver01.json"
    validation_path = output_dir / "validation_ver01.json"

    records = load_annotations(annotations_path)
    ensure_clean_dir(output_dir, overwrite=args.overwrite)
    codec = choose_codec(args.codec)
    print(f"[INFO] Using codec: {codec}")

    manifest_rows: list[dict[str, object]] = []
    for row_index, record in iter_records(records, args.start_index, args.limit):
        input_path = video_dir / record.video_path
        output_path = output_dir / input_path.name
        info = {
            "index": row_index,
            "video_path": record.video_path,
            "selected_class": record.selected_class,
            "selected_subclass": record.selected_subclass,
            "instruction": record.instruction,
        }
        if not input_path.exists():
            info["status"] = "missing_input"
            manifest_rows.append(info)
            raise FileNotFoundError(f"Input video not found: {input_path}")

        result = process_video(record, input_path, output_path, codec=codec)
        if codec != result["codec"]:
            print(f"[INFO] Switching codec for remaining videos: {codec} -> {result['codec']}")
            codec = str(result["codec"])
        info.update(result)
        manifest_rows.append(info)
        print(
            f"[INFO] {row_index:03d} {record.selected_class} / {record.selected_subclass} -> {output_path.name}"
        )

    write_manifest(manifest_path, manifest_rows)
    validation = validate_outputs(records, video_dir, output_dir, args.start_index, args.limit)
    write_manifest(validation_path, validation)
    if validation["status"] != "ok":
        raise RuntimeError(
            "Output validation failed: "
            f"missing={len(validation['missing'])} "
            f"extra={len(validation['extra'])} "
            f"frame_mismatches={len(validation['frame_mismatches'])}"
        )
    make_zip(output_dir, output_zip)
    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Wrote validation: {validation_path}")
    print(f"[INFO] Wrote zip: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
