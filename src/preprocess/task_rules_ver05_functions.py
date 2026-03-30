#!/usr/bin/env python3
"""Utility functions for task_rules_ver05 action tuning.

This module is designed for iterative designer-side tuning:
- one action per function
- each action function uses signature:
  (video_path_in, video_path_out, params)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


def read_task_rules(file_path_json: str | Path, video_id: int = 0) -> dict[str, Any]:
	"""Load task-rules JSON and return dictionary payload.

	The video_id argument is kept for compatibility with requested API shape.
	"""
	_ = video_id
	path = Path(file_path_json)
	return json.loads(path.read_text(encoding="utf-8"))


def _load_video(video_path_in: str | Path) -> tuple[list[np.ndarray], float, int, int]:
	cap = cv2.VideoCapture(str(video_path_in))
	if not cap.isOpened():
		raise RuntimeError(f"cannot open video: {video_path_in}")
	fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames: list[np.ndarray] = []
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		frames.append(frame)
	cap.release()
	if not frames:
		raise RuntimeError(f"no frames: {video_path_in}")
	return frames, fps, width, height


def _write_video(video_path_out: str | Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> None:
	out_path = Path(video_path_out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(out_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)
	if not writer.isOpened():
		raise RuntimeError(f"cannot open writer: {video_path_out}")
	for frame in frames:
		if frame.shape[1] != width or frame.shape[0] != height:
			frame = cv2.resize(frame, (width, height))
		writer.write(frame)
	writer.release()


def _process_video(
	video_path_in: str | Path,
	video_path_out: str | Path,
	params: dict[str, Any],
	fn: Callable[[np.ndarray, int, int, int, dict[str, Any]], np.ndarray],
) -> None:
	frames, fps, width, height = _load_video(video_path_in)
	out_frames: list[np.ndarray] = []
	n = len(frames)
	for i, frame in enumerate(frames):
		out_frames.append(fn(frame, i, n, width, params))
	_write_video(video_path_out, out_frames, fps, width, height)


def _identity_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width, params)
	return frame


def _zoom_in_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n)
	max_scale = float(params.get("max_scale", 1.3))
	h, w = frame.shape[:2]
	scale = max(1.0, max_scale)
	nw, nh = int(w / scale), int(h / scale)
	x0 = (w - nw) // 2
	y0 = (h - nh) // 2
	crop = frame[y0 : y0 + nh, x0 : x0 + nw]
	return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _zoom_out_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	min_scale = float(params.get("min_scale", 0.8))
	h, w = frame.shape[:2]
	scale = min(1.0, max(0.2, min_scale))
	sw, sh = int(w * scale), int(h * scale)
	small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
	canvas = np.zeros_like(frame)
	x0 = (w - sw) // 2
	y0 = (h - sh) // 2
	canvas[y0 : y0 + sh, x0 : x0 + sw] = small
	return canvas


def _dolly_in_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = width
	start_scale = float(params.get("start_scale", 1.0))
	end_scale = float(params.get("end_scale", 1.3))
	t = i / max(1, n - 1)
	scale = start_scale + (end_scale - start_scale) * t
	return _zoom_in_frame(frame, i, n, width, {"max_scale": scale})


def _dolly_out_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = width
	start_scale = float(params.get("start_scale", 1.0))
	end_scale = float(params.get("end_scale", 0.8))
	t = i / max(1, n - 1)
	scale = start_scale + (end_scale - start_scale) * t
	return _zoom_out_frame(frame, i, n, width, {"min_scale": scale})


def _perspective_warp_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	h, w = frame.shape[:2]
	strength = float(params.get("strength", 0.05))
	dx = int(w * strength)
	src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
	dst = np.float32([[dx, 0], [w - 1 - dx, 0], [0, h - 1], [w - 1, h - 1]])
	m = cv2.getPerspectiveTransform(src, dst)
	return cv2.warpPerspective(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _orbit_camera_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	h, w = frame.shape[:2]
	max_shift_ratio = float(params.get("max_shift_ratio", 0.1))
	max_shift = int(w * max_shift_ratio)
	phase = (i / max(1, n - 1)) * 2.0 - 1.0
	shift = int(phase * max_shift)
	m = np.float32([[1, 0, shift], [0, 1, 0]])
	return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _replace_background_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	blur_background = bool(params.get("blur_background", True))
	h, w = frame.shape[:2]
	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)
	bg = cv2.GaussianBlur(frame, (0, 0), 9) if blur_background else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if bg.ndim == 2:
		bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
	fg = cv2.bitwise_and(frame, frame, mask=mask)
	inv = cv2.bitwise_not(mask)
	bg_part = cv2.bitwise_and(bg, bg, mask=inv)
	return cv2.add(fg, bg_part)


def _change_color_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	hue_shift = int(params.get("hue_shift", 12))
	sat_scale = float(params.get("sat_scale", 1.1))
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
	hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
	hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
	return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _inpaint_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	h, w = frame.shape[:2]
	radius = float(params.get("inpaint_radius", 5))
	box = params.get("box", [int(w * 0.4), int(h * 0.35), int(w * 0.2), int(h * 0.3)])
	x, y, bw, bh = [int(v) for v in box]
	x = max(0, min(w - 1, x))
	y = max(0, min(h - 1, y))
	bw = max(1, min(w - x, bw))
	bh = max(1, min(h - y, bh))
	mask = np.zeros((h, w), dtype=np.uint8)
	mask[y : y + bh, x : x + bw] = 255
	return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def _stylize_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	sigma = int(params.get("sigma", 75))
	blend = float(params.get("blend", 0.35))
	smooth = cv2.bilateralFilter(frame, 9, sigma, sigma)
	return cv2.addWeighted(smooth, blend, frame, 1.0 - blend, 0)


def _add_effect_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	mode = str(params.get("mode", "brightness"))
	if mode == "blur":
		k = int(params.get("k", 7))
		k = k + 1 if k % 2 == 0 else k
		return cv2.GaussianBlur(frame, (k, k), 0)
	alpha = float(params.get("alpha", 1.1))
	beta = float(params.get("beta", 10.0))
	return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def _histogram_match_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	gain = float(params.get("gain", 1.05))
	gamma = float(params.get("gamma", 0.95))
	lut = np.array([np.clip(((x / 255.0) ** gamma) * 255 * gain, 0, 255) for x in range(256)], dtype=np.uint8)
	return cv2.LUT(frame, lut)


def _sharpness_frame(frame: np.ndarray, i: int, n: int, width: int, params: dict[str, Any]) -> np.ndarray:
	_ = (i, n, width)
	strength = float(params.get("strength", 0.5))
	blur = cv2.GaussianBlur(frame, (0, 0), 2.0)
	return cv2.addWeighted(frame, 1.0 + strength, blur, -strength, 0)


# one action = one function
def zoom_in(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _zoom_in_frame)


def zoom_out(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _zoom_out_frame)


def dolly_in(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _dolly_in_frame)


def dolly_out(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _dolly_out_frame)


def change_camera_angle(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _perspective_warp_frame)


def orbit_camera(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _orbit_camera_frame)


def replace_background(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _replace_background_frame)


def change_color(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _change_color_frame)


def add_object(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def remove_object(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _inpaint_frame)


def replace_object(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def edit_motion(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def apply_style(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _stylize_frame)


def add_effect(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _add_effect_frame)


def increase_amount(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def edit_expression(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_foreground(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_objects(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_identity(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_focus(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_framing(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_layout(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def preserve_material_appearance(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def align_replacement(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def match_appearance(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _histogram_match_frame)


def match_lighting(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _histogram_match_frame)


def match_background_camera_properties(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def match_effect_lighting(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def match_scene_interaction(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_instances(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_edit(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_motion(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_style(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_effect(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_composite(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def stabilize_inpaint(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def refine_mask(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def blend_instances(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def inpaint_background(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _inpaint_frame)


def adjust_perspective(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _perspective_warp_frame)


def track_effect(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


def enhance_style_details(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _sharpness_frame)


def stabilize_object(video_path_in, video_path_out, params):
	_process_video(video_path_in, video_path_out, params, _identity_frame)


ACTION_FUNCTIONS: dict[str, Callable[[str | Path, str | Path, dict[str, Any]], None]] = {
	"zoom_in": zoom_in,
	"zoom_out": zoom_out,
	"dolly_in": dolly_in,
	"dolly_out": dolly_out,
	"change_camera_angle": change_camera_angle,
	"orbit_camera": orbit_camera,
	"replace_background": replace_background,
	"change_color": change_color,
	"add_object": add_object,
	"remove_object": remove_object,
	"replace_object": replace_object,
	"edit_motion": edit_motion,
	"apply_style": apply_style,
	"add_effect": add_effect,
	"increase_amount": increase_amount,
	"edit_expression": edit_expression,
	"preserve_foreground": preserve_foreground,
	"preserve_objects": preserve_objects,
	"preserve_identity": preserve_identity,
	"preserve_focus": preserve_focus,
	"preserve_framing": preserve_framing,
	"preserve_layout": preserve_layout,
	"preserve_material_appearance": preserve_material_appearance,
	"align_replacement": align_replacement,
	"match_appearance": match_appearance,
	"match_lighting": match_lighting,
	"match_background_camera_properties": match_background_camera_properties,
	"match_effect_lighting": match_effect_lighting,
	"match_scene_interaction": match_scene_interaction,
	"stabilize_instances": stabilize_instances,
	"stabilize_edit": stabilize_edit,
	"stabilize_motion": stabilize_motion,
	"stabilize_style": stabilize_style,
	"stabilize_effect": stabilize_effect,
	"stabilize_composite": stabilize_composite,
	"stabilize_inpaint": stabilize_inpaint,
	"refine_mask": refine_mask,
	"blend_instances": blend_instances,
	"inpaint_background": inpaint_background,
	"adjust_perspective": adjust_perspective,
	"track_effect": track_effect,
	"enhance_style_details": enhance_style_details,
	"stabilize_object": stabilize_object,
}


def run_action(action: str, video_path_in: str | Path, video_path_out: str | Path, params: dict[str, Any] | None = None) -> None:
	fn = ACTION_FUNCTIONS.get(action)
	if fn is None:
		raise KeyError(f"unknown action: {action}")
	fn(video_path_in, video_path_out, params or {})

