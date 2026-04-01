#!/usr/bin/env python3
"""Proxy evaluation for the rule-based GIVE baseline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/workspace/data/annotations.jsonl")
    parser.add_argument("--input-dir", default="/workspace/data/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_sample_frames(path: Path) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set([0, max(0, count // 2), max(0, count - 1)]))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    cap.release()
    return frames, count


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def center_border_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    h, w = a.shape[:2]
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    center_a = a[y0:y1, x0:x1]
    center_b = b[y0:y1, x0:x1]
    center = mean_abs_diff(center_a, center_b)
    mask = np.ones((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = False
    border = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))[mask]))
    return center, border


def estimate_zoom_score(inp: np.ndarray, out: np.ndarray, zoom_in: bool) -> float:
    h, w = inp.shape[:2]
    scale = 1.24 if zoom_in else 0.84
    if zoom_in:
        crop_w = max(2, int(round(w / scale)))
        crop_h = max(2, int(round(h / scale)))
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        ref = cv2.resize(inp[y0:y0 + crop_h, x0:x0 + crop_w], (w, h))
    else:
        small = cv2.resize(inp, (max(2, int(w * scale)), max(2, int(h * scale))))
        ref = np.zeros_like(inp)
        x0 = (w - small.shape[1]) // 2
        y0 = (h - small.shape[0]) // 2
        ref[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    raw = mean_abs_diff(inp, out)
    target = mean_abs_diff(ref, out)
    return raw - target


def edge_strength(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean())


def saturation_mean(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean())


def hue_target_score(frame: np.ndarray, color_name: str) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    center = hsv[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    hue_targets = {
        "violet": 138,
        "purple": 145,
        "blue": 108,
        "navy blue": 112,
        "red": 0,
        "green": 60,
    }
    target = hue_targets.get(color_name, 128)
    hue = center[..., 0].astype(np.float32)
    sat = center[..., 1].astype(np.float32)
    return float(np.mean((180.0 - np.minimum(np.abs(hue - target), 180.0 - np.abs(hue - target))) * (sat / 255.0)))


def pixelation_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    gy = np.abs(np.diff(gray.astype(np.float32), axis=0))
    return float((gx.mean() + gy.mean()) / 2.0)


def target_color_from_instruction(text: str) -> str | None:
    lower = text.lower()
    for color in ["navy blue", "violet", "purple", "red", "blue", "green"]:
        if color in lower:
            return color
    return None


def evaluate_pair(annotation: dict[str, Any], input_frames: list[np.ndarray], output_frames: list[np.ndarray]) -> dict[str, float]:
    inp = input_frames[-1]
    out = output_frames[-1]
    center_diff, border_diff = center_border_diff(inp, out)
    metrics: dict[str, float] = {
        "mean_abs_diff": mean_abs_diff(inp, out),
        "center_diff": center_diff,
        "border_diff": border_diff,
        "saturation_delta": saturation_mean(out) - saturation_mean(inp),
        "edge_delta": edge_strength(out) - edge_strength(inp),
        "pixelation_delta": pixelation_score(out) - pixelation_score(inp),
    }
    cls = annotation["selected_class"]
    sub = annotation["selected_subclass"]
    if cls == "Camera Motion Editing":
        if sub in {"Zoom in", "Dolly in"}:
            metrics["zoom_proxy"] = estimate_zoom_score(inp, out, zoom_in=True)
        elif sub == "Zoom out":
            metrics["zoom_proxy"] = estimate_zoom_score(inp, out, zoom_in=False)
    elif cls == "Visual Effect Editing" and sub == "Background Change":
        metrics["bg_change_ratio"] = border_diff / max(center_diff, 1e-6)
    elif cls == "Attribute Editing" and sub == "Color adjustment":
        color = target_color_from_instruction(annotation["instruction"])
        if color:
            metrics["target_hue_score"] = hue_target_score(out, color) - hue_target_score(inp, color)
    elif cls == "Style Editing":
        metrics["style_energy"] = abs(metrics["saturation_delta"]) + abs(metrics["edge_delta"]) + abs(metrics["pixelation_delta"])
    return metrics


def main() -> int:
    args = parse_args()
    annotations = load_annotations(Path(args.annotations))
    if args.limit > 0:
        annotations = annotations[:args.limit]

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)

    for annotation in annotations:
        name = annotation["video_path"]
        input_path = input_dir / name
        output_path = output_dir / name
        if not output_path.exists():
            continue
        input_frames, input_count = read_sample_frames(input_path)
        output_frames, output_count = read_sample_frames(output_path)
        if not input_frames or not output_frames:
            continue
        metrics = evaluate_pair(annotation, input_frames, output_frames)
        row = {
            "video_path": name,
            "selected_class": annotation["selected_class"],
            "selected_subclass": annotation["selected_subclass"],
            "input_count": input_count,
            "output_count": output_count,
            **metrics,
        }
        rows.append(row)
        for key, value in metrics.items():
            grouped[f"{annotation['selected_subclass']}::{key}"].append(value)

    summary = {
        "rows": rows,
        "summary": {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for key, values in sorted(grouped.items())
        },
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote report: {args.report}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
