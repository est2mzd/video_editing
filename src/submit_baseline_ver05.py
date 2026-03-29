#!/usr/bin/env python3
"""submit_baseline_ver05.py

背景:
    GIVE では instruction-following だけでなく、編集の排他性・時間的一貫性が重要で、
    出力動画の frame 数と解像度の一致が必須。

意図:
    GT action を起点に task -> tool を明示的にルーティングし、
    未対応ツールや GPU OOM でも処理全体を止めずに完走させる。

目的:
    1) instruction を読み込む
    2) GT task を読み込み、action ごとに task_rules_ver05.json へルーティング
    3) ツールで処理し、未対応は pass-through
    4) frame 数・解像度を維持して mp4/zip を生成
    5) manifest/validation/task_decomposition を logs に残す

ルーティング理由（runtime probe based）:
    - OpenCV: 推論+mp4 生成まで成功（最優先）
    - GroundingDINO: 推論+mp4 生成成功（mask前段として利用可）
    - SAM2: 推論+mp4 生成成功（mask候補として利用可）
    - RAFT: 推論+mp4 生成成功（flow系候補）
    - VACE: 推論+mp4 生成成功（実行時間長め）
    - Wan: checkpoint 不足で失敗
    ただし submit_baseline_ver05 の task executor は OpenCV ベースで安定運用し、
    その他ツールは runtime probe 結果を記録して routing 可否へ反映する。

結論:
    本スクリプトは GT-first で安全に実行できる提出ベースライン。
    失敗時は pass-through/fallback を優先し、提出フォーマットを壊さない。

Usage:
    python -m src.submit_baseline_ver05 [options]
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import shutil
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ver19 パーサー
from src.parse.instruction_parser_ver19 import (
    build_predictions,
    parse_annotations_jsonl,
    MULTI_CFG_BEST,
)


# ---------------------------------------------------------------------------
# ロギング設定
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, name: str = "ver05") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(sh)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ver05 competition pipeline")
    p.add_argument("--annotations", default="/workspace/data/annotations.jsonl")
    p.add_argument("--gt", default="/workspace/data/annotations_gt_task_ver10.json")
    p.add_argument("--routing-source", choices=["gt", "prediction"], default="gt")
    p.add_argument("--third-party-dir", default="/workspace/third_party")
    p.add_argument("--video-dir", default="/workspace/data/videos")
    p.add_argument("--output-dir", default="/workspace/logs/submit/submission_ver05_videos")
    p.add_argument("--output-zip", default="/workspace/logs/submit/submission_ver05.zip")
    p.add_argument(
        "--task-rules",
        default="/workspace/logs/submit/submission_ver05_json/task_rules_ver05.json",
    )
    p.add_argument(
        "--tool-probe-report",
        default="/workspace/logs/submit/submission_ver05_json/tool_runtime_probe_ver05.json",
    )
    p.add_argument("--log-dir", default="/workspace/logs/submit")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--codec", default="libx264")
    return p.parse_args()


# ---------------------------------------------------------------------------
# タスクルール読み込み
# ---------------------------------------------------------------------------

def load_task_rules(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["actions"]


def ensure_task_rules_file(task_rules_path: Path, logger: logging.Logger) -> Path:
    """task-rules JSON の保存先を logs 配下に統一し、必要なら旧パスから複製する。"""
    if task_rules_path.exists():
        return task_rules_path

    legacy_candidates = [
        Path("/workspace/configs/task_rules_ver05.json"),
    ]
    for legacy in legacy_candidates:
        if legacy.exists():
            task_rules_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, task_rules_path)
            logger.warning(
                f"Task-rules not found at {task_rules_path}; copied from legacy path {legacy}"
            )
            return task_rules_path

    raise FileNotFoundError(
        f"task-rules file not found: {task_rules_path} (and no legacy source found)"
    )


def get_action_rule(rules: dict[str, dict], action: str) -> dict:
    return rules.get(action, rules.get("_default", {"primary_tool": "passthrough", "method": "identity", "params": {}}))


TOOL_PRIORITY = ["opencv", "raft", "vace", "wan"]


def _tool_rank(tool: str) -> int:
    return TOOL_PRIORITY.index(tool) if tool in TOOL_PRIORITY else 999


def build_tool_chain(rule: dict[str, Any]) -> list[tuple[str, str]]:
    primary_tool = str(rule.get("primary_tool", "opencv"))
    primary_method = str(rule.get("method", "identity"))
    secondary_tool = str(rule.get("secondary_tool", "raft"))
    secondary_method = str(rule.get("secondary_method", "identity"))

    ordered = sorted(
        [(primary_tool, primary_method), (secondary_tool, secondary_method)],
        key=lambda x: _tool_rank(x[0]),
    )

    uniq: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tool, method in ordered:
        if tool in seen:
            continue
        seen.add(tool)
        uniq.append((tool, method))
    return uniq[:2]


def discover_available_tools(third_party_dir: Path) -> set[str]:
    available = {"opencv", "passthrough"}
    if not third_party_dir.exists():
        return available

    repo_to_tool = {
        "sam2": "sam2_opencv",
        "segment-anything": "sam_opencv",
        "vace": "vace",
        "wan2.1": "wan",
        "groundingdino": "groundingdino",
        "raft": "raft",
    }
    for child in third_party_dir.iterdir():
        if not child.is_dir():
            continue
        key = child.name.lower()
        for repo_hint, tool_name in repo_to_tool.items():
            if repo_hint in key:
                available.add(tool_name)
    return available


def probe_tool_capabilities(
    third_party_dir: Path,
    log_dir: Path,
) -> dict[str, dict[str, Any]]:
    """利用可能ツールを事前診断する。"""
    caps: dict[str, dict[str, Any]] = {}

    def mark(name: str, ok: bool, reason: str = "") -> None:
        caps[name] = {"available": bool(ok), "reason": reason}

    # OpenCV
    try:
        _ = cv2.__version__
        mark("opencv", True, f"cv2={cv2.__version__}")
    except Exception as e:
        mark("opencv", False, f"cv2 unavailable: {e}")

    def import_probe(label: str, import_stmt: str, extra_paths: list[Path]) -> None:
        py = [str(p.resolve()) for p in extra_paths if p.exists()]
        env = dict(**__import__("os").environ)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(py + ([prev] if prev else []))
        cmd = [
            "python3",
            "-c",
            (
                "import importlib; "
                f"importlib.import_module('{import_stmt}'); "
                "print('ok')"
            ),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.returncode == 0:
            mark(label, True, f"import ok: {import_stmt}")
        else:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            tail = err[-1] if err else "import failed"
            mark(label, False, f"import ng: {import_stmt} ({tail})")

    # third_party + import probes
    import_probe("raft", "core.raft", [third_party_dir / "RAFT"])
    import_probe("vace", "vace", [third_party_dir / "VACE"])
    import_probe("wan", "wan", [third_party_dir / "Wan2.1"])
    import_probe(
        "groundingdino",
        "groundingdino.util.inference",
        [third_party_dir / "GroundingDINO"],
    )
    import_probe("sam2_opencv", "sam2", [third_party_dir / "sam2"])

    # lightweight import probes for key modules (best effort)
    try:
        importlib.import_module("cv2")
    except Exception as e:
        mark("opencv", False, f"import failed: {e}")

    # mp4 writer probe (ffmpeg + cv2 fallback)
    log_dir.mkdir(parents=True, exist_ok=True)
    probe_mp4 = log_dir / "tool_probe_mp4.mp4"
    try:
        frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(5)]
        encode_video_frames(frames, probe_mp4, 96, 64, 10.0, "libx264")
        ok_frames = count_frames_ffprobe(probe_mp4)
        ok_size = probe_resolution(probe_mp4)
        if ok_frames == 5 and ok_size == (96, 64):
            mark("mp4_writer", True, "ffmpeg encoder ok")
        else:
            mark("mp4_writer", False, f"ffmpeg probe mismatch frames={ok_frames} size={ok_size}")
    except Exception as e:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(probe_mp4), fourcc, 10.0, (96, 64))
            if not writer.isOpened():
                raise RuntimeError("cv2 VideoWriter open failed")
            for f in frames:
                writer.write(f)
            writer.release()
            ok_frames = count_frames_ffprobe(probe_mp4)
            ok_size = probe_resolution(probe_mp4)
            if ok_frames == 5 and ok_size == (96, 64):
                mark("mp4_writer", True, f"cv2/mp4v fallback ok after ffmpeg failure: {e}")
            else:
                mark("mp4_writer", False, f"cv2 fallback mismatch after ffmpeg failure: {e}")
        except Exception as e2:
            mark("mp4_writer", False, f"both ffmpeg/cv2 failed: {e}; {e2}")

    return caps


def load_runtime_probe_capabilities(report_path: Path) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        tool = str(row.get("tool", ""))
        status = str(row.get("status", ""))
        mp4_ok = bool(row.get("mp4_info", {}).get("exists", False))
        available = status == "success" and mp4_ok
        out[tool] = {
            "available": available,
            "reason": row.get("error", "runtime probe success"),
            "gpu_before": row.get("gpu_before", {}),
            "gpu_after": row.get("gpu_after", {}),
            "elapsed_sec": row.get("elapsed_sec", None),
        }
    return out


def apply_capabilities_to_rules(
    rules: dict[str, dict[str, Any]],
    caps: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """2段ルーティングを capability に応じて調整する。"""
    adjusted: dict[str, dict[str, Any]] = {}
    change_log: dict[str, Any] = {}
    for action, rule in rules.items():
        r = dict(rule)
        p = str(r.get("primary_tool", "opencv"))
        s = str(r.get("secondary_tool", "raft"))

        p_ok = bool(caps.get(p, {}).get("available", p in {"opencv", "passthrough"}))
        s_ok = bool(caps.get(s, {}).get("available", s in {"opencv", "passthrough"}))

        before = {"primary_tool": p, "secondary_tool": s}
        if not p_ok and s_ok:
            r["primary_tool"], r["secondary_tool"] = s, "passthrough"
            r["method"] = str(r.get("secondary_method", "identity"))
            r["secondary_method"] = "identity"
        elif not p_ok and not s_ok:
            r["primary_tool"], r["secondary_tool"] = "passthrough", "passthrough"
            r["method"] = "identity"
            r["secondary_method"] = "identity"
        elif p_ok and not s_ok:
            r["secondary_tool"] = "passthrough"
            r["secondary_method"] = "identity"

        # mask pipeline is only valid when both groundingdino and sam2 are available
        if "mask_pipeline" in r:
            gd_ok = bool(caps.get("groundingdino", {}).get("available", False))
            sam_ok = bool(caps.get("sam2_opencv", {}).get("available", False))
            if not (gd_ok and sam_ok):
                r["mask_pipeline"] = []
                r["mask_pipeline_disabled_reason"] = "groundingdino/sam2 runtime probe failed"

        after = {
            "primary_tool": r.get("primary_tool"),
            "secondary_tool": r.get("secondary_tool"),
        }
        if before != after:
            change_log[action] = {"before": before, "after": after}

        adjusted[action] = r
    return adjusted, change_log


def load_gt_task_rows(gt_path: Path, annotations_path: Path) -> list[dict[str, Any]]:
    gt_rows = json.loads(gt_path.read_text(encoding="utf-8"))
    ann_rows = parse_annotations_jsonl(annotations_path)
    gt_by_video: dict[str, dict[str, Any]] = {
        str(r.get("video_path", "")): r for r in gt_rows if str(r.get("video_path", ""))
    }

    out: list[dict[str, Any]] = []
    for ann in ann_rows:
        video_path = str(ann.get("video_path", ""))
        if not video_path:
            continue
        gt_row = gt_by_video.get(video_path, {})
        tasks_raw = gt_row.get("tasks", [])
        tasks = tasks_raw if isinstance(tasks_raw, list) else []
        out.append(
            {
                "video_path": video_path,
                "instruction": str(ann.get("instruction", "")),
                "prediction": {"tasks": tasks},
                "routing_source": "gt",
            }
        )
    return out


# ---------------------------------------------------------------------------
# 動画 I/O ユーティリティ
# ---------------------------------------------------------------------------

def read_video(path: Path) -> tuple[list[np.ndarray], float, int, int]:
    """動画をフレームリストとして読み込む。(frames, fps, width, height)"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")
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
        raise RuntimeError(f"No frames read: {path}")
    return frames, fps, width, height


def choose_fourcc(codec: str, output_path: Path) -> str:
    """利用可能なコーデックを選択して返す。"""
    candidates = [codec] if codec not in ("auto", "") else []
    candidates += ["avc1", "mp4v"]
    for cc in candidates:
        try:
            fcc = cv2.VideoWriter_fourcc(*cc[:4].ljust(4))
            tmp = str(output_path.with_suffix(".tmp.mp4"))
            w = cv2.VideoWriter(tmp, fcc, 30.0, (64, 64))
            if w.isOpened():
                w.release()
                Path(tmp).unlink(missing_ok=True)
                return cc
        except Exception:
            pass
    return "mp4v"


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


def write_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str,
) -> None:
    """フレームリストを mp4 に書き込む。frame数・解像度は保証する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized_frames: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if w != width or h != height:
            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(f)

    # 1st: ffmpeg encoder
    try:
        encode_video_frames(resized_frames, output_path, width, height, fps, codec)
    except Exception:
        # 2nd: cv2/mp4v fallback (grounding_dino_sam_02.py と同系統)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
        for f in resized_frames:
            out.write(f)
        out.release()

    out_frames = count_frames_ffprobe(output_path)
    out_res = probe_resolution(output_path)
    if out_frames != len(resized_frames) or out_res != (width, height):
        raise RuntimeError(
            f"Output mismatch: frames={out_frames}/{len(resized_frames)} "
            f"size={out_res}/{(width, height)}"
        )


# ---------------------------------------------------------------------------
# フレーム処理: action ごとの変換ロジック
# ---------------------------------------------------------------------------

def apply_zoom(
    frame: np.ndarray,
    progress: float,
    zoom_in: bool,
    max_scale: float = 1.3,
    min_scale: float = 0.8,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if zoom_in:
        scale = 1.0 + (max_scale - 1.0) * progress
    else:
        scale = 1.0 - (1.0 - min_scale) * progress

    if abs(scale - 1.0) < 1e-4:
        return frame

    if scale > 1.0:
        # crop center then resize back
        new_w = int(w / scale)
        new_h = int(h / scale)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # shrink then pad
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(frame)
        ox = (w - new_w) // 2
        oy = (h - new_h) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        return canvas


def apply_perspective_warp(frame: np.ndarray, strength: float = 0.07) -> np.ndarray:
    h, w = frame.shape[:2]
    s = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * s, h * s],
        [w * (1 - s), 0],
        [w * (1 - s * 0.5), h],
        [w * s * 0.5, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def extract_target_color(instruction: str) -> str | None:
    color_names = [
        "navy blue", "violet", "purple", "red", "blue", "green",
        "yellow", "orange", "pink", "black", "white", "silver",
    ]
    lower = instruction.lower()
    for c in color_names:
        if c in lower:
            return c
    return None


HSV_BOUNDS: dict[str, tuple[tuple, tuple]] = {
    "red":      ((0, 70, 40),   (12, 255, 255)),
    "orange":   ((8, 70, 40),   (25, 255, 255)),
    "yellow":   ((20, 60, 40),  (38, 255, 255)),
    "green":    ((35, 40, 30),  (90, 255, 255)),
    "blue":     ((90, 40, 30),  (130, 255, 255)),
    "navy blue":((100, 40, 20), (125, 255, 180)),
    "violet":   ((125, 40, 30), (155, 255, 255)),
    "purple":   ((125, 40, 30), (155, 255, 255)),
    "pink":     ((150, 30, 40), (179, 255, 255)),
    "black":    ((0, 0, 0),     (179, 255, 55)),
    "white":    ((0, 0, 180),   (179, 70, 255)),
    "silver":   ((0, 0, 80),    (179, 50, 230)),
}


def apply_color_change(frame: np.ndarray, instruction: str) -> np.ndarray:
    color_name = extract_target_color(instruction)
    if color_name is None or color_name not in HSV_BOUNDS:
        return frame
    low, high = HSV_BOUNDS[color_name]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    target_h = (int(low[0]) + int(high[0])) // 2
    hsv[:, :, 0] = target_h
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_style(frame: np.ndarray) -> np.ndarray:
    # Fast style approximation for stable long-batch processing.
    smooth = cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)
    return cv2.addWeighted(frame, 0.7, smooth, 0.3, 0)


def apply_inpaint(frame: np.ndarray, radius: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    if mask.sum() == 0:
        return frame
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def apply_background_blur(frame: np.ndarray) -> np.ndarray:
    """背景ぼかし (GrabCut で前景抽出 → 背景ガウシアン)"""
    h, w = frame.shape[:2]
    mask_gc = np.zeros((h, w), dtype=np.uint8)
    rect = (w // 8, h // 8, w * 3 // 4, h * 3 // 4)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame, mask_gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.uint8)
    except Exception:
        fg_mask = np.ones((h, w), dtype=np.uint8)

    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    fg_mask_3ch = fg_mask[:, :, None]
    return (frame * fg_mask_3ch + blurred * (1 - fg_mask_3ch)).astype(np.uint8)


def build_mask_from_grounding_sam(
    frame: np.ndarray,
    task: dict,
    rule: dict,
    available_tools: set[str],
    logger: logging.Logger,
) -> np.ndarray | None:
    """GroundingDINO + SAM を想定したマスク生成の入口。

    現段階ではランタイム統合を行わず、ツールが無い/失敗時は None を返して
    下流で pass-through または OpenCV 近似へフォールバックする。
    """
    pipeline = rule.get("mask_pipeline", [])
    if not isinstance(pipeline, list) or not pipeline:
        return None

    missing = [t for t in pipeline if t not in available_tools]
    if missing:
        logger.warning(f"  mask pipeline unavailable for {task.get('action', '')}: missing={missing}")
        return None

    # TODO: GroundingDINO/SAM 実統合時に差し替え
    logger.debug(f"  mask pipeline requested ({pipeline}) but runtime integration is pending; fallback to OpenCV heuristic")
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w // 8, h // 8), (w * 7 // 8, h * 7 // 8), 255, -1)
    return mask


def apply_sharpness(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32) * strength
    kernel[1, 1] = 1.0 + 8.0 * strength
    return cv2.filter2D(frame, -1, kernel)


def apply_horizontal_shift(frame: np.ndarray, progress: float, max_ratio: float = 0.1) -> np.ndarray:
    h, w = frame.shape[:2]
    shift = int(w * max_ratio * progress)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    return cv2.warpAffine(frame, M, (w, h))


def apply_histogram_match(frame: np.ndarray) -> np.ndarray:
    # 自己参照のhistogram equalization（参照画像なし）
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ---------------------------------------------------------------------------
# タスク実行
# ---------------------------------------------------------------------------

def apply_task_to_frames(
    frames: list[np.ndarray],
    task: dict,
    rule: dict,
    tool_name: str,
    available_tools: set[str],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """1 タスクをフレームリスト全体に適用する。"""
    method = rule.get("method", "identity")
    params = rule.get("params", {})
    action = task.get("action", "")
    n = len(frames)

    if tool_name not in {
        "opencv",
        "sam2_opencv",
        "sam_opencv",
        "groundingdino",
        "raft",
        "vace",
        "wan",
        "passthrough",
    }:
        raise RuntimeError(f"unsupported tool: {tool_name}")

    if tool_name in {"raft", "vace", "wan"}:
        raise RuntimeError(f"tool not integrated yet: {tool_name}")

    if method == "identity" or method is None:
        return frames

    static_mask = build_mask_from_grounding_sam(frames[0], task, rule, available_tools, logger)

    result: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        progress = i / max(n - 1, 1)
        try:
            if method == "crop_resize":
                out = apply_zoom(frame, progress, zoom_in=True, max_scale=params.get("max_scale", 1.3))

            elif method == "resize_pad":
                out = apply_zoom(frame, progress, zoom_in=False, min_scale=params.get("min_scale", 0.8))

            elif method == "progressive_crop_resize":
                out = apply_zoom(
                    frame, progress, zoom_in=True,
                    max_scale=params.get("end_scale", 1.3),
                )

            elif method == "progressive_resize_pad":
                out = apply_zoom(
                    frame, progress, zoom_in=False,
                    min_scale=params.get("end_scale", 0.8),
                )

            elif method == "perspective_warp":
                out = apply_perspective_warp(frame, params.get("strength", 0.07))

            elif method == "horizontal_shift":
                out = apply_horizontal_shift(frame, progress, params.get("max_shift_ratio", 0.1))

            elif method == "hsv_retarget":
                out = apply_color_change(frame, instruction)
                if static_mask is not None:
                    out = np.where(static_mask[:, :, None] > 0, out, frame)

            elif method == "segment_and_replace":
                if static_mask is not None:
                    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
                    out = np.where(static_mask[:, :, None] > 0, frame, blurred)
                else:
                    out = apply_background_blur(frame)

            elif method == "opencv_blur":
                out = apply_background_blur(frame)

            elif method == "inpaint":
                if static_mask is not None:
                    out = cv2.inpaint(frame, static_mask, params.get("inpaint_radius", 5), cv2.INPAINT_TELEA)
                else:
                    out = apply_inpaint(frame, params.get("inpaint_radius", 5))

            elif method == "stylize":
                out = apply_style(frame)

            elif method == "blur_or_brightness":
                out = cv2.GaussianBlur(frame, (5, 5), 0)

            elif method == "sharpness":
                out = apply_sharpness(frame, params.get("strength", 0.5))

            elif method == "histogram_match":
                out = apply_histogram_match(frame)

            else:
                out = frame

            result.append(out)

        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower() and "memory" in str(e).lower():
                raise RuntimeError(str(e))
            logger.warning(f"  frame {i} [{action}/{method}] error: {e} → passthrough")
            result.append(frame)

    return result


# ---------------------------------------------------------------------------
# 1 動画の処理
# ---------------------------------------------------------------------------

def process_one_video(
    row: dict,
    video_dir: Path,
    output_dir: Path,
    rules: dict[str, dict],
    codec: str,
    available_tools: set[str],
    logger: logging.Logger,
) -> dict:
    video_name = row["video_path"]
    input_path = video_dir / video_name
    output_path = output_dir / video_name
    instruction = row.get("instruction", "")
    tasks = row.get("prediction", {}).get("tasks", [])

    logger.info(f"[START] {video_name}  tasks={[t['action'] for t in tasks]}")

    frames, fps, width, height = read_video(input_path)
    n_frames_in = len(frames)

    skipped_actions: list[str] = []
    applied_actions: list[str] = []

    for task in tasks:
        action = task.get("action", "")
        rule = get_action_rule(rules, action)
        chain = build_tool_chain(rule)
        applied = False
        for idx, (tool_name, method_name) in enumerate(chain, start=1):
            if tool_name == "passthrough":
                continue
            if tool_name not in available_tools:
                logger.warning(f"  tool unavailable: {tool_name} for {action} (choice#{idx})")
                continue

            try:
                run_rule = dict(rule)
                run_rule["method"] = method_name
                frames = apply_task_to_frames(frames, task, run_rule, tool_name, available_tools, instruction, logger)
                applied_actions.append(f"{action}[{tool_name}]")
                logger.debug(f"  applied: {action} / {tool_name} / {method_name}")
                applied = True
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"  OOM on {action} with {tool_name}: {e}")
                else:
                    logger.warning(f"  runtime error on {action} with {tool_name}: {e}")
            except Exception as e:
                logger.warning(f"  unexpected error on {action} with {tool_name}: {e}")

        if not applied:
            skipped_actions.append(action)

    # フレーム数・解像度の検証と強制補正
    n_frames_out = len(frames)
    if n_frames_out != n_frames_in:
        logger.warning(f"  frame count mismatch: {n_frames_out} != {n_frames_in} → padding/truncate")
        if n_frames_out < n_frames_in:
            frames += [frames[-1]] * (n_frames_in - n_frames_out)
        else:
            frames = frames[:n_frames_in]

    write_video(frames, output_path, fps, width, height, codec)
    logger.info(f"[DONE] {video_name}  applied={applied_actions}  skipped={skipped_actions}")

    return {
        "video_path": video_name,
        "status": "ok",
        "input_frames": n_frames_in,
        "output_frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
        "applied_actions": applied_actions,
        "skipped_actions": skipped_actions,
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    logger = setup_logger(log_dir, "submit_baseline_ver05")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir)
    gt_path = Path(args.gt)
    task_rules_path = ensure_task_rules_file(Path(args.task_rules), logger)
    rules = load_task_rules(task_rules_path)
    available_tools = discover_available_tools(Path(args.third_party_dir))

    probe_report_path = Path(args.tool_probe_report)
    if not probe_report_path.exists():
        fallback_candidates = [
            Path("/workspace/logs/submit/tool_runtime_probe_ver05.json"),
            Path("/workspace/logs/experiments/tool_runtime_probe_ver05.json"),
        ]
        for legacy_probe_path in fallback_candidates:
            if legacy_probe_path.exists():
                logger.warning(
                    f"Probe report not found at {probe_report_path}; fallback to legacy path {legacy_probe_path}"
                )
                probe_report_path = legacy_probe_path
                break

    runtime_caps = load_runtime_probe_capabilities(probe_report_path)
    if runtime_caps:
        tool_caps = runtime_caps
        logger.info(f"Loaded runtime probe report: {probe_report_path}")
    else:
        tool_caps = probe_tool_capabilities(Path(args.third_party_dir), Path(args.log_dir))
        logger.info("Runtime probe report not found; using import/mp4 probe")

    # submit_baseline_ver05.py 内で実際に task 実行へ統合済みのツールのみ有効化する
    integrated_tools = {
        "opencv",
        "groundingdino",
        "sam2_opencv",
        "sam_opencv",
        "passthrough",
    }
    for t, info in tool_caps.items():
        if t not in integrated_tools and bool(info.get("available", False)):
            info["available"] = False
            reason = str(info.get("reason", ""))
            info["reason"] = (
                reason + " | disabled in routing: not integrated in submit_baseline_ver05"
            ).strip(" |")

    adjusted_rules, routing_changes = apply_capabilities_to_rules(rules, tool_caps)
    rules = adjusted_rules
    logger.info(f"Available tools: {sorted(available_tools)}")
    logger.info("Tool capability probe done")

    # --- 1. instruction 読み込み ---
    logger.info("Step 1: Loading annotations")
    records = parse_annotations_jsonl(Path(args.annotations))
    logger.info(f"  {len(records)} records loaded")

    # --- 2. タスク分解 / ルーティング元選択 ---
    logger.info(f"Step 2: Building tasks (source={args.routing_source})")
    try:
        if args.routing_source == "gt":
            predictions = load_gt_task_rows(gt_path, Path(args.annotations))
        else:
            predictions = build_predictions(records, gt_path, mode="multi", cfg=MULTI_CFG_BEST)
            for row in predictions:
                row["routing_source"] = "prediction"
    except Exception as e:
        logger.error(f"Task decomposition failed: {e}")
        return 1

    # index/limit
    predictions = predictions[args.start_index:]
    if args.limit > 0:
        predictions = predictions[:args.limit]
    logger.info(f"  {len(predictions)} videos to process")

    # --- 3. 動画処理 ---
    logger.info("Step 3: Video processing")
    manifest: list[dict] = []
    errors: list[dict] = []

    codec = args.codec if args.codec not in ("auto", "") else "mp4v"

    for row in predictions:
        video_name = row["video_path"]
        input_path = video_dir / video_name
        if not input_path.exists():
            logger.warning(f"  MISSING: {video_name}")
            errors.append({"video_path": video_name, "error": "input not found"})
            continue
        try:
            result = process_one_video(row, video_dir, output_dir, rules, codec, available_tools, logger)
            manifest.append(result)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"  FAILED: {video_name}: {e}\n{tb}")
            errors.append({"video_path": video_name, "error": str(e)})

    # --- 4. 出力検証 ---
    logger.info("Step 4: Validation")
    expected = {row["video_path"] for row in predictions}
    actual = {p.name for p in output_dir.glob("*.mp4")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    validation = {
        "expected": len(expected),
        "actual": len(actual),
        "missing": missing,
        "extra": extra,
        "errors": errors,
        "output_dir": str(output_dir),
        "status": "ok" if not missing else "incomplete",
    }
    logger.info(f"  expected={len(expected)} actual={len(actual)} missing={len(missing)}")

    # ログ出力
    log_output_dir = Path(args.log_dir)
    log_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir = log_output_dir / "submission_ver05_json"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    (json_output_dir / "manifest_ver05.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (json_output_dir / "validation_ver05.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    task_decomp_path = json_output_dir / "task_decomposition_ver05.json"
    task_decomp_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    routing_summary_path = json_output_dir / "routing_summary_ver05.json"
    routing_by_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in predictions:
        for task in row.get("prediction", {}).get("tasks", []):
            action = str(task.get("action", ""))
            primary_tool = str(get_action_rule(rules, action).get("primary_tool", "passthrough"))
            routing_by_action[action][primary_tool] += 1
    routing_summary_path.write_text(
        json.dumps({k: dict(v) for k, v in sorted(routing_by_action.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (json_output_dir / "tool_capabilities_ver05.json").write_text(
        json.dumps(tool_caps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (json_output_dir / "routing_adjustments_ver05.json").write_text(
        json.dumps(routing_changes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  manifest → {json_output_dir}/manifest_ver05.json")
    logger.info(f"  task decomposition → {task_decomp_path}")
    logger.info(f"  routing summary → {routing_summary_path}")
    logger.info(f"  tool capabilities → {json_output_dir}/tool_capabilities_ver05.json")
    logger.info(f"  routing adjustments → {json_output_dir}/routing_adjustments_ver05.json")

    # --- 5. ZIP 作成 ---
    logger.info("Step 5: Creating zip")
    output_zip = Path(args.output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mp4 in sorted(output_dir.glob("*.mp4")):
            zf.write(mp4, arcname=mp4.name)
    logger.info(f"  zip → {output_zip}  ({output_zip.stat().st_size // 1024} KB)")

    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
