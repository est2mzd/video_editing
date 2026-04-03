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
from parse.other_trials.instruction_parser_ver19 import (
    build_predictions,
    parse_annotations_jsonl,
    MULTI_CFG_BEST,
)
from src.postprocess import dispatcher as task_rule_funcs


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
# フレーム処理関数は src/postprocess/dispatcher.py を参照
# ---------------------------------------------------------------------------


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
    method = str(rule.get("method", "identity"))
    params = dict(rule.get("params", {}))
    action = str(task.get("action", ""))

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

    try:
        return task_rule_funcs.run_method(
            action=action,
            targets=task.get("targets", task.get("target")),
            frames=frames,
            params=params,
            instruction=instruction,
            logger=logger,
            method=method,
        )
    except Exception as e:
        if "out of memory" in str(e).lower() or ("cuda" in str(e).lower() and "memory" in str(e).lower()):
            raise RuntimeError(str(e))
        logger.warning(f"  task error [{action}/{method}] {e} → passthrough")
        return frames


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
