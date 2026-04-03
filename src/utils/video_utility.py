from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from IPython.display import HTML, Video, display


def load_video(path: str | Path) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {path}")
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
        raise RuntimeError(f"no frames: {path}")
    return frames, fps, width, height


def write_video(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float,
    width: int,
    height: int,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")
    except Exception:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open writer: {path}")
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()


def show_before_after(video_in: str | Path, video_out: str | Path, width: int = 560) -> None:
    display(HTML(f"<h3>Before: {Path(video_in).name}</h3>"))
    display(Video(Path(video_in).as_posix(), embed=True, width=width))
    display(HTML(f"<h3>After: {Path(video_out).name}</h3>"))
    display(Video(Path(video_out).as_posix(), embed=True, width=width))
    
    
def show_video(video_path: str | Path, width: int = 560) -> None:
    display(Video(Path(video_path).as_posix(), embed=True, width=width))
