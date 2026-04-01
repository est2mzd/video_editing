from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable


def resolve_video_name_for_progress(params: dict[str, Any]) -> str:
    raw = (
        params.get("video_name")
        or params.get("input_video_name")
        or params.get("mp4_name")
        or params.get("video_path")
        or params.get("_video_path")
        or ""
    )
    text = str(raw)
    if not text:
        return "unknown.mp4"
    return Path(text).name


def iter_frames_with_progress(
    iterable: Iterable,
    params: dict[str, Any],
    action_hint: str,
    stage: str,
):
    action = str(params.get("action", params.get("_action", action_hint)))
    video_name = resolve_video_name_for_progress(params)
    return tqdm(
        iterable,
        desc=f"{action}:{stage} [{video_name}]",
        unit="frame",
        leave=False,
        dynamic_ncols=True,
    )
