from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.postprocess import add_object_versions as versions


def add_object_frames(
    frames: list[np.ndarray],
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Route add_object execution to selected implementation version.

    Tools: add_object_versions ver1-ver9 (DINO/SAM/RAFT/XMem/OpenCV mix).
    Steps:
    1. Read `add_object_version` selector from params.
    2. Map aliases to concrete version function.
    3. Execute selected pipeline and return processed frames.
    """
    version = str(params.get("add_object_version", "ver2")).lower()
    if version in {"ver1", "1", "first", "first_frame"}:
        return versions.add_object_frames_ver1(
            frames, params, instruction, logger
        )
    if version in {"ver3", "3", "fixed_bbox"}:
        return versions.add_object_frames_ver3(
            frames, params, instruction, logger
        )
    if version in {"ver4", "4", "tracked"}:
        return versions.add_object_frames_ver4(
            frames, params, instruction, logger
        )
    if version in {"ver5", "5", "tracked_sam_fusion", "hybrid"}:
        return versions.add_object_frames_ver5(
            frames, params, instruction, logger
        )
    if version in {"ver6", "6", "xmem", "xmem_hybrid", "tracked_sam_xmem"}:
        return versions.add_object_frames_ver6(
            frames, params, instruction, logger
        )
    if version in {"ver7", "7", "xmem_dup"}:
        return versions.add_object_frames_ver7(
            frames, params, instruction, logger
        )
    if version in {"ver8", "8", "iou_dup", "mask_iou_dup"}:
        return versions.add_object_frames_ver8(
            frames, params, instruction, logger
        )
    if version in {"ver9", "9", "center_dup", "ema_center_dup"}:
        return versions.add_object_frames_ver9(
            frames, params, instruction, logger
        )
    return versions.add_object_frames_ver2(frames, params, instruction, logger)
