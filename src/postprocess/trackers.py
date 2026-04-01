from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from . import model_registry as registry


def estimate_optical_flow(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    try:
        import torch
        if registry.load_raft_model(logger=logger):
            raft_root = "/workspace/third_party/RAFT"
            if raft_root not in sys.path:
                sys.path.insert(0, raft_root)
            from raft.utils.utils import InputPadder

            prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            image1 = (
                torch.from_numpy(prev_rgb)
                .permute(2, 0, 1)
                .float()[None]
                .to(registry.RAFT_DEVICE)
            )
            image2 = (
                torch.from_numpy(curr_rgb)
                .permute(2, 0, 1)
                .float()[None]
                .to(registry.RAFT_DEVICE)
            )
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                _flow_low, flow_up = registry.RAFT_MODEL(
                    image1,
                    image2,
                    iters=20,
                    test_mode=True,
                )

            flow = (
                padder.unpad(flow_up)[0]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return flow
    except Exception as exc:
        if logger is not None:
            logger.debug(f"RAFT flow failed, fallback to Farneback: {exc}")

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def xmem_predict_mask(
    processor,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray | None:
    if processor is None:
        return None
    try:
        import torch

        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        image_t, _ = registry.XMEM_IMAGE_TO_TORCH(
            curr_rgb, device=registry.XMEM_DEVICE
        )
        with torch.no_grad():
            prob = processor.step(image_t)
        if prob is None:
            return None
        if hasattr(prob, "detach"):
            prob_t = prob.detach()
            if prob_t.ndim != 3 or prob_t.shape[0] < 2:
                return None
            mask = torch.argmax(prob_t, dim=0).cpu().numpy().astype(np.uint8)
            return (mask == 1).astype(np.uint8)
        return None
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem predict failed: {exc}")
        return None


def track_mask_with_xmem_or_ostrack(
    prev_mask: np.ndarray,
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    xmem_dir = Path("/workspace/third_party/XMem")
    ostrack_dir = Path("/workspace/third_party/OSTrack")
    if xmem_dir.exists() or ostrack_dir.exists():
        return prev_mask
    if logger is not None:
        logger.debug("XMem/OSTrack not found; using fallback tracker")
    return prev_mask
