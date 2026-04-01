from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

GROUNDING_DINO_MODEL: Any = None
GROUNDING_DINO_TRANSFORMS: Any = None
SAM_PREDICTOR: Any = None
RAFT_MODEL: Any = None
RAFT_DEVICE: str | None = None
XMEM_NETWORK: Any = None
XMEM_DEVICE: str | None = None
XMEM_IMAGE_TO_TORCH: Any = None
XMEMInferenceCore: Any = None


def resolve_grounding_dino_checkpoint_path() -> Path | None:
    candidates = [
        Path("/workspace/weights/groundingdino_swint_ogc.pth"),
        Path(
            "/workspace/third_party/GroundingDINO/weights/"
            "groundingdino_swint_ogc.pth"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_grounding_dino_model(logger: logging.Logger | None = None) -> bool:
    global GROUNDING_DINO_MODEL, GROUNDING_DINO_TRANSFORMS
    if (
        GROUNDING_DINO_MODEL is not None
        and GROUNDING_DINO_TRANSFORMS is not None
    ):
        return True
    try:
        import torch
        from groundingdino.datasets import transforms as T
        from groundingdino.util.inference import load_model

        config_path = (
            "/workspace/third_party/GroundingDINO/groundingdino/config/"
            "GroundingDINO_SwinT_OGC.py"
        )
        checkpoint = resolve_grounding_dino_checkpoint_path()
        if checkpoint is None or not Path(config_path).exists():
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        GROUNDING_DINO_MODEL = load_model(
            config_path, str(checkpoint), device=device
        )
        GROUNDING_DINO_TRANSFORMS = T
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"GroundingDINO load failed: {exc}")
        return False


def load_sam_predictor(logger: logging.Logger | None = None) -> bool:
    global SAM_PREDICTOR
    if SAM_PREDICTOR is not None:
        return True
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        sam_checkpoint = Path("/workspace/weights/sam_vit_h_4b8939.pth")
        if not sam_checkpoint.exists():
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        SAM_PREDICTOR = SamPredictor(sam)
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"SAM load failed: {exc}")
        return False


def load_raft_model(logger: logging.Logger | None = None) -> bool:
    global RAFT_MODEL, RAFT_DEVICE
    if RAFT_MODEL is not None and RAFT_DEVICE is not None:
        return True
    try:
        import torch
        from types import SimpleNamespace

        raft_root = Path("/workspace/third_party/RAFT")
        model_path = raft_root / "models" / "raft-things.pth"
        if not raft_root.exists() or not model_path.exists():
            return False

        if str(raft_root) not in sys.path:
            sys.path.insert(0, str(raft_root))

        from raft.raft import RAFT

        args = SimpleNamespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RAFT(args)

        state = torch.load(str(model_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        RAFT_MODEL = model
        RAFT_DEVICE = device
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"RAFT load failed: {exc}")
        return False


def find_xmem_model_path(params: dict[str, Any] | None = None) -> Path | None:
    candidates: list[Path] = []
    if params is not None:
        explicit = params.get("xmem_model_path")
        if explicit:
            candidates.append(Path(str(explicit)))
    candidates.extend([
        Path("/workspace/third_party/XMem/saves/XMem.pth"),
        Path("/workspace/weights/XMem.pth"),
        Path("/workspace/weights/xmem.pth"),
        Path("/workspace/models/XMem.pth"),
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


def load_xmem_model(
    params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    global XMEM_NETWORK, XMEM_DEVICE, XMEM_IMAGE_TO_TORCH, XMEMInferenceCore
    if (
        XMEM_NETWORK is not None
        and XMEM_DEVICE is not None
        and XMEM_IMAGE_TO_TORCH is not None
        and XMEMInferenceCore is not None
    ):
        return True
    try:
        import torch

        xmem_root = Path("/workspace/third_party/XMem")
        model_path = find_xmem_model_path(params=params)
        if not xmem_root.exists() or model_path is None:
            if logger is not None:
                logger.debug("XMem repo or weight file not found")
            return False

        if str(xmem_root) not in sys.path:
            sys.path.insert(0, str(xmem_root))

        from model.network import XMem as XMemNetwork
        from inference.inference_core import InferenceCore
        from inference.interact.interactive_utils import image_to_torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {
            "model": str(model_path),
            "top_k": int((params or {}).get("xmem_top_k", 30)),
            "mem_every": int((params or {}).get("xmem_mem_every", 5)),
            "deep_update_every": int(
                (params or {}).get("xmem_deep_update_every", -1)
            ),
            "enable_long_term": bool(
                (params or {}).get("xmem_enable_long_term", True)
            ),
            "enable_long_term_count_usage": bool(
                (params or {}).get(
                    "xmem_enable_long_term_count_usage", True
                )
            ),
            "max_mid_term_frames": int(
                (params or {}).get("xmem_max_mid_term_frames", 10)
            ),
            "min_mid_term_frames": int(
                (params or {}).get("xmem_min_mid_term_frames", 5)
            ),
            "num_prototypes": int(
                (params or {}).get("xmem_num_prototypes", 128)
            ),
            "max_long_term_elements": int(
                (params or {}).get("xmem_max_long_term_elements", 10000)
            ),
            "single_object": True,
        }
        network = XMemNetwork(
            config,
            model_path=str(model_path),
            map_location=device,
        )
        network.to(device)
        network.eval()

        XMEM_NETWORK = network
        XMEM_DEVICE = device
        XMEM_IMAGE_TO_TORCH = image_to_torch
        XMEMInferenceCore = InferenceCore
        return True
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem load failed: {exc}")
        return False


def make_xmem_processor(
    first_bgr,
    first_mask,
    params: dict[str, Any],
    logger: logging.Logger | None = None,
):
    if not load_xmem_model(params=params, logger=logger):
        return None
    try:
        import cv2
        import torch

        config = {
            "top_k": int(params.get("xmem_top_k", 30)),
            "mem_every": int(params.get("xmem_mem_every", 5)),
            "deep_update_every": int(params.get("xmem_deep_update_every", -1)),
            "enable_long_term": bool(
                params.get("xmem_enable_long_term", True)
            ),
            "enable_long_term_count_usage": bool(
                params.get("xmem_enable_long_term_count_usage", True)
            ),
            "max_mid_term_frames": int(
                params.get("xmem_max_mid_term_frames", 10)
            ),
            "min_mid_term_frames": int(
                params.get("xmem_min_mid_term_frames", 5)
            ),
            "num_prototypes": int(params.get("xmem_num_prototypes", 128)),
            "max_long_term_elements": int(
                params.get("xmem_max_long_term_elements", 10000)
            ),
        }
        processor = XMEMInferenceCore(XMEM_NETWORK, config)
        processor.set_all_labels([1])

        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        image_t, _ = XMEM_IMAGE_TO_TORCH(first_rgb, device=XMEM_DEVICE)
        mask_t = torch.from_numpy(
            (first_mask > 0).astype("float32")
        ).unsqueeze(0).to(XMEM_DEVICE)
        _ = processor.step(image_t, mask_t)
        return processor
    except Exception as exc:
        if logger is not None:
            logger.debug(f"XMem processor init failed: {exc}")
        return None
