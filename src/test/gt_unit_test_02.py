from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("/workspace")
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.gt_unit_runner import (  # noqa: E402
    Scenario,
    build_annotation_index,
    load_annotations_jsonl,
    run_default_scenarios,
    run_single_case,
)

ANNOTATION_JSONL_PATH = WORKSPACE_ROOT / "data" / "annotations.jsonl"
VIDEO_ROOT = WORKSPACE_ROOT / "data" / "videos"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "logs" / "test"

DEFAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="add_object_05_last_add_object",
        mp4_path="1s9DER1bpm0_10_0to213.mp4",
        action="add_object",
        target="buffalo",
        params_override={
            "add_object_version": "ver9",
            "target": "buffalo",
            "temporal_smooth_alpha": 0.7,
            "xmem_mem_every": 5,
        },
    ),
    Scenario(
        name="apply_style_03_last_apply_style",
        mp4_path="94msufYZzaQ_26_0to273.mp4",
        action="apply_style",
        target="full_frame",
        params_override={"style": "oil_painting"},
    ),
    Scenario(
        name="dolly_in_02_dolly_in",
        mp4_path="wyzi9GNZFMU_0_0to121.mp4",
        action="dolly_in",
        target="man's face",
        params_override={"end_scale": 0.5},
    ),
    Scenario(
        name="zoom_in_01_zoom_in",
        mp4_path="_pQAUwy0yWs_0_119to277.mp4",
        action="zoom_in",
        target="face",
        params_override={"zoom_factor": 1.0},
    ),
]


def build_manual_plan_map() -> dict[str, dict[str, Any]]:
    return {
        "1s9DER1bpm0_10_0to213.mp4": {
            "action": "add_object",
            "target": "buffalo",
            "params": {
                "count": 2,
                "position": ["background", "mid-ground"],
                "spatial_distribution": "background",
                "density": "dense",
                "add_object_version": "ver9",
                "temporal_smooth_alpha": 0.7,
                "xmem_mem_every": 5,
            },
        },
        "94msufYZzaQ_26_0to273.mp4": {
            "action": "apply_style",
            "target": "full_frame",
            "params": {"style": "oil_painting"},
        },
        "wyzi9GNZFMU_0_0to121.mp4": {
            "action": "dolly_in",
            "target": "man's face",
            "params": {
                "motion_type": "dolly_in",
                "start_framing": "medium_shot",
                "end_framing": "close_up",
                "end_scale": 0.5,
            },
        },
        "_pQAUwy0yWs_0_119to277.mp4": {
            "action": "zoom_in",
            "target": "face",
            "params": {
                "motion_type": "zoom_in",
                "speed": "gradual",
                "zoom_factor": 1.0,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "annotations.jsonl-driven unit test runner "
            "(no prediction mode, no task_rules json)"
        )
    )
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--action", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="output folder",
    )
    parser.add_argument(
        "--run-default-scenarios",
        action="store_true",
        help="run 4 notebook-equivalent default scenarios",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="optional frame limit for quick validation",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="process every N-th frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ann_rows = load_annotations_jsonl(ANNOTATION_JSONL_PATH)
    annotation_index = build_annotation_index(ann_rows)
    plan_map = build_manual_plan_map()

    output_dir = Path(args.output_dir)

    if args.run_default_scenarios:
        outputs = run_default_scenarios(
            scenarios=DEFAULT_SCENARIOS,
            annotation_index=annotation_index,
            plan_map=plan_map,
            video_root=VIDEO_ROOT,
            output_dir=output_dir,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            logger_name="gt_unit_test_02",
        )
        print("\n=== summary ===")
        for p in outputs:
            print(str(p))
        return

    if args.mp4_path and args.action and args.target:
        out_path = run_single_case(
            annotation_index=annotation_index,
            plan_map=plan_map,
            video_root=VIDEO_ROOT,
            mp4_path=args.mp4_path,
            action=args.action,
            target=args.target,
            output_dir=output_dir,
            params_override=None,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            logger_name="gt_unit_test_02",
        )
        print(str(out_path))
        return

    raise SystemExit(
        "Specify either --run-default-scenarios "
        "or all of --mp4-path --action --target"
    )


if __name__ == "__main__":
    main()
