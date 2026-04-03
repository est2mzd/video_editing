from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path("/workspace")
SRC_ROOT = WORKSPACE_ROOT / "src"
for _p in [str(SRC_ROOT), str(WORKSPACE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backup import task_rules_ver05_functions as _backup_funcs  # noqa: E402
from utils.video_utility import load_video, write_video  # noqa: E402


ANNOTATION_JSONL_PATH = WORKSPACE_ROOT / "data" / "annotations.jsonl"
VIDEO_ROOT = WORKSPACE_ROOT / "data" / "videos"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "logs" / "test"
RULES_PATH = (
    WORKSPACE_ROOT
    / "logs"
    / "submit"
    / "submission_ver05_json"
    / "task_rules_ver05.json"
)

# task_rules_ver05.json を読み込む
_rules: dict[str, Any] = {}
if RULES_PATH.exists():
    _rules_payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    _rules = _rules_payload.get("actions", {})


def _get_rule_method_and_params(action: str) -> tuple[str, dict[str, Any]]:
    """Return (method, base_params) for the given action."""
    rule = dict(_rules.get(action, {}))
    method = str(rule.get("method", "identity"))
    base_params = dict(rule.get("params", {}))
    return method, base_params


def run_method(
    action: str,
    targets: Any,
    frames: list,
    params: dict[str, Any],
    instruction: str,
    logger: logging.Logger,
) -> list:
    """Wrapper that mirrors notebook run_action_core:
    1. Look up method + base_params from task_rules_ver05.json.
    2. Merge base_params <- params (params override).
     3. Call backup task_rules_ver05_functions.run_method with
         method-based routing.
    """
    method, base_params = _get_rule_method_and_params(action)
    merged = dict(base_params)
    merged.update(params)
    return _backup_funcs.run_method(
        method=method,
        frames=frames,
        params=merged,
        instruction=instruction,
        logger=logger,
    )


@dataclass
class Scenario:
    name: str
    mp4_path: str
    action: str
    target: str
    params_override: dict[str, Any]


# notebook 相当の4ケース
DEFAULT_SCENARIOS: list[Scenario] = [
    # add_object_05.ipynb の最終 add_object（ver9 + buffalo）
    # notebookは target="buffalo" を強制上書きし、
    # temporal_smooth_alpha=0.7 と xmem_mem_every=5 を使う
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
    # apply_style_03.ipynb の apply_style (style='oil_painting' に強制上書き)
    Scenario(
        name="apply_style_03_last_apply_style",
        mp4_path="94msufYZzaQ_26_0to273.mp4",
        action="apply_style",
        target="full_frame",
        params_override={"style": "oil_painting"},
    ),
    # dolly_in_02.ipynb の dolly_in (end_scale=0.5 でZOOM量を上げる)
    Scenario(
        name="dolly_in_02_dolly_in",
        mp4_path="wyzi9GNZFMU_0_0to121.mp4",
        action="dolly_in",
        target="man's face",
        params_override={"end_scale": 0.5},
    ),
    # zoom_in_01.ipynb の zoom_in (zoom_factor=1.0 で notebook 相当)
    Scenario(
        name="zoom_in_01_zoom_in",
        mp4_path="_pQAUwy0yWs_0_119to277.mp4",
        action="zoom_in",
        target="face",
        params_override={"zoom_factor": 1.0},
    ),
]


def _normalize(text: str) -> str:
    return str(text or "").strip().lower()


def _resolve_video_path(mp4_path: str) -> Path:
    p = Path(mp4_path)
    if p.exists():
        return p
    candidate = VIDEO_ROOT / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"video not found: {mp4_path}")


def load_annotations_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no}: json object expected")
            rows.append(obj)
    return rows


def build_annotation_index(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = Path(str(row.get("video_path", ""))).name
        if key:
            index[key] = row
    return index


def build_manual_plan_map() -> dict[str, dict[str, Any]]:
    # 予測なしで使う固定プラン。
    # 将来は Predictor 実装を注入して差し替え可能な構造にする。
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
                "temporal_smooth_alpha": 0.7,  # notebook
                "xmem_mem_every": 5,           # notebook
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
                "end_scale": 0.5,  # notebook: params['end_scale'] = 0.5
            },
        },
        "_pQAUwy0yWs_0_119to277.mp4": {
            "action": "zoom_in",
            "target": "face",
            "params": {
                "motion_type": "zoom_in",
                "speed": "gradual",
                "zoom_factor": 1.0,  # notebook: zoom_factor = 1.0
            },
        },
    }


def resolve_action_plan(
    annotation_row: dict[str, Any],
    plan_map: dict[str, dict[str, Any]],
    fallback_action: str,
    fallback_target: str,
    fallback_params: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any], str]:
    instruction = str(annotation_row.get("instruction", ""))
    video_name = Path(str(annotation_row.get("video_path", ""))).name

    # 1) annotations.jsonl に action/target/params が直接ある場合は最優先でそのまま使用（予測しない）
    direct_action = annotation_row.get("action")
    direct_target = annotation_row.get("target")
    direct_params = annotation_row.get("params")
    if (
        direct_action is not None
        and direct_target is not None
        and isinstance(direct_params, dict)
    ):
        params = copy.deepcopy(direct_params)
        action = str(direct_action)
        target = str(direct_target)
        params["action"] = action
        params["target"] = target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return action, target, params, instruction

    # 2) 既知ケースは固定プランをそのまま使用（予測しない）
    if video_name in plan_map:
        plan = plan_map[video_name]
        action = str(plan["action"])
        target = str(plan["target"])
        params = copy.deepcopy(plan.get("params") or {})
        params["action"] = action
        params["target"] = target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return action, target, params, instruction

    # 3) 単体CLI指定がある場合のフォールバック
    if fallback_action and fallback_target:
        params = copy.deepcopy(fallback_params or {})
        params["action"] = fallback_action
        params["target"] = fallback_target
        params["video_path"] = video_name
        params["instruction"] = instruction
        return fallback_action, fallback_target, params, instruction

    raise LookupError(
        "no direct/manual plan for video="
        f"{video_name}; add action/target/params to annotations.jsonl"
    )


def run_single_case(
    annotation_index: dict[str, dict[str, Any]],
    plan_map: dict[str, dict[str, Any]],
    mp4_path: str,
    action: str,
    target: str,
    output_dir: Path,
    params_override: dict[str, Any] | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    retry: int = 2,
    allow_fallback_output: bool = True,
) -> Path:
    params_override = params_override or {}
    resolved_video = _resolve_video_path(mp4_path)

    ann_row = annotation_index.get(resolved_video.name)
    if ann_row is None:
        raise LookupError(
            f"video not found in annotations.jsonl: {resolved_video.name}"
        )

    resolved_action, resolved_target, params, instruction = (
        resolve_action_plan(
            annotation_row=ann_row,
            plan_map=plan_map,
            fallback_action=action,
            fallback_target=target,
            fallback_params=params_override,
        )
    )
    params.update(params_override)

    frames, fps, width, height = load_video(resolved_video)
    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]
    if frame_stride > 1:
        frames = frames[::frame_stride]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = resolved_video.stem
    action_norm = _normalize(resolved_action).replace(" ", "_")
    out_path = output_dir / f"{stem}__{action_norm}.mp4"

    logger = logging.getLogger("gt_unit_test")
    last_error: Exception | None = None

    for attempt in range(1, retry + 1):
        try:
            out_frames = run_method(
                action=resolved_action,
                targets=resolved_target,
                frames=frames,
                params=copy.deepcopy(params),
                instruction=instruction,
                logger=logger,
            )
            write_video(out_path, out_frames, fps, width, height)
            print(
                "[OK] "
                f"attempt={attempt} video={resolved_video.name} "
                f"action={resolved_action} target={resolved_target} "
                f"out={out_path}"
            )
            return out_path
        except Exception as e:
            last_error = e
            print(
                "[RETRY] "
                f"attempt={attempt}/{retry} video={resolved_video.name} "
                f"action={resolved_action} error={e}"
            )

    if allow_fallback_output:
        fallback_path = output_dir / f"{stem}__{action_norm}__fallback.mp4"
        write_video(fallback_path, frames, fps, width, height)
        print(
            "[FALLBACK] original video exported because action failed. "
            f"out={fallback_path} error={last_error}"
        )
        return fallback_path

    assert last_error is not None
    raise RuntimeError(f"failed after retries: {last_error}")


def run_default_scenarios(
    annotation_index: dict[str, dict[str, Any]],
    plan_map: dict[str, dict[str, Any]],
    output_dir: Path,
    max_frames: int | None,
    frame_stride: int,
    retry: int,
    allow_fallback_output: bool,
) -> list[Path]:
    outputs: list[Path] = []
    for scenario in DEFAULT_SCENARIOS:
        print(f"\n=== scenario: {scenario.name} ===")
        out_path = run_single_case(
            annotation_index=annotation_index,
            plan_map=plan_map,
            mp4_path=scenario.mp4_path,
            action=scenario.action,
            target=scenario.target,
            output_dir=output_dir,
            params_override=scenario.params_override,
            max_frames=max_frames,
            frame_stride=frame_stride,
            retry=retry,
            allow_fallback_output=allow_fallback_output,
        )
        outputs.append(out_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "annotations.jsonl-driven unit test runner "
            "(no prediction mode)"
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
        default=10,
        help="process every N-th frame (default: 10 for full-video runs)",
    )
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument(
        "--no-fallback-output",
        action="store_true",
        help="do not export original video when all retries fail",
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
    allow_fallback_output = not args.no_fallback_output

    if args.run_default_scenarios:
        outputs = run_default_scenarios(
            annotation_index=annotation_index,
            plan_map=plan_map,
            output_dir=output_dir,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            retry=args.retry,
            allow_fallback_output=allow_fallback_output,
        )
        print("\n=== summary ===")
        for p in outputs:
            print(str(p))
        return

    if args.mp4_path and args.action and args.target:
        out_path = run_single_case(
            annotation_index=annotation_index,
            plan_map=plan_map,
            mp4_path=args.mp4_path,
            action=args.action,
            target=args.target,
            output_dir=output_dir,
            params_override=None,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            retry=args.retry,
            allow_fallback_output=allow_fallback_output,
        )
        print(str(out_path))
        return

    raise SystemExit(
        "Specify either --run-default-scenarios "
        "or all of --mp4-path --action --target"
    )


if __name__ == "__main__":
    main()
