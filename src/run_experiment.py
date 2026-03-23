#!/usr/bin/env python3
"""Unified CLI entrypoint.

This file is the single Python entrypoint called by scripts/run.sh.
"""

import argparse
import csv
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipeline.video_processor import VideoProcessor
from src.utils.vace_executor import VaceExecutor


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VACE experiment")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/base.yaml",
        help="Override config path (merged on top of configs/base.yaml)",
    )
    parser.add_argument(
        "input", nargs="?", default="data/default/train/0000.parquet"
    )
    parser.add_argument(
        "prompt", nargs="?", default="Make the scene more rainy"
    )
    parser.add_argument(
        "output", nargs="?", default="data/work/submission_videos.zip"
    )
    parser.add_argument("rows", nargs="?", type=int, default=0)
    parser.add_argument("strict", nargs="?", type=int, default=1)
    parser.add_argument("--python-bin", default="/usr/bin/python3")
    return parser.parse_args(argv)


def deep_merge_dict(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge override into base and return merged dict."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load one YAML file as dict."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"Config file is empty: {path}")
    return data


def load_config(override_config_path: str) -> Dict[str, Any]:
    """Load base config then merge override config on top."""
    base_path = Path("configs/base.yaml")
    override_path = Path(override_config_path)

    base_cfg = _load_yaml(base_path)

    # baseのみ指定時はそのまま返す
    if override_path.resolve() == base_path.resolve():
        return base_cfg

    override_cfg = _load_yaml(override_path)
    return deep_merge_dict(base_cfg, override_cfg)


def resolve_stage(input_path: Path) -> str:
    """Infer stage folder from input path, fallback to infer."""
    parts = {p.lower() for p in input_path.parts}
    for stage in ("train", "eval", "infer"):
        if stage in parts:
            return stage
    return "infer"


def resolve_output_base_dir(config: Dict[str, Any]) -> Path:
    """Resolve the configured experiment base directory."""
    output_cfg = config.get("output", {})
    base_dir = str(output_cfg.get("base_dir", "/workspace/logs")).strip()
    return Path(base_dir or "/workspace/logs")


def build_experiment_context(
    config: Dict[str, Any], input_path: Path
) -> Dict[str, str]:
    """Create experiment id and directory context."""
    output_cfg = config.get("output", {})
    exp_name = (
        str(output_cfg.get("exp_name", "default_exp")).strip()
        or "default_exp"
    )
    exp_content = str(output_cfg.get("exp_content", "")).strip()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{exp_name}_{now}"
    stage = resolve_stage(input_path)

    exp_dir = resolve_output_base_dir(config) / stage / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "exp_name": exp_name,
        "exp_content": exp_content,
        "exp_id": exp_id,
        "stage": stage,
        "exp_dir": str(exp_dir),
    }


def append_experiment_summary(
    config: Dict[str, Any], summary: Dict[str, str]
) -> None:
    """Append one record per run to <base_dir>/exp_summary/exp_summary.csv."""
    summary_dir = resolve_output_base_dir(config) / "exp_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "exp_summary.csv"

    need_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(
                [
                    "timestamp",
                    "exp_id",
                    "exp_content",
                    "stage",
                    "status",
                    "input",
                    "output",
                ]
            )
        writer.writerow(
            [
                summary.get("timestamp", ""),
                summary.get("exp_id", ""),
                summary.get("exp_content", ""),
                summary.get("stage", ""),
                summary.get("status", ""),
                summary.get("input", ""),
                summary.get("output", ""),
            ]
        )


def run_directory(
    config: Dict[str, Any],
    input_dir: Path,
    output_target: Path,
    default_prompt: str,
    python_bin: str,
    limit_rows: int,
    strict_no_fallback: bool,
) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    vace_repo = Path(model_cfg.get("vace_repo", "./third_party/VACE"))
    vace_ckpt_dir = Path(model_cfg.get("model_path", ""))

    executor = VaceExecutor(vace_repo, vace_ckpt_dir, python_bin)
    if strict_no_fallback and not executor.is_available():
        raise RuntimeError("STRICT_NO_FALLBACK=1 but VACE is not available")

    processor = VideoProcessor(config, executor, strict_no_fallback)
    return processor.process_directory(
        input_dir=input_dir,
        output_target=output_target,
        default_prompt=default_prompt,
        limit_rows=limit_rows,
    )


def run_single_mp4(
    config: Dict[str, Any], input_file: Path, prompt: str
) -> Dict[str, Any]:
    from src.pipeline.run_experiment import (
        ExperimentRunner as SingleVideoRunner,
    )

    runner = SingleVideoRunner(config)
    return runner.run(input_video=str(input_file), prompt=prompt)


def run_parquet(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError("Parquet path is not wired yet in this refactor")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    exp_ctx: Dict[str, str] = {}
    run_status = "error"
    run_output = ""

    try:
        config = load_config(args.config)
        input_path = Path(args.input)
        strict_mode = bool(args.strict)

        exp_ctx = build_experiment_context(config, input_path)

        runtime_cfg = dict(config.get("runtime", {}))
        runtime_cfg["exp_id"] = exp_ctx["exp_id"]
        runtime_cfg["exp_dir"] = exp_ctx["exp_dir"]
        runtime_cfg["stage"] = exp_ctx["stage"]
        config["runtime"] = runtime_cfg

        exp_dir = Path(exp_ctx["exp_dir"])
        requested_output = Path(args.output)
        output_name = requested_output.name or "submission.zip"
        output_path = exp_dir / output_name

        with open(
            exp_dir / "config_snapshot.yaml", "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        if input_path.is_dir():
            result = run_directory(
                config=config,
                input_dir=input_path,
                output_target=output_path,
                default_prompt=args.prompt,
                python_bin=args.python_bin,
                limit_rows=args.rows,
                strict_no_fallback=strict_mode,
            )
        elif input_path.suffix.lower() == ".mp4":
            result = run_single_mp4(
                config=config,
                input_file=input_path,
                prompt=args.prompt,
            )
        elif input_path.suffix.lower() == ".parquet":
            result = run_parquet(
                config=config,
                input_file=input_path,
                output_target=output_path,
                default_prompt=args.prompt,
                python_bin=args.python_bin,
                strict_no_fallback=strict_mode,
            )
        else:
            raise ValueError(f"Unsupported input type: {input_path}")

        run_status = str(result.get("status", "ok"))
        run_output = str(result.get("output_path", output_path))

        with open(exp_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[RESULT] {result}")
        return 0 if result.get("status") == "ok" else 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        run_status = "error"
        run_output = str(Path(args.output))
        return 1
    finally:
        if exp_ctx:
            append_experiment_summary(
                config,
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "exp_id": exp_ctx.get("exp_id", ""),
                    "exp_content": exp_ctx.get("exp_content", ""),
                    "stage": exp_ctx.get("stage", ""),
                    "status": run_status,
                    "input": args.input,
                    "output": run_output,
                }
            )


if __name__ == "__main__":
    sys.exit(main())
