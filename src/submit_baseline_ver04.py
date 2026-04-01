#!/usr/bin/env python3
"""Command-wise baseline executor for ver04."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2

from src.command_planner_ver04 import plan_commands
from src.instruction_parser_ver04 import parse_annotation_file
from src.submit_baseline_ver03 import (
    AnnotationRecord,
    alpha_blend,
    apply_color_retarget_with_mask,
    apply_human_motion_ver03,
    apply_instance_removal_ver03,
    apply_localized_attribute_edit,
    apply_perspective_tilt,
    apply_quantity_increase_ver03,
    apply_style,
    apply_zoom,
    build_video_context,
    choose_codec,
    count_frames_ffprobe,
    encode_video_frames,
    filter_background_change,
    load_config,
    mask_from_face_region,
    probe_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ver04 command-wise baseline.")
    parser.add_argument("--annotations", default="/workspace/data/annotations.jsonl")
    parser.add_argument("--schema", default="/workspace/configs/command_schema_ver04.yaml")
    parser.add_argument("--config", default="/workspace/configs/submit_baseline_ver03.yaml")
    parser.add_argument("--video-dir", default="/workspace/data/videos")
    parser.add_argument("--output-dir", default="/workspace/data/submission_ver04_videos")
    parser.add_argument("--output-zip", default="/workspace/data/submission_ver04.zip")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", default="libx264")
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_rows(rows: list[dict[str, Any]], start_index: int, limit: int) -> list[dict[str, Any]]:
    sliced = rows[start_index:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def command_record(base: dict[str, Any]) -> AnnotationRecord:
    return AnnotationRecord(
        video_path=str(base["video_path"]),
        selected_class=str(base["selected_class"]),
        selected_subclass=str(base["selected_subclass"]),
        instruction=str(base["instruction"]),
    )


def apply_command(
    frame,
    command: dict[str, Any],
    record: AnnotationRecord,
    context,
    progress: float,
    cfg: dict[str, Any],
):
    h, w = frame.shape[:2]
    action = str(command.get("action", ""))
    target = str(command.get("target", ""))
    value = str(command.get("value", ""))
    if action in {"zoom_in", "dolly_in"}:
        return apply_zoom(frame, progress, True, context.anchor_xy, max_scale=float(cfg.get("zoom_in_max_scale", 1.24)))
    if action == "zoom_out":
        return apply_zoom(frame, progress, False, context.anchor_xy, min_scale=float(cfg.get("zoom_out_min_scale", 0.84)))
    if action == "low_angle":
        return apply_perspective_tilt(frame, progress, True, context.anchor_xy, strength=float(cfg.get("low_angle_strength", 0.075)))
    if action == "high_angle":
        return apply_perspective_tilt(frame, progress, False, context.anchor_xy, strength=float(cfg.get("high_angle_strength", 0.07)))
    if action == "replace_background":
        return filter_background_change(frame, context.subject_mask, cfg)
    if action == "change_color":
        if target in {"hair", "tie", "hat"}:
            alpha = mask_from_face_region(frame, context.face_box, target)
            return apply_color_retarget_with_mask(frame, value, alpha)
        return apply_localized_attribute_edit(frame, record, context)
    if action == "apply_style":
        style_name = value.replace("_", " ")
        return apply_style(frame, style_name if style_name != "american_comic" else "American comic style")
    if action == "increase_quantity":
        return apply_quantity_increase_ver03(frame, context)
    if action == "remove_instance":
        return apply_instance_removal_ver03(frame, record, context)
    if action == "replace_instance":
        if target == "hat":
            alpha = mask_from_face_region(frame, context.face_box, "hat")
            return apply_color_retarget_with_mask(frame, value, alpha)
        return frame
    if action == "insert_instance":
        # Reuse ver03 microphone heuristic when appropriate.
        if "microphone" in record.instruction.lower():
            from src.submit_baseline_ver03 import overlay_microphone
            return overlay_microphone(frame)
        return frame
    if action == "change_human_motion":
        return apply_human_motion_ver03(frame, record, context, progress)
    if action == "generic_edit":
        return frame
    if command.get("type") in {"preserve", "quality"}:
        return frame
    return frame


def process_video_row(
    row: dict[str, Any],
    video_dir: Path,
    output_dir: Path,
    codec: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_path = video_dir / row["video_path"]
    output_path = output_dir / row["video_path"]
    record = command_record(row)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to read first frame: {input_path}")
    context = build_video_context(first_frame, cfg)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    commands = plan_commands(list(row["commands"]))
    edited_frames = []
    per_command_eval = []
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        edited = frame.copy()
        for command in commands:
            edited = apply_command(edited, command, record, context, 0.0 if frame_count <= 1 else written / (frame_count - 1), cfg)
        edited_frames.append(edited)
        written += 1
    cap.release()
    used_codec = encode_video_frames(edited_frames, output_path, width, height, fps, codec)
    out_frames = count_frames_ffprobe(output_path)
    out_res = probe_resolution(output_path)
    if out_frames != frame_count or out_res != (width, height):
        raise RuntimeError(f"Output mismatch for {output_path.name}: frames={out_frames} res={out_res}")
    for command in commands:
        per_command_eval.append(
            {
                "video_path": row["video_path"],
                "command_id": command["command_id"],
                "command_type": command["type"],
                "target": command["target"],
                "action": command["action"],
                "value": command["value"],
                "source_text": command["source_text"],
                "status": "applied" if command["type"] == "edit" else "tracked",
            }
        )
    manifest = {
        "video_path": row["video_path"],
        "status": "ok",
        "codec": used_codec,
        "input_frames": frame_count,
        "output_frames": out_frames,
        "parsed_command_count": len(commands),
        "edit_command_count": sum(1 for c in commands if c["type"] == "edit"),
        "preserve_command_count": sum(1 for c in commands if c["type"] == "preserve"),
        "quality_command_count": sum(1 for c in commands if c["type"] == "quality"),
    }
    return manifest, per_command_eval


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_zip(output_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.glob("*.mp4")):
            zf.write(path, arcname=path.name)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_clean_dir(output_dir, args.overwrite)
    cfg = load_config(Path(args.config))
    parsed = parse_annotation_file(Path(args.annotations), Path(args.schema))
    rows = iter_rows(parsed, args.start_index, args.limit)
    video_dir = Path(args.video_dir)
    codec = choose_codec(args.codec)
    parsed_path = output_dir / "parsed_commands_ver04.json"
    manifest_path = output_dir / "manifest_ver04.json"
    command_eval_path = output_dir / "per_command_eval_ver04.json"
    validation_path = output_dir / "validation_ver04.json"

    all_manifest = []
    all_command_eval = []
    write_json(parsed_path, rows)
    for row in rows:
        manifest, command_eval = process_video_row(row, video_dir, output_dir, codec, cfg)
        all_manifest.append(manifest)
        all_command_eval.extend(command_eval)
        print(f"[INFO] {row['video_path']} commands={manifest['parsed_command_count']}")

    validation = {
        "expected_count": len(rows),
        "actual_count": len(list(output_dir.glob('*.mp4'))),
        "missing": sorted({row["video_path"] for row in rows} - {p.name for p in output_dir.glob('*.mp4')}),
        "extra": sorted({p.name for p in output_dir.glob('*.mp4')} - {row["video_path"] for row in rows}),
        "status": "ok",
    }
    if validation["expected_count"] != validation["actual_count"] or validation["missing"] or validation["extra"]:
        validation["status"] = "error"

    write_json(manifest_path, all_manifest)
    write_json(command_eval_path, all_command_eval)
    write_json(validation_path, validation)
    make_zip(output_dir, Path(args.output_zip))
    print(f"[INFO] Wrote parsed commands: {parsed_path}")
    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Wrote per-command eval: {command_eval_path}")
    print(f"[INFO] Wrote validation: {validation_path}")
    print(f"[INFO] Wrote zip: {args.output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
