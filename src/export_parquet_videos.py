#!/usr/bin/env python3
"""Export mp4 files from parquet and write metadata.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export embedded mp4 bytes from a parquet file."
    )
    parser.add_argument(
        "--parquet",
        default="/workspace/data/0000.parquet",
        help="Path to source parquet file.",
    )
    parser.add_argument(
        "--annotations",
        default="/workspace/data/annotations.jsonl",
        help="Path to annotation jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/data/videos",
        help="Directory to write .mp4 files into.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="/workspace/data/metadata.csv",
        help="Path to metadata.csv output.",
    )
    return parser.parse_args()


def load_annotation_paths(path: Path) -> list[str]:
    paths: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            paths.append(str(obj["video_path"]))
    return paths


def extract_video_record(record: Any) -> tuple[str, bytes]:
    if not isinstance(record, dict):
        raise TypeError(f"Unexpected parquet cell type: {type(record)!r}")

    if "path" not in record or "bytes" not in record:
        raise KeyError("Expected parquet record to contain 'path' and 'bytes'")

    video_path = str(record["path"])
    video_bytes = record["bytes"]
    if not isinstance(video_bytes, (bytes, bytearray)):
        raise TypeError(f"Unexpected video bytes type: {type(video_bytes)!r}")
    return video_path, bytes(video_bytes)


def probe_video(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open written video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_path": str(video_path.resolve()),
        "frame": frame_count,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
    }


def main() -> int:
    args = parse_args()

    parquet_path = Path(args.parquet)
    annotation_path = Path(args.annotations)
    output_dir = Path(args.output_dir)
    metadata_csv = Path(args.metadata_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    annotation_paths = load_annotation_paths(annotation_path)
    df = pd.read_parquet(parquet_path)

    if "video" not in df.columns:
        raise KeyError("Expected parquet to contain a 'video' column")

    video_records = [extract_video_record(row) for row in df["video"].tolist()]
    parquet_paths = [path for path, _ in video_records]

    annotation_set = set(annotation_paths)
    parquet_set = set(parquet_paths)
    if annotation_set != parquet_set:
        missing_in_parquet = sorted(annotation_set - parquet_set)
        missing_in_annotations = sorted(parquet_set - annotation_set)
        raise RuntimeError(
            "Annotation/parquet filename mismatch. "
            f"missing_in_parquet={missing_in_parquet[:5]} "
            f"missing_in_annotations={missing_in_annotations[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for idx, (video_name, video_bytes) in enumerate(video_records):
        output_path = output_dir / video_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)
        rows.append(probe_video(output_path))
        if (idx + 1) % 10 == 0 or idx == len(video_records) - 1:
            print(f"[INFO] Exported {idx + 1}/{len(video_records)} videos")

    rows.sort(key=lambda row: Path(row["video_path"]).name)
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "frame",
                "fps",
                "duration",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote videos to: {output_dir}")
    print(f"[INFO] Wrote metadata to: {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
