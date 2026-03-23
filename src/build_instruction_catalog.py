#!/usr/bin/env python3
"""Build a reusable catalog of atomic instructions grouped by paraphrase.

The script loads free-form instructions from JSONL/CSV/Parquet, splits them
into sentence-level commands, normalizes each command into a semantic key, and
exports a wide CSV:

    id,inst_1,inst_2,...

Each row corresponds to one normalized atomic instruction, and each column is a
surface-form variant observed in the source data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "all",
    "across",
    "against",
    "around",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "it",
    "its",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "throughout",
    "to",
    "up",
    "while",
    "with",
}

COLOR_WORDS = {
    "black",
    "blonde",
    "blue",
    "brown",
    "dark",
    "electric",
    "gold",
    "golden",
    "green",
    "grey",
    "gray",
    "hot",
    "indigo",
    "navy",
    "neon",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "violet",
    "white",
    "yellow",
}

STYLE_WORDS = {
    "anime",
    "cartoon",
    "comic",
    "cyberpunk",
    "pixel",
    "pixelart",
    "retro",
    "ukiyoe",
    "watercolor",
    "woodblock",
}

GENERIC_QUALITY_TOKENS = {
    "abrupt",
    "accurate",
    "across",
    "adjusted",
    "adjustment",
    "anti",
    "artifact",
    "artifacts",
    "blend",
    "blending",
    "bleeding",
    "clean",
    "clearly",
    "clip",
    "color",
    "composition",
    "consistent",
    "consistency",
    "defined",
    "detail",
    "distortion",
    "duration",
    "edge",
    "edges",
    "effect",
    "entire",
    "exactly",
    "flicker",
    "flickering",
    "focus",
    "frame",
    "frames",
    "high",
    "jitter",
    "layered",
    "legible",
    "light",
    "lighting",
    "look",
    "maintain",
    "masked",
    "match",
    "matching",
    "motion",
    "natural",
    "naturally",
    "noise",
    "perfectly",
    "position",
    "precisely",
    "preserve",
    "preserved",
    "professional",
    "quality",
    "realistic",
    "remain",
    "seamless",
    "sharp",
    "sharpness",
    "smooth",
    "stable",
    "steady",
    "subtle",
    "temporally",
    "temporal",
    "throughout",
    "transition",
    "video",
    "visible",
    "without",
    "zero",
}

SUBJECT_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bman'?s\b"), "subject"),
    (re.compile(r"\bwoman'?s\b"), "subject"),
    (re.compile(r"\bbusinessman'?s\b"), "subject"),
    (re.compile(r"\bspeaker'?s\b"), "subject"),
    (re.compile(r"\bpresenter'?s\b"), "subject"),
    (re.compile(r"\bplayer'?s\b"), "subject"),
    (re.compile(r"\bhe\b"), "subject"),
    (re.compile(r"\bhim\b"), "subject"),
    (re.compile(r"\bhis\b"), "subject"),
    (re.compile(r"\bshe\b"), "subject"),
    (re.compile(r"\bher\b"), "subject"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/workspace/data/annotations.jsonl",
        help="Input file or directory. Supports .jsonl, .csv, .parquet.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/data/instructions.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--text-column",
        default="instruction",
        help="Preferred column/key name for instruction text.",
    )
    return parser.parse_args()


def load_records(path: Path, text_column: str) -> list[str]:
    if path.is_dir():
        records: list[str] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".jsonl", ".csv", ".parquet"}:
                records.extend(load_records(child, text_column))
        return records

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path, text_column)
    if suffix == ".csv":
        return load_csv(path, text_column)
    if suffix == ".parquet":
        return load_parquet(path, text_column)
    raise ValueError(f"Unsupported input format: {path}")


def load_jsonl(path: Path, text_column: str) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = find_instruction_field(obj, text_column)
            if text:
                rows.append(text)
    return rows


def load_csv(path: Path, text_column: str) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            text = find_instruction_field(record, text_column)
            if text:
                rows.append(text)
    return rows


def load_parquet(path: Path, text_column: str) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(path)
    if text_column in df.columns:
        return [str(value) for value in df[text_column].dropna().tolist()]

    for candidate in ("instruction", "instructions", "prompt", "text"):
        if candidate in df.columns:
            return [str(value) for value in df[candidate].dropna().tolist()]

    raise KeyError(f"No instruction-like column found in {path}")


def find_instruction_field(obj: object, preferred_key: str) -> str:
    if isinstance(obj, dict):
        for key in (preferred_key, "instruction", "instructions", "prompt", "text"):
            value = obj.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        for value in obj.values():
            nested = find_instruction_field(value, preferred_key)
            if nested:
                return nested
    return ""


def split_atomic_commands(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = [part.strip(" ,") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts or [text]


def normalize_surface(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text[0].upper() + text[1:]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("'", "")
    text = text.replace("lower-third", "lower third")
    text = text.replace("zoom-in", "zoom in")
    text = text.replace("zoom-out", "zoom out")
    text = text.replace("dolly-in", "dolly in")
    text = text.replace("dolly-out", "dolly out")
    text = text.replace("16-bit", "16 bit")
    for pattern, replacement in SUBJECT_NORMALIZERS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prune_tokens(phrase: str) -> str:
    norm = normalize_text(phrase)
    tokens: list[str] = []
    for token in norm.split():
        if token in STOPWORDS:
            continue
        if len(token) == 1:
            continue
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def extract_after(text: str, anchor: str) -> str:
    idx = text.find(anchor)
    if idx < 0:
        return ""
    return text[idx + len(anchor) :].strip()


def compact_phrase(phrase: str, max_tokens: int = 8) -> str:
    kept = prune_tokens(phrase).split()
    return " ".join(kept[:max_tokens])


def normalize_style(phrase: str) -> str:
    norm = prune_tokens(phrase).replace(" ", "_")
    return norm or "style"


def salient_fingerprint(phrase: str, max_tokens: int = 4) -> str:
    tokens: list[str] = []
    for token in prune_tokens(phrase).split():
        if token in GENERIC_QUALITY_TOKENS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    if not tokens:
        return "generic"
    ordered = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return "_".join(ordered[:max_tokens])


def infer_key(command: str) -> str:
    raw = command.strip()
    text = normalize_text(raw)

    camera_patterns = [
        ("dolly in", "camera:dolly_in"),
        ("dolly out", "camera:dolly_out"),
        ("zoom in", "camera:zoom_in"),
        ("zoom out", "camera:zoom_out"),
        ("low angle", "camera:low_angle"),
        ("high angle", "camera:high_angle"),
        ("birds eye", "camera:birdseye"),
        ("overhead", "camera:overhead"),
    ]
    for needle, prefix in camera_patterns:
        if needle in text:
            target = ""
            for marker in ("toward ", "towards ", "on ", "to ", "at "):
                if marker in text:
                    target = compact_phrase(extract_after(text, marker), 5)
                    if target:
                        break
            return f"{prefix}:{target or 'scene'}"

    if text.startswith(("change the color of ", "change ", "adjust the color of ")):
        target = ""
        value = ""
        if " color to " in text:
            left, right = text.split(" color to ", 1)
            target = compact_phrase(left.replace("change the color of", "").replace("change", ""), 5)
            value = compact_phrase(right, 4)
        elif " to " in text:
            left, right = text.split(" to ", 1)
            target = compact_phrase(left.replace("change", ""), 5)
            value = compact_phrase(right, 4)
        return f"edit:change_color:{target or 'target'}:{value or 'value'}"

    if text.startswith("replace "):
        remainder = text[len("replace ") :]
        if " with " in remainder:
            before, after = remainder.split(" with ", 1)
            return (
                "edit:replace:"
                f"{compact_phrase(before, 6) or 'source'}:"
                f"{compact_phrase(after, 6) or 'target'}"
            )
        return f"edit:replace:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith(("remove ", "carefully remove ")):
        remainder = text.split("remove ", 1)[1]
        return f"edit:remove:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith(("insert ", "add ", "place ")):
        action = "insert" if text.startswith("insert ") else "add" if text.startswith("add ") else "place"
        remainder = text.split(" ", 1)[1]
        return f"edit:{action}:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith("increase the number of "):
        remainder = text[len("increase the number of ") :]
        return f"edit:increase_count:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith("increase the amount of "):
        remainder = text[len("increase the amount of ") :]
        return f"edit:increase_amount:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith("transform ") or text.startswith("apply ") or " style" in text:
        for marker in (" into a ", " into an ", " into ", " to a ", " to an ", " style"):
            if marker in text:
                maybe_style = extract_after(text, marker).replace(" style", "")
                style = normalize_style(maybe_style)
                if any(word in style for word in STYLE_WORDS) or style:
                    return f"edit:style:{style}"

    if text.startswith("modify the video so that "):
        remainder = text[len("modify the video so that ") :]
        return f"edit:motion:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith("modify the video to "):
        remainder = text[len("modify the video to ") :]
        return f"edit:modify:{compact_phrase(remainder, 8) or 'target'}"

    if text.startswith("preserve ") or text.startswith("keep "):
        action = "preserve" if text.startswith("preserve ") else "keep"
        remainder = text.split(" ", 1)[1]
        return f"preserve:{action}:{salient_fingerprint(remainder)}"

    if " remain " in text and any(word in text for word in ("must", "should", "remain")):
        left = text.split(" remain ", 1)[0]
        return f"preserve:remain:{salient_fingerprint(left)}"

    if text.startswith("maintain "):
        if "temporal consistency" in text:
            return f"quality:temporal_consistency:{salient_fingerprint(text)}"
        if "clean" in text and "edge" in text:
            return f"quality:clean_edges:{salient_fingerprint(text)}"
        if "sharp focus" in text or "sharpness" in text:
            return f"quality:sharp_focus:{salient_fingerprint(text)}"
        if "lighting" in text and "perspective" in text:
            return f"quality:lighting_perspective_match:{salient_fingerprint(text)}"
        return f"quality:maintain:{salient_fingerprint(text)}"

    if text.startswith("ensure "):
        if any(needle in text for needle in ("temporally consistent", "temporal consistency")):
            return f"quality:temporal_consistency:{salient_fingerprint(text)}"
        if any(needle in text for needle in ("no flickering", "without any flickering", "zero flickering")):
            if "edge" in text or "mask" in text or "outline" in text:
                return f"quality:no_flicker_clean_edges:{salient_fingerprint(text)}"
            return f"quality:no_flicker:{salient_fingerprint(text)}"
        if "clean edges" in text or "sharp edges" in text or "cleanly masked" in text or "cleanly matted" in text:
            return f"quality:clean_edges:{salient_fingerprint(text)}"
        if "lighting" in text and "shadow" in text:
            return f"quality:lighting_shadow_match:{salient_fingerprint(text)}"
        if "sharp focus" in text or "subject sharpness" in text:
            return f"quality:sharp_focus:{salient_fingerprint(text)}"
        if "seamless" in text and "integration" in text:
            return f"quality:seamless_integration:{salient_fingerprint(text)}"
        if "perspective" in text:
            return f"quality:perspective_match:{salient_fingerprint(text)}"
        return f"quality:ensure:{salient_fingerprint(text)}"

    if text.startswith("the ") and " must remain " in text:
        left = text.split(" must remain ", 1)[0]
        return f"preserve:must_remain:{salient_fingerprint(left)}"

    if text.startswith("the ") and any(token in text for token in (" should ", " must ")):
        if "flickering" in text or "artifact" in text:
            return f"quality:no_flicker:{salient_fingerprint(text)}"
        if "light" in text or "lighting" in text:
            return f"quality:lighting_match:{salient_fingerprint(text)}"
        return f"quality:statement:{salient_fingerprint(text)}"

    if text.startswith("throughout the entire"):
        return "quality:full_duration"

    return f"other:{compact_phrase(text, 10) or 'command'}"


def build_catalog(instructions: Iterable[str]) -> OrderedDict[str, list[str]]:
    groups: OrderedDict[str, list[str]] = OrderedDict()
    seen_by_group: dict[str, set[str]] = {}

    for instruction in instructions:
        for command in split_atomic_commands(instruction):
            surface = normalize_surface(command)
            if not surface:
                continue
            key = infer_key(surface)
            if key not in groups:
                groups[key] = []
                seen_by_group[key] = set()
            signature = normalize_text(surface)
            if signature in seen_by_group[key]:
                continue
            groups[key].append(surface)
            seen_by_group[key].add(signature)
    return groups


def write_catalog(groups: OrderedDict[str, list[str]], output_path: Path) -> None:
    max_width = max((len(variants) for variants in groups.values()), default=0)
    fieldnames = ["id"] + [f"inst_{idx}" for idx in range(1, max_width + 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, variants in enumerate(groups.values(), start=1):
            row = {"id": idx}
            for variant_idx, variant in enumerate(variants, start=1):
                row[f"inst_{variant_idx}"] = variant
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    instructions = load_records(input_path, args.text_column)
    groups = build_catalog(instructions)
    write_catalog(groups, output_path)
    print(f"[INFO] Loaded {len(instructions)} instructions")
    print(f"[INFO] Built {len(groups)} unique atomic-instruction groups")
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
