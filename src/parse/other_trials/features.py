from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

COLOR_WORDS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "violet",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "gold",
    "beige",
    "brown",
    "navy",
    "emerald",
    "metallic",
    "neon",
]
STYLE_WORDS = ["anime", "cyberpunk", "ghibli", "watercolor", "oil painting", "pixel", "ukiyo-e"]
SHOT_TERMS = [
    "extreme wide shot",
    "wide shot",
    "medium shot",
    "close-up",
    "close up",
    "tight close-up",
    "tight close up",
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "another": 1, "second": 1, "additional": 1}
MASS_NOUN_HINTS = ["jam", "cream", "sauce", "water", "juice", "paint", "powder", "fog", "smoke"]
MOTION_CUES = ["nod", "wave", "turn", "tilt", "rotate", "spin", "shake", "look up", "raise", "hop", "toast"]
EXPRESSION_CUES = ["expression", "smile", "fear", "shock", "pensive", "joyous"]


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("_", " ").strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    char_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return 0.6 * token_overlap + 0.4 * char_ratio


def clone_json(value):
    return json.loads(json.dumps(value))


def merge_dict(base: dict, override: dict) -> dict:
    merged = clone_json(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def clean_candidate(text: str) -> str:
    candidate = (text or "").strip(" .,:;\n\t")
    candidate = re.sub(r"^(the|a|an|entire|current|existing|original)\s+", "", candidate, flags=re.IGNORECASE)
    for marker in [" throughout", " during", " across", " while", " with no", " without", ". ensure", " ensure", ". maintain", " maintain", ", ensuring", ", while"]:
        idx = candidate.lower().find(marker)
        if idx >= 0:
            candidate = candidate[:idx]
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;\n\t")
    return candidate.lower()


def singularize_target(text: str) -> str:
    value = clean_candidate(text)
    replacements = {
        "pastries": "pastry",
        "rhinos and buffalos": "rhino_and_buffalo",
        "rhinos and buffaloes": "rhino_and_buffalo",
        "towel animals": "towel_animal",
        "speed bumps": "speed_bump",
        "jumping baby characters": "jumping_baby_character",
        "cars": "car",
    }
    if value in replacements:
        return replacements[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def parse_count_hint(text: str):
    lowered = text.lower()
    match = re.search(r"\b(\d+)\b(?:\s+more)?", lowered)
    if match:
        return int(match.group(1))
    for token, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return value
    return None


def detect_colors(text: str) -> list[str]:
    lowered = text.lower()
    return [c for c in COLOR_WORDS if re.search(rf"\b{re.escape(c)}\b", lowered)]


def detect_positions(text: str) -> list[str]:
    lowered = text.lower()
    positions = []
    for cue in [
        "left side",
        "right side",
        "center",
        "foreground",
        "background",
        "mid-ground",
        "midground",
        "on the desk",
        "on the plate",
        "on the tray",
        "on the baking tray",
        "in the background",
        "in the foreground",
        "behind",
        "in front of",
        "adjacent",
        "next to",
        "on the left",
        "on the right",
    ]:
        if cue in lowered:
            positions.append(cue.replace("midground", "mid-ground"))
    return positions


def best_examples(record: dict, records: list[dict], action: str, k: int = 3) -> list[dict]:
    scored = []
    for candidate in records:
        if candidate["video_path"] == record["video_path"] and candidate.get("variant") == record.get("variant"):
            continue
        score = text_similarity(record["instruction"], candidate["instruction"])
        if normalize_text(record["class"]) == normalize_text(candidate["class"]):
            score += 0.15
        if normalize_text(record["subclass"]) == normalize_text(candidate["subclass"]):
            score += 0.15
        if candidate["gt_primary"]["action"] == action:
            score += 0.25
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:k]]
