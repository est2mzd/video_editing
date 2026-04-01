#!/usr/bin/env python3
"""Build short atomic instructions and constraints for VACE evaluation.

Outputs:
  - instructions.csv: unique short edit commands
  - constraints.csv / constrains.csv: unique short constraints
  - instruction_constraint_map.csv: mapping from each source instruction to its parts
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


COMMAND_VERBS = (
    "apply",
    "change",
    "replace",
    "remove",
    "increase",
    "decrease",
    "add",
    "place",
    "transform",
    "modify",
    "perform",
    "shift",
    "adjust",
    "execute",
    "enhance",
    "animate",
    "make",
    "turn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/workspace/data/annotations.jsonl")
    parser.add_argument("--outdir", default="/workspace/data")
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            obj = json.loads(line)
            rows.append(
                {
                    "source_id": str(idx),
                    "video_path": str(obj.get("video_path", "")),
                    "selected_class": str(obj.get("selected_class", "")),
                    "selected_subclass": str(obj.get("selected_subclass", "")),
                    "instruction": str(obj.get("instruction", "")).strip(),
                }
            )
    return rows


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [part.strip(" ,") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_text(text: str) -> str:
    text = text.strip().strip(".").strip(",")
    text = re.sub(r"\s+", " ", text)
    return text


def sentence_stem(text: str) -> str:
    lowered = clean_text(text).lower()
    lowered = lowered.replace("'", "")
    lowered = lowered.replace("zoom-in", "zoom in")
    lowered = lowered.replace("zoom-out", "zoom out")
    lowered = lowered.replace("dolly-in", "dolly in")
    lowered = lowered.replace("dolly-out", "dolly out")
    lowered = lowered.replace("lower-third", "lower third")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def shorten_command(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r",\s*starting from .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*gradually .*?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*making .*?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+while .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+without .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+with no .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+avoiding .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+to create .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+to capture .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+throughout the entire (?:duration of the )?video(?: sequence)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+throughout the edit", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def simplify_phrase(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\b(directly|slightly|subtly|slowly|gradually|smoothly|carefully|gently|rapidly|continuously)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(in the scene|in the video|throughout the video|throughout the clip|throughout the sequence)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(on the left side of the frame|on the right side of the screen|behind the speaker|in the background)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(featuring|using|characterized by|to focus on|to reveal more of).*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(the|a|an)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(existing|entire|plain|solid|blurred|blurry|bright|vibrant|sleek|modern|cinematic|dimly lit|professional-grade)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(man|woman|businessman|speaker|presenter|subject|character)s\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshade of\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\blying on ground in mid ground and background areas\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return clean_text(text)


def prettify_command(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\bmans\b", "the man's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwomans\b", "the woman's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbusinessmans\b", "the businessman's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsubjects\b", "the subject's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcharacters\b", "the character's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bman his left hand\b", "the man's left hand", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwoman her head\b", "the woman's head", text, flags=re.IGNORECASE)
    return titlecase_first(text)


def split_compound_commands(commands: list[str]) -> list[str]:
    out: list[str] = []
    for command in commands:
        low = sentence_stem(command)

        m = re.match(r"add more (.+?) and (.+)", low)
        if m:
            out.append(prettify_command(f"Add more {simplify_phrase(m.group(1))}"))
            out.append(prettify_command(f"Add more {simplify_phrase(m.group(2))}"))
            continue

        m = re.match(r"remove (.+?) and (.+)", low)
        if m:
            left = simplify_phrase(m.group(1))
            right = simplify_phrase(m.group(2))
            right = re.sub(r"\bfrom video inpainting.*$", "", right, flags=re.IGNORECASE).strip()
            if left:
                out.append(prettify_command(f"Remove {left}"))
            if right:
                out.append(prettify_command(f"Remove {right}"))
            continue

        m = re.match(r"replace (.+?) and (.+?) with (.+)", low)
        if m:
            out.append(prettify_command(f"Replace {simplify_phrase(m.group(1))} with {simplify_phrase(m.group(3))}"))
            out.append(prettify_command(f"Replace {simplify_phrase(m.group(2))} with {simplify_phrase(m.group(3))}"))
            continue

        m = re.match(r"change (.+?) and transform (.+?) into (.+)", low)
        if m:
            out.append(prettify_command(f"Change {simplify_phrase(m.group(1))}"))
            out.append(prettify_command(f"Transform {simplify_phrase(m.group(2))} into {simplify_phrase(m.group(3))}"))
            continue

        m = re.match(r"increase (.+?) and introduce (.+)", low)
        if m:
            out.append(prettify_command(f"Increase {simplify_phrase(m.group(1))}"))
            out.append(prettify_command(f"Introduce {simplify_phrase(m.group(2))}"))
            continue

        m = re.match(r"edit (.+?) to (.+?) and perform (.+)", low)
        if m:
            out.append(prettify_command(f"Make {simplify_phrase(m.group(1))} {simplify_phrase(m.group(2))}"))
            out.append(prettify_command(simplify_phrase(m.group(3))))
            continue

        m = re.match(r"(.+?) and add (.+)", low)
        if m:
            out.append(prettify_command(m.group(1)))
            out.append(prettify_command(f"Add {simplify_phrase(m.group(2))}"))
            continue

        m = re.match(r"(.+?) adding (.+)", low)
        if m and low.startswith("apply "):
            out.append(prettify_command(m.group(1)))
            out.append(prettify_command(f"Add {simplify_phrase(m.group(2))}"))
            continue

        m = re.match(r"(.+?) to (look up .+?) and perform (.+)", low)
        if m:
            out.append(prettify_command(m.group(2)))
            out.append(prettify_command(m.group(3)))
            continue

        m = re.match(r"(.+?) turns (.+?) and (offers .+|starts laughing)", low)
        if m:
            subject = simplify_phrase(m.group(1))
            turn = simplify_phrase(m.group(2))
            out.append(prettify_command(f"{subject} turns {turn}"))
            out.append(prettify_command(m.group(3)))
            continue

        m = re.match(r"(.+?) and (.+)", low)
        if m and any(token in low for token in (" turns ", " nod ", " look up ", " starts laughing", " smile")):
            left = simplify_phrase(m.group(1))
            right = simplify_phrase(m.group(2))
            if left:
                out.append(prettify_command(left))
            if right:
                out.append(prettify_command(right))
            continue

        out.append(command)
    return unique_texts(out)


def summarize_command(text: str) -> list[str]:
    text = titlecase_first(shorten_command(text))
    low = sentence_stem(text)

    m = re.search(r"dolly in .*?(?:toward|towards) (.+)", low)
    if m:
        return [prettify_command(f"Dolly in on {simplify_phrase(m.group(1))}")]

    if "dolly in" in low:
        return ["Dolly in"]

    if "dolly out" in low:
        return ["Dolly out"]

    m = re.search(r"zoom in .*? on (.+)", low)
    if m:
        return [prettify_command(f"Zoom in on {simplify_phrase(m.group(1))}")]
    if "zoom in" in low:
        return ["Zoom in"]

    m = re.search(r"zoom out .*?(?:reveal more of )(.+)", low)
    if m:
        return [prettify_command(f"Zoom out to reveal {simplify_phrase(m.group(1))}")]
    if "zoom out" in low:
        return ["Zoom out"]

    m = re.match(r"change the color of (.+) to (.+)", low)
    if m:
        return [prettify_command(f"Change {simplify_phrase(m.group(1))} to {simplify_phrase(m.group(2))}")]

    m = re.match(r"change (.+) color to (.+)", low)
    if m:
        return [prettify_command(f"Change {simplify_phrase(m.group(1))} to {simplify_phrase(m.group(2))}")]

    m = re.match(r"change (.+) to (.+)", low)
    if m and "camera perspective" not in low and "shot to" not in low:
        return [prettify_command(f"Change {simplify_phrase(m.group(1))} to {simplify_phrase(m.group(2))}")]

    m = re.match(r"replace (.+) with (.+)", low)
    if m:
        return [prettify_command(f"Replace {simplify_phrase(m.group(1))} with {simplify_phrase(m.group(2))}")]

    m = re.match(r"remove (.+)", low)
    if m:
        return [prettify_command(f"Remove {simplify_phrase(m.group(1))}")]

    m = re.match(r"increase the number of (.+)", low)
    if m:
        return [prettify_command(f"Add more {simplify_phrase(m.group(1))}")]

    m = re.match(r"increase the amount of (.+)", low)
    if m:
        return [prettify_command(f"Add more {simplify_phrase(m.group(1))}")]

    m = re.match(r"add (.+)", low)
    if m:
        return [prettify_command(f"Add {simplify_phrase(m.group(1))}")]

    m = re.match(r"place (.+)", low)
    if m:
        return [prettify_command(f"Place {simplify_phrase(m.group(1))}")]

    m = re.match(r"transform the entire (?:video|scene) into (?:a |an )?(.+?) style", low)
    if m:
        return [prettify_command(f"Apply {simplify_phrase(m.group(1))} style")]

    m = re.match(r"transform the (?:entire )?(?:video|scene) into (.+)", low)
    if m:
        return [prettify_command(f"Apply {simplify_phrase(m.group(1))} style")]

    m = re.match(r"apply (.+?) style", low)
    if m:
        return [prettify_command(f"Apply {simplify_phrase(m.group(1))} style")]

    m = re.match(r"modify the video so that (.+)", low)
    if m:
        body = m.group(1)
        out: list[str] = []
        if "raises" in body and "wave" in body:
            who = simplify_phrase(body.split(" raises ", 1)[0])
            hand = ""
            mm = re.search(r"raises (.+?) to wave", body)
            if mm:
                hand = simplify_phrase(mm.group(1))
            out.append(prettify_command(f"Raise {who} {hand}".strip()))
            out.append("Wave to the camera")
            return unique_texts(out)
        return [prettify_command(simplify_phrase(body))]

    m = re.match(r"modify the video so (.+)", low)
    if m:
        body = m.group(1)
        if "tilts" in body and "head" in body:
            who = simplify_phrase(body.split(" tilts ", 1)[0])
            tail = simplify_phrase(body.split(" tilts ", 1)[1])
            return [prettify_command(f"Tilt {who} {tail}")]
        return [prettify_command(simplify_phrase(body))]

    m = re.match(r"modify the video to adopt (.+)", low)
    if m:
        return [prettify_command(f"Apply {simplify_phrase(m.group(1))}")]

    m = re.match(r"modify the (.+?) facial expression to transition .*? into (.+)", low)
    if m:
        return [prettify_command(f"Make {simplify_phrase(m.group(1))} {simplify_phrase(m.group(2))}")]

    m = re.match(r"adjust the camera perspective to (.+)", low)
    if m:
        return [prettify_command(f"Set camera to {simplify_phrase(m.group(1))}")]

    m = re.match(r"change the camera perspective to (.+)", low)
    if m:
        return [prettify_command(f"Set camera to {simplify_phrase(m.group(1))}")]

    m = re.match(r"change the shot to (.+)", low)
    if m:
        return [prettify_command(f"Set shot to {simplify_phrase(m.group(1))}")]

    m = re.match(r"shift the camera to (.+)", low)
    if m:
        return [prettify_command(f"Shift camera to {simplify_phrase(m.group(1))}")]

    m = re.match(r"perform (.+)", low)
    if m:
        return [prettify_command(simplify_phrase(m.group(1)))]

    m = re.match(r"enhance (.+?) by adding (.+)", low)
    if m:
        return unique_texts(
            [
                prettify_command(f"Enhance {simplify_phrase(m.group(1))}"),
                prettify_command(f"Add {simplify_phrase(m.group(2))}"),
            ]
        )

    return [text]


def listify_items(text: str) -> list[str]:
    text = clean_text(text)
    text = re.sub(r"\bincluding\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bexactly as they are\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bperfectly\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcompletely\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bprecisely\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe existing\b", "", text, flags=re.IGNORECASE)
    text = clean_text(text)
    parts = re.split(r",| and ", text)
    return [clean_text(p) for p in parts if clean_text(p)]


def titlecase_first(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    return text[0].upper() + text[1:]


def constraint_from_with_fragment(fragment: str) -> list[str]:
    fragment = clean_text(fragment)
    if not fragment:
        return []
    low = fragment.lower()
    out: list[str] = []
    if low.startswith("no "):
        items = re.split(r" or | and |,", fragment[3:])
        for item in items:
            item = clean_text(item)
            if item:
                out.append(f"Avoid {item}")
        return out
    if "flickering" in low or "artifacts" in low or "jitter" in low or "distortion" in low:
        items = re.split(r" or | and |,", fragment)
        for item in items:
            item = clean_text(re.sub(r"^(no|any)\s+", "", item, flags=re.IGNORECASE))
            if item:
                out.append(f"Avoid {item}")
        return out
    out.append(titlecase_first(fragment))
    return out


def extract_inline_constraints(sentence: str) -> tuple[str, list[str], list[str]]:
    text = clean_text(sentence)
    constraints: list[str] = []
    extra_commands: list[str] = []

    patterns = [
        (r",\s*starting from (.+?)(?=(?:,?\s+and ending in\b|,?\s+ending in\b|,?\s+while\b|$))", "Start from {x}"),
        (r"(?:,|\s)and ending in (.+?)(?=(?:,?\s+while\b|$))", "End in {x}"),
        (r",\s*making (.+?)(?=(?:,?\s+while\b|$))", None),
        (r"\s+while keeping (.+)$", "Keep {x}"),
        (r"\s+while preserving (.+)$", "Preserve {x}"),
        (r"\s+while maintaining (.+)$", "Maintain {x}"),
        (r"\s+while ensuring (.+)$", "Ensure {x}"),
        (r"\s+without (.+)$", None),
        (r"\s+with no (.+)$", None),
        (r"\s+avoiding (.+)$", "Avoid {x}"),
    ]

    main = text
    for pattern, template in patterns:
        m = re.search(pattern, main, flags=re.IGNORECASE)
        if not m:
            continue
        frag = clean_text(m.group(1))
        if pattern.startswith(r",\s*making "):
            extra_commands.append(titlecase_first(f"Make {frag}"))
        elif pattern.endswith(r"\s+without (.+)$") or pattern.endswith(r"\s+with no (.+)$"):
            constraints.extend(constraint_from_with_fragment(frag))
        elif template is not None:
            constraints.append(titlecase_first(template.format(x=frag)))
        main = clean_text(re.sub(pattern, "", main, flags=re.IGNORECASE))

    by_add = re.search(r"\bby adding (.+)$", main, flags=re.IGNORECASE)
    if by_add:
        extra_commands.append(titlecase_first(f"Add {clean_text(by_add.group(1))}"))
        main = clean_text(re.sub(r"\bby adding .+$", "", main, flags=re.IGNORECASE))

    return main, constraints, extra_commands


def split_preserve_items(sentence: str, lead: str) -> list[str]:
    body = clean_text(re.sub(rf"^{lead}\s+", "", sentence, flags=re.IGNORECASE))
    items = listify_items(body)
    if not items:
        return []
    prefix = "Keep" if lead.lower() == "keep" else "Preserve"
    return [titlecase_first(f"{prefix} {item}") for item in items]


def keyword_constraints(text: str) -> list[str]:
    low = sentence_stem(text)
    out: list[str] = []

    if "temporal consistency" in low or "temporally consistent" in low or "consistent across all frames" in low:
        out.append("Maintain temporal consistency")
    if "flicker" in low:
        if "around " in low:
            out.append(titlecase_first(f"Avoid flickering around {clean_text(low.split('around ', 1)[1])}"))
        else:
            out.append("Avoid flickering")
    if "artifact" in low:
        if "around " in low:
            out.append(titlecase_first(f"Avoid artifacts around {clean_text(low.split('around ', 1)[1])}"))
        else:
            out.append("Avoid artifacts")
    if "jitter" in low:
        out.append("Avoid jitter")
    if "distortion" in low:
        out.append("Avoid distortion")
    if "ghosting" in low:
        out.append("Avoid ghosting")
    if "halo" in low:
        out.append("Avoid haloing")
    if "white fringe" in low or "white fringes" in low or "white outline" in low:
        out.append("Avoid white fringing")
    if "edge bleeding" in low or "color bleeding" in low or "bleeding onto" in low:
        out.append("Avoid color bleeding")
    if "green spill" in low:
        out.append("Avoid green spill")
    if "visible seam" in low:
        out.append("Avoid visible seams")
    if "clean edges" in low or "sharp edges" in low or "anti aliased edges" in low:
        out.append("Keep edges clean")
    if "cleanly masked" in low or "cleanly matted" in low or "silhouette" in low or "outline" in low:
        out.append("Keep subject edges clean")
    if "mask accurately follows" in low or "mask precisely follows" in low:
        out.append("Keep the mask aligned")
    if "sharp focus" in low or "sharpness" in low:
        out.append("Keep the subject sharp")
    if "centered" in low:
        out.append("Keep the subject centered")
    if "lighting" in low and "shadow" in low:
        out.append("Match lighting and shadows")
    elif "lighting" in low and ("match" in low or "adjust" in low or "reflect" in low or "complement" in low):
        out.append("Match lighting")
    if "shadows and highlights" in low:
        out.append("Match shadows and highlights")
    if "3d animation style" in low:
        out.append("Match the original 3D animation style")
    if "perspective" in low and ("match" in low or "adjust" in low or "consistent" in low or "natural" in low):
        out.append("Match perspective")
    if "depth of field" in low:
        out.append("Match depth of field")
    if "spatial alignment" in low:
        out.append("Keep spatial alignment")
    if "position relative" in low or ("position" in low and "stable" in low):
        out.append("Keep the position stable")
    if "accurately positioned" in low or "positioned within" in low:
        out.append("Keep the position accurate")
    if "conforms to the shape" in low or "conform to the shape" in low:
        out.append("Match the original shape")
    if "texture" in low and ("clearly defined" in low or "maintained" in low):
        out.append("Keep texture detail")
    if "color graded" in low or "color graded" in low or "color-graded" in text.lower():
        out.append("Match color grading")
    if "seamless integration" in low or "seamless blend" in low or "seamlessly integrates" in low:
        out.append("Keep the integration seamless")
    if "background stable" in low:
        out.append("Keep the background stable")
    if "move in sync" in low or "synchronized" in low:
        out.append("Keep motion synchronized")
    if "consistent distance and speed" in low:
        out.append("Keep distance and speed consistent")
    if "legible" in low:
        out.append("Keep text legible")
    if "visible" in low and "remain" in low:
        out.append("Keep key elements visible")
    if "professional" in low and "composite" in low:
        out.append("Keep a clean composite look")
    if "smooth" in low and ("motion" in low or "move" in low or "animation" in low):
        out.append("Keep the motion smooth")
    if "natural" in low and ("motion" in low or "appearance" in low or "look" in low):
        out.append("Keep the motion natural")
    if "fluid" in low and ("motion" in low or "arms" in low):
        out.append("Keep the motion fluid")
    if "pulse rhythmically" in low or "glow should pulse" in low:
        out.append("Make the glow pulse rhythmically")
    if "focus on the presenter" in low:
        out.append("Keep the presenter in focus")
    if "progressively over the sequence" in low:
        out.append("Make the change progressive")
    if "peak of the smile by the final frame" in low:
        out.append("Reach the full smile by the final frame")
    return unique_texts(out)


def split_remain_sentence(sentence: str) -> list[str]:
    text = clean_text(sentence)
    m = re.match(r"^(?:the |all )(.+?)\s+(?:must|should)\s+remain\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return []
    left = clean_text(m.group(1))
    right = clean_text(m.group(2))
    items = listify_items(left)
    if any(word in right.lower() for word in ("unchanged", "unaffected", "untouched", "intact")):
        return [titlecase_first(f"Preserve {item}") for item in items]
    if "visible" in right.lower():
        return [titlecase_first(f"Keep {item} visible") for item in items]
    if "legible" in right.lower():
        return [titlecase_first(f"Keep {item} legible") for item in items]
    if "stable" in right.lower():
        return [titlecase_first(f"Keep {item} stable") for item in items]
    return [titlecase_first(f"Keep {item} {right}") for item in items]


def split_ensure_sentence(sentence: str) -> list[str]:
    text = clean_text(re.sub(r"^(ensure|maintain)\s+(that\s+)?", "", sentence, flags=re.IGNORECASE))
    out = keyword_constraints(text)
    if not out:
        out.append(titlecase_first(text))
    return unique_texts(out)


def split_declarative_constraint(sentence: str) -> list[str]:
    text = clean_text(sentence)
    low = text.lower()
    out: list[str] = []

    if low.startswith("throughout the entire video"):
        remainder = clean_text(re.sub(r"^throughout the entire video,?\s*", "", text, flags=re.IGNORECASE))
        if remainder:
            return split_ensure_sentence(remainder)

    if low.startswith("throughout the entire sequence"):
        remainder = clean_text(re.sub(r"^throughout the entire sequence,?\s*", "", text, flags=re.IGNORECASE))
        if remainder:
            return split_ensure_sentence(remainder)

    if low.startswith("this ") and ("must" in low or "should" in low):
        text = re.sub(r"^this\s+\w+\s+(?:must|should)\s+", "", text, flags=re.IGNORECASE)
        out.extend(keyword_constraints(text))
        return unique_texts(out) or [titlecase_first(clean_text(text))]

    if low.startswith("all edges"):
        if "sharp and clean" in low or "clean" in low:
            out.append("Keep edges clean")
        if "no flickering" in low:
            out.append("Avoid flickering")
        return unique_texts(out)

    if low.startswith("the final output should") or low.startswith("the resulting video should"):
        if "natural perspective" in low:
            out.append("Keep perspective natural")
        out.extend(keyword_constraints(text))
        return unique_texts(out) or [titlecase_first(text)]

    if "should appear smooth" in low or "should appear natural" in low:
        if "smooth" in low:
            out.append("Keep the motion smooth")
        if "natural" in low:
            out.append("Keep the motion natural")
        if "no flickering" in low:
            out.append("Avoid flickering")
        return unique_texts(out)

    if "should pulse" in low:
        out.append("Make the glow pulse rhythmically")
        return out

    if "must remain" in low or "should remain" in low:
        return split_remain_sentence(text)

    return []


def is_command_sentence(sentence: str) -> bool:
    low = sentence_stem(sentence)
    return any(low.startswith(verb + " ") for verb in COMMAND_VERBS) or low.startswith("please edit ") or low.startswith("edit ")


def split_command_sentence(sentence: str) -> tuple[list[str], list[str]]:
    sentence = re.sub(r"^please\s+", "", sentence, flags=re.IGNORECASE)
    main, constraints, extra_commands = extract_inline_constraints(sentence)
    commands: list[str] = []
    if shorten_command(main):
        commands.extend(summarize_command(main))
    for x in extra_commands:
        if shorten_command(x):
            commands.extend(summarize_command(x))

    extra: list[str] = []
    for command in commands:
        low = sentence_stem(command)
        if " by " in low:
            prefix, suffix = re.split(r"\bby\b", command, maxsplit=1, flags=re.IGNORECASE)
            prefix = titlecase_first(shorten_command(prefix))
            suffix = titlecase_first(shorten_command(f"Add {clean_text(suffix)}"))
            if prefix:
                extra.append(prefix)
            if suffix:
                extra.append(suffix)
        else:
            extra.append(command)
    return split_compound_commands(unique_texts(extra)), unique_texts(constraints)


def unique_texts(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = sentence_stem(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(titlecase_first(cleaned))
    return out


def canonical_key(text: str) -> str:
    return sentence_stem(text)


def parse_instruction(text: str) -> tuple[list[str], list[str]]:
    commands: list[str] = []
    constraints: list[str] = []

    for sentence in split_sentences(text):
        if is_command_sentence(sentence):
            cmd_parts, con_parts = split_command_sentence(sentence)
            commands.extend(cmd_parts)
            constraints.extend(con_parts)
            continue

        low = sentence_stem(sentence)
        if low.startswith("preserve ") or low.startswith("carefully preserve "):
            constraints.extend(split_preserve_items(sentence, "Preserve"))
        elif low.startswith("keep ") or low.startswith("carefully keep "):
            constraints.extend(split_preserve_items(sentence, "Keep"))
        elif low.startswith("ensure ") or low.startswith("maintain "):
            constraints.extend(split_ensure_sentence(sentence))
        else:
            declarative = split_declarative_constraint(sentence)
            if declarative:
                constraints.extend(declarative)
            else:
                constraints.extend(keyword_constraints(sentence) or [titlecase_first(sentence)])

    return unique_texts(commands), unique_texts(constraints)


def write_unique_csv(path: Path, id_name: str, value_name: str, values: list[str]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_name, value_name])
        writer.writeheader()
        for idx, value in enumerate(values, start=1):
            writer.writerow({id_name: idx, value_name: value})
            mapping[canonical_key(value)] = idx
    return mapping


def write_mapping_csv(
    path: Path,
    rows: list[dict[str, str]],
    instruction_id_map: dict[str, int],
    constraint_id_map: dict[str, int],
) -> None:
    fieldnames = [
        "source_id",
        "video_path",
        "selected_class",
        "selected_subclass",
        "component_type",
        "component_id",
        "component_text",
        "original_instruction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            commands, constraints = parse_instruction(row["instruction"])
            for command in commands:
                writer.writerow(
                    {
                        "source_id": row["source_id"],
                        "video_path": row["video_path"],
                        "selected_class": row["selected_class"],
                        "selected_subclass": row["selected_subclass"],
                        "component_type": "instruction",
                        "component_id": instruction_id_map[canonical_key(command)],
                        "component_text": command,
                        "original_instruction": row["instruction"],
                    }
                )
            for constraint in constraints:
                writer.writerow(
                    {
                        "source_id": row["source_id"],
                        "video_path": row["video_path"],
                        "selected_class": row["selected_class"],
                        "selected_subclass": row["selected_subclass"],
                        "component_type": "constraint",
                        "component_id": constraint_id_map[canonical_key(constraint)],
                        "component_text": constraint,
                        "original_instruction": row["instruction"],
                    }
                )


def main() -> None:
    args = parse_args()
    rows = load_annotations(Path(args.input))

    all_commands: list[str] = []
    all_constraints: list[str] = []
    for row in rows:
        commands, constraints = parse_instruction(row["instruction"])
        all_commands.extend(commands)
        all_constraints.extend(constraints)

    unique_commands = unique_texts(all_commands)
    unique_constraints = unique_texts(all_constraints)

    outdir = Path(args.outdir)
    instruction_map = write_unique_csv(outdir / "instructions.csv", "instruction_id", "instruction", unique_commands)
    constraint_map = write_unique_csv(outdir / "constraints.csv", "constraint_id", "constraint", unique_constraints)
    write_unique_csv(outdir / "constrains.csv", "constraint_id", "constraint", unique_constraints)
    write_mapping_csv(outdir / "instruction_constraint_map.csv", rows, instruction_map, constraint_map)

    print(f"[INFO] Source instructions: {len(rows)}")
    print(f"[INFO] Unique atomic instructions: {len(unique_commands)}")
    print(f"[INFO] Unique atomic constraints: {len(unique_constraints)}")
    print(f"[INFO] Wrote {outdir / 'instructions.csv'}")
    print(f"[INFO] Wrote {outdir / 'constraints.csv'}")
    print(f"[INFO] Wrote {outdir / 'instruction_constraint_map.csv'}")


if __name__ == "__main__":
    main()
