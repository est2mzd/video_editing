from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import json
import re
import colorsys

import cv2
import numpy as np
from tqdm.auto import tqdm
import webcolors

# Project imports
sys.path.append('/workspace/src')
#sys.path.append('/workspace/src/utils')

from postprocess.detectors import detect_all_boxes, get_sam_mask_from_box
from utils.video_utility import load_video, write_video, show_before_after


@dataclass
class ChangeColorConfig:
    """Configuration for LLM-based color change pipeline."""
    input_video: str
    output_video: str
    instruction: str
    dino_box_threshold: float = 0.30
    dino_text_threshold: float = 0.25
    max_frames: int | None = None
    target_hue: int = 110
    saturation_scale: float = 1.2
    saturation_min: float = 80.0
    saturation_boost_gain: float = 0.2
    value_scale: float = 1.0
    value_min: float = 30.0
    transition_speed: float = 1.0


# ──────────────────────────────────────────────
# Color name → OpenCV hue (0-180) conversion
# ──────────────────────────────────────────────

def _rgb_to_cv2_hue(r: int, g: int, b: int) -> int:
    """RGB (0-255) → OpenCV hue (0-180)."""
    h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return int(h * 180)


def color_name_to_hue_rulebase(color_name: str) -> int | None:
    """Rule-based: color name → OpenCV hue (0-180).

    Uses webcolors CSS3 name database (140+ colors) with alias pre-processing.
    Returns None if the name cannot be resolved.
    """
    ALIASES: dict[str, str] = {
        # single-word aliases
        'violet': 'blueviolet',
        'aqua': 'cyan',
        'fuchsia': 'magenta',
        'sky': 'deepskyblue',
        'forest': 'forestgreen',
        'olive': 'olivedrab',
        'mint': 'mediumspringgreen',
        'teal': 'teal',
        'turquoise': 'turquoise',
        'coral': 'coral',
        'salmon': 'salmon',
        'gold': 'gold',
        'khaki': 'khaki',
        'indigo': 'indigo',
        'lavender': 'lavender',
        'lilac': 'plum',
        'mauve': 'rosybrown',
        'maroon': 'maroon',
        'amber': 'orange',
        'crimson': 'crimson',
        'scarlet': 'crimson',
        'burgundy': 'darkred',
        'beige': 'beige',
        'ivory': 'ivory',
        'tan': 'tan',
        'brown': 'brown',
        'chocolate': 'chocolate',
        'sienna': 'sienna',
        'rose': 'deeppink',
        'lime': 'lime',
        'emerald': 'mediumseagreen',
        'cobalt': 'steelblue',
        # multi-word aliases
        'navy blue': 'navy',
        'deep navy blue': 'darkblue',
        'deep navy': 'darkblue',
        'royal blue': 'royalblue',
        'sky blue': 'deepskyblue',
        'hot pink': 'hotpink',
        'light blue': 'lightblue',
        'dark red': 'darkred',
        'dark blue': 'darkblue',
        'dark green': 'darkgreen',
        'sea green': 'seagreen',
        'dark purple': 'purple',
        'light purple': 'mediumpurple',
        'light green': 'lightgreen',
        'light yellow': 'lightyellow',
        'light pink': 'lightpink',
        'emerald green': 'mediumseagreen',
        'neon blue': 'dodgerblue',
        'neon green': 'lime',
        'neon pink': 'hotpink',
        'deep blue': 'mediumblue',
        'deep green': 'darkgreen',
        'deep red': 'darkred',
        'deep burgundy': 'darkred',
        'metallic red': 'red',
        'metallic green': 'green',
        'metallic blue': 'blue',
        'bright red': 'red',
        'bright blue': 'blue',
        'bright green': 'limegreen',
        'vibrant red': 'red',
        'vibrant blue': 'blue',
        'vibrant green': 'limegreen',
    }

    normalized = color_name.lower().strip()
    resolved = ALIASES.get(normalized, normalized)

    # Exact CSS3 lookup (with alias resolution)
    for attempt in (resolved, normalized):
        try:
            rgb = webcolors.name_to_hex(attempt)
            r = int(rgb[1:3], 16)
            g = int(rgb[3:5], 16)
            b = int(rgb[5:7], 16)
            return _rgb_to_cv2_hue(r, g, b)
        except (ValueError, AttributeError):
            pass

    # Partial match against all CSS3 names
    all_names = webcolors.names('css3')
    for css_name in all_names:
        if normalized in css_name or css_name in normalized:
            rgb = webcolors.name_to_hex(css_name)
            r = int(rgb[1:3], 16)
            g = int(rgb[3:5], 16)
            b = int(rgb[5:7], 16)
            return _rgb_to_cv2_hue(r, g, b)

    # Word-by-word fallback: try each word individually
    for word in normalized.split():
        if word == normalized:
            continue
        word_resolved = ALIASES.get(word, word)
        try:
            rgb = webcolors.name_to_hex(word_resolved)
            r = int(rgb[1:3], 16)
            g = int(rgb[3:5], 16)
            b = int(rgb[5:7], 16)
            return _rgb_to_cv2_hue(r, g, b)
        except (ValueError, AttributeError):
            pass

    return None


def color_name_to_hue_llm(color_name: str) -> int:
    """LLM fallback: ask GPT for the RGB value of an unknown color name."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{
                'role': 'user',
                'content': (
                    f'What is the typical RGB value of the color "{color_name}"?\n'
                    'Reply with only three integers separated by commas, e.g. 128,0,255'
                ),
            }],
            max_tokens=20,
        )
        rgb_str = response.choices[0].message.content.strip()
        r, g, b = [int(x.strip()) for x in rgb_str.split(',')]
        hue = _rgb_to_cv2_hue(r, g, b)
        print(f'LLM resolved "{color_name}" → RGB({r},{g},{b}) → hue={hue}')
        return hue
    except Exception as e:
        print(f'LLM fallback failed ({e}), defaulting to blue hue=110')
        return 110


def color_name_to_hue(color_name: str) -> int:
    """Convert color name to OpenCV hue (0-180).

    Strategy: rule-base (webcolors CSS3 + aliases) → LLM fallback.
    """
    hue = color_name_to_hue_rulebase(color_name)
    if hue is not None:
        print(f'Rule-base resolved "{color_name}" → hue={hue}')
        return hue
    print(f'"{color_name}" not in rule-base, trying LLM...')
    return color_name_to_hue_llm(color_name)


# ──────────────────────────────────────────────
# Core processing functions
# ──────────────────────────────────────────────

def apply_color_change(
    frame_bgr: np.ndarray,
    target_mask_u8: np.ndarray,
    target_hue: int,
    saturation_scale: float = 1.2,
    saturation_min: float = 80.0,
    value_scale: float = 1.0,
    value_min: float = 30.0,
) -> np.ndarray:
    """Apply color change by directly setting target hue on masked region."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = target_mask_u8.astype(bool)

    hsv_out = hsv.copy()
    hsv_out[mask, 0] = float(target_hue)
    hsv_out[mask, 1] = np.clip(
        np.maximum(hsv[mask, 1] * saturation_scale, saturation_min), 0.0, 255.0
    )
    hsv_out[mask, 2] = np.clip(
        np.maximum(hsv[mask, 2] * value_scale, value_min), 0.0, 255.0
    )
    return cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2BGR)


def estimate_target_mask_gdino_sam(
    frame_bgr: np.ndarray,
    target_prompt: str,
    dino_box_threshold: float = 0.30,
    dino_text_threshold: float = 0.25,
) -> np.ndarray:
    """Estimate target object mask using GroundingDINO + SAM."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_all_boxes(
        frame_rgb,
        text_prompt=target_prompt,
        box_threshold=dino_box_threshold,
        text_threshold=dino_text_threshold,
    )
    if not boxes:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    mask = get_sam_mask_from_box(frame_rgb, best_box).astype(np.uint8)
    return mask


def run_change_color_gradual_pipeline(
    input_video: str,
    output_video: str,
    target_prompt: str,
    target_color: str,
    dino_box_threshold: float = 0.30,
    dino_text_threshold: float = 0.25,
    max_frames: int | None = None,
    saturation_scale: float = 1.2,
    saturation_min: float = 80.0,
    saturation_boost_gain: float = 0.2,
    value_scale: float = 1.0,
    value_min: float = 30.0,
    gradual: bool = True,
    transition_speed: float = 1.0,
) -> dict:
    """Run color change with gradual hue transition across frames.

    Args:
        gradual: If False, apply target color immediately from the first frame.
        transition_speed: Controls how fast the color transition completes.
            1.0 = linear, full transition by the last frame (default).
            2.0 = full transition by the midpoint (faster).
            0.5 = only 50% transitioned at the last frame (slower).
    """
    target_hue = color_name_to_hue(target_color)

    frames, fps, width, height = load_video(input_video)
    if max_frames is not None:
        frames = frames[:max_frames]

    out_frames: list[np.ndarray] = []
    num_frames = len(frames)

    for frame_idx, frame in enumerate(tqdm(frames, desc='change_color_gradual')):
        if gradual:
            raw_progress = frame_idx / max(num_frames - 1, 1)
            progress = min(raw_progress * transition_speed, 1.0)
        else:
            progress = 1.0

        target_mask = estimate_target_mask_gdino_sam(
            frame_bgr=frame,
            target_prompt=target_prompt,
            dino_box_threshold=dino_box_threshold,
            dino_text_threshold=dino_text_threshold,
        )

        if np.any(target_mask):
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            mask = target_mask.astype(bool)
            original_hue = float(np.median(hsv_frame[mask, 0])) if mask.any() else 0.0
            interpolated_hue = int(original_hue + (target_hue - original_hue) * progress)
        else:
            interpolated_hue = target_hue

        edited = apply_color_change(
            frame_bgr=frame,
            target_mask_u8=target_mask,
            target_hue=interpolated_hue,
            saturation_scale=1.0 + float(saturation_boost_gain) * progress * saturation_scale,
            saturation_min=saturation_min,
            value_scale=value_scale,
            value_min=value_min,
        )
        out_frames.append(edited)

    write_video(output_video, out_frames, fps, width, height)

    return {
        'input_video': input_video,
        'output_video': output_video,
        'target_prompt': target_prompt,
        'target_color': target_color,
        'target_hue': target_hue,
        'saturation_scale': saturation_scale,
        'saturation_min': saturation_min,
        'saturation_boost_gain': saturation_boost_gain,
        'value_scale': value_scale,
        'value_min': value_min,
        'transition_speed': transition_speed,
        'gradual_requested': gradual,
        'fps': fps,
        'width': width,
        'height': height,
        'num_frames': num_frames,
        'gradual': gradual,
    }

# ──────────────────────────────────────────────
# Instruction parser: extract target object & colors
# ──────────────────────────────────────────────

@dataclass
class ColorChangeInstruction:
    """Parsed result of a color-change instruction."""
    target_object: str          # noun to recolor (e.g. "hair", "shirt")
    to_color: str               # target color name (cleaned)
    from_color: str | None = None  # original color (cleaned, if mentioned)


# ─── Color word sets ──────────────────────────
_COLOR_WORDS = {
    'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'indigo',
    'cyan', 'magenta', 'pink', 'white', 'black', 'gray', 'grey', 'brown',
    'gold', 'golden', 'silver', 'beige', 'ivory', 'coral', 'salmon', 'teal',
    'turquoise', 'lavender', 'lilac', 'maroon', 'crimson', 'scarlet', 'amber',
    'rose', 'lime', 'mint', 'navy', 'cobalt', 'khaki', 'tan', 'sienna',
    'chocolate', 'burgundy', 'mauve', 'emerald',
}

# Adjective/function words that prefix a color but are NOT the color itself
_COLOR_MODIFIERS = {
    'a', 'an', 'the', 'of',
    'vibrant', 'metallic', 'deep', 'bright', 'neon', 'bold',
    'solid', 'light', 'dark', 'rich', 'warm', 'cool', 'dull', 'pastel', 'vivid',
    'muted', 'subtle', 'electric', 'glossy', 'matte', 'soft', 'intense', 'lush',
    'earthy', 'neutral', 'sophisticated', 'classic',
    'shade', 'hue', 'tone', 'tint', 'color', 'colour',
}

# Object adjective modifiers that add little value for GroundingDINO
_OBJECT_MODIFIERS = {'vibrant', 'solid', 'multicolored', 'iconic', 'bold', 'patterned'}

# Regex: clause boundary that terminates an object/color phrase
_CLAUSE_STOP_RE = re.compile(
    r'\s+(?:throughout|while|ensuring|without|across|during|except|keeping|'
    r'maintaining|preserving|to\s+(?:keep|maintain|ensure|create|preserve|make)|'
    r'with\s+no|for\s+a\s+|to\s+give|giving|adding|featuring|using)\b',
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def clean_to_color(raw: str) -> str:
    """Strip modifier adjectives and clause junk; return the core color (1-2 words).

    Examples:
      'a vibrant shade of violet throughout the entire video' → 'violet'
      'a deep navy blue throughout'                          → 'navy blue'
      'a vibrant metallic emerald green throughout'          → 'emerald green'
      'deep blue color while keeping the lighting'           → 'blue'
      'a deep burgundy'                                      → 'burgundy'
    """
    # Stop at clause boundaries
    raw = _CLAUSE_STOP_RE.split(raw)[0]
    # Strip trailing decoration words
    raw = re.sub(r'\s+(?:color|hue|shade|tint|tone)\s*$', '', raw.strip(), flags=re.IGNORECASE)

    words = raw.lower().split()
    # Strip leading modifier words (including 'of')
    while words and words[0] in _COLOR_MODIFIERS:
        words.pop(0)

    if not words:
        return raw.strip()

    # Find the first color word, then check if next word is also a color → 2-word result
    for i, w in enumerate(words):
        base = w.rstrip('.,')
        if base in _COLOR_WORDS:
            # Look ahead: "navy blue", "emerald green", etc.
            if i + 1 < len(words):
                nxt = words[i + 1].rstrip('.,')
                if nxt in _COLOR_WORDS:
                    return f'{base} {nxt}'
            return base

    # Fallback: first 1-2 remaining words
    return ' '.join(words[:2])


def shorten_object(raw: str) -> str:
    """Trim extracted object to a GroundingDINO-friendly length (≤3 words).

    Examples:
      'exterior color of the blue luxury car'               → 'blue luxury car'
      "woman's hair color"                                  → 'hair'
      "news presenter's suit"                               → 'suit'
      'multicolored floral shirt of the girl on the right'  → 'floral shirt'
      'bright red foreground surface'                       → 'red foreground surface'
      'magenta porsche in the foreground'                   → 'magenta porsche'
    """
    s = raw.lower().strip()

    # Strip leading structural prefixes (order matters)
    for prefix_pat in [
        r'^(?:exterior|interior)\s+color\s+of\s+(?:the\s+|a\s+|an\s+)?',
        r'^(?:exterior|interior)\s+',
        r'^color\s+of\s+(?:the\s+|a\s+|an\s+)?',
        r'^[\w\s]+?\'s\s+',       # possessive (single or multi-word): "woman's", "news presenter's"
        r'^(?:the\s+|a\s+|an\s+)',
    ]:
        s_new = re.sub(prefix_pat, '', s)
        if s_new != s:
            s = s_new
            break  # apply only the first matching prefix

    # Strip trailing relative/prepositional clauses
    s = re.sub(r'\s+(?:of\s+the|of\s+a|in\s+the|at\s+the|from\s+the|behind\s+the|on\s+the)\b.*$', '', s)
    s = re.sub(r'\s+on\s+(?:the\s+)?(?:left|right|middle|top|bottom)\b.*$', '', s)
    # Strip trailing " color"
    s = re.sub(r'\s+color\s*$', '', s)

    # Strip leading non-color object modifiers
    words = s.split()
    while len(words) > 1 and words[0] in _OBJECT_MODIFIERS:
        words.pop(0)

    # Limit to 3 words; prefer last 3 if still longer
    if len(words) > 3:
        words = words[-3:]

    return ' '.join(words).strip()


def parse_color_change_instruction_rulebase(instruction: str) -> ColorChangeInstruction | None:
    """Rule-based parser: extract target object, from-color, and to-color.

    Handles patterns including:
      - "Change the woman's hair color to a vibrant shade of violet"
      - "Change/Modify the color of X's Y (from A) to B"
      - "Modify the exterior color of the blue car to metallic emerald green"
      - "Change the bright red surface to a deep blue color while..."
      - "Make the shirt red" / "Dye her hair golden"
    Returns None if the pattern cannot be matched.
    """
    text = _normalize(instruction)
    _stop = (r'(?:\s+throughout|\s+while|\s+ensuring|\s+to\s+(?:keep|maintain|ensure|create|preserve|make)'
             r'|\s+keeping|\s+maintaining|\s+preserving|\s*[,.]|$)')

    # ---- Pattern A: change/modify (the) [word] color of (possessive) X from A to B ----
    m = re.search(
        r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)?color\s+of\s+'
        r'(?:the\s+|a\s+|an\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+from\s+([\w\s]+?)\s+(?:to|into)\s+([\w\s]+?)' + _stop,
        text,
    )
    if m:
        obj = shorten_object(m.group(1).strip())
        from_c = clean_to_color(m.group(2).strip())
        to_c = clean_to_color(m.group(3).strip())
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, from_color=from_c or None, to_color=to_c)

    # ---- Pattern B: change/modify (the) [word] color of (possessive) X to B ----
    m = re.search(
        r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)?color\s+of\s+'
        r'(?:the\s+|a\s+|an\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+(?:to|into)\s+([\w\s]+?)' + _stop,
        text,
    )
    if m:
        obj = shorten_object(m.group(1).strip())
        to_c = clean_to_color(m.group(2).strip())
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, to_color=to_c)

    # ---- Pattern C: verb (possessive) X color from A to B ----
    m = re.search(
        r'(?:change|turn|dye|color|paint|make|set|modify)\s+'
        r'(?:the\s+|his\s+|her\s+|their\s+|its\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+(?:color\s+)?from\s+([\w\s]+?)\s+(?:to|into)\s+([\w\s]+?)' + _stop,
        text,
    )
    if m:
        obj = shorten_object(re.sub(r'\s*color$', '', m.group(1).strip()).strip())
        from_c = clean_to_color(m.group(2).strip())
        to_c = clean_to_color(m.group(3).strip())
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, from_color=from_c or None, to_color=to_c)

    # ---- Pattern D: verb (possessive) X color to B ----
    m = re.search(
        r'(?:change|turn|dye|color|paint|make|set|modify)\s+'
        r'(?:the\s+|his\s+|her\s+|their\s+|its\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+(?:color\s+)?(?:to|into)\s+([\w\s]+?)' + _stop,
        text,
    )
    if m:
        obj = shorten_object(re.sub(r'\s*color$', '', m.group(1).strip()).strip())
        to_c = clean_to_color(m.group(2).strip())
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, to_color=to_c)

    # ---- Pattern E: dye/paint ... {color_word} (no "to") ----
    _color_alt = '|'.join(_COLOR_WORDS)
    m = re.search(
        r'(?:dye|paint)\s+'
        r'(?:the\s+|his\s+|her\s+|their\s+|its\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+(' + _color_alt + r')' + _stop,
        text,
    )
    if m:
        obj = shorten_object(m.group(1).strip())
        to_c = m.group(2).strip()
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, to_color=to_c)

    # ---- Pattern F: make ... {color_word} ----
    m = re.search(
        r'make\s+'
        r'(?:the\s+|his\s+|her\s+|their\s+|its\s+)?(?:[\w\s]+?\'s\s+)?'
        r'([\w\s]+?)\s+(' + _color_alt + r')' + _stop,
        text,
    )
    if m:
        obj = shorten_object(m.group(1).strip())
        to_c = m.group(2).strip()
        if obj and to_c:
            return ColorChangeInstruction(target_object=obj, to_color=to_c)

    return None


def parse_color_change_instruction_llm(instruction: str) -> ColorChangeInstruction:
    """LLM fallback: ask GPT to extract object and colors from instruction."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are an information extractor. Given a video editing instruction, extract:\n'
                        '1. target_object: short noun phrase (≤3 words) for the object being recolored\n'
                        '2. to_color: the new color name (1-2 words, no modifier adjectives)\n'
                        '3. from_color: original color (1-2 words, null if not mentioned)\n'
                        'Reply with JSON only, e.g. '
                        '{"target_object": "hair", "to_color": "violet", "from_color": null}'
                    ),
                },
                {'role': 'user', 'content': instruction},
            ],
            max_tokens=80,
            response_format={'type': 'json_object'},
        )
        data = json.loads(response.choices[0].message.content)
        result = ColorChangeInstruction(
            target_object=data.get('target_object', ''),
            to_color=data.get('to_color', ''),
            from_color=data.get('from_color') or None,
        )
        print(f'LLM parsed: {result}')
        return result
    except Exception as e:
        print(f'LLM parse failed ({e})')
        return ColorChangeInstruction(target_object='', to_color='')


def parse_color_change_instruction(instruction: str) -> ColorChangeInstruction:
    """Parse a color-change instruction to extract target object and colors.

    Strategy: rule-base (regex + clean_to_color + shorten_object) → LLM fallback.
    """
    result = parse_color_change_instruction_rulebase(instruction)
    if result is not None and result.target_object and result.to_color:
        print(f'Rule-base parsed: {result}')
        return result
    print(f'Rule-base failed for: "{instruction[:60]}...", trying LLM...')
    return parse_color_change_instruction_llm(instruction)


# ---- Smoke-test with annotation-style instructions ----
_smoke_tests = [
    "Change the woman's hair color to a vibrant shade of violet throughout the entire duration of the video.",
    "Change the color of the subject's red necktie to a deep navy blue throughout the entire video.",
    "Modify the exterior color of the blue luxury car to a vibrant metallic emerald green throughout.",
    "Change the bright red foreground surface to a deep blue color while keeping the lighting consistent.",
    "Change the color of the character's orange fur suit to a vibrant neon blue.",
    "Change the color of the silver sports car to a vibrant metallic red while maintaining its glossy finish.",
    "Change the color of the man's blue jacket to a dark emerald green.",
    "Change the multicolored floral shirt of the girl on the right to a solid deep blue color.",
    "Change the color of the news presenter's suit from navy blue to a deep burgundy.",
    "Change the color of the magenta Porsche in the foreground to a metallic emerald green.",
    "Make the shirt red",
    "Dye her hair golden",
]