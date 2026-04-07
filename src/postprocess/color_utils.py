from __future__ import annotations


def extract_target_color(instruction: str) -> str | None:
    """Extract a target color keyword from natural-language instruction text.

    Tools: none (string matching only).
    Steps:
    1. Lowercase instruction text.
    2. Scan predefined color phrases in deterministic order.
    3. Return first matched color name or None.
    """
    names = [
        "navy blue", "violet", "purple", "red", "blue", "green",
        "yellow", "orange", "pink", "black", "white", "silver",
    ]
    text = (instruction or "").lower()
    for name in names:
        if name in text:
            return name
    return None


def target_color_bgr(name: str | None) -> tuple[int, int, int]:
    """Map a color name to OpenCV BGR tuple used in synthetic fills.

    Tools: OpenCV color convention only (BGR ordering).
    Steps:
    1. Normalize color name to lowercase.
    2. Look up fixed LUT tuned for visible replacement colors.
    3. Return default orange-like fallback when unknown.
    """
    lut = {
        "red": (40, 40, 220),
        "orange": (0, 128, 255),
        "yellow": (40, 220, 220),
        "green": (60, 180, 60),
        "blue": (220, 80, 40),
        "navy blue": (160, 60, 20),
        "violet": (200, 80, 180),
        "purple": (180, 60, 180),
        "pink": (180, 120, 255),
        "black": (16, 16, 16),
        "white": (240, 240, 240),
        "silver": (180, 180, 180),
    }
    return lut.get((name or "").lower(), (0, 128, 255))
