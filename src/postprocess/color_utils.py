from __future__ import annotations


def extract_target_color(instruction: str) -> str | None:
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
