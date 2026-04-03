
import cv2
import numpy as np

from postprocess.trials import apply_style_common as common

# ===============================
# Style normalization
# ===============================
from .apply_style_common import (
    APPLY_STYLES,
    STYLE_ALIASES,
)

_STYLE_ALIASES = dict(STYLE_ALIASES)

def normalize_style(style: str) -> str:
    """Normalize style text to ver1 canonical labels."""
    return common.normalize_style_alias(
        style,
        aliases=_STYLE_ALIASES,
        fallback="cleaned",
    )

# ===============================
# Style functions (same as before)
# ===============================


def style_pixel(frame):
    """Create a pixel-art look using OpenCV resize quantization.



    Tools:
    - OpenCV resize with nearest-neighbor upsampling.

    Steps:
    1. Downsample frame to coarse resolution.
    2. Upsample with INTER_NEAREST to preserve block edges.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w//8, h//8))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def style_cyberpunk(frame):
    """Apply a high-contrast cyberpunk-like grade.

    Tools:
    - OpenCV convertScaleAbs and NumPy intensity scaling.

    Steps:
    1. Increase contrast and shift brightness.
    2. Apply a global darkening multiplier for moody tone.
    """
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=-30)
    return (frame * 0.8).astype(np.uint8)


def style_comic(frame):
    """Apply comic rendering via edge extraction and bilateral smoothing.

    Tools:
    - OpenCV Canny, bilateralFilter, and subtract compositing.

    Steps:
    1. Detect edges from grayscale image.
    2. Smooth colors while preserving boundaries.
    3. Subtract edges from smoothed color image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    return cv2.subtract(color, edges)


def style_anime(frame):
    """Apply anime-like stylization from smoothed colors plus line edges.

    Tools:
    - OpenCV bilateralFilter, Canny, and channel conversion.

    Steps:
    1. Denoise/flatten color regions with bilateral filtering.
    2. Extract edge map and convert it to 3 channels.
    3. Subtract edges to emphasize cel-style boundaries.
    """
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    edges = cv2.Canny(frame, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.subtract(color, edges)


def style_ghibli(frame):
    """Apply a soft painted look using strong bilateral smoothing.

    Tools:
    - OpenCV bilateralFilter and convertScaleAbs.

    Steps:
    1. Smooth texture while preserving major edges.
    2. Slightly raise exposure/contrast for warm tone.
    """
    frame = cv2.bilateralFilter(frame, 15, 100, 100)
    return cv2.convertScaleAbs(frame, alpha=1.1, beta=20)


def style_ghibli_stronger(frame):
    """Apply stronger Ghibli-like abstraction with color quantization.

    Tools:
    - OpenCV bilateralFilter, kmeans, Canny, and subtract.

    Steps:
    1. Heavily smooth the frame.
    2. Quantize colors with k-means clustering.
    3. Extract edges and subtract them from quantized colors.
    """
    # 強め平滑化
    img = cv2.bilateralFilter(frame, 25, 150, 150)

    # 色を減らす（重要）
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    K = 8
    _, label, center = cv2.kmeans(
        Z,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((frame.shape))

    # エッジ
    edges = cv2.Canny(frame, 80, 120)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.subtract(img, edges)


def style_watercolor(frame):
    """Apply watercolor effect using OpenCV stylization filter.

    Tools:
    - OpenCV stylization.

    Steps:
    1. Run stylization with watercolor-friendly sigma parameters.
    2. Return stylized BGR frame.
    """
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)


def style_oil(frame):
    """Apply oil painting effect via OpenCV xphoto module.

    Tools:
    - OpenCV xphoto.oilPainting.

    Steps:
    1. Use local neighborhood painting simulation.
    2. Return brush-stroked frame.
    """
    return cv2.xphoto.oilPainting(frame, 7, 1)


def style_ukiyoe(frame):
    """Apply ukiyo-e inspired grade with smoothing and tone boost.

    Tools:
    - OpenCV bilateralFilter and convertScaleAbs.

    Steps:
    1. Smooth fine texture while keeping boundaries.
    2. Lift contrast and brightness for print-like palette.
    """
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

# ===============================
# Dispatcher (robust)
# ===============================


def apply_style_frame(frame, style):
    """Dispatch one frame to a selected style function.

    Tools:
    - Canonical style normalization and local style operators.

    Steps:


    1. Normalize input style token.
    2. Route to style-specific transform.
    3. Return original frame when style is unsupported.
    """

    style = normalize_style(style)

    if style == "pixel_art":
        return style_pixel(frame)
    elif style == "cyberpunk":
        return style_cyberpunk(frame)
    elif style == "american_comic":
        return style_comic(frame)
    elif style == "anime":
        return style_anime(frame)
    elif style == "ghibli":
        # return style_ghibli(frame)
        return style_ghibli_stronger(frame)
    elif style == "watercolor":
        return style_watercolor(frame)
    elif style == "oil_painting":
        return style_oil(frame)
    elif style == "ukiyoe":
        return style_ukiyoe(frame)
    else:
        return frame


def apply_style_frames(frames, style):
    """Apply style transform frame-by-frame for a video sequence.

    Tools:
    - Python list comprehension over apply_style_frame.

    Steps:
    1. Iterate through all frames in order.
    2. Stylize each frame with the same style.
    3. Return stylized frame list.
    """
    return [apply_style_frame(f, style) for f in frames]
