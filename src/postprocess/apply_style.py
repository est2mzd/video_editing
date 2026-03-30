
import cv2
import numpy as np

# ===============================
# Style normalization
# ===============================
def normalize_style(style: str) -> str:
    s = style.lower().replace("-", "").replace("_", "").strip()

    mapping = {
        "cyberpunk": ["cyberpunk", "cyber", "neon"],
        "pixel_art": ["pixel", "pixelart", "8bit"],
        "american_comic": ["comic", "americancomic", "cartooncomic"],
        "anime": ["anime", "japanimation"],
        "ghibli": ["ghibli", "miyazaki", "studio ghibli"],
        "watercolor": ["watercolor", "watercolour"],
        "oil_painting": ["oil", "oilpainting"],
        "ukiyoe": ["ukiyoe", "ukiyo", "japaneseprint"],
    }

    for key, vals in mapping.items():
        for v in vals:
            if v in s:
                return key

    return s

# ===============================
# Style functions (same as before)
# ===============================
def style_pixel(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w//8, h//8))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def style_cyberpunk(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=-30)
    return (frame * 0.8).astype(np.uint8)


def style_comic(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    return cv2.subtract(color, edges)


def style_anime(frame):
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    edges = cv2.Canny(frame, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.subtract(color, edges)


def style_ghibli(frame):
    frame = cv2.bilateralFilter(frame, 15, 100, 100)
    return cv2.convertScaleAbs(frame, alpha=1.1, beta=20)

def style_ghibli_stronger(frame):
    # 強め平滑化
    img = cv2.bilateralFilter(frame, 25, 150, 150)

    # 色を減らす（重要）
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    K = 8
    _,label,center=cv2.kmeans(Z,K,None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),
        10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((frame.shape))

    # エッジ
    edges = cv2.Canny(frame, 80, 120)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.subtract(img, edges)



def style_watercolor(frame):
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)


def style_oil(frame):
    return cv2.xphoto.oilPainting(frame, 7, 1)


def style_ukiyoe(frame):
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

# ===============================
# Dispatcher (robust)
# ===============================
def apply_style_frame(frame, style):

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
    return [apply_style_frame(f, style) for f in frames]
