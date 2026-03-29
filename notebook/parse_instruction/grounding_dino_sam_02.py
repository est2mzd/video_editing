#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!which python
#!python3 -m pip install addict


# In[2]:


# ============================================
# video → mask → zoom in
# ============================================

import cv2
import torch
import numpy as np
from pathlib import Path

# ==============================
# 入力
# ==============================
mp4_path = "/workspace/data/videos/_pQAUwy0yWs_0_119to277.mp4"
instruction = "zoom in on face"

# ==============================
# 1. 動画から1フレーム取得
# ==============================
frames_rgb = []
cap = cv2.VideoCapture(mp4_path)

# 動画のフレーム数を取得
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for _ in range(frame_count):
    ret, frame = cap.read() # frameはnumpy.ndarray. channel orderはBGR
    # bgr → rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break
    frames_rgb.append(frame_rgb)
cap.release()

if not frames_rgb:
    raise RuntimeError("Failed to read video")

image_source = frames_rgb[0].copy()
h, w, _ = image_source.shape

print(f"Video loaded: {mp4_path}, frame size: {w}x{h}")
print(f"Frame number: {len(frames_rgb)} ")


# In[3]:


print(type(frame))
print(frame.shape)


# In[4]:


# mp4を表示する.
from IPython.display import Video
Video(mp4_path, embed=True, width=640, height=360)


# In[5]:


# ==============================
# 2. GroundingDINO
# ==============================
from groundingdino.util.inference import load_model, predict
from PIL import Image

CONFIG_PATH = "/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = load_model(CONFIG_PATH, CHECKPOINT, device=device)

TEXT_PROMPT = "face . person ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

from groundingdino.datasets import transforms as T

def tranform_image(image_np_rgb):
    transform = T.Compose([
        # 短辺を800にリサイズするという意味. max_sizeは長辺の最大値
        T.RandomResize([800], max_size=1333), # [800] <- listが 1個 のときは、randomにならない.
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    
    # Image を import
    image_pil_rgb = Image.fromarray(image_np_rgb)
    image_pil_transformed, _ = transform(image_pil_rgb, None)
    return image_pil_transformed    
    
    
# DINOはRGB想定
image_pil_transformed = tranform_image(frames_rgb[0])

boxes, logits, phrases = predict(
    model=dino_model,
    image=image_pil_transformed,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device,
)

if len(boxes) == 0:
    raise RuntimeError("No object detected")


# In[6]:


print(f"logits = {logits}")
print(f"phrases = {phrases}")
print(f"boxes = {boxes}")


# In[7]:


def box_cxcywh_to_xyxy(boxes):
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

# GroundingDINOは[0,1]正規化 → pixelへ
boxes_xyxy = box_cxcywh_to_xyxy(boxes)
boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h])
boxes_xyxy = boxes_xyxy.cpu().numpy()


# In[8]:


# BBox と　image を重ね描きする
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(frames_rgb[0])
colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
for i, box_xyxy in enumerate(boxes_xyxy):
    x1, y1, x2, y2 = box_xyxy
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor=colors[i % len(colors)], linewidth=2)
    plt.gca().add_patch(rect)
plt.axis('off')


# In[9]:


import subprocess
import json

def get_video_info_ffprobe(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    # duration（秒）
    duration = float(data["format"]["duration"])

    # fps（avg_frame_rate優先）
    fps_str = data["streams"][0]["avg_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den != 0 else 0

    return duration, fps


# In[10]:


# ============================================
# video → bbox追従 → zoom in（サイズ維持）
# ============================================

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ==============================
# 入力
# ==============================
mp4_path = "/workspace/data/videos/_pQAUwy0yWs_0_119to277.mp4"
instruction = "zoom in on face"

# 使用例
duration, fps = get_video_info_ffprobe(mp4_path)

print("duration(sec):", duration)
print("fps:", fps)
print("frame_count:", int(duration * fps))

fps = int(round(fps))
print("fps rounded:", fps)

# ==============================
# 1. 動画読み込み
# 背景意図：
# ・フレーム単位処理のため全フレーム保持
# ・RGBに統一（DINO前提）
# ==============================
frames_rgb = []
cap = cv2.VideoCapture(mp4_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames_rgb.append(frame_rgb)

cap.release()

if not frames_rgb:
    raise RuntimeError("Failed to read video")

h, w, _ = frames_rgb[0].shape
T_total = len(frames_rgb)

print(f"Video loaded: {w}x{h}, frames={T_total}")


# In[11]:


# ==============================
# 2. GroundingDINOロード
# 背景意図：
# ・毎フレームで対象再検出（tracking無しで簡易対応）
# ==============================
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T

CONFIG_PATH = "/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
dino_model = load_model(CONFIG_PATH, CHECKPOINT, device=device)

TEXT_PROMPT = "face . person ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


# In[12]:


# ==============================
# 3. 前処理
# 背景意図：
# ・DINO入力形式に統一
# ==============================
def transform_image(image_np_rgb):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(image_np_rgb)
    image_tensor, _ = transform(image_pil, None)
    return image_tensor

# ==============================
# 4. bbox変換
# ==============================
def box_cxcywh_to_xyxy(boxes):
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

# ==============================
# 5. bbox選択（face優先）
# 背景意図：
# ・複数検出から一意に決定しないとズームがブレる
# ==============================
def select_target_box(boxes_xyxy, logits, phrases):
    # face優先
    for i, p in enumerate(phrases):
        if "face" in p:
            return boxes_xyxy[i]

    # fallback: 最大面積
    areas = (boxes_xyxy[:,2] - boxes_xyxy[:,0]) * (boxes_xyxy[:,3] - boxes_xyxy[:,1])
    return boxes_xyxy[np.argmax(areas)]

# ==============================
# 6. bbox平滑化
# 背景意図：
# ・フレームごとの揺れを抑制（重要）
# ==============================
def smooth_bbox(prev_box, curr_box, alpha=0.7):
    if prev_box is None:
        return curr_box
    return alpha * prev_box + (1 - alpha) * curr_box

# ==============================
# 7. zoom処理
# 背景意図：
# ・frame進行に応じてbboxを徐々に縮小（＝ズームイン）
# ==============================
def zoom_bbox(box, t_ratio, zoom_strength=0.6):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1)
    bh = (y2 - y1)

    scale = 1.0 - zoom_strength * t_ratio  # 徐々に縮小

    new_w = bw * scale
    new_h = bh * scale

    x1_new = cx - new_w / 2
    y1_new = cy - new_h / 2
    x2_new = cx + new_w / 2
    y2_new = cy + new_h / 2

    return np.array([x1_new, y1_new, x2_new, y2_new])

# ==============================
# 8. crop + resize
# 背景意図：
# ・最終出力サイズは固定（コンペ要件）
# ==============================
def crop_and_resize(frame, box, w, h):
    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        return frame

    resized = cv2.resize(cropped, (w, h))
    return resized


# In[13]:


# ==============================
# 9. メイン処理
# 背景意図：
# ・全フレームに対して一貫した処理
# ==============================
output_frames = []
prev_box = None

for t, frame in enumerate(frames_rgb):

    # ---- DINO ----
    image_tensor = transform_image(frame)

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    if len(boxes) == 0:
        output_frames.append(frame)
        continue

    # ---- bbox変換 ----
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h])
    boxes_xyxy = boxes_xyxy.cpu().numpy()

    # ---- target選択 ----
    target_box = select_target_box(boxes_xyxy, logits, phrases)

    # ---- 平滑化 ----
    target_box = smooth_bbox(prev_box, target_box)
    prev_box = target_box

    # ---- zoom ----
    t_ratio = t / T_total
    zoomed_box = zoom_bbox(target_box, t_ratio)

    # ---- crop + resize ----
    out_frame = crop_and_resize(frame, zoomed_box, w, h)

    output_frames.append(out_frame)

# ==============================
# 10. 動画保存
# 背景意図：
# ・結果確認用
# ==============================
out_path = "/workspace/workspace/output/output_zoom.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for f in output_frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

writer.release()

print(f"Saved: {out_path}")


# In[14]:


# mp4を表示する.
from IPython.display import Video
Video(out_path, embed=True, width=640, height=360)


# In[15]:


# output_frames　を 列数指定、行数は自動で調整して表示する.
import matplotlib.pyplot as plt
cols = 5
rows = (len(output_frames) + cols - 1) // cols
plt.figure(figsize=(20, 4 * rows))
for i, frame in enumerate(output_frames):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(frame)
    plt.axis('off')
plt.tight_layout()


# In[16]:


# ============================================
# video → stable zoom（完全修正版）
# ============================================

import cv2
import torch
import numpy as np
from PIL import Image

# ==============================
# 入力
# ==============================
mp4_path = "/workspace/data/videos/_pQAUwy0yWs_0_119to277.mp4"
out_path = "/workspace/notebook/output/zoom_fixed.mp4"

zoom_factor = 1.0  # ← ユーザー指定
# fps = 30

# ==============================
# 1. 動画読み込み
# ==============================
frames_rgb = []
cap = cv2.VideoCapture(mp4_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

h, w, _ = frames_rgb[0].shape
T_total = len(frames_rgb)

print(f"frames={T_total}, size={w}x{h}")

# ==============================
# 2. DINO
# ==============================
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T

CONFIG_PATH = "/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(CONFIG_PATH, CHECKPOINT, device=device)

def transform_image(image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return transform(Image.fromarray(image), None)[0]

def box_cxcywh_to_xyxy(boxes):
    cx, cy, bw, bh = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    return torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1)

# ==============================
# 3. 初回bbox取得（←ここ重要）
# 背景意図：
# ・ズームスケールは1フレーム目で固定
# ==============================
img0 = frames_rgb[0]
tensor0 = transform_image(img0)

boxes, logits, phrases = predict(
    model=model,
    image=tensor0,
    caption="face . person .",
    box_threshold=0.35,
    text_threshold=0.25,
    device=device,
)

boxes = box_cxcywh_to_xyxy(boxes)
boxes = boxes * torch.tensor([w,h,w,h])
boxes = boxes.cpu().numpy()

# face優先
target_box = boxes[0]

x1,y1,x2,y2 = target_box
cx = (x1+x2)/2
cy = (y1+y2)/2
bw = (x2-x1)
bh = (y2-y1)

# ==============================
# 4. ズームスケジュール生成（固定）
# 背景意図：
# ・frameごとに同じスケール規則を使う
# ==============================

# bboxの長辺
bbox_long = max(bw, bh)

# 最終的に画面いっぱいになるスケール
target_scale = (bbox_long / max(w,h)) / zoom_factor

# t=0 → 1.0（全体）
# t=1 → target_scale
scales = np.linspace(1.0, target_scale, T_total)

# ==============================
# 5. 各フレーム処理
# ==============================
output_frames = []

for t, frame in enumerate(frames_rgb):

    scale = scales[t]

    # ---- 現在のcropサイズ（アスペクト比維持） ----
    crop_w = int(w * scale)
    crop_h = int(h * scale)

    # 中心固定
    x1 = int(round(cx - crop_w/2))
    y1 = int(round(cy - crop_h/2))
    x2 = int(round(cx + crop_w/2))
    y2 = int(round(cy + crop_h/2))

    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2]

    # resizeで元サイズへ戻す
    resized = cv2.resize(cropped, (w,h), interpolation=cv2.INTER_LINEAR)

    output_frames.append(resized)


# In[17]:


fps


# In[22]:


# ==============================
# 6. 動画保存（破損対策）
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ← 安定
out_path2 = out_path.replace(".mp4", "_v2.mp4")  # テスト用パス
writer = cv2.VideoWriter(out_path2, fourcc, 30, (w,h))
print("writer opened:", writer.isOpened())

for f in output_frames:
    # RGB→BGR（重要）
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

writer.release()

print("Saved:", out_path2)


# In[23]:


# mp4を表示する.
from IPython.display import Video
Video(out_path2, embed=True, width=640, height=360)


# In[25]:


import subprocess
import os 
tmp_dir = "/workspace/notebook/output/tmp_frames"
os.makedirs(tmp_dir, exist_ok=True)
out_path_3 = out_path.replace(".mp4", "_v3.mp4")  # テスト用パス

# フレーム保存
for i, f in enumerate(output_frames):
    path = f"{tmp_dir}/{i:05d}.png"
    cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

# ffmpegで動画化
cmd = f"""
ffmpeg -y -framerate {fps} -i {tmp_dir}/%05d.png \
-c:v libx264 -pix_fmt yuv420p {out_path_3}
"""
subprocess.run(cmd, shell=True)

# mp4を表示する.
from IPython.display import Video
Video(out_path_3, embed=True, width=640, height=360)


# In[20]:


# output_frames　を 列数指定、行数は自動で調整して表示する.
import matplotlib.pyplot as plt
cols = 5
rows = (len(output_frames) + cols - 1) // cols
plt.figure(figsize=(20, 4 * rows))
for i, frame in enumerate(output_frames):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(frame)
    plt.axis('off')
plt.tight_layout()


# In[21]:


# 各フレームの サイズの分布を表示する. width vs height
frame_sizes = [(f.shape[1], f.shape[0]) for f in output_frames]
widths, heights = zip(*frame_sizes)
min_width, max_width = min(widths), max(widths)
min_height, max_height = min(heights), max(heights)
print(f"Width: min={min_width}, max={max_width}")
print(f"Height: min={min_height}, max={max_height}")
plt.figure(figsize=(10,5))
plt.plot(widths, heights, 'o')
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Frame Size Distribution')
plt.grid(True)
plt.xlim(1919, 1921)
plt.ylim(1079, 1081)
plt.show()


# In[ ]:




