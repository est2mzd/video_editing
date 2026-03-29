#!/usr/bin/env python
# coding: utf-8

# # task_02_background
# zoom in 処理 — GroundingDINO でターゲットを検出し、フレームごとにズームクロップして動画を出力する。

# ## 1. パラメータ設定

# In[10]:


# ==============================
# パラメータ設定
# ==============================

# 入力・出力
mp4_path = "/workspace/data/videos/DaUJkmBvTKM_2_0to150.mp4"
out_path = "/workspace/notebook/output/background_change.mp4"
# instruction = "background change"

annotation = {"video_path": "DaUJkmBvTKM_2_0to150.mp4", "selected_class": "Visual Effect Editing", "selected_subclass": "Background Change", "instruction": "Replace the entire outdoor background behind the speaker with a sleek, modern automotive showroom featuring soft studio lighting and blurred cars in the distance. Ensure that the speaker's outline, including the fine edges of his head and shoulders, is cleanly masked with no edge flickering or artifacts across all frames. The existing lower-third text and the 'an' logo in the top right must remain perfectly legible and completely unaffected by the background modification. Maintain a high-quality, professional look where the new background's lighting and perspective naturally complement the speaker's position in the frame."}

# ズーム
zoom_factor = 1.0  # ← ユーザー指定

# GroundingDINO
CONFIG_PATH = "/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "sky" #"face . person ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# SAM
SAM_CHECKPOINT = "/workspace/weights/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"


# ## 2. インポート

# In[11]:


# ==============================
# インポート
# ==============================

import cv2
import torch
import numpy as np
import subprocess
import json
import os
import sys
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
from IPython.display import Video

from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T

sys.path.append("/workspace/third_party/segment-anything")
from segment_anything import sam_model_registry, SamPredictor


# ## 3. 関数定義

# In[12]:


# ==============================
# 関数定義
# ==============================

def get_video_info_ffprobe(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    fps_str = data["streams"][0]["avg_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den != 0 else 0
    width = int(data["streams"][0]["width"])
    height = int(data["streams"][0]["height"])
    return {"duration": duration, "fps": fps, "width": width, "height": height}


def transform_image(image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(image), None)[0]


def box_cxcywh_to_xyxy(boxes):
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


# ## 4. 動画読み込み

# In[13]:


# ==============================
# 動画読み込み
# ==============================

video_info = get_video_info_ffprobe(mp4_path)

print("duration(sec):", video_info["duration"])
print("fps:", video_info["fps"])
print("frame_count:", int(video_info["duration"] * video_info["fps"]))
print(f"shape: {video_info['width']}x{video_info['height']}")

fps = video_info["fps"]

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


# ## 5. モデルロード

# In[14]:


# ==============================
# モデルロード
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

# GroundingDINO
model_dino = load_model(CONFIG_PATH, CHECKPOINT, device=device)

# SAM
model_sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
model_sam.to(device=device)
predictor = SamPredictor(model_sam)


# ## 6. 処理

# In[15]:


def select_target_box(boxes_xyxy, logits, phrases):
    # face優先
    for i, p in enumerate(phrases):
        if "face" in p:
            return boxes_xyxy[i]

    # fallback: 最大面積
    areas = (boxes_xyxy[:,2] - boxes_xyxy[:,0]) * (boxes_xyxy[:,3] - boxes_xyxy[:,1])
    return boxes_xyxy[np.argmax(areas)]


# In[21]:


# ==============================
# メイン処理（DINO + SAM + 背景変更）
# ==============================
output_frames = []
prev_box = None
all_bboxes = []

for t, frame in enumerate(frames_rgb):

    # -----------------------------
    # 1. DINO
    # -----------------------------
    image_tensor = transform_image(frame)

    boxes, logits, phrases = predict(
        model=model_dino,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    if len(boxes) == 0:
        output_frames.append(frame)
        continue

    # -----------------------------
    # 2. bbox変換（添付コード通り）
    # -----------------------------
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h])
    boxes_xyxy = boxes_xyxy.cpu().numpy()
    all_bboxes.append(boxes_xyxy)
    # -----------------------------
    # 3. target選択
    # -----------------------------
    target_box = select_target_box(boxes_xyxy, logits, phrases)

    # -----------------------------
    # 4. SAM（重要）
    # -----------------------------
    predictor.set_image(frame)

    masks, scores, _ = predictor.predict(
        box=target_box,
        multimask_output=False
    )

    mask = masks[0]  # (H,W)

    # -----------------------------
    # 5. 背景生成（仮）
    # -----------------------------
    bg = np.full_like(frame, [255,128,0])

    # -----------------------------
    # 6. 合成
    # -----------------------------
    mask_3ch = np.repeat(mask[:,:,None], 3, axis=2)

    output = np.where(mask_3ch, frame, bg)

    output_frames.append(output)


# In[22]:


# デバッグ用: 最初の数フレームだけ画像とBBoxをpltで表示
import matplotlib.colors as mcolors
frame_id = 0
frame = frames_rgb[frame_id]
boxes_xyxy = all_bboxes[frame_id]
logits = logits.cpu().numpy()
phrases = phrases    
colors_str = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
colors_num = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
# 画像の表示
plt.imshow(frame)

# bboxの上書き
for i, bbox in enumerate(boxes_xyxy):
    x1, y1, x2, y2 = map(int, bbox)
    color = colors_num[i % len(colors_num)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    plt.text(x1, y1, phrases[i], color=colors_str[i % len(colors_str)], fontsize=12)
    print(f"Box {i}: {phrases[i]}, Score: {logits[i]:.2f}, Coords: ({x1}, {y1}), ({x2}, {y2})")


# ## 7. 動画出力

# In[17]:


# ==============================
# 動画出力
# ==============================

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

# mp4を表示する
Video(out_path_3, embed=True, width=640, height=360)


# ## 8. 結果確認

# In[18]:


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

