import torch
import cv2
import numpy as np
from PIL import Image
from groundingdino.datasets import transforms as T
from groundingdino.util.inference import load_model, predict

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
CONFIG_PATH = "/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT = "/workspace/weights/groundingdino_swint_ogc.pth"


def transform_image(image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(image), None)[0]


def box_cxcywh_to_xyxy(boxes):
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def get_bbox(tensor0, zoom_target):
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CONFIG_PATH, CHECKPOINT, device=torch.device(device))

    # predict bboxes
    boxes, logits, phrases = predict(
        model=model,
        image=tensor0,
        caption=zoom_target,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )
    return boxes, logits, phrases


def zoom_in(frames, w, h, zoom_target="face . person .", zoom_factor=1.0):
    # get 1st frame
    img0 = frames[0]
    tensor0 = transform_image(img0)

    # get bbox
    boxes, _, _ = get_bbox(tensor0, zoom_target)
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = boxes * torch.tensor([w, h, w, h])
    boxes = boxes.cpu().numpy()

    # select the first box as the target
    target_box = boxes[0]
    print(f"Target box: {target_box}")

    x1, y1, x2, y2 = target_box
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    bw = (x2-x1)
    bh = (y2-y1)

    # bboxの長辺
    bbox_long = max(bw, bh)

    # 最終的に画面いっぱいになるスケール
    target_scale = (bbox_long / max(w, h)) / zoom_factor

    # t=0 → 1.0（全体）
    # t=1 → target_scale
    scales = np.linspace(1.0, target_scale, len(frames))

    # ==============================
    # 5. 各フレーム処理
    # ==============================
    output_frames = []

    for t, frame in enumerate(frames):

        scale = scales[t]

        # ---- 現在のcropサイズ（アスペクト比維持） ----
        crop_w = int(w * scale)
        crop_h = int(h * scale)

        # 中心固定
        x1 = int(cx - crop_w/2)
        y1 = int(cy - crop_h/2)
        x2 = int(cx + crop_w/2)
        y2 = int(cy + crop_h/2)

        # はみ出し補正（ここが本質）
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        cropped = frame[y1:y2, x1:x2]

        # resizeで元サイズへ戻す
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        output_frames.append(resized)

    return output_frames
