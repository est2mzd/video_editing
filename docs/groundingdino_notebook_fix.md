# GroundingDINO test.ipynb 実行検証レポート

> 作成日: 2026-03-22  
> 対象ファイル: `third_party/GroundingDINO/test.ipynb`

---

## 概要

`test.ipynb` をそのまま実行すると複数のエラーが発生し、推論が完了しなかった。
`bertwarper.py` に互換パッチを当て、ノートブックのパス定義を整理することで正常実行できた。

最終的な実行結果:
- `chair . person . dog .` のプロンプトに対し **dog × 2 件を検出**
- 出力画像 `annotated_image.jpg` を保存

---

## 実行環境

| 項目 | 値 |
|---|---|
| OS | Ubuntu 22.04.4 LTS (dev container) |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu124 |
| transformers | 5.3.0 |
| CUDA | 利用可能 (NVIDIA GPU) |

---

## Before: 変更前の問題

### Cell 1
```python
!which python
!python3 -m pip install addict
```
→ **成功** (問題なし)

---

### Cell 2 (推論セル) — 問題箇所

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "../04-06-segment-anything/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = ".asset/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
...
```

#### エラー 1: `FileNotFoundError`

```
FileNotFoundError: [Errno 2] No such file or directory:
  '../04-06-segment-anything/weights/groundingdino_swint_ogc.pth'
```

**原因**: 重みファイルのパスが古いディレクトリ構成を前提としていた。
実際の重みは `third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth` に存在する。

---

#### エラー 2: `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`

```
File groundingdino/models/GroundingDINO/bertwarper.py:29
    self.get_head_mask = bert_model.get_head_mask
AttributeError: 'BertModel' object has no attribute 'get_head_mask'
```

**原因**: GroundingDINO が想定する `transformers~=4.33` では  
`BertModel` インスタンスが `get_head_mask` を持つが、  
環境内の `transformers==5.3.0` ではこのメソッドが削除された。

---

#### エラー 3: `TypeError: to() received an invalid combination of arguments`

```
File transformers/modeling_utils.py:952
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, ...)
```

**原因**: `transformers 5` で `get_extended_attention_mask` の引数仕様が変わり、  
3 番目の引数が `device` から受け付けなくなった (内部的に `dtype` を要求するようになった)。

---

## After: 変更内容

### 1. `bertwarper.py` への互換パッチ

ファイル: `third_party/GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py`

#### 追加 import

```python
import inspect
```

#### `__init__` の変更

```diff
- self.get_head_mask = bert_model.get_head_mask
+ self._extended_attention_uses_dtype = (
+     "dtype" in inspect.signature(self.get_extended_attention_mask).parameters
+ )
+ if hasattr(bert_model, "get_head_mask"):
+     self.get_head_mask = bert_model.get_head_mask
+ else:
+     self.get_head_mask = self._compat_get_head_mask
```

#### 追加メソッド 2 件

```python
def _compat_get_extended_attention_mask(self, attention_mask, input_shape, device):
    """transformers 4 (device引数) と 5 (dtype引数) の両方に対応"""
    if self._extended_attention_uses_dtype:
        dtype = self.embeddings.word_embeddings.weight.dtype
        return self.get_extended_attention_mask(attention_mask, input_shape, dtype=dtype)
    return self.get_extended_attention_mask(attention_mask, input_shape, device)

def _compat_get_head_mask(self, head_mask, num_hidden_layers):
    """transformers 5 で削除された get_head_mask の互換実装"""
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    head_mask = head_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
    return head_mask
```

#### `forward` の変更

```diff
- extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
-     attention_mask, input_shape, device
- )
+ extended_attention_mask: torch.Tensor = self._compat_get_extended_attention_mask(
+     attention_mask, input_shape, device
+ )
```

---

### 2. `test.ipynb` Cell 2 の書き換え

重みパスのハードコードを廃止し、複数候補から自動解決するよう変更した。

```python
from pathlib import Path
import torch, cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"

CHECKPOINT_CANDIDATES = [
    ROOT / "weights/groundingdino_swint_ogc.pth",
    ROOT / "../weights/groundingdino_swint_ogc.pth",
    ROOT / "../../checkpoints/groundingdino_swint_ogc.pth",
]
checkpoint_path = next((p for p in CHECKPOINT_CANDIDATES if p.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_CANDIDATES}")

IMAGE_CANDIDATES = [
    ROOT / ".asset/cat_dog.jpeg",
    ROOT / "demo/test.jpg",
]
image_path = next((p for p in IMAGE_CANDIDATES if p.exists()), None)
if image_path is None:
    raise FileNotFoundError(f"image not found: {IMAGE_CANDIDATES}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(str(CONFIG_PATH), str(checkpoint_path), device=device)

TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(str(image_path))
boxes, logits, phrases = predict(
    model=model, image=image, caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD, device=device,
)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(str(ROOT / "annotated_image.jpg"), annotated_frame)
print(f"detections={len(phrases)}", phrases)
```

---

## 実行条件まとめ

| 条件 | 内容 |
|---|---|
| 重みファイル | `third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth` が存在すること |
| 入力画像 | `third_party/GroundingDINO/.asset/cat_dog.jpeg` が存在すること |
| transformers | 5.x 系でも動作可（互換パッチ適用済み）。4.33 系でも引き続き動作 |
| GPU | CUDA 有効で高速実行。`device` を自動判定するため CPU 環境でも起動可 |
| 作業ディレクトリ | ノートブックは `third_party/GroundingDINO/` 配下で起動すること |
| pip パッケージ | `addict`, `supervision>=0.22`, `timm`, `opencv-python` が必要（環境ではインストール済み） |

---

## 残存 Warning

以下は機能に影響しない警告で、無視してよい。

| Warning | 意味 |
|---|---|
| `timm.models.layers is deprecated` | timm 2.x への移行催促。動作は問題なし |
| `torch.meshgrid: indexing argument` | PyTorch 2.x での非推奨警告。動作は問題なし |
| `torch.cuda.amp.autocast` deprecated | amp API の新構文への移行催促。動作は問題なし |
| `torch.load weights_only=False` | セキュリティ推奨の警告。信頼できる重みファイルなら無視可 |
| BertModel `UNEXPECTED keys` | GroundingDINO が MLM ヘッドを必要としないため。正常動作 |
