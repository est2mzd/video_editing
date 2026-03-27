# Attribute Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 入出力 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|--------|-----------|---------------|--------|------|-------------|----------------|
| Color adjustment | OpenCV（色変換/LUT） | 入力:画像/動画 → 出力:色変換画像/動画 | 不要 | 不要 | cv2 | 色変更は画素変換で可能 | 高速・完全制御 | 超軽量 |
| Color adjustment | Histogram Matching | 入力:画像 → 出力:色分布変換画像 | 不要 | 不要 | skimage | 色分布を合わせる | 自然な変換 | 超軽量 |
| Color adjustment | Segmentation + Color | 入力:画像 → mask → 出力:部分色変更画像 | COCO | SAM | https://github.com/facebookresearch/segment-anything | 部分的変更が必要 | 高精度制御 | 中（6〜8GB） |
| Color adjustment | Diffusion Editing | 入力:画像+テキスト → 出力:生成画像 | LAION | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 高品質変換 | 自然だが不安定 | 重い（10〜16GB） |
| Color adjustment | ControlNet | 入力:画像+条件 → 出力:生成画像 | LAION | ControlNet | https://github.com/lllyasviel/ControlNet | 条件付き色変更 | 制御性あり | 重い（10〜16GB） |
| Color adjustment | Video Color Model | 入力:動画 → 出力:色変換動画 | DAVIS / YouTube-VOS | Deep Video Colorization | https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization | 動画一貫性対応 | temporal安定 | 中（8〜10GB） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象決定
   ├─ 全体 → 全フレーム
   └─ 部分 → segmentation（SAM）

   ↓
② 色変換
   ├─ OpenCV（LUT / HSV変換）
   ├─ Histogram Matching
   └─ diffusion（必要な場合）

   ↓
③ temporal補正
   → RAFTで色の揺れ防止

   ↓
④ 出力
```

---

## Subclass別まとめ

### Color adjustment

```
手段：OpenCV（LUT / HSV）
```

---

### ✔ 本質
```
色 = 画素値変換
```

---

### ✔ 手段①（最適）
OpenCV（HSV / LUT）

- 完全制御可能
- 最速
- 最も安定

---

### ✔ 手段②
Histogram Matching

- 元画像に自然に合わせる
- 簡単で効果大

---

### ✔ 手段③（部分変更）
SAM + OpenCV

- 特定領域のみ色変更

---

### ✔ 手段④（高品質）
Stable Diffusion

- 見た目は良い
- ただし不安定

---

### ✔ 手段⑤（動画特化）
Video Colorization

- temporal一貫性あり

---

### ✔ 結論

安定 → OpenCV  
自然 → Histogram Matching  
高品質 → diffusion  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | ◎ | ◎ | ◎ | ◎ |
| Histogram | ○ | ◎ | ○ | ◎ |
| Segmentation | ◎ | ◎ | ○ | ○ |
| Diffusion | ◎ | △ | △ | △ |
| Video Model | ○ | ○ | △ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| SAM | 6〜8GB |
| Stable Diffusion | 10〜16GB |
| ControlNet | 10〜16GB |
| Video Colorization | 8〜10GB |

---

## 設計思想

- 色変更はCVで十分  
- diffusionは不要な場合が多い  
- 部分変更のみsegmentation  

---

## 判断フロー

```
① 全体の色変更か？
   → YES → OpenCV

② 部分変更か？
   → YES → SAM + OpenCV

③ 自然さ必要か？
   → YES → Histogram

④ 見た目重視か？
   → YES → diffusion
```

---

## 最終結論

```
基本 → OpenCV（LUT / HSV）
部分 → SAM + OpenCV
必要なら → Histogram
特殊 → diffusion
```
