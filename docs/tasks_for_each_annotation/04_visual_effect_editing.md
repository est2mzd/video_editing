# Visual Effect Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|-----------|---------------|--------|------|-------------|----------------|
| Background Change | OpenCV（クロマキー/置換） | 不要 | 不要 | cv2 | 単純背景は置換可能 | 高速だが精度低い | 超軽量 |
| Background Change | Segmentation + Replace | COCO | SAM | https://github.com/facebookresearch/segment-anything | 背景分離が必要 | 高精度マスク | 中（6〜8GB） |
| Background Change | Text-conditioned Segmentation | COCO | GroundingDINO + SAM | https://github.com/IDEA-Research/GroundingDINO | 指示に基づく対象抽出 | 柔軟性高い | 中（8GB） |
| Background Change | Video Editing | WebVid | VACE / Wan | https://github.com/Vchitect/VACE | 動画編集特化 | 高品質・一貫性課題 | 重い（16GB〜） |
| Background Change | Video Inpainting | YouTube-VOS | E2FGVI | https://github.com/MCG-NKU/E2FGVI | 背景補完必要 | 時系列安定 | 中（8〜10GB） |
| Decoration effect | OpenCV（フィルタ） | 不要 | 不要 | cv2 | エフェクトは画像処理 | 軽量・制御可能 | 超軽量 |
| Decoration effect | Diffusion Editing | LAION | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 見た目変化に強い | 高品質 | 重い（10〜16GB） |
| Decoration effect | ControlNet | LAION | ControlNet | https://github.com/lllyasviel/ControlNet | 条件付き制御 | 安定性向上 | 重い（10〜16GB） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象分離
   ├─ Background → segmentation（SAM / DINO）
   └─ Decoration → 全体 or 部分指定

   ↓
② 編集処理
   ├─ Background Change
   │     → mask抽出 → 背景置換 or inpainting
   │     → 高品質なら VACE
   │
   └─ Decoration
         → OpenCVフィルタ
         → diffusion（必要なら）

   ↓
③ フレーム間補正
   → RAFTで安定化
   → XMemでmask維持

   ↓
④ 出力
```

---

## Subclass別まとめ

### Background Change

```
手段：Segmentation + Replace
```

- 背景変更 = マスク精度が全て

#### 手段①（推奨）
GroundingDINO + SAM  
- 高精度で対象抽出

#### 手段②
E2FGVI  
- 背景補完（穴埋め）

#### 手段③
VACE  
- end-to-endだが不安定あり

#### 結論
Segmentationベースが最も安定  

---

### Decoration effect

```
手段：OpenCV or Diffusion
```

- エフェクト = 見た目変化

#### 手段①
OpenCV  
- 軽量・制御可能

#### 手段②
Stable Diffusion  
- 高品質

#### 手段③
ControlNet  
- 制御性向上

#### 結論
軽量 → OpenCV  
高品質 → diffusion  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | 低 | ◎ | ◎ | ◎ |
| Segmentation | 高 | ◎ | ○ | ○ |
| Inpainting | 中 | ○ | △ | ○ |
| Diffusion | 高 | △ | △ | △ |
| VACE | 高 | △ | △ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| SAM | 6〜8GB |
| GroundingDINO | 8GB |
| E2FGVI | 8〜10GB |
| Stable Diffusion | 10〜16GB |
| ControlNet | 10〜16GB |
| VACE | 16GB〜 |

---

## 設計思想

- 背景変更は segmentation が核心  
- decoration は軽量でも対応可能  
- diffusionは補助的に使う  

---

## 判断フロー

```
① 背景を変えるか？
   → YES → segmentation

② 見た目加工か？
   → YES → OpenCV or diffusion

③ temporal崩れるか？
   → YES → RAFT

④ 精度足りないか？
   → YES → ControlNet / VACE
```

---

## 最終結論

```
Background → GroundingDINO + SAM + Inpainting
Decoration → OpenCV or Diffusion
全体 → RAFT
```
