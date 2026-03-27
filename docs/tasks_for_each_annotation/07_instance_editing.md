# Instance Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 入出力 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|--------|-----------|---------------|--------|------|-------------|----------------|
| Instance Insertion | OpenCV（貼り付け） | 画像→画像 | 不要 | 不要 | cv2 | 単純合成可能 | 不自然になりやすい | 超軽量 |
| Instance Insertion | Copy-Paste + Segmentation | 画像＋マスク→画像 | COCO | SAM | https://github.com/facebookresearch/segment-anything | マスク必要 | 制御しやすい | 中（6〜8GB） |
| Instance Insertion | PISCO | 画像＋条件→画像 | COCO / LVIS | PISCO | https://github.com/amazon-science/pisco | 挿入特化モデル | 高精度・位置制御可能 | 重い（12〜16GB） |
| Instance Insertion | Diffusion | 画像＋テキスト→画像 | LAION | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 自然生成可能 | 高品質だが不安定 | 重い（10〜16GB） |
| Instance Replacement | OpenCV（置換） | 画像→画像 | 不要 | 不要 | cv2 | 単純置換可能 | 境界不自然 | 超軽量 |
| Instance Replacement | Segmentation + Diffusion | 画像＋マスク＋テキスト→画像 | COCO | SAM + SD | https://github.com/facebookresearch/segment-anything | 対象置換 | 高品質 | 重い |
| Instance Replacement | ControlNet | 画像＋条件→画像 | LAION | ControlNet | https://github.com/lllyasviel/ControlNet | 構造維持 | 制御性高い | 重い |
| Instance Removal | OpenCV（塗りつぶし） | 画像→画像 | 不要 | 不要 | cv2 | 簡易削除 | 品質低い | 超軽量 |
| Instance Removal | Inpainting | 画像＋マスク→画像 | Places2 | LaMa | https://github.com/advimman/lama | 欠損補完 | 高品質 | 中（8GB） |
| Instance Removal | Video Inpainting | 動画＋マスク→動画 | YouTube-VOS | E2FGVI | https://github.com/MCG-NKU/E2FGVI | 動画一貫性 | 高品質 | 中（8〜10GB） | YouTube-VOS | E2FGVI | https://github.com/MCG-NKU/E2FGVI | 動画一貫性 | 高品質 | 中（8〜10GB） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象検出
   → GroundingDINO + SAM（mask取得）

   ↓
② 編集処理
   ├─ Insertion
   │     → PISCO or Copy-Paste
   │
   ├─ Replacement
   │     → mask + diffusion
   │
   └─ Removal
         → inpainting（LaMa / E2FGVI）

   ↓
③ temporal補正
   → RAFT
   → XMem（mask追跡）

   ↓
④ 出力
```

---

## Subclass別まとめ

### Instance Insertion

```
手段：PISCO / Copy-Paste
```

- 物体追加 + 自然な配置

#### 結論
安定 → Copy-Paste  
高品質 → PISCO  

---

### Instance Replacement

```
手段：Segmentation + Diffusion
```

- 対象を別のものに置換

#### 結論
Diffusionベース  

---

### Instance Removal

```
手段：Inpainting
```

- 対象削除 + 背景補完

#### 結論
静止画 → LaMa  
動画 → E2FGVI  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | 低 | ◎ | ◎ | ◎ |
| Copy-Paste | 中 | ◎ | ◎ | ○ |
| PISCO | ◎ | ○ | ◎ | △ |
| Inpainting | ◎ | ○ | △ | ○ |
| Diffusion | ◎ | △ | △ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| SAM | 6〜8GB |
| PISCO | 12〜16GB |
| Stable Diffusion | 10〜16GB |
| ControlNet | 10〜16GB |
| LaMa | 8GB |
| E2FGVI | 8〜10GB |

---

## 最終結論

```
Insertion → PISCO
Replacement → SAM + Diffusion
Removal → LaMa / E2FGVI
全体 → RAFT
```
