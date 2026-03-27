# Camera Angle Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 入出力 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|--------|-----------|---------------|--------|------|-------------|----------------|
| High angle | OpenCV（幾何変換） | 画像→画像 | 不要 | 不要 | cv2 | 疑似的な視点変更 | 不自然になりやすい | 超軽量 |
| High angle | Depth + Warp | 画像→depth→画像 | KITTI / NYU | MiDaS | https://github.com/isl-org/MiDaS | 視点変更はdepth必要 | 近似だが実用可能 | 中（6〜8GB） |
| High angle | Video Diffusion | 画像＋テキスト→動画 | WebVid | Wan / VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | 視点変更を生成 | 高品質・不安定 | 重い（16GB〜） |
| High angle | NeRF | 画像群→3D→画像 | LLFF | NeRF | https://github.com/bmild/nerf | 真の3D視点変更 | 最も正確 | 超重い（24GB〜） |
| Low angle | OpenCV（幾何変換） | 画像→画像 | 不要 | 不要 | cv2 | 疑似視点変更 | 不自然 | 超軽量 |
| Low angle | Depth + Warp | 画像→depth→画像 | KITTI / NYU | Depth-Anything | https://github.com/LiheYoung/Depth-Anything | depthで視点変化 | 実用レベル | 中（8GB） |
| Low angle | Video Diffusion | 画像＋テキスト→動画 | WebVid | Wan / VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | 視点生成 | 高品質・不安定 | 重い |
| Low angle | NeRF | 画像群→3D→画像 | LLFF | NeRF | https://github.com/bmild/nerf | 3D再構成 | 最も自然 | 超重い |


| Subclass | 手段 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|-----------|---------------|--------|------|-------------|----------------|
| High angle | OpenCV（幾何変換） | 不要 | 不要 | cv2 | 疑似的な視点変更 | 不自然になりやすい | 超軽量 |
| High angle | Depth + Warp | KITTI / NYU | MiDaS | https://github.com/isl-org/MiDaS | 視点変更はdepth必要 | 近似だが実用可能 | 中（6〜8GB） |
| High angle | Video Diffusion | WebVid | Wan / VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | 視点変更を生成 | 高品質・不安定 | 重い（16GB〜） |
| High angle | NeRF | LLFF | NeRF | https://github.com/bmild/nerf | 真の3D視点変更 | 最も正確 | 超重い（24GB〜） |
| Low angle | OpenCV（幾何変換） | 不要 | 不要 | cv2 | 疑似視点変更 | 不自然 | 超軽量 |
| Low angle | Depth + Warp | KITTI / NYU | Depth-Anything | https://github.com/LiheYoung/Depth-Anything | depthで視点変化 | 実用レベル | 中（8GB） |
| Low angle | Video Diffusion | WebVid | Wan / VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | 視点生成 | 高品質・不安定 | 重い |
| Low angle | NeRF | LLFF | NeRF | https://github.com/bmild/nerf | 3D再構成 | 最も自然 | 超重い |

---

## 実戦構成

```
[入力動画]
   ↓
① depth推定
   → MiDaS / Depth-Anything

   ↓
② 視点変換
   ├─ High angle → 上方向視点へwarp
   └─ Low angle → 下方向視点へwarp

   ↓
③ 補完処理
   → inpainting（欠損領域）

   ↓
④ temporal補正
   → RAFT

   ↓
⑤ 出力
```

---

## Subclass別まとめ

### High angle

```
手段：Depth + Warp
```

- 上から見下ろす視点

#### 手段①（推奨）
MiDaS  
- 軽量で実用

#### 手段②
Diffusion  
- 見た目良いが不安定

#### 手段③
NeRF  
- 完全3Dだが非現実的

#### 結論
現実 → Depth  
研究 → NeRF  

---

### Low angle

```
手段：Depth + Warp
```

- 下から見上げる視点

#### 手段①（推奨）
Depth-Anything  
- 高精度depth

#### 手段②
Diffusion  
- 見た目良い

#### 手段③
NeRF  
- 理想解

#### 結論
Depthベース  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | 低 | ◎ | ◎ | ◎ |
| Depth | 中 | ○ | ○ | ○ |
| Diffusion | 高 | △ | △ | △ |
| NeRF | ◎ | ◎ | ◎ | × |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| MiDaS | 6GB |
| Depth-Anything | 8GB |
| Stable Diffusion | 10〜16GB |
| VideoCrafter | 16GB〜 |
| NeRF | 24GB〜 |

---

## 設計思想

- 視点変更 = 3D問題  
- depthなしでは破綻  
- diffusionは補助  

---

## 判断フロー

```
① 視点変更か？
   → YES → depth

② 高品質必要か？
   → YES → diffusion

③ 完全3D必要か？
   → YES → NeRF

④ 軽量必要か？
   → YES → depthのみ
```

---

## 最終結論

```
基本 → Depth + Warp
高品質 → diffusion追加
理想 → NeRF
全体 → RAFT
```
