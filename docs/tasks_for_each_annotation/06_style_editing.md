# Style Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|-----------|---------------|--------|------|-------------|----------------|
| Watercolor | Diffusion | LAION-5B | Stable Diffusion | https://github.com/CompVis/stable-diffusion | スタイル変換に最適 | 高品質 | 重い（10〜16GB） |
| Ghibli | Diffusion | LAION | SD + LoRA | https://github.com/cloneofsimo/lora | 特定スタイル学習可能 | 高品質・再現性あり | 重い（10〜16GB） |
| Ukiyo-e | Diffusion | WikiArt | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 伝統絵画スタイル | 高品質 | 重い |
| American comic style | Diffusion | LAION | SD + ControlNet | https://github.com/lllyasviel/ControlNet | 線画・色制御 | 制御性あり | 重い |
| Cyberpunk | Diffusion | LAION | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 色調変換に強い | 高品質 | 重い |
| Anime | Diffusion | Danbooru | AnythingV3 | https://github.com/AUTOMATIC1111/stable-diffusion-webui | アニメ特化 | 非常に高品質 | 重い |
| Oil painting | Diffusion | WikiArt | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 絵画風変換 | 高品質 | 重い |
| Pixel | OpenCV（downsample） | 不要 | 不要 | cv2 | 解像度低下で実現 | 完全制御 | 超軽量 |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象範囲
   ├─ 全体 → full frame
   └─ 部分 → segmentation（SAM）

   ↓
② スタイル変換
   ├─ Diffusion（基本）
   ├─ LoRA（特定スタイル）
   └─ OpenCV（Pixelのみ）

   ↓
③ temporal補正
   → RAFT
   → consistency処理（重要）

   ↓
④ 出力
```

---

## Subclass別まとめ

### Watercolor / Oil painting / Ukiyo-e / Cyberpunk

```
手段：Stable Diffusion
```

- 汎用スタイル変換
- プロンプトで制御可能

---

### Ghibli / Anime

```
手段：LoRA / 特化モデル
```

- 特定スタイルは専用モデルが強い
- 再現性が高い

---

### American comic

```
手段：ControlNet + Diffusion
```

- 線画制御が重要
- 輪郭維持

---

### Pixel

```
手段：OpenCV
```

- downsampleで完全再現

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | ○ | ◎ | ◎ | ◎ |
| Diffusion | ◎ | △ | △ | △ |
| LoRA | ◎ | ○ | ○ | △ |
| ControlNet | ◎ | ○ | ◎ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| Stable Diffusion | 10〜16GB |
| LoRA | 10〜16GB |
| ControlNet | 10〜16GB |

---

## 設計思想

- Style = diffusionが基本
- 特定スタイルはLoRA
- Pixelのみ例外

---

## 判断フロー

```
① Pixelか？
   → YES → OpenCV

② 特定スタイルか？（Anime/Ghibli）
   → YES → LoRA

③ 一般スタイルか？
   → YES → Diffusion

④ 制御必要か？
   → YES → ControlNet
```

---

## 最終結論

```
基本 → Stable Diffusion
特化 → LoRA
制御 → ControlNet
例外 → PixelはOpenCV
```
