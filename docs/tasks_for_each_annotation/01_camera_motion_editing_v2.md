# Camera Motion Editing（詳細版）

## 一覧表

| Subclass | 手段 | 入出力 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|--------|-----------|---------------|--------|------|-------------|----------------|
| Zoom in | OpenCV（crop+resize） | 入力:画像/動画 → 出力:拡大画像/動画 | 不要 | 不要 | cv2 | 幾何変換で十分 | 高速・完全安定 | 超軽量（CPU可） |
| Zoom in | Video Diffusion | 入力:動画+テキスト → 出力:編集動画 | WebVid-10M | VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | テキスト駆動編集対応 | 高品質だが制御弱い | 重い（16GB以上推奨） |
| Zoom out | OpenCV（縮小＋padding） | 入力:画像/動画 → 出力:縮小画像/動画 | 不要 | 不要 | cv2 | 逆zoomは単純変換 | 安定だが背景が不自然 | 超軽量 |
| Zoom out | Inpainting | 入力:画像+mask → 出力:補完画像 | Places2 | LaMa | https://github.com/advimman/lama | 背景補完が必要 | 背景自然化可能 | 軽〜中（8GB程度） |
| Dolly in | Depth + Warp | 入力:画像 → depth → 出力:視点変化画像 | KITTI / NYU | MiDaS | https://github.com/isl-org/MiDaS | 視点移動はdepth必要 | 自然だが歪みあり | 中（6〜8GB） |
| Dolly in | Video Diffusion | 入力:動画+テキスト → 出力:動画 | WebVid | Wan2.1 | https://github.com/Wan-Video/Wan2.1 | 生成ベース視点変化 | 高品質・不安定あり | 重い（16GB〜） |
| Arc shot | NeRF系 | 入力:多視点画像 → 出力:新視点レンダリング | LLFF | NeRF | https://github.com/bmild/nerf | 真の視点変更 | 最も正確 | 超重い（学習必要） |
| Arc shot | Depth + 2D warp | 入力:画像 → depth → 出力:回転画像 | KITTI | Depth-Anything | https://github.com/LiheYoung/Depth-Anything | 擬似3D回転 | 近似的・軽量 | 中（8GB） |
| Zoom系全般 | Optical Flow補正 | 入力:連続フレーム → 出力:補正動画 | FlyingChairs | RAFT | https://github.com/princeton-vl/RAFT | temporal安定化必須 | フリッカ防止 | 中（8GB） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象 or 全体の領域決定
   ├─ Zoom → 全体
   ├─ Dolly → 中心対象（人物など）
   └─ Arc → 対象＋背景（広め）

   ↓
② 手段選択（Subclassごと）
   ├─ Zoom in / out
   │     → OpenCV（crop / resize / padding）
   │
   ├─ Dolly in
   │     → MiDaS（depth）＋ warping
   │     → 高品質が必要なら Wan / VideoCrafter
   │
   └─ Arc shot
         → Depth-Anything + warping（現実解）
         → 余裕があれば NeRF（研究用途）

   ↓
③ フレーム間補正（必須）
   → RAFT（optical flow）でブレ補正
   → 必要なら XMem でマスク追跡

   ↓
④ 出力動画（fps / frame数維持）
```

---

## Subclass別まとめ

### Zoom in / Zoom out
```
手段：OpenCVのみ
```

- crop / resize / paddingで完結
- 学習モデル不要
- 最も安全

---

### Dolly in
```
手段：MiDaS（depth）＋ warp
```

- 疑似3Dで視点移動
- 軽くて現実的

代替：
```
Wan / VideoCrafter
```

- 見た目は良いが不安定

---

### Arc shot
```
手段：Depth + warp（現実解）
```

理想：
```
NeRF
```

---

## 設計思想

### ① 幾何で済むものは学習を使わない
Zoomは完全に2D変換

### ② 視点変化はdepthが必要
Dolly / Arcは3D問題

### ③ diffusionは万能ではない
フリッカ・制御不能

### ④ temporal補正は必須
RAFTで安定化

---

## 判断フロー

```
① これは「画面の拡大縮小」か？
   → YES → OpenCV

② 視点が変わるか？
   → YES → depth系

③ 見た目重視か？
   → YES → diffusion追加

④ 動画が揺れるか？
   → YES → RAFT追加
```

---

## 最終結論

```
Zoom → OpenCV（確定）

Dolly → depth（MiDaS）＋ warp

Arc → depthベース（妥協） or NeRF（研究）

全体 → RAFTで時間安定化
```
