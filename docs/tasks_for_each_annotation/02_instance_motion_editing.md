# Instance Motion Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|-----------|---------------|--------|------|-------------|----------------|
| Human motion | OpenCV（座標移動） | 不要 | 不要 | cv2 | 単純移動は可能 | 不自然になりやすい | 超軽量 |
| Human motion | Pose制御 | COCO Keypoints | OpenPose | https://github.com/CMU-Perceptual-Computing-Lab/openpose | 人の動きは骨格で表現 | 安定・制御しやすい | 中（6〜8GB） |
| Human motion | Diffusion + Pose | DeepFashion / COCO | ControlNet (pose) | https://github.com/lllyasviel/ControlNet | 条件付き生成が可能 | 高品質・制御可能 | 重い（10〜16GB） |
| Human motion | Motion Transfer | VoxCeleb | First Order Motion | https://github.com/AliaksandrSiarohin/first-order-model | 動きの転写 | 顔中心で強い | 中（8GB） |
| object motion | OpenCV（平行移動） | 不要 | 不要 | cv2 | bbox移動で可能 | 簡単だが破綻あり | 超軽量 |
| object motion | Tracking + Warp | GOT-10k | OSTrack | https://github.com/botaoye/OSTrack | 位置追跡が必要 | 安定した追従 | 中（8GB） |
| object motion | Optical Flow | FlyingChairs | RAFT | https://github.com/princeton-vl/RAFT | フレーム間変化取得 | 滑らかな動き | 中（8GB） |
| object motion | Video Diffusion | WebVid | VideoCrafter | https://github.com/AILab-CVC/VideoCrafter | 動きを生成 | 高品質・制御弱い | 重い（16GB〜） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象検出
   ├─ Human → pose推定
   └─ Object → bbox / mask取得

   ↓
② 動き生成
   ├─ Human
   │     → Pose制御（OpenPose / ControlNet）
   │
   └─ Object
         → Tracking（OSTrack）＋位置変換
         → または Optical Flow（RAFT）

   ↓
③ フレーム間補正（必須）
   → RAFTで補間・滑らか化
   → 必要なら XMemでmask追跡

   ↓
④ 出力動画
```

---

## Subclass別まとめ

### Human motion

```
手段：Pose制御（OpenPose / ControlNet）
```

- 人の動き = 骨格（keypoints）

#### 手段①（推奨）
OpenPose  
- 安定・軽量

#### 手段②（高品質）
ControlNet（pose）  
- 見た目が自然

#### 手段③（特殊）
First Order Motion  
- 動き転写

#### 結論
安定 → OpenPose  
高品質 → ControlNet  

---

### object motion

```
手段：Tracking + 位置変換
```

- 物体 = 座標移動

#### 手段①
OpenCV移動  
- 簡単だが不自然

#### 手段②（推奨）
OSTrack  
- 安定追跡

#### 手段③
RAFT  
- 滑らか補正

#### 手段④
Video Diffusion  
- 制御弱い

#### 結論
Tracking + RAFT  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | 低 | ◎ | ◎ | ◎ |
| Pose系 | 中 | ◎ | ◎ | ○ |
| Tracking | 中 | ◎ | ○ | ○ |
| Optical Flow | 中 | ○ | △ | ○ |
| Diffusion | 高 | △ | △ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| OpenPose | 6GB |
| ControlNet | 10〜16GB |
| OSTrack | 8GB |
| RAFT | 8GB |
| VideoCrafter | 16GB〜 |

---

## 設計思想

- Human → pose  
- Object → tracking  
- temporal必須  

---

## 判断フロー

```
① 人か？ → Pose
② 物体か？ → Tracking
③ 揺れる？ → RAFT
④ 見た目重視？ → diffusion
```

---

## 最終結論

```
Human → OpenPose / ControlNet
Object → OSTrack + 座標変換
全体 → RAFT
```
