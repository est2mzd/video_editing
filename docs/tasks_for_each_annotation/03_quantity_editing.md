# Quantity Editing（詳細版）

---

## 一覧表

| Subclass | 手段 | 学習データ | 学習済みモデル | コード | 根拠 | 性能 / 特徴 | 重さ（GPU/計算） |
|----------|------|-----------|---------------|--------|------|-------------|----------------|
| Increase | OpenCV（コピー＆ペースト） | 不要 | 不要 | cv2 | 単純複製で増加可能 | 高速・制御容易だが不自然 | 超軽量 |
| Increase | Segmentation + Copy-Paste | COCO | SAM | https://github.com/facebookresearch/segment-anything | マスクで対象抽出 | 制御しやすい・破綻少 | 中（6〜8GB） |
| Increase | Instance Insertion（生成） | COCO / LVIS | PISCO | https://github.com/amazon-science/pisco | 挿入特化 | 位置・スケール制御良 | 重い（12〜16GB） |
| Increase | Diffusion生成 | LAION | Stable Diffusion | https://github.com/CompVis/stable-diffusion | 新規生成で増加 | 高品質だが制御弱い | 重い（10〜16GB） |
| Increase | Controlled Diffusion | LAION | ControlNet | https://github.com/lllyasviel/ControlNet | 条件付き生成 | 構造維持・安定向上 | 重い（10〜16GB） |
| Increase | Video Insertion | YouTube-VOS | E2FGVI | https://github.com/MCG-NKU/E2FGVI | 動画一貫性 | 時系列安定 | 中（8〜10GB） |
| Increase | Tracking補助 | GOT-10k | OSTrack | https://github.com/botaoye/OSTrack | 配置後追跡 | 動きの一貫性確保 | 中（8GB） |
| Increase | Optical Flow補正 | FlyingChairs | RAFT | https://github.com/princeton-vl/RAFT | フレーム間補間 | フリッカ低減 | 中（8GB） |

---

## 実戦構成

```
[入力動画]
   ↓
① 対象抽出
   → GroundingDINO + SAM（mask）

   ↓
② 増加（Insertion）
   ├─ 簡易：Copy-Paste（複製）
   ├─ 高品質：PISCO
   └─ 生成：Diffusion / ControlNet

   ↓
③ 配置最適化
   → スケール・位置・オクルージョン調整

   ↓
④ 時系列安定化
   → OSTrack（追跡）
   → RAFT（補間）

   ↓
⑤ 出力
```

---

## Subclass別まとめ

### Increase

```
手段：Copy-Paste / PISCO / Diffusion
```

#### ✔ 本質
```
同一または新規インスタンスを自然に増やす
```

#### 手段①（最速・安定）
OpenCV / Copy-Paste  
- 同一物体を複製  
- 制御容易・破綻少（ただしリアリズムは低め）

#### 手段②（推奨）
SAM + Copy-Paste  
- マスクで切り出し → 複製  
- 背景とのブレンドで品質向上

#### 手段③（高品質・制御）
PISCO  
- 挿入専用  
- 位置・スケール・オクルージョンに強い

#### 手段④（生成）
Stable Diffusion / ControlNet  
- 新規物体を生成  
- 見た目は良いが位置制御が難しい

#### 手段⑤（動画安定）
OSTrack + RAFT  
- 挿入後の動き・フリッカを抑制

#### 結論
安定 → Copy-Paste  
高品質・制御 → PISCO  
見た目生成 → Diffusion  

---

## モデル比較

| 手段 | 精度 | 安定性 | 制御性 | 実装難易度 |
|------|------|--------|--------|------------|
| OpenCV | 低 | ◎ | ◎ | ◎ |
| Copy-Paste（+SAM） | 中 | ◎ | ◎ | ○ |
| PISCO | ◎ | ○ | ◎ | △ |
| Diffusion | ◎ | △ | △ | △ |
| ControlNet | ◎ | ○ | ○ | △ |

---

## GPU負荷

| モデル | VRAM |
|--------|------|
| OpenCV | 0GB |
| SAM | 6〜8GB |
| PISCO | 12〜16GB |
| Stable Diffusion | 10〜16GB |
| ControlNet | 10〜16GB |
| OSTrack | 8GB |
| RAFT | 8GB |
| E2FGVI | 8〜10GB |

---

## 設計思想

- 「増やす」は **挿入問題（Instance Insertion）に帰着**
- まず **mask精度（SAM）** を担保
- 生成は補助、**配置制御が本質**
- 動画では **tracking / flow** が必須

---

## 判断フロー

```
① 同一物体を増やすか？
   → YES → Copy-Paste（+SAM）

② 位置・スケールを厳密に制御したいか？
   → YES → PISCO

③ 新規見た目を生成したいか？
   → YES → Diffusion / ControlNet

④ 動画で不安定か？
   → YES → OSTrack + RAFT
```

---

## 最終結論

```
基本 → SAM + Copy-Paste
高品質 → PISCO
生成 → Diffusion / ControlNet
動画 → OSTrack + RAFT
```
