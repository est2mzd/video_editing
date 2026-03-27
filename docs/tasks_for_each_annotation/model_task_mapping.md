# タスク別 × モデル対応表（具体命令ベース）

## 前提
instruction を「実際に出る指示」に分解（zoom / wave / change background など）

---

## 対応表

| タスク（instruction例） | Wan I2V (14B) | Wan T2V (1.3B) | VACE | GroundingDINO + SAM | RAFT | OpenCV |
|------------------------|---------------|----------------|------|---------------------|------|--------|
| zoom in（ズーム） | ◎ | ◎ | △ | × | × | ◎ |
| pan（カメラ移動） | ◎ | ◎ | × | × | × | ◎ |
| wave hands（手を振る） | ◎ | ◎ | × | × | △ | × |
| walk forward（前進） | ◎ | ◎ | × | × | △ | × |
| change background（背景変更） | ○ | ○ | ◎ | ◎ | × | ◎ |
| blur background（背景ぼかし） | △ | △ | ◎ | ◎ | × | ◎ |
| replace object（物体置換） | ○ | ○ | ◎ | ◎ | × | ○ |
| remove object（物体削除） | ○ | ○ | ◎ | ◎ | × | ◎ |
| recolor object（色変更） | △ | △ | ◎ | ◎ | × | ◎ |
| add object（物体追加） | ◎ | ◎ | △ | × | × | △ |
| face close-up（顔ズーム） | ◎ | ◎ | △ | ◎ | × | ◎ |
| dolly in on face | ◎ | ◎ | × | ◎ | × | ◎ |
| keep subject, change env | ○ | ○ | ◎ | ◎ | × | ◎ |
| static edit（静止編集） | △ | △ | ◎ | ◎ | × | ◎ |
| motion generation（新規動作生成） | ◎ | ◎ | × | × | △ | × |

---

## 記号の定義

| 記号 | 意味 |
|------|------|
| ◎ | 直接対応（モデル単体で成立） |
| ○ | 条件付きで対応（品質 or 制御弱い） |
| △ | 補助的に可能（本来用途ではない） |
| × | 不可 |

---

## 根拠

### Wan I2V / T2V
- `wan_i2v.generate(...)` により動画フレーム列を生成
- promptのみで motion を生成

対応：
- zoom / pan / wave / walk → ◎
- 背景変更 → ○（制御弱い）

---

### VACE
- 入力：既存動画 + instruction
- 出力：編集後動画
- フレーム編集ベース

対応：
- 背景変更 / 色変更 / 物体削除 → ◎
- motion生成 → ×

---

### GroundingDINO + SAM
- DINO：text → bbox
- SAM：bbox → mask

対応：
- 対象領域抽出 → ◎
- 編集 → ×

---

### RAFT
- optical flow 推定モデル

対応：
- motion解析 → △
- motion生成 → ×

---

### OpenCV
- affine / inpaint / blur / color操作

対応：
- zoom / pan → ◎
- 背景変更 → ◎（mask前提）
- motion生成 → ×

---

## 最重要ポイント

### VACEの限界
```
wave / walk はできない
```

理由：新規フレーム生成なし

---

### Wanの役割
```
motion専用
```

---

### OpenCVの強み
```
zoom / pan は最強（軽い・確実）
```

---

## 実務向けまとめ

| タスク | ベスト手段 |
|--------|------------|
| zoom / pan | OpenCV |
| wave / walk | Wan |
| background change | VACE + SAM |
| object edit | VACE |
| face操作 | DINO + SAM + VACE |

---

## 最終結論
```
VACE単体ではコンペは勝てない（motionが抜ける）
```

必要構成：
```
LLM
↓
DINO + SAM（mask）
↓
VACE（編集）
↓
OpenCV（幾何）
↓
Wan（必要ならmotion）
```
