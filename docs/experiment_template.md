# Experiment Report Template
<!-- 背景意図: 実験ごとにコピーして使う標準フォーマット -->

---

## 1. Experiment ID
<!-- 背景意図: run_experiment.pyで生成されたIDと一致させる -->
- ID: XXXX

---

## 2. Objective（目的）
<!-- 背景意図: この実験で何を検証したいかを明確化 -->
- What:
- Why:

---

## 3. Hypothesis（仮説）
<!-- 背景意図: 結果の解釈を可能にするため事前に仮説を書く -->
- If ..., then ...

---

## 4. Setup（設定）
<!-- 背景意図: 再現性確保のため全条件を記録 -->

### 4.1 Input
<!-- 背景意図: データ依存を明確化 -->
- Video:
- Frames:

### 4.2 Model
<!-- 背景意図: モデル依存を記録 -->
- Model:
- Checkpoint:

### 4.3 Config
<!-- 背景意図: base.yamlとの差分を書く -->
- long_side:
- steps:
- seed:

### 4.4 Prompt
<!-- 背景意図: VLM系では最重要パラメータ -->