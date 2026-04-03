# Parser Cheat Analysis (Function-level)

## 目的
- 既存 parser 系コードにおけるチート情報混入箇所を関数単位で特定
- no-cheat 方針（instruction-only）を満たすための修正方針を明確化

## チート判定基準
- 予測時に以下を参照したらチート:
  - GT の tasks/action/target/params
  - video_path に紐づく GT 値
  - class/subclass ラベル（instruction 以外のメタ情報）

## 主要ファイル分析

### 1. src/parse/instruction_parser_ver19.py
- `build_noun_bank(base_records)`
  - `(video_path, action) -> GT由来 noun list` を構築
  - 問題: video_path と GT task に依存した辞書を作るため、未知データ予測でリーク
- `choose_gt_task_by_action(video_path, action, video_to_tasks)`
  - 予測時に GT tasks を直接参照
  - 問題: target/params のリーク
- `predict_single(...)`
  - `action_source == gt_match` で GT action を優先
  - `target_source == gt_target_exact` で GT target を直接利用
  - `params_source == gt_params` で GT params をコピー
  - 問題: action/target/params が GT 依存
- `predict_multi(...)`
  - `multi_source == gt_tasks_exact` で GT tasks をそのまま返す
  - 問題: 完全リーク
- `build_predictions(records, gt_path, ...)`
  - GT をロードし `video_to_tasks` を生成、予測時に利用
  - 問題: 本質的に「検索型GT参照」

結論: ver19 は設計上 no-cheat 不適合。

### 2. src/parse/data_loading.py
- `load_base_records(raw_path, gt_path)`
  - `gt_tasks` と `gt_primary` を record に埋め込む
  - 評価用途としては妥当だが、予測関数へ渡すとリーク経路になる
- `load_grouped_unknown_records(...)`
  - grouped instruction に `gt_tasks/gt_primary` を同梱
  - 問題: 予測器にそのまま渡すとチート可能

結論: データローダは評価用と推論用を厳密分離すべき。

### 3. src/parse/models.py
- `baseline_action(record)`
  - `record['class']`, `record['subclass']` を使用
  - 問題: instruction-only 違反
- `infer_action(record)`
  - class/subclass を補助信号として使用
  - 問題: instruction-only 違反
- `_predict_improved(record, records, ...)`
  - `best_examples(...)` が `records`（GTラベル付き）から近傍検索
  - `template_target` に GT target を使用
  - 問題: retrieval経由でGTリーク
- `predict_v11d_ensemble(...)`
  - seed prediction と GT近傍合議を使用
  - 問題: instruction-only 違反

結論: 現行 improved/retrieval/ensemble は no-cheat 不適合。

## 排除方針
1. 予測器入力を instruction のみへ固定
2. GT は評価時のみ参照
3. class/subclass/video_path は予測ロジックから切り離す
4. params は空 dict（または instruction から抽出可能な最小値のみ）

## 実装（ver01）
- 追加: `src/parse/parser_no_cheat_rulebase_ver01.py`
  - `NoCheatRuleParserV01.pred(instruction)`
  - action/target を instruction 正規表現のみで推定
- 追加: `src/parse/validate_no_cheat_rulebase_ver01.py`
  - GT と grouped の両評価を実施
  - 目標:
    - GT: action > 80%, target > 80%
    - grouped: action >= 70%, target >= 70%
- 追加: `scripts/run_validate_no_cheat_rulebase_ver01.sh`
  - ログ付き実行

## 次の改善ポイント（チートなしの範囲）
- action regex の競合解消（優先順位 + スコア改善）
- target 抽出を構文ベースへ拡張
- `preserve_*` を主タスクに誤採用しないガード追加
