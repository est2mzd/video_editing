# refactoring_04_build_code

## 目的
- `/workspace/src/test/gt_unit_test.py` を作成し、以下4つの notebook 相当ケースを実行できるようにした。
	- `add_object_05.ipynb` の最終 add_object
	- `apply_style_03.ipynb` の最終 apply_style
	- `dolly_in_02.ipynb` の dolly_in
	- `zoom_in_01.ipynb` の zoom_in
- 入力は `mp4_path, action, target`。
- 出力 mp4 は `/workspace/logs/test/` に保存。

## 実装方針
- 既存コードを優先利用。
-	- 動画入出力: `src/utils/video_utility.py`
-	- アクション実行: `src/postprocess/dispatcher.py` の `run_method`
- `data/annotations.jsonl` を一次ソースとして読み込む。
- `action/target/params` は予測せず、以下の順でそのまま使用する。
	1. `annotations.jsonl` の各行に `action/target/params` がある場合は最優先で採用。
	2. 4つの notebook 相当ケースは固定の manual plan を採用。
	3. 単体CLI指定 (`--mp4-path --action --target`) は最終フォールバック。
- 将来の予測モード差し替えを想定し、`resolve_action_plan(...)` で解決点を一箇所に集約。
- 失敗時は再試行。全失敗時はフォールバックとして元動画を書き出し、処理の完了を保証。

## 追加した主な機能 (`src/test/gt_unit_test.py`)
- `Scenario` dataclass
	- 4ケースの固定シナリオを保持。
- JSONL読み込み
	- `load_annotations_jsonl(...)`: `annotations.jsonl` を1行1JSONで読み込む。
	- `build_annotation_index(...)`: `video_path` の basename をキーに索引化。
- アクションプラン解決
	- `build_manual_plan_map(...)`: 予測なしで使う固定プラン。
	- `resolve_action_plan(...)`: direct fields / manual plan / CLI fallback の順で `action/target/params` を確定。
- 単体実行
	- `run_single_case(...)`: `load_video -> run_method -> write_video`。
	- `--retry` 回再試行。
	- 全失敗時は `__fallback.mp4` を出力。
- 一括実行
	- `run_default_scenarios(...)`: 4ケースを順に実行。
- CLI
	- `--run-default-scenarios`
	- `--mp4-path --action --target`
	- `--output-dir`
	- `--max-frames` (検証用の短尺実行)
	- `--retry`
	- `--no-fallback-output`

## 実行コマンド

### 4ケース一括
```bash
python /workspace/src/test/gt_unit_test.py --run-default-scenarios --output-dir /workspace/logs/test
```

### 4ケース一括 (短尺検証)
```bash
python /workspace/src/test/gt_unit_test.py --run-default-scenarios --max-frames 12 --output-dir /workspace/logs/test
```

### 単体実行
```bash
python /workspace/src/test/gt_unit_test.py \
	--mp4-path /workspace/data/videos/wyzi9GNZFMU_0_0to121.mp4 \
	--action dolly_in \
	--target "man's face" \
	--output-dir /workspace/logs/test
```

## 出力先
- 通常出力: `/workspace/logs/test/<video_stem>__<action>.mp4`
- フォールバック出力: `/workspace/logs/test/<video_stem>__<action>__fallback.mp4`

## 備考
- 重いモデル依存のあるアクション（特に add_object 系）は環境依存で失敗する場合がある。
- その場合もフォールバック出力で処理フローを完了させ、失敗内容は標準出力に表示する。
- 将来予測モードを導入する場合は、`resolve_action_plan(...)` のみ差し替える想定。

---

## まとめ

### 実施した変更の全体像

本リファクタリングは2フェーズで構成される。

**フェーズ1: テストハーネス構築**

- `src/test/gt_unit_test.py` を新規作成。
- 入力ソースを `annotations_gt_task_ver10.json` から `annotations.jsonl` に変更。
- アクション・ターゲット・パラメータは予測せず、以下優先順でそのまま採用するノーコード予測設計とした。
  1. `annotations.jsonl` 各行に `action/target/params` が直接ある場合
  2. notebook 相当4ケースの固定 manual plan
  3. CLI 指定による単体フォールバック

**フェーズ2: postprocess 挙動の notebook 互換化**

| ファイル | 変更内容 |
|---|---|
| `src/postprocess/dispatcher.py` | `_resolve_action` を追加し、`action` / `params.action` / `params._action` / `params.method` の優先順で解決。notebook 形式の `_action`・`_constraints` も受理。`targets` キーも認識。 |
| `src/postprocess/detectors.py` | `split_target_keywords` に typo 吸収を追加（`mas` → `man`）。`build_detection_prompts` を新設し、target / instruction から複数の検出プロンプト候補を生成。 |
| `src/postprocess/camera_ops.py` | `stable_zoom_in` の検出ロジックを複数プロンプト順次試行 → union box フォールバックに改良。possessive 表記や表記揺れによる検出失敗を低減。 |

### 確認済み動作

| ケース | 入力ターゲット | 結果 |
|---|---|---|
| add_object_05 (buffalo) | `buffalo` | OK |
| apply_style_03 (oil_painting) | `full_frame` | OK |
| dolly_in_02 | `man's face` | OK |
| dolly_in (typo) | `mas's face` | OK (aliasで `man` に正規化) |
| zoom_in_01 | `camera_view` | OK |

### 設計上の拡張ポイント

- **予測モード導入**: `resolve_action_plan(...)` の内部に Predictor クラスを注入するだけで切り替え可能。
- **新規アクション対応**: `dispatcher.py` の `method_to_action` テーブルに追記するだけで notebook 形式 `method` からのルーティングが自動拡張。
- **typo 辞書拡充**: `detectors.py` の `token_aliases` に追記するだけで表記ゆれ吸収が拡張可能。
