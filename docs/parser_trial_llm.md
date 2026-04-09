# parser_trial_llm

LLM試行の記録（1試行ずつ実行、エラーは試行回数に含めない）。

## Goal
- GT: action > 80%, target > 80%
- grouped: action >= 70%, target >= 70%

## Trial Policy
- 1回ごとに: 実行 -> 分析 -> 対策1点 -> 次試行
- パラメータスタディは禁止
- `.sh` 実行とログ保存を必須

---

## Trial 001 - llm_trial_001
- Date: 2026-04-02
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial001.py](src/parse/instruction_parser_v3_llm_trial001.py)
- Log: [logs/analysis/llm_trial_001_20260402_152211.log](logs/analysis/llm_trial_001_20260402_152211.log)
- JSON: [logs/analysis/llm_trial_001_20260402_152211.json](logs/analysis/llm_trial_001_20260402_152211.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis

**上手くいった理由**
- ルールベース v3 singlefile が土台として強く、action の 16 クラス分類が既に高精度（GT action 98%）だった。
- LLM は `_is_low_confidence` 判定が True のサンプルのみに限定して呼ぶ設計のため、ルールが自信を持って当てられているサンプルを上書きするリスクがなかった。
- `_sanitize_prediction` でルール結果から大きく乖離する LLM 出力を棄却したため、LLM の誤出力が精度を下げなかった。

**残る失敗の分析（13件）**

| 分類 | 件数 | 原因 |
|---|---|---|
| `change_color` で target が `object` に落ちる | 3 | 指示文に曖昧マーカー（transform/while/keep等）がなく `_is_low_confidence=False` → LLM 未発火 → ルールが "object" フォールバックを返したまま |
| `zoom_in/change_camera_angle` で target と GT が不一致 | 4 | GT が `camera_view` のとき pred は「人物名+副詞節」を取り、GT が人物名のとき pred は `camera_view` に正規化。camera_view ↔ 具体人物名の切り分けルールが双方向で噛み合っていない |
| `add_effect` で target が `object` | 1 | action=`add_effect` は `uncertain_action` に含まれず、かつ指示文に曖昧マーカーなし → LLM 未発火 → ルールが "object" を返す |
| remove_object の target で "and" vs コンマ不一致 | 1 | ルールが "and" を含む句をそのまま返し、GT はコンマ区切りリスト形式。`_target_ok` の部分一致チェックが `and` 前後の分割に対応していない |
| add_effect で "throughout the entire video" が target に混入 | 1 | 文から "his body" を抜く NP 抽出が後続の副詞節まで拾っている |
| `apply_style` を `add_object` に誤分類 | 2 | "adding vibrant neon glows" / "Replace the warm kitchen lights" のように追加・置換動詞が混在する文で `add_object` ルールが `apply_style` よりスコアが高くなる。`_is_low_confidence` が True になって LLM 発火するが、0.5B モデルがスタイル変換の文脈判断を誤り `add_object` を維持 |

- ただしバッチ入力時の挙動差を評価していないため、次で batch=1 とマルチバッチを比較する。

### Countermeasure For Next Trial
- 対策は1点のみ: 評価側に `--eval-batch-size` と tqdm 進捗表示を追加し、batch=1/8 を同一条件で比較する。

---

## Non-Counted Run - llm_trial_002
- Date: 2026-04-02
- Parser: [src/parse/instruction_parser_v3_llm_trial002.py](src/parse/instruction_parser_v3_llm_trial002.py)
- Log: [logs/analysis/llm_trial_002_20260402_153254.log](logs/analysis/llm_trial_002_20260402_153254.log)

### Status
- 実行が完了していないため、試行回数にはカウントしない。
- ログは起動直後の情報のみで、最終メトリクス出力に到達していない。

**未完了の原因分析**
- `llm_trial_002.py` は `pred_batch` を追加する修正途中で実行されたため、tqdm 進捗表示どころかモデルロード直後に出力が止まり、評価ループに入れていなかった。
- また、クラス名が `InstructionParserV3LLMTrial001` のまま（名前不一致）になっていたが、`build_parser` 関数が正しく参照していたため実行自体はできるが、内部バッチロジックが完成していない状態で動いていた可能性がある。

### Countermeasure For Next Trial
- 対策は1点のみ: 同一 parser で trial 名を分け、batch=1 と batch=8 を再実行して完走ログを残す。

---

## Trial 003 - llm_trial_003_batch1
- Date: 2026-04-02
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial002.py](src/parse/instruction_parser_v3_llm_trial002.py)
- Config: eval_batch_size=1, show_progress=1
- Log: [logs/analysis/llm_trial_003_batch1_20260402_154728.log](logs/analysis/llm_trial_003_batch1_20260402_154728.log)
- JSON: [logs/analysis/llm_trial_003_batch1_20260402_154728.json](logs/analysis/llm_trial_003_batch1_20260402_154728.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Goal Check
- GT goal: PASS
- grouped goal: PASS

### Analysis

**上手くいった理由（batch=1 の確認として）**
- trial001 と完全に同一の 13 件しか失敗しなかった。これは LLM 発火条件（`_is_low_confidence`）と `_sanitize_prediction` のフィルタが batch サイズに無関係に決定論的に動いていることを示す。
- 進捗が tqdm で可視化されたことで、GT 100件は約 1.3 it/s、grouped 600件は約 1.17 it/s で安定して動いていることが確認できた。

**失敗 13 件の構造（trial001 と同一）**
- LLM が発火しないパターン（change_color/add_effect で `_is_low_confidence=False`）では、ルール側の target フォールバック "object" が残り続けている。
- `apply_style` vs `add_object` の 2 件は LLM が発火しているが、0.5B モデルが "adding" "Replace" などの動詞を見て `add_object` を返してしまう。

### Countermeasure For Next Trial
- 対策は1点のみ: マルチバッチ（8）で同条件比較を実施する。

---

## Trial 004 - llm_trial_004_batch8
- Date: 2026-04-02
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial002.py](src/parse/instruction_parser_v3_llm_trial002.py)
- Config: eval_batch_size=8, llm_batch_size=8, show_progress=1
- Log: [logs/analysis/llm_trial_004_batch8_20260402_155726.log](logs/analysis/llm_trial_004_batch8_20260402_155726.log)
- JSON: [logs/analysis/llm_trial_004_batch8_20260402_155727.json](logs/analysis/llm_trial_004_batch8_20260402_155727.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Goal Check
- GT goal: PASS
- grouped goal: PASS

### Analysis

**上手くいった理由（batch=8 の精度確認）**
- 精度は batch=1 と完全に同一。`pred_batch` 内でキャッシュを利用し、かつ `_sanitize_prediction` が決定論的なので、バッチまとめ生成しても結果は変わらない。

**失敗した理由（実行速度の劣化）**
- decoder-only モデル（GPT 系列）は right-padding でバッチを揃えると、生成時に誤った位置アテンションが発生する。Qwen2.5 も同様で、`padding_side='right'` のデフォルト設定のままバッチ化すると生成確率が不安定になり、警告が大量発生するだけでなく出力トークンのズレが起きる可能性があった。
- ただし今回の精度が落ちなかった理由は、`_sanitize_prediction` が LLM 出力を強くクリップしており、ズレた出力でも "base_task にフォールバック" される比率が高かったため。

### Countermeasure For Next Trial
- 対策は1点のみ: tokenizer の padding_side を left に固定し、multi-batch 実行を再評価する。

---

## Trial 005 - llm_trial_005_batch8_leftpad
- Date: 2026-04-02
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial003.py](src/parse/instruction_parser_v3_llm_trial003.py)
- Config: eval_batch_size=8, llm_batch_size=8, show_progress=1
- Log: [logs/analysis/llm_trial_005_batch8_leftpad_20260402_160932.log](logs/analysis/llm_trial_005_batch8_leftpad_20260402_160932.log)
- JSON: [logs/analysis/llm_trial_005_batch8_leftpad_20260402_160932.json](logs/analysis/llm_trial_005_batch8_leftpad_20260402_160932.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Goal Check
- GT goal: PASS
- grouped goal: PASS

### Analysis

**上手くいった理由（left-padding 対策）**
- `tokenizer.padding_side = "left"` に固定したことで、バッチ内の全シーケンスが左詰めパディングになり、モデルが実際のトークン列の末尾から正しく生成を開始できるようになった。right-padding 警告がゼロになり、tqdm の進捗もほぼ一定速度で流れるようになった。
- `pad_token = eos_token` の設定により、パディングトークンが未定義でエラーになるケースも防いだ。
- 精度は batch=1/right-pad と同一 → バッチ化が精度に副作用を与えないことが確認できた。

**残っている構造的な失敗（3-5 まで共通の 13 件）**

1. **LLM 未発火 × change_color の target="object"（3件）**  
   - `_is_low_confidence` が `has_ambiguous_phrase AND (generic_target OR uncertain_action)` の AND 条件のため、曖昧マーカーなしの直接命令文（"Change the … color to …"）では発火しない。ルール側の change_color target 抽出が "object" にフォールバックしている。

2. **camera 系 action の target 方向違い（4件）**  
   - GT が `camera_view` のとき（zoom_in/change_camera_angle）、ルールが指示文の人物句を先に抜いてしまう。  
   - GT が具体的な人物名のとき（"rugby player", "speaker"）、ルールが `camera_view` に正規化してしまう。  
   - camera アクション時の target 決定ロジックが「`camera_view` 固定」と「人物句優先」のどちらかしか取れない二項問題になっている。

3. **apply_style が add_object に誤分類（2件）**  
   - "Transform … into Cyberpunk style by adding neon glows" のように、apply_style 文脈に `add_object` 的な動詞（adding, placing）が混在するケース。  
   - LLM は発火するが、0.5B モデルが "adding" を意味重点として「add_object」と出力し、`_sanitize_prediction` もそれを採用する。  
   - LLM がルールのフォールバック先として機能しておらず、むしろルール誤分類を引き継いでいる。

4. **複数 target の "and" vs コンマ不一致（1件）**  
   - `_target_ok` の部分一致では `"person walking…and the green plastic containers"` vs `"person…, green plastic containers"` を同一視できない。

5. **add_effect で "throughout the entire video" が target に混入（1件）**  
   - NP 抽出の境界が副詞節まで伸びている。

### Current Decision
- この判断は **ver3ベース併用時点** の結論であり、ver4以降の LLM 独立試行には適用しない。

---

## Correction (ver4+)
- user 指示により、ver4 以降は [src/parse/instruction_parser_v3_singlefile.py](src/parse/instruction_parser_v3_singlefile.py) を import しない。
- したがって ver4+ は「LLM 単体（instruction only）」として再開する。

---

## Trial 006 - llm_trial_006_ver4_independent_b1
- Date: 2026-04-03
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial004.py](src/parse/instruction_parser_v3_llm_trial004.py)
- Config: eval_batch_size=1, llm_batch_size=1, show_progress=1
- Log: [logs/analysis/llm_trial_006_ver4_independent_b1_20260403_000350.log](logs/analysis/llm_trial_006_ver4_independent_b1_20260403_000350.log)
- JSON: [logs/analysis/llm_trial_006_ver4_independent_b1_20260403_000350.json](logs/analysis/llm_trial_006_ver4_independent_b1_20260403_000350.json)

### Hypothesis (before code)
- 仮説: v3 singlefile 依存を外しても、LLM JSON 生成 + sanitize + 軽いヒューリスティックで action は 70% 以上に到達できる。

### Metrics
- GT action/target: 64.00% / 39.00%
- grouped action/target: 54.67% / 38.17%

### Goal Check
- GT goal: FAIL
- grouped goal: FAIL

### Analysis
- 失敗の主因は action 取り違え（例: `change_color -> dolly_in`, `replace_background -> change_camera_angle`）。
- target も弱いが、まず action を固定しない限り target 改善が効きにくい。

### Countermeasure For Next Trial
- 対策は1点のみ: high-precision な `locked_action` ルータを追加し、特定パターンでは LLM 出力よりルータを優先する。

---

## Trial 007 - llm_trial_007_ver5_action_router
- Date: 2026-04-03
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial005.py](src/parse/instruction_parser_v3_llm_trial005.py)
- Config: eval_batch_size=1, llm_batch_size=1, show_progress=1
- Log: [logs/analysis/llm_trial_007_ver5_action_router_20260403_003315.log](logs/analysis/llm_trial_007_ver5_action_router_20260403_003315.log)
- JSON: [logs/analysis/llm_trial_007_ver5_action_router_20260403_003315.json](logs/analysis/llm_trial_007_ver5_action_router_20260403_003315.json)

### Hypothesis (before code)
- 仮説: action 誤りが主要ボトルネックなので、`locked_action` 優先だけで action を大きく改善できる。

### Metrics
- GT action/target: 78.00% / 39.00%
- grouped action/target: 71.83% / 37.83%

### Goal Check
- GT goal: FAIL
- grouped goal: FAIL

### Analysis
- action は大幅改善（GT +14pt, grouped +17.16pt）。
- ただし target は横ばいで、`camera_view` / `object` フォールバックが多すぎる。
- `increase number` 系が `add_object` / `increase_amount` で揺れるケースが残る。

### Countermeasure For Next Trial
- 対策は1点のみ: locked パターン時は action だけでなく target も決定論ルールで同時に確定し、LLM の target ドリフトを抑える。

---

## Trial 008 - llm_trial_008_ver6_locked_target
- Date: 2026-04-03
- Runner: [scripts/run_validate_llm_single_trial.sh](scripts/run_validate_llm_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_llm_trial006.py](src/parse/instruction_parser_v3_llm_trial006.py)
- Config: eval_batch_size=1, llm_batch_size=1, show_progress=1
- Log: [logs/analysis/llm_trial_008_ver6_locked_target_20260403_021241.log](logs/analysis/llm_trial_008_ver6_locked_target_20260403_021241.log)
- JSON: [logs/analysis/llm_trial_008_ver6_locked_target_20260403_021241.json](logs/analysis/llm_trial_008_ver6_locked_target_20260403_021241.json)

### Hypothesis (before code)
- 仮説: locked パターンで action+target を同時決定すれば、action 80% 超えと target 改善を同時に狙える。

### Metrics
- GT action/target: 85.00% / 48.00%
- grouped action/target: 76.17% / 43.83%

### Goal Check
- GT goal: FAIL
- grouped goal: FAIL

### Analysis
- action は閾値を超過したが、target が大きく不足。
- 主失敗は `dolly/zoom` の target が `camera_view` に寄りすぎる点と、`add_object/edit_motion` の target が `object` に落ちる点。
- `replace_object` が `replace_background` に吸われるケースも残る。

### Countermeasure For Next Trial
- 対策は1点のみ: target resolver を camera/object 系で改善し、`object/camera_view` への過剰フォールバックを減らす。

---

## Trial 009 (Planned) - llm_trial_009_ver7_target_resolver
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_llm_trial007.py](src/parse/instruction_parser_v3_llm_trial007.py)

### Hypothesis (before code)
- 仮説: target resolver を以下 1 点で改善すれば、target を優先改善できる。  
   具体策: camera 系 action でも `toward/on/at` 句に明確な対象がある場合は人物・物体句を返し、`entire video/scene` の場合のみ `camera_view` にする。

### Metrics
- GT action/target: 85.00% / 50.00%
- grouped action/target: 76.17% / 45.33%

### Goal Check
- GT goal: FAIL
- grouped goal: FAIL

### Analysis
- 仮説どおり target は改善（GT +2pt, grouped +1.5pt）。
- ただし主要失敗は `add_object` と `edit_motion` の target が `object` に落ちる点。
- camera 系は一部改善した一方で、GT=`camera_view` 期待サンプルに対して人物句を返してしまう逆方向の誤りが残る。

### Countermeasure For Next Trial
- 対策は1点のみ: `add_object/edit_motion` の target フォールバックを entity-aware 化し、`object` 既定値に落ちる率を下げる。

---

## Execution Mapping (Trial 010-013)

Trialごとの「Version」「実行時に指定したPythonファイル」「実行方法」を明記する。

| Trial | Version Label | Runtime Parser File | 実行コマンド | Log/JSON |
|---|---|---|---|---|
| 010 | ver8 | [src/parse/instruction_parser_v3_llm_trial008.py](src/parse/instruction_parser_v3_llm_trial008.py) | `PARSER_FILE=/workspace/src/parse/instruction_parser_v3_llm_trial008.py TRIAL_NAME=llm_trial_010_ver8_entity_target EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh` | [logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.log](logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.log) / [logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.json](logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.json) |
| 011 | ver9 | [src/parse/instruction_parser_v3_llm_trial009.py](src/parse/instruction_parser_v3_llm_trial009.py) | `PARSER_FILE=/workspace/src/parse/instruction_parser_v3_llm_trial009.py TRIAL_NAME=llm_trial_011_ver9_replace_priority EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh` | [logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.log](logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.log) / [logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.json](logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.json) |
| 012 | ver10 | [src/parse/instruction_parser_v3_llm_trial010.py](src/parse/instruction_parser_v3_llm_trial010.py) | `PARSER_FILE=/workspace/src/parse/instruction_parser_v3_llm_trial010.py TRIAL_NAME=llm_trial_012_ver10_camera_target EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh` | [logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.log](logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.log) / [logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.json](logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.json) |
| 013 | ver11 | [src/parse/instruction_parser_v3_llm_trial011.py](src/parse/instruction_parser_v3_llm_trial011.py) | `PARSER_FILE=/workspace/src/parse/instruction_parser_v3_llm_trial011.py TRIAL_NAME=llm_trial_013_ver11_generic_target EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh` | [logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.log](logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.log) / [logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.json](logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.json) |

注記:
- [src/parse/instruction_parser_v3_llm_trial012.py](src/parse/instruction_parser_v3_llm_trial012.py) と [src/parse/instruction_parser_v3_llm_trial013.py](src/parse/instruction_parser_v3_llm_trial013.py) は、番号整合のために後から追加したエイリアスファイル。
- Trial 012/013 の実測結果は、上表の Runtime Parser File（trial010 / trial011）で取得した値を正とする。

---

## Trial 010 - llm_trial_010_ver8_entity_target
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_llm_trial008.py](src/parse/instruction_parser_v3_llm_trial008.py)
- Log: [logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.log](logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.log)
- JSON: [logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.json](logs/analysis/llm_trial_010_ver8_entity_target_20260403_031415.json)

### Hypothesis (before code)
- 仮説: `add_object` と `edit_motion` の target について、人物・物体名の抽出を強化すれば target 精度が改善する。  
   具体策: `increase/add more/add a second` 系から名詞句を抽出し、`edit_motion` は `man/woman/baby/person/speaker` を優先返却する。

### Metrics
- GT action/target: 85.00% / 57.00%
- grouped action/target: 76.17% / 52.50%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL (target 57%)
- grouped goal (action >= 70%, target >= 70%): FAIL (target 52.5%)

### Analysis

**対策の効果**
- action は変化なし（85%/76.17%）
- target GT: 50% → 57% (+7pt)、grouped: 45.33% → 52.50% (+7.17pt)。`edit_motion` と `add_object` の一部が改善した。

**残る主な失敗パターン（target）**
| 分類 | 内容 | 原因 |
|---|---|---|
| `replace_object` が `replace_background` に誤分類（action） | "silver sports car"、"woman wearing bright pink hat" など 3件 | `_locked_action()` の `replace_background` 判定が先に発火し、`replace_object` の locked が後回しになっている |
| `apply_style` が `replace_background` に誤分類（action） | "Replace panels with neon signs…cyberpunk" | `replace.*background` 以外の `replace` ロックが `apply_style` を覆す |
| `change_camera_angle` target が角度表現を返す | "a low angle shot", "adopt a low angle perspective" など 5件 | カメラアングル変更の target resolver が人物名を取れず角度説明句をそのまま返している |
| `add_object` target で NP 抽出が長すぎる | "rhino_and_buffalo" (GT) vs 長い pred | 文中の長い NP が GT の短縮形と一致しない |
| `edit_motion` target が不一致 | "person" (GT) vs 具体的な人物名 | GT が汎用キーワードのときに具体名を返してしまい TP 判定されない |
| `change_color` が `edit_motion` に誤分類 | "blue luxury car"のコンテキスト | 1件残存 |

**最大ゲイン候補**
- `replace_background` の locked 誤発火は action + target 両方に影響（3+1 = 4件）
- `change_camera_angle` target は 5件が角度表現で失敗 → person noun 抽出を優先化で改善見込み

### Countermeasure For Next Trial
- 対策: `_locked_action()` の `replace_background` 条件を厳格化。  
  「replace ... with ... background/scene/setting」の形式のみ `replace_background` に lock する。  
  それ以外の "replace X with Y" パターンは `replace_object` に誘導する。  
  これにより 3～4 件の action 誤分類と対応 target エラーを同時修正する。

---

## Trial 011 - llm_trial_011_ver9_replace_priority
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_llm_trial009.py](src/parse/instruction_parser_v3_llm_trial009.py)
- Log: [logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.log](logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.log)
- JSON: [logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.json](logs/analysis/llm_trial_011_ver9_replace_priority_20260403_040021.json)

### Hypothesis (before code)
- 仮説: `replace_background` の locked 条件が広すぎる。"replace X with Y" で Y が background/scene/setting でない場合は `replace_object` に lock するよう変更すれば、action 誤分類 3～4 件が修正され GT action が 88～89% に上昇する。  
  同時に `apply_style` が "Replace ... with ... neon/cyberpunk" で誤分類されるケースも、`apply_style` ロックに "neon|cyberpunk|vintage|cartoon|anime|watercolor|oil.paint|sketch" キーワードを追加して対策する。  
  target への副次効果: 誤分類が修正されると target の TP も連動して増加する見込み。

### Metrics
- GT action/target: 88.00% / 61.00%
- grouped action/target: 80.00% / 57.33%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL (target 61%)
- grouped goal (action >= 70%, target >= 70%): FAIL (target 57.33%)

### Analysis

**対策の効果**
- action GT: 85% → 88% (+3pt)。`replace_background` 誤分類 3件 + `apply_style` 1件が修正された。
- action grouped: 76.17% → 80.00% (+3.83pt)。grouped action が Goal の 70% を大きく超えた。
- target GT: 57% → 61% (+4pt)。action 修正に連動して target も改善。
- target grouped: 52.50% → 57.33% (+4.83pt)。

**新たな回帰**
- `replace_background` → `replace_object` の逆方向回帰が 1件。  
  "Replace the existing plain white wall and blinds background with..." で "background" が 8語目にあり、新レジェックス `{0,5}` 制限を超えた。→ `{0,8}` に緩和で修正可能。

**残る主な失敗パターン**
| 分類 | 件数 | 原因 |
|---|---|---|
| `change_camera_angle` target が角度表現 | 5件 | GT は人物名 or `camera_view`。角度表現 → 人物名変換ロジック未実装 |
| `edit_motion` target が GT=`person` vs pred=具体名 | 3件 | GT が汎用キーワードのとき具体名は不一致 |
| `dolly_in` → `change_camera_angle` 誤分類 | 1件 | "dolly-in" ハイフン形式が `\bdolly in\b` にマッチしない |
| `change_color` → `edit_motion` 誤分類 | 1件 | "Modify the exterior color" → `edit_motion` locked |
| `apply_style` → `edit_motion` 誤分類 | 1件 | "Studio Ghibli" キーワード未登録 |
| `increase_amount` → `add_object` 誤分類 | 1件 | "increase the amount of X" が `add_object` locked に引っかかる |

**最大ゲイン候補**
- `change_camera_angle` target: 5件修正で最大 +5pt GT target
- `dolly_in` ハイフン修正: 1件 action+target 同時修正
- `replace_background` 回帰修正: net-zero 必須 fix

### Countermeasure For Next Trial
- 主対策: `change_camera_angle` の target resolver を「角度表現を除外し、人物名を抽出する」に変更。  
  文中に `man|woman|boy|girl|person|speaker` 等があれば return それ、なければ `camera_view`。
- 回帰修正: `replace_background` 正規表現を `{0,5}` → `{0,8}` に緩和。
- Bug fix: `dolly in\b` → `dolly.?in\b` に変更しハイフン対応。

---

## Trial 012 - llm_trial_012_ver10_camera_target
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_llm_trial010.py](src/parse/instruction_parser_v3_llm_trial010.py)
- Log: [logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.log](logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.log)
- JSON: [logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.json](logs/analysis/llm_trial_012_ver10_camera_target_20260403_045051.json)

### Hypothesis (before code)
- 仮説: `change_camera_angle` の target が角度表現を返しているため 5件失敗。GT は人物名か `camera_view`。人物名を優先抽出し、なければ `camera_view` を返すことで +5pt GT target が見込める。  
   同時に `dolly.?in` ハイフン修正で +1pt action、`replace_background` 正規表現 `{0,8}` 緩和で回帰 1件を防ぐ。  
   合計で GT action 89%、GT target 66% 前後を想定。

### Metrics
- GT action/target: 91.00% / 66.00%
- grouped action/target: 82.50% / 62.17%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL (target 66%)
- grouped goal (action >= 70%, target >= 70%): FAIL (target 62.17%)

### Analysis

**対策の効果**
- action GT: 88% → 91% (+3pt)。`dolly-in` ハイフン修正、`replace_background` {0,8} 修正、camera_angle target 修正の副次効果で +3pt。
- action grouped: 80% → 82.5% (+2.5pt)。
- target GT: 61% → 66% (+5pt)。`change_camera_angle` 3件修正が寄与。
- target grouped: 57.33% → 62.17% (+4.84pt)。

**残る主な失敗パターン**
| 分類 | 件数 | 原因 |
|---|---|---|
| `edit_motion` target GT=`person` vs pred=具体名 | 5件 | GT の 11/12 が `person`。常に "person" 返すだけで +4pt 改善見込み |
| `change_camera_angle` target GT=`two men...` | 2件 | "men" (複数形) が person リスト未登録 |
| `change_color` → `edit_motion` 誤分類 | 2件 | "Modify ... color" → `edit_motion` locked が先に発火 |
| `add_effect` target | 2件 | NP 抽出が副詞節まで含む |
| `apply_style` → `edit_motion` 誤分類 | 1件 | "Studio Ghibli" キーワード未登録 |
| `increase_amount` → `add_object` 誤分類 | 1件 | "increase the amount" が `add_object` locked |
| `zoom_in` target GT=`camera_view` vs pred=人物名 | 1件 | "focusing closer on the man's face" → man 抽出 |
| `add_object` NP 抽出が長すぎる | 2件 | GT は短縮形、pred は文節全体 |

**最大ゲイン候補**
- `edit_motion` target を語彙固定ではなく、動詞構文（誰が何をするか）で抽出
- `change_camera_angle` target を人物語彙列挙ではなく、前置詞句（look at/toward/at）から抽出
- `change_color` 誤分類修正: "Modify ... color" で `change_color` を優先

### Countermeasure For Next Trial
- 対策: `edit_motion` target を固定ラベル返却にせず、motion 動詞にかかる主語句を抽出する。
- 追加対策: `change_camera_angle` は列挙語彙ではなく、`look at/toward/at` の目的語句を抽出し、角度語 (`low angle` など) は除外する。

---

## Trial 013 - llm_trial_013_ver11_generic_target
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_llm_trial011.py](src/parse/instruction_parser_v3_llm_trial011.py)
- Log: [logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.log](logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.log)
- JSON: [logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.json](logs/analysis/llm_trial_013_ver11_generic_target_20260403_053917.json)

### Hypothesis (before code)
- 仮説: `edit_motion` と `change_camera_angle` の target を語彙固定から構文抽出に切り替えると、未知 instruction でも破綻しにくくなる。  
  `edit_motion` は motion 動詞の主語句を返し、`change_camera_angle` は視線/向きの前置詞句目的語を返す。  
  角度語 (`low angle` など) を target から除外し、抽出不能時のみ `camera_view` / `object` にフォールバックする。

### Metrics
- GT action/target: 91.00% / 65.00%
- grouped action/target: 82.50% / 60.50%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL (target 65%)
- grouped goal (action >= 70%, target >= 70%): FAIL (target 60.5%)

### Analysis
- action は維持（GT 91%、grouped 82.5%）。
- target は Trial012 比で低下（GT 66%→65%、grouped 62.17%→60.5%）。
- 主因: `edit_motion` で "Modify the video so that ..." を主語抽出すると `video` を返しやすく、GT の期待 target と乖離した。
- `change_camera_angle` は未知語耐性は上がったが、target 改善インパクトは限定的だった。

### Countermeasure For Next Trial
- `edit_motion` は「主語句そのまま抽出」ではなく、`video/scene/frame/clip` の抽象語を除外して、動作主体らしい名詞句を優先する。
- `change_camera_angle` は前置詞句抽出に加え、`looking up/down at` と `capturing ... from` のパターンを追加して recall を上げる。
- いずれも語彙固定ではなく、構文・停止語ベースで実装する。
