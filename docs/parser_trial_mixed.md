# parser_trial_mixed

LLM + RuleBase の混合試行ログ。

## Goal
- GT: action > 80%, target > 80%
- grouped: action >= 70%, target >= 70%

## Mixed Parser
- File: [src/parse/prototype_instruction_parser_v3_mixed_trial001.py](src/parse/prototype_instruction_parser_v3_mixed_trial001.py)
- Modes:
	- `MIX_MODE=llm_main`: LLM を主、RuleBase を補助
	- `MIX_MODE=rule_main`: RuleBase を主、LLM を補助

## Execution Mapping
- Trial 番号は「実行回数」の番号。
- Parser ファイル番号は「実装バージョン」の番号。
- そのため、Trial 004 が `prototype_instruction_parser_v3_mixed_trial004.py` を意味するわけではない。

| Trial | Executed Parser File | MIX_MODE | Trial Name |
| --- | --- | --- | --- |
| 001 | `prototype_instruction_parser_v3_mixed_trial001.py` | `llm_main` | `mixed_trial_001_llm_main` |
| 002 | `prototype_instruction_parser_v3_mixed_trial001.py` | `rule_main` | `mixed_trial_002_rule_main` |
| 003 | `prototype_instruction_parser_v3_mixed_trial002.py` | `llm_main` | `mixed_trial_003_llm_main_regexfix` |
| 004 | `prototype_instruction_parser_v3_mixed_trial002.py` | `rule_main` | `mixed_trial_004_rule_main_regexfix` |
| 005 | `prototype_instruction_parser_v3_mixed_trial003.py` | `rule_main` | `mixed_trial_005_rule_main_action_fixes_v3_batched` |

## Trial 001 - mixed_trial_001_llm_main
- Date: 2026-04-03
- Parser: [src/parse/prototype_instruction_parser_v3_mixed_trial001.py](src/parse/prototype_instruction_parser_v3_mixed_trial001.py)
- Mode: `MIX_MODE=llm_main`
- Command:
	`MIX_MODE=llm_main PARSER_FILE=/workspace/src/parse/prototype_instruction_parser_v3_mixed_trial001.py TRIAL_NAME=mixed_trial_001_llm_main EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh`
- Log: [logs/analysis/mixed_trial_001_llm_main_20260403_064458.log](logs/analysis/mixed_trial_001_llm_main_20260403_064458.log)
- JSON: [logs/analysis/mixed_trial_001_llm_main_20260403_064459.json](logs/analysis/mixed_trial_001_llm_main_20260403_064459.json)

### Metrics
- GT action/target: 91.00% / 65.00%
- grouped action/target: 82.67% / 60.50%

### Goal Check
- GT: FAIL
- grouped: FAIL

### Analysis
- LLM 主導でも action は高水準を維持。
- target は [src/parse/prototype_instruction_parser_v3_llm_trial013.py](src/parse/prototype_instruction_parser_v3_llm_trial013.py) と同等帯で頭打ち。
- RuleBase 補助は効いているが、target の曖昧文で改善が不足。

---

## Trial 002 - mixed_trial_002_rule_main
- Date: 2026-04-03
- Parser: [src/parse/prototype_instruction_parser_v3_mixed_trial001.py](src/parse/prototype_instruction_parser_v3_mixed_trial001.py)
- Mode: `MIX_MODE=rule_main`
- Command:
	`MIX_MODE=rule_main PARSER_FILE=/workspace/src/parse/prototype_instruction_parser_v3_mixed_trial001.py TRIAL_NAME=mixed_trial_002_rule_main EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh`
- Log: [logs/analysis/mixed_trial_002_rule_main_20260403_071353.log](logs/analysis/mixed_trial_002_rule_main_20260403_071353.log)
- JSON: [logs/analysis/mixed_trial_002_rule_main_20260403_071353.json](logs/analysis/mixed_trial_002_rule_main_20260403_071353.json)

### Metrics
- GT action/target: 92.00% / 67.00%
- grouped action/target: 85.00% / 62.83%

### Goal Check
- GT: FAIL
- grouped: FAIL

### Analysis
- RuleBase 主導の方が、現状では action/target ともに上回った。
- 特に grouped target が +2.33pt（60.50% → 62.83%）改善。
- ただし target 80/70 目標にはまだ不足。

## Interim Conclusion (after Trial 002)
- 現時点ベストは `rule_main`。
- 次の改善軸:
	- RuleBase の target 抽出で抽象語（video/scene/frame）を除外
	- LLM 補助を「action rescue」ではなく「target refinement」に限定適用
	- `change_color` / `add_effect` の target 抽出を優先改善

---

## Trial 003 - mixed_trial_003_llm_main_regexfix
- Date: 2026-04-03
- Parser: [src/parse/prototype_instruction_parser_v3_mixed_trial002.py](src/parse/prototype_instruction_parser_v3_mixed_trial002.py)
- Mode: `MIX_MODE=llm_main`
- Changes from Trial 001: `\\b` → `\b` regex fix in motion_verbs/camera patterns; stopword (video/scene/frame/clip/shot) removal for edit_motion candidate
- Command:
	`MIX_MODE=llm_main PARSER_FILE=/workspace/src/parse/prototype_instruction_parser_v3_mixed_trial002.py TRIAL_NAME=mixed_trial_003_llm_main_regexfix EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh`
- Log: [logs/analysis/mixed_trial_003_llm_main_regexfix_20260403_075956.log](logs/analysis/mixed_trial_003_llm_main_regexfix_20260403_075956.log)
- JSON: [logs/analysis/mixed_trial_003_llm_main_regexfix_20260403_075956.json](logs/analysis/mixed_trial_003_llm_main_regexfix_20260403_075956.json)

### Metrics
- GT action/target: 91.00% / 64.00%
- grouped action/target: 82.67% / 58.67%

### Goal Check
- GT: FAIL
- grouped: FAIL

### Analysis
- regex fix により edit_motion の誤パターンが解消されたが、llm_main では target が逆に -1pt（65→64%）。
- LLM 依存の抽出で word boundary 修正が一部を正しく修正したが、別のケースで微減。
- llm_main は rule_main に比べ安定性が低い。以降は rule_main 単独で深化させる方針。

---

## Trial 004 - mixed_trial_004_rule_main_regexfix
- Date: 2026-04-03
- Parser: [src/parse/prototype_instruction_parser_v3_mixed_trial002.py](src/parse/prototype_instruction_parser_v3_mixed_trial002.py)
- Mode: `MIX_MODE=rule_main`
- Changes from Trial 002: 同上（regex fix + stopword removal）
- Command:
	`MIX_MODE=rule_main PARSER_FILE=/workspace/src/parse/prototype_instruction_parser_v3_mixed_trial002.py TRIAL_NAME=mixed_trial_004_rule_main_regexfix EVAL_BATCH_SIZE=1 SHOW_PROGRESS=1 LLM_BATCH_SIZE=1 bash /workspace/scripts/run_validate_llm_single_trial.sh`
- Log: [logs/analysis/mixed_trial_004_rule_main_regexfix_20260403_082841.log](logs/analysis/mixed_trial_004_rule_main_regexfix_20260403_082841.log)
- JSON: [logs/analysis/mixed_trial_004_rule_main_regexfix_20260403_082841.json](logs/analysis/mixed_trial_004_rule_main_regexfix_20260403_082841.json)

### Metrics
- GT action/target: 92.00% / 66.00%
- grouped action/target: 85.00% / 61.00%

### Goal Check
- GT: FAIL
- grouped: FAIL

### Analysis
- regex 修正は rule_main にも効果を維持（92%/67% → 92%/66%、-1pt は誤差範囲）。
- 依然として action 誤り 4 件、target 誤り 16 件。
- action 誤り内訳:
  - `increase_amount → add_object` (1件): "Increase the amount of X" を add_object にロック
  - `change_color → edit_motion` (1件): "Modify.*color" が edit_motion 優先で処理される
  - `apply_style → edit_motion` (1件): "Studio Ghibli" が apply_style に未登録
  - `edit_motion → dolly_in` (1件): "tilts her head" が dolly_in にマッチ
- target 誤り主因: edit_motion target に person が来るべきケース（5件）で NP 抽出が "please_edit_woman_to" 等の誤句を拾う

---

## Trial 005 - mixed_trial_005_rule_main_action_fixes_v3_batched
- Date: 2026-04-03
- Parser: [src/parse/prototype_instruction_parser_v3_mixed_trial003.py](src/parse/prototype_instruction_parser_v3_mixed_trial003.py)
- Mode: `MIX_MODE=rule_main`
- Command:
	`MIX_MODE=rule_main PARSER_FILE=/workspace/src/parse/prototype_instruction_parser_v3_mixed_trial003.py TRIAL_NAME=mixed_trial_005_rule_main_action_fixes_v3_batched EVAL_BATCH_SIZE=16 SHOW_PROGRESS=1 LLM_BATCH_SIZE=8 bash /workspace/scripts/run_validate_llm_single_trial.sh`
- Log: [logs/analysis/mixed_trial_005_rule_main_action_fixes_v3_batched_20260403_094801.log](logs/analysis/mixed_trial_005_rule_main_action_fixes_v3_batched_20260403_094801.log)
- JSON: [logs/analysis/mixed_trial_005_rule_main_action_fixes_v3_batched_20260403_094801.json](logs/analysis/mixed_trial_005_rule_main_action_fixes_v3_batched_20260403_094801.json)

### Metrics
- GT action/target: 97.00% / 67.00%
- grouped action/target: 94.00% / 62.50%

### Goal Check
- GT: FAIL（action は達成、target 未達）
- grouped: FAIL（action は達成、target 未達）

### Changes from Trial 004
1. **_locked_action fix**: `increase.*amount` を `increase_amount` に修正（以前は `add_object`）
2. **_locked_action fix**: `modify.*color|change.*color` を `change_color` lock に追加
3. **_locked_action expansion**: `studio ghibli|ghibli|aesthetic|inspired|transform.*aesthetic|transform.*style|pixel.?art|ukiyo` を `apply_style` 判定に追加
4. **LLM rescue restriction**: rule_main で edit_motion に対する camera action rescue を禁止（`dolly_in/out`, `zoom_in/out`, `orbit_camera`, `change_camera_angle`）
5. **Critical fallback fix**: `_heuristic_prediction()` が `_locked_action()` を先に呼ぶよう修正（LLM 例外時に lock 判定がバイパスされていた不具合を解消）

### Analysis
- Trial 004（92% / 66%）比で、GT action は +5pt、GT target は +1pt。
- grouped も action +9pt、target +1.5pt と改善。
- action 側の主要誤り（`increase_amount`, `apply_style`, `change_color`, `edit_motion→dolly_in`）は実質解消。
- 未達の主因は target 抽出で、特に `edit_motion` の主語抽出誤りと冗長 NP 抽出が残存。

### Remaining Issues (from failure samples)
- `edit_motion` target: `person` 相当を短く正規化できず、`man`, `she_slowly`, `modify_to_have_man` などに崩れる
- `change_camera_angle` / `zoom_in` target: `camera_view` と人物部位ターゲットの使い分けが不十分
- `add_object` / `add_effect` / `remove_object` target: 句境界処理不足で target が冗長化

## Consolidated Conclusion (through Trial 005)
- 現時点ベスト: Trial 005（rule_main, mixed_trial003）
- 達成:
	- GT action > 80%: 達成（97%）
	- grouped action >= 70%: 達成（94%）
- 未達:
	- GT target > 80%: 67%
	- grouped target >= 70%: 62.5%
- 次の改善軸は action ではなく target 抽出専用の改善（主語抽出・句境界・短文化正規化）。
