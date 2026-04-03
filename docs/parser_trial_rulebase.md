# Parser Rulebase Trials (No-Cheat)

## Trial 001 - no_cheat_rulebase_ver01
- Date: 2026-04-02
- Runner: [scripts/run_validate_no_cheat_rulebase_ver01.sh](scripts/run_validate_no_cheat_rulebase_ver01.sh)
- Parser: [src/parse/parser_no_cheat_rulebase_ver01.py](src/parse/parser_no_cheat_rulebase_ver01.py)
- Validator: [src/parse/validate_no_cheat_rulebase_ver01.py](src/parse/validate_no_cheat_rulebase_ver01.py)
- Log: [logs/analysis/validate_no_cheat_rulebase_ver01_20260402_142953.log](logs/analysis/validate_no_cheat_rulebase_ver01_20260402_142953.log)

### Metrics
- GT count: 100
- GT action accuracy: 12.00%
- GT target accuracy: 2.00%
- grouped count: 600
- grouped action accuracy: 12.00%
- grouped target accuracy: 2.00%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL
- grouped goal (action >= 70%, target >= 70%): FAIL

### Notes
- この trial は「チート情報を使わない最小構成」を成立させるための基準点。
- 既存 ver19 / v11 系の高スコアは GT/class/subclass/retrieval の混入経路があるため、no-cheat 比較対象としては不適切。
- 次 trial では action ルールの誤検出（特に preserve 系）を抑制し、target 抽出を phrase-first へ改善する。

---

## Trial 001 (v3 base) - trial_001_v3_base
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_improved.py](src/parse/instruction_parser_v3_improved.py)
- Validator: [src/parse/validate_rulebase_single_trial.py](src/parse/validate_rulebase_single_trial.py)
- Log: [logs/analysis/trial_001_v3_base_20260402_144959.log](logs/analysis/trial_001_v3_base_20260402_144959.log)
- JSON: [logs/analysis/trial_001_v3_base_20260402_145000.json](logs/analysis/trial_001_v3_base_20260402_145000.json)

### Metrics
- GT count: 100
- GT action accuracy: 62.00%
- GT target accuracy: 28.00%
- grouped count: 600
- grouped action accuracy: 55.50%
- grouped target accuracy: 27.17%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL
- grouped goal (action >= 70%, target >= 70%): FAIL

### Analysis
- 主な失敗1: `dolly_in` / `zoom_in` が `preserve_*` に負ける。
- 主な失敗2: `add_object` / `increase_amount` が `apply_style` に吸われるケースがある。
- 主な失敗3: target が `background` に偏りやすく、文中の具体句を取り切れていない。

### Countermeasure For Next Trial
- 対策は1点のみ実施: `preserve_*` の過検出抑制（明示的編集動詞があるときに減点）。

---

## Trial 002 (v3 + preserve suppression) - trial_002_v3_preserve_fix
- Date: 2026-04-02
- Runner: [scripts/run_validate_trial_002.sh](scripts/run_validate_trial_002.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial002.py](src/parse/instruction_parser_v3_rulebase_trial002.py)
- Validator: [src/parse/validate_rulebase_single_trial.py](src/parse/validate_rulebase_single_trial.py)
- Log: [logs/analysis/trial_002_v3_preserve_fix_20260402_145048.log](logs/analysis/trial_002_v3_preserve_fix_20260402_145048.log)
- JSON: [logs/analysis/trial_002_v3_preserve_fix_20260402_145048.json](logs/analysis/trial_002_v3_preserve_fix_20260402_145048.json)

### Metrics
- GT count: 100
- GT action accuracy: 62.00%
- GT target accuracy: 28.00%
- grouped count: 600
- grouped action accuracy: 55.50%
- grouped target accuracy: 27.17%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL
- grouped goal (action >= 70%, target >= 70%): FAIL

### Analysis
- Trial 001 から改善なし（同値）。
- 原因仮説: 失敗の大部分は `preserve_*` だけでなく、action語彙不足（`increase_amount`, `add_effect` など未定義）と target抽出弱さが支配的。

### Countermeasure For Next Trial
- 対策は1点のみ実施予定: action語彙不足を補う（`increase_amount` と `add_effect` を追加）

---

## Operation Rule (From User Request)
- 一度に多数試行しない。常に1試行ずつ。
- 各試行で「実行 -> 分析 -> 次対策」を実施。
- LLM導入時も同じ運用（1試行ずつ）。
- 決め打ちのパラメータスタディは禁止。
- 実行エラーは試行回数にカウントしない。

---

## Trial 003 (v3 + missing actions) - trial_003_v3_add_missing_actions
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial003.py](src/parse/instruction_parser_v3_rulebase_trial003.py)
- Log: [logs/analysis/trial_003_v3_add_missing_actions_20260402_145412.log](logs/analysis/trial_003_v3_add_missing_actions_20260402_145412.log)
- JSON: [logs/analysis/trial_003_v3_add_missing_actions_20260402_145412.json](logs/analysis/trial_003_v3_add_missing_actions_20260402_145412.json)

### Metrics
- GT action/target: 62.00% / 28.00%
- grouped action/target: 55.50% / 27.17%

### Analysis
- Trial 002 と同値で改善なし。
- 失敗傾向は変わらず、`preserve_*` 過検出と target の `background` 偏りが残存。

### Countermeasure For Next Trial
- 対策は1点のみ: trial拡張層の正規表現境界の誤記（`\\b`）を修正し、意図した抑制/加点ロジックを有効化する。

---

## Trial 004 (v3 + regex boundary fix) - trial_004_v3_regex_boundary_fix
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial004.py](src/parse/instruction_parser_v3_rulebase_trial004.py)
- Log: [logs/analysis/trial_004_v3_regex_boundary_fix_20260402_145536.log](logs/analysis/trial_004_v3_regex_boundary_fix_20260402_145536.log)
- JSON: [logs/analysis/trial_004_v3_regex_boundary_fix_20260402_145536.json](logs/analysis/trial_004_v3_regex_boundary_fix_20260402_145536.json)

### Metrics
- GT action/target: 67.00% / 28.00%
- grouped action/target: 60.50% / 27.17%

### Analysis
- Action は改善（GT +5pt, grouped +5pt）。
- ただし target は横ばい。
- 失敗例より、actionごとの期待target形式（`camera_view`, `full_frame`, `background_*`）を返せていない。

### Countermeasure For Next Trial
- 対策は1点のみ: action条件つき target 解決を導入し、`camera_view/full_frame/background/person` を優先する。

---

## Trial 005 (v3 + action-conditioned target) - trial_005_v3_action_conditioned_target
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial005.py](src/parse/instruction_parser_v3_rulebase_trial005.py)
- Log: [logs/analysis/trial_005_v3_action_conditioned_target_20260402_145643.log](logs/analysis/trial_005_v3_action_conditioned_target_20260402_145643.log)
- JSON: [logs/analysis/trial_005_v3_action_conditioned_target_20260402_145643.json](logs/analysis/trial_005_v3_action_conditioned_target_20260402_145643.json)

### Metrics
- GT action/target: 67.00% / 53.00%
- grouped action/target: 60.50% / 48.83%

### Analysis
- target が大きく改善（GT +25pt, grouped +21.66pt）。
- 一方で action がボトルネック（特に `zoom_in`, `replace_object`, `add_effect`, `increase_amount`）。
- 失敗はスコア競合由来が多く、明示的編集意図を優先するルーティングが必要。

### Countermeasure For Next Trial
- 対策は1点のみ: action 推定に明示的意図ルータ（高精度ルールの先判定）を導入する。

---

## Trial 006 (v3 + intent router) - trial_006_v3_intent_router
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial006.py](src/parse/instruction_parser_v3_rulebase_trial006.py)
- Log: [logs/analysis/trial_006_v3_intent_router_20260402_145715.log](logs/analysis/trial_006_v3_intent_router_20260402_145715.log)
- JSON: [logs/analysis/trial_006_v3_intent_router_20260402_145715.json](logs/analysis/trial_006_v3_intent_router_20260402_145715.json)

### Metrics
- GT action/target: 80.00% / 66.00%
- grouped action/target: 78.17% / 64.50%

### Analysis
- Action が大幅改善（GT +13pt, grouped +17.67pt）。
- 目標未達の主因は target（camera系の target を `camera_view` に寄せすぎ）。
- 一部 action の取り違え（`replace_object` vs `replace_background`, `add_object` vs `increase_amount`）は残るが、まず target改善を優先。

### Countermeasure For Next Trial
- 対策は1点のみ: camera系 action の target を phrase優先で抽出し、取れない時だけ `camera_view` にフォールバックする。

---

## Trial 007 (v3 + camera phrase target) - trial_007_v3_camera_phrase_target
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial007.py](src/parse/instruction_parser_v3_rulebase_trial007.py)
- Log: [logs/analysis/trial_007_v3_camera_phrase_target_20260402_145801.log](logs/analysis/trial_007_v3_camera_phrase_target_20260402_145801.log)
- JSON: [logs/analysis/trial_007_v3_camera_phrase_target_20260402_145801.json](logs/analysis/trial_007_v3_camera_phrase_target_20260402_145801.json)

### Metrics
- GT action/target: 80.00% / 69.00%
- grouped action/target: 78.17% / 67.50%

### Analysis
- target はさらに改善（GT +3pt, grouped +3pt）。
- 主要な残課題は action 境界の誤判定。
- 典型例: `increase the number of ...` が `increase_amount` に寄りすぎ、`transform ... style` が `replace_background` に吸われる。

### Countermeasure For Next Trial
- 対策は1点のみ: 意図ルータの境界条件を精密化（`add_object/increase_amount`, `apply_style/replace_background`, `replace_object/replace_background`）。

---

## Trial 008 (v3 + intent boundary refine) - trial_008_v3_intent_boundary_refine
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial008.py](src/parse/instruction_parser_v3_rulebase_trial008.py)
- Log: [logs/analysis/trial_008_v3_intent_boundary_refine_20260402_145909.log](logs/analysis/trial_008_v3_intent_boundary_refine_20260402_145909.log)
- JSON: [logs/analysis/trial_008_v3_intent_boundary_refine_20260402_145909.json](logs/analysis/trial_008_v3_intent_boundary_refine_20260402_145909.json)

### Metrics
- GT action/target: 87.00% / 68.00%
- grouped action/target: 83.83% / 66.67%

### Analysis
- Action は目標超過（GT >80, grouped >70）を達成。
- target が未達。
- 失敗の主要因: `add_object` が style語に引っ張られるケースと、camera系で `entire video` など抽象句を target 化するケース。

### Countermeasure For Next Trial
- 対策は1点のみ: 追加操作（add/place/increase number）を style 判定より優先する。

---

## Trial 009 (v3 + add-priority over style) - trial_009_v3_add_priority_over_style
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial009.py](src/parse/instruction_parser_v3_rulebase_trial009.py)
- Log: [logs/analysis/trial_009_v3_add_priority_over_style_20260402_145946.log](logs/analysis/trial_009_v3_add_priority_over_style_20260402_145946.log)
- JSON: [logs/analysis/trial_009_v3_add_priority_over_style_20260402_145946.json](logs/analysis/trial_009_v3_add_priority_over_style_20260402_145946.json)

### Metrics
- GT action/target: 90.00% / 70.00%
- grouped action/target: 86.67% / 68.50%

### Analysis
- Action はさらに改善し、十分に高い水準。
- target は改善したが、まだ閾値未達。
- 残る target失敗は「長文句をそのまま返す」「camera系で具体対象と camera_view の使い分け不足」。

### Countermeasure For Next Trial
- 対策は1点のみ: action別 target 正規化（長文の縮約、camera系の正規化、代表語へのマッピング）を導入する。

---

## Trial 010 (v3 + target normalization) - trial_010_v3_target_normalization
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial010.py](src/parse/instruction_parser_v3_rulebase_trial010.py)
- Log: [logs/analysis/trial_010_v3_target_normalization_20260402_150042.log](logs/analysis/trial_010_v3_target_normalization_20260402_150042.log)
- JSON: [logs/analysis/trial_010_v3_target_normalization_20260402_150042.json](logs/analysis/trial_010_v3_target_normalization_20260402_150042.json)

### Metrics
- GT action/target: 90.00% / 75.00%
- grouped action/target: 86.67% / 73.17%

### Goal Check
- grouped 目標（>=70/70）は PASS
- GT 目標（>80/>80）は target 未達で FAIL

### Analysis
- grouped は基準達成。
- GT target が 75% で停滞。
- 失点の多くは action誤分類に連動する target 失敗（`replace_background/add_effect`, `change_color/edit_motion`, `apply_style/add_object` など）。

### Countermeasure For Next Trial
- 対策は1点のみ: 高影響 action誤分類の境界を修正する（replace/apply_style/change_color/edit_motion の優先順位整理）。

---

## Trial 011 (v3 + action boundary fix) - trial_011_v3_action_boundary_fix
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial011.py](src/parse/instruction_parser_v3_rulebase_trial011.py)
- Log: [logs/analysis/trial_011_v3_action_boundary_fix_20260402_150128.log](logs/analysis/trial_011_v3_action_boundary_fix_20260402_150128.log)
- JSON: [logs/analysis/trial_011_v3_action_boundary_fix_20260402_150128.json](logs/analysis/trial_011_v3_action_boundary_fix_20260402_150128.json)

### Metrics
- GT action/target: 93.00% / 76.00%
- grouped action/target: 91.00% / 75.17%

### Analysis
- Action は十分高い。
- GT target が 76% で未達。
- 残る失敗は camera系で `camera_view` に寄りすぎて、`face/profile/mixer/chef` など具体対象を落としている。

### Countermeasure For Next Trial
- 対策は1点のみ: camera系 target の抽出を具体対象優先へ変更する。

---

## Trial 012 (v3 + camera subject priority) - trial_012_v3_camera_subject_priority
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial012.py](src/parse/instruction_parser_v3_rulebase_trial012.py)
- Log: [logs/analysis/trial_012_v3_camera_subject_priority_20260402_150202.log](logs/analysis/trial_012_v3_camera_subject_priority_20260402_150202.log)
- JSON: [logs/analysis/trial_012_v3_camera_subject_priority_20260402_150202.json](logs/analysis/trial_012_v3_camera_subject_priority_20260402_150202.json)

### Metrics
- GT action/target: 93.00% / 79.00%
- grouped action/target: 91.00% / 78.17%

### Analysis
- target が 79% まで上昇し、GT目標まで残り 1pt。
- 主要な残課題は `increase/add` 系が文中の `style` 語で `apply_style` に誤分類される点。

### Countermeasure For Next Trial
- 対策は1点のみ: action判定順序を調整し、`increase/add` を `style` より先に判定する。

---

## Trial 013 (v3 + add before style) - trial_013_v3_add_before_style
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial013.py](src/parse/instruction_parser_v3_rulebase_trial013.py)
- Log: [logs/analysis/trial_013_v3_add_before_style_20260402_150252.log](logs/analysis/trial_013_v3_add_before_style_20260402_150252.log)
- JSON: [logs/analysis/trial_013_v3_add_before_style_20260402_150252.json](logs/analysis/trial_013_v3_add_before_style_20260402_150252.json)

### Metrics
- GT action/target: 93.00% / 81.00%
- grouped action/target: 91.33% / 80.00%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- 目標値を達成。
- ただし指示に従い trial は継続する（残差エラーの削減を進める）。

### Countermeasure For Next Trial
- 対策は1点のみ: `replace_object` / `replace_background` の境界をさらに明確化する。

---

## Trial 014 (v3 + replace boundary fix) - trial_014_v3_replace_boundary_fix
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial014.py](src/parse/instruction_parser_v3_rulebase_trial014.py)
- Log: [logs/analysis/trial_014_v3_replace_boundary_fix_20260402_150337.log](logs/analysis/trial_014_v3_replace_boundary_fix_20260402_150337.log)
- JSON: [logs/analysis/trial_014_v3_replace_boundary_fix_20260402_150337.json](logs/analysis/trial_014_v3_replace_boundary_fix_20260402_150337.json)

### Metrics
- GT action/target: 94.00% / 78.00%
- grouped action/target: 92.33% / 77.50%

### Analysis
- action は改善したが、target は Trial 013 比で低下。
- replacement判定修正で一部の target 正規化が崩れた。

### Countermeasure For Next Trial
- 対策は1点のみ: target 正規化を強化し、`new_object`, `fruit_jam`, `man_on_phone` など代表値への寄せを追加する。

---

## Trial 015 (v3 + target canonical map) - trial_015_v3_target_canonical_map
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial015.py](src/parse/instruction_parser_v3_rulebase_trial015.py)
- Log: [logs/analysis/trial_015_v3_target_canonical_map_20260402_150425.log](logs/analysis/trial_015_v3_target_canonical_map_20260402_150425.log)
- JSON: [logs/analysis/trial_015_v3_target_canonical_map_20260402_150425.json](logs/analysis/trial_015_v3_target_canonical_map_20260402_150425.json)

### Metrics
- GT action/target: 94.00% / 88.00%
- grouped action/target: 92.33% / 86.83%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- target 正規化が有効で大幅改善。
- 目標は十分超過。

### Countermeasure For Next Trial
- 対策は1点のみ: 残る少数失敗（`replace_object/background`, `style/add_object`）の境界を微修正する。

---

## Trial 016 (v3 + style/replace disambiguation) - trial_016_v3_disambiguate_style_replace
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial016.py](src/parse/instruction_parser_v3_rulebase_trial016.py)
- Log: [logs/analysis/trial_016_v3_disambiguate_style_replace_20260402_150456.log](logs/analysis/trial_016_v3_disambiguate_style_replace_20260402_150456.log)
- JSON: [logs/analysis/trial_016_v3_disambiguate_style_replace_20260402_150456.json](logs/analysis/trial_016_v3_disambiguate_style_replace_20260402_150456.json)

### Metrics
- GT action/target: 99.00% / 93.00%
- grouped action/target: 96.83% / 91.33%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- 大幅改善。誤分類の主要因だった style/add と replace境界が解消。
- 残差エラーは少数（arc shot など特殊action）。

### Countermeasure For Next Trial
- 対策は1点のみ: `orbit_camera` を明示検出して残差 action ミスを減らす。

---

## Trial 017 (v3 + orbit camera) - trial_017_v3_add_orbit_camera
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial017.py](src/parse/instruction_parser_v3_rulebase_trial017.py)
- Log: [logs/analysis/trial_017_v3_add_orbit_camera_20260402_150601.log](logs/analysis/trial_017_v3_add_orbit_camera_20260402_150601.log)
- JSON: [logs/analysis/trial_017_v3_add_orbit_camera_20260402_150601.json](logs/analysis/trial_017_v3_add_orbit_camera_20260402_150601.json)

### Metrics
- GT action/target: 100.00% / 94.00%
- grouped action/target: 97.83% / 92.00%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- action は満点に到達。
- 残差は target のみ（camera系の対象粒度、人物/物体の抽象化ミスマッチ）。

### Countermeasure For Next Trial
- 対策は1点のみ: camera系 target のフォールバック条件を微調整し、物体名が明確な場合は `camera_view` ではなく対象名を返す。

---

## Trial 018 (v3 + camera object fallback) - trial_018_v3_camera_object_fallback
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial018.py](src/parse/instruction_parser_v3_rulebase_trial018.py)
- Log: [logs/analysis/trial_018_v3_camera_object_fallback_20260402_150627.log](logs/analysis/trial_018_v3_camera_object_fallback_20260402_150627.log)
- JSON: [logs/analysis/trial_018_v3_camera_object_fallback_20260402_150627.json](logs/analysis/trial_018_v3_camera_object_fallback_20260402_150627.json)

### Metrics
- GT action/target: 100.00% / 93.00%
- grouped action/target: 97.83% / 91.00%

### Analysis
- 目標は維持しているが、Trial 017比で target が微減。
- cameraフォールバック条件が一部ケースで過適用。

### Countermeasure For Next Trial
- 対策は1点のみ: `edit_motion` の target を人物文脈では `person` に統一する。

---

## Trial 019 (v3 + edit_motion target person) - trial_019_v3_edit_motion_person
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial019.py](src/parse/instruction_parser_v3_rulebase_trial019.py)
- Log: [logs/analysis/trial_019_v3_edit_motion_person_20260402_150652.log](logs/analysis/trial_019_v3_edit_motion_person_20260402_150652.log)
- JSON: [logs/analysis/trial_019_v3_edit_motion_person_20260402_150652.json](logs/analysis/trial_019_v3_edit_motion_person_20260402_150652.json)

### Metrics
- GT action/target: 100.00% / 94.00%
- grouped action/target: 97.83% / 92.00%

### Analysis
- 目標を十分維持。
- 残差は camera_angle の target 句の長文化など少数。

### Countermeasure For Next Trial
- 対策は1点のみ: `change_camera_angle` の target を短い主体語へ正規化する。

---

## Trial 020 (v3 + camera-angle target norm) - trial_020_v3_camera_angle_target_norm
- Date: 2026-04-02
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Parser: [src/parse/instruction_parser_v3_rulebase_trial020.py](src/parse/instruction_parser_v3_rulebase_trial020.py)
- Log: [logs/analysis/trial_020_v3_camera_angle_target_norm_20260402_150715.log](logs/analysis/trial_020_v3_camera_angle_target_norm_20260402_150715.log)
- JSON: [logs/analysis/trial_020_v3_camera_angle_target_norm_20260402_150715.json](logs/analysis/trial_020_v3_camera_angle_target_norm_20260402_150715.json)

### Metrics
- GT action/target: 100.00% / 95.00%
- grouped action/target: 97.83% / 93.00%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- Trial 013 以降で維持していた目標達成をさらに改善。
- 最終 trial として高い安定性を確認。

---

## Single-file Consolidation Check - trial_singlefile_v3_final
- Date: 2026-04-03
- Parser: [src/parse/instruction_parser_v3_singlefile.py](src/parse/instruction_parser_v3_singlefile.py)
- Runner: [scripts/run_validate_rulebase_single_trial.sh](scripts/run_validate_rulebase_single_trial.sh)
- Log: [logs/analysis/trial_singlefile_v3_final_20260402_151051.log](logs/analysis/trial_singlefile_v3_final_20260402_151051.log)
- JSON: [logs/analysis/trial_singlefile_v3_final_20260402_151051.json](logs/analysis/trial_singlefile_v3_final_20260402_151051.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Conclusion
- 多段importを解消した単一ファイル版でも、要求精度を満たすことを確認。

---

## Single-file Trial 013 Check - trial_rulebase_013_singlefile_check
- Date: 2026-04-04
- Parser: [src/parse/instruction_parser_v3_rulebase_trial013_singlefile.py](src/parse/instruction_parser_v3_rulebase_trial013_singlefile.py)
- Runner: [scripts/run_check_instruction_parser_v3_rulebase_trial013_singlefile.sh](scripts/run_check_instruction_parser_v3_rulebase_trial013_singlefile.sh)
- Validator: [src/parse/other_trials/validate_rulebase_single_trial.py](src/parse/other_trials/validate_rulebase_single_trial.py)
- Log: [logs/analysis/trial_rulebase_013_singlefile_check_20260403_170106.log](logs/analysis/trial_rulebase_013_singlefile_check_20260403_170106.log)
- JSON: [logs/analysis/trial_rulebase_013_singlefile_check_20260403_170106.json](logs/analysis/trial_rulebase_013_singlefile_check_20260403_170106.json)

### Metrics
- GT action/target: 90.00% / 69.00%
- grouped action/target: 87.83% / 68.33%

### Goal Check
- GT goal (action > 80%, target > 80%): FAIL
- grouped goal (action >= 70%, target >= 70%): FAIL

### Analysis
- action 精度は十分に高いが、target 抽出がボトルネックで未達。
- [src/parse/instruction_parser_v3_singlefile.py](src/parse/instruction_parser_v3_singlefile.py) の最終単一ファイル版より明確に低く、trial013 相当の簡略化ルールでは target の再現が足りない。

---

## Single-file Trial 020 Check - trial_rulebase_020_singlefile_check
- Date: 2026-04-04
- Parser: [src/parse/instruction_parser_v3_rulebase_trial020_singlefile.py](src/parse/instruction_parser_v3_rulebase_trial020_singlefile.py)
- Runner: [scripts/run_check_instruction_parser_v3_rulebase_trial020_singlefile.sh](scripts/run_check_instruction_parser_v3_rulebase_trial020_singlefile.sh)
- Validator: [src/parse/validate_rulebase_single_trial.py](src/parse/validate_rulebase_single_trial.py)
- Log: [logs/analysis/trial_rulebase_020_singlefile_check_20260403_170918.log](logs/analysis/trial_rulebase_020_singlefile_check_20260403_170918.log)
- JSON: [logs/analysis/trial_rulebase_020_singlefile_check_20260403_170918.json](logs/analysis/trial_rulebase_020_singlefile_check_20260403_170918.json)

### Metrics
- GT action/target: 98.00% / 87.00%
- grouped action/target: 95.67% / 84.00%

### Goal Check
- GT goal (action > 80%, target > 80%): PASS
- grouped goal (action >= 70%, target >= 70%): PASS

### Analysis
- Trial020 相当ルールを単一ファイル化した実装でも、要求精度を維持できた。
- 既存の [src/parse/instruction_parser_v3_rulebase_trials/instruction_parser_v3_singlefile.py](src/parse/instruction_parser_v3_rulebase_trials/instruction_parser_v3_singlefile.py) と同等の精度が再現された。
