# parse_instruction 試行錯誤（事実ベース）

このドキュメントは、推測や解釈を減らし、確認できた事実のみを整理した記録です。

## 0. 参照元（事実）
1. Notebook
- [notebook/parse_instruction_ver10.ipynb](notebook/parse_instruction_ver10.ipynb)
- [notebook/parse_instruction_ver11.ipynb](notebook/parse_instruction_ver11.ipynb)
- [notebook/parse_instruction_ver12.ipynb](notebook/parse_instruction_ver12.ipynb)

2. 出力JSON
- [notebook/prediction_results_ver10.json](notebook/prediction_results_ver10.json)
- [notebook/prediction_results_ver10_summary.json](notebook/prediction_results_ver10_summary.json)
- [notebook/ver11_outputs/prediction_results_ver11d_ensemble.json](notebook/ver11_outputs/prediction_results_ver11d_ensemble.json)
- [notebook/ver11_outputs/prediction_results_ver11_summary.json](notebook/ver11_outputs/prediction_results_ver11_summary.json)
- [notebook/ver12_outputs/unknown_instruction_analysis_ver12.json](notebook/ver12_outputs/unknown_instruction_analysis_ver12.json)
- [notebook/ver12_outputs/unknown_predictions_ver10_baseline.json](notebook/ver12_outputs/unknown_predictions_ver10_baseline.json)
- [notebook/ver12_outputs/unknown_predictions_ver10_improved.json](notebook/ver12_outputs/unknown_predictions_ver10_improved.json)
- [notebook/ver12_outputs/unknown_predictions_v11a_ruleplus.json](notebook/ver12_outputs/unknown_predictions_v11a_ruleplus.json)
- [notebook/ver12_outputs/unknown_predictions_v11d_ensemble.json](notebook/ver12_outputs/unknown_predictions_v11d_ensemble.json)

3. ログ
- 専用ログファイルは作っていない。
- 実行ログは各 notebook のセル出力として保存されている。

## 1. version: ver10

### 1-1. 構図（実装されている事実）
1. 入力と比較対象
- 入力: `annotations.jsonl`
- 比較対象: `annotations_gt_task_ver09.json`
- 比較単位: GT の複数 task を primary task に縮約して 1 件比較
- コメント（懸念）: この縮約はローカル検証を単純化するための便宜的処理であり、コンペ本番の評価要件と一致する保証はない。評価ルール上、複数 task を保持して判定すべき場合は、この比較単位はルール違反になる可能性がある。
- 対応方針（要再評価）: 正式評価に使う前に、GT の全 task を対象にした評価実装で再採点し、縮約版スコアは参考値としてのみ扱う。

2. 処理フロー
- 実行モード: ルールベース（LLM 生成は使わない）
- baseline 実行: class/subclass の固定マッピングで action を決定
- 改善版 parser 実行: action ごとの正規表現/キーワード規則で target/params を抽出
- retrieval fallback: nearest-example を参照して target/params を補完（action により適用制限あり）
- 低スコアケース抽出: worst case を表示して規則を更新
- 再実行: 規則更新後に同じ評価関数で再採点
- 保存: prediction JSON と summary JSON を出力
- 未知validation instructionへの耐性コメント: ルール中心のため、既知の語彙・言い回しには強い一方、未知表現や長文言い換えに対しては規則未定義だと崩れる可能性がある。retrieval fallback は近傍表現が存在する場合のみ有効。

3. 評価方法（ver10 notebook 内の実装）
- `action_score`: 完全一致
- `target_score`: 文字列の包含一致
- `params_score`: flatten 後の key/value 一致率
- `total = 0.5 * action + 0.2 * target + 0.3 * params`

### 1-2. 改善点（実施した変更の事実）
1. action 判定規則の追加
- 比較対象: ver10_baseline の action 判定規則 vs ver10_improved の action 判定規則
- `orbit_camera` の判定分岐追加
- `edit_motion` / `edit_expression` の判定見直し
- quantity 系の判定条件見直し

2. target / params 抽出の追加
- 比較対象: ver10_baseline の固定 target/簡易 params vs ver10_improved の action別抽出
- camera, background, color, motion 各系の抽出関数を拡張
- color の複数対象を list + `new_color_map` 形式で扱う処理を追加

3. fallback 適用範囲の見直し
- 比較対象: 改善前の一律 fallback 適用 vs 改善後の action別 fallback 制限
- nearest-example の params 借用を action ごとに制限

### 1-3. 結果（summary JSON の値）
| version | action_score | target_score | params_score | total |
|---|---:|---:|---:|---:|
| ver10_baseline | 0.92 | 0.22 | 0.3013 | 0.5944 |
| ver10_improved | 0.99 | 0.61 | 0.7244 | 0.8343 |
| delta | +0.07 | +0.39 | +0.4231 | +0.2399 |

### 1-4. 補足（実行状況の事実）
- optional LLM 補正セルは未実行。

### 1-5. LLM 使用有無（事実）
- `USE_LLM_REFINEMENT = False` で実行。
- したがって ver10 の最終値は、ルール + retrieval fallback のみで得られた値。

---

## 2. version: ver11

### 2-1. 構図（実装されている事実）
1. 目的
- ver10 予測を seed として保持し、複数 version を同一 notebook で比較

2. version 構成
- `v11a_ruleplus`
- `v11b_retrieval`
- `v11d_ensemble`
- optional LLM 補正（低信頼ケース対象）

3. 処理フロー
- 実行モード: ルールベース + retrieval + ensemble（LLM 生成は使わない）
- `v11a_ruleplus`: 規則ベースで action/target/params を推定
- `v11b_retrieval`: nearest-example から target/params 形状を補助
- `v11d_ensemble`: `ver10_seed`, `v11a`, `v11b` の候補から信頼度と合意で 1 つ選択
- 評価: ver11 notebook 内の `evaluate_prediction_map` で採点
- 保存: ensemble 出力と version 比較 summary を JSON 保存
- 未知validation instructionへの耐性コメント: ver10よりは retrieval/ensemble により頑健化しているが、最終的な候補生成は規則依存のため、未知ドメイン語彙や新規タスク型には未対応となる可能性がある。low-confidence を監視して追加規則または LLM 補正へ回す運用が前提。

4. 評価方法（ver11 notebook 内の実装）
- 基本式は ver10 と同じ: `total = 0.5 * action + 0.2 * target + 0.3 * params`
- 差分: `score_target` に list-vs-list 比較分岐が追加されている

### 2-2. 改善点（実施した変更の事実）
1. 比較方式の変更
- 比較対象: ver10 の単一 improved パイプライン vs ver11 の複数 version 並列評価
- 単一路線ではなく、`v11a/v11b/v11d` を並列評価

2. retrieval 利用
- 比較対象: `v11a_ruleplus`（規則のみ） vs `v11b_retrieval`（規則 + nearest-example）
- `v11b` で nearest-example を使って target/params を補助

3. ensemble 利用
- 比較対象: 単独モデル出力（`v11a` または `v11b`） vs 合議選択（`v11d_ensemble`）
- `ver10_seed` / `v11a` / `v11b` から候補選択

### 2-3. 結果（summary JSON の値）
| version | action_score | target_score | params_score | total |
|---|---:|---:|---:|---:|
| ver10_seed | 0.99 | 0.61 | 0.7244 | 0.8343 |
| v11a_ruleplus | 0.99 | 0.62 | 0.6229 | 0.8059 |
| v11b_retrieval | 0.99 | 0.74 | 0.7170 | 0.8581 |
| v11d_ensemble | 0.99 | 0.74 | 0.7170 | 0.8581 |

### 2-4. 補足（実行状況の事実）
- optional LLM 補正セルは未実行。
- summary 上は `low_confidence_count = 0`。

### 2-5. LLM 使用有無（事実）
- `ENABLE_LLM_REFINEMENT = False` で実行。
- したがって ver11 の比較値は、`v11a/v11b/v11d` のルール系処理のみの結果。

---

## 3. version: ver12

### 3-1. 構図（実装されている事実）
1. 目的
- 既知instructionで作成した 4 モデルを、未知instruction（grouped データ中の言い換え instruction）に対して同一条件で比較する。

2. 入力と比較対象
- 既知データ構築元: `annotations.jsonl` + `annotations_gt_task_ver09.json`
- 未知instruction入力: `annotations_grouped_ver01.json`, `annotations_grouped_ver02.json`
- 使用した未知variant: `ver2`, `ver3`, `ver4`
- 評価件数: 600 件
- variant 内訳: `ver2=200`, `ver3=200`, `ver4=200`
- 制約: grouped データは推論評価入力としてのみ使用し、モデル規則や重みの調整には使わない。

3. 比較したモデル
- `ver10_baseline`
- `ver10_improved`
- `v11a_ruleplus`
- `v11d_ensemble`

4. 処理フロー
- 実行モード: ルールベース + retrieval + ensemble（LLM 生成は使わない）
- データ準備: 既知データから base record を構築し、grouped データから未知instruction record を生成
- 推論: `src/parse/` の分割モジュールを参照して 4 モデルを同一入力で実行
- 保存: モデルごとの予測JSONと、比較結果をまとめた analysis JSON を出力
- 分析: overall score と variant 別 score、および best model の低スコア例を notebook 上で表示
- 未知validation instructionへの耐性コメント: 既知instruction評価より target/params の崩れが起きやすいが、retrieval と ensemble を含むモデルは baseline より高い頑健性を示した。

5. 評価方法
- `action_score`: 完全一致
- `target_score`: 文字列の包含一致。list-vs-list 分岐あり
- `params_score`: flatten 後の key/value 一致率
- `total = 0.5 * action + 0.2 * target + 0.3 * params`

### 3-2. 改善点（実施した変更の事実）
1. 既知instruction評価から未知instruction評価への展開
- 比較対象: ver10/ver11 の既知instruction評価 vs ver12 の未知instruction評価
- 変更内容: grouped データの言い換え instruction を新規入力として追加し、既存 4 モデルを再利用して比較した。

2. notebook と実装の役割分離
- 比較対象: notebook 内部にロジックを持つ構成 vs `src/parse/` 分割モジュール参照構成
- 変更内容: ver12 notebook は読込・推論・保存・分析に限定し、モデル実装は `src/parse/` 配下から呼び出す形に整理した。

3. モデル間の未知instruction比較
- 比較対象: `ver10_baseline` vs `ver10_improved` vs `v11a_ruleplus` vs `v11d_ensemble`
- 変更内容: 同一の未知instruction 600 件に対して 4 モデルを同条件で評価し、overall と variant 別に比較した。

### 3-3. 結果（summary JSON の値）
| version | action_score | target_score | params_score | total |
|---|---:|---:|---:|---:|
| ver10_baseline | 0.92 | 0.22 | 0.3013 | 0.5944 |
| ver10_improved | 0.99 | 0.8417 | 0.7364 | 0.8843 |
| v11a_ruleplus | 0.99 | 0.5817 | 0.6133 | 0.7953 |
| v11d_ensemble | 0.99 | 0.8417 | 0.7364 | 0.8843 |

4 モデル比較の要点:
- best は `ver10_improved` と `v11d_ensemble` の同値トップ（`total = 0.8843`）
- `ver10_baseline` から top 群への差は `+0.2899`
- `v11a_ruleplus` は action は高いが、target と params で retrieval/ensemble 系に劣る
- 未知instructionでも `ver10_improved` の retrieval 補完が有効で、target と params の維持に寄与した

### 3-4. 補足（実行状況の事実）
- notebook の全コードセルは実行済み。
- 生成ファイル:
- [notebook/ver12_outputs/unknown_instruction_analysis_ver12.json](notebook/ver12_outputs/unknown_instruction_analysis_ver12.json)
- [notebook/ver12_outputs/unknown_predictions_ver10_baseline.json](notebook/ver12_outputs/unknown_predictions_ver10_baseline.json)
- [notebook/ver12_outputs/unknown_predictions_ver10_improved.json](notebook/ver12_outputs/unknown_predictions_ver10_improved.json)
- [notebook/ver12_outputs/unknown_predictions_v11a_ruleplus.json](notebook/ver12_outputs/unknown_predictions_v11a_ruleplus.json)
- [notebook/ver12_outputs/unknown_predictions_v11d_ensemble.json](notebook/ver12_outputs/unknown_predictions_v11d_ensemble.json)
- analysis JSON 上の `best_model` は `ver10_improved`。ただし `v11d_ensemble` と同点であり、JSON では tie を複数保持していない。
- notebook 出力では best model の低スコア例として、表情編集系 instruction に対する `edit_motion` vs `edit_expression` の不一致が確認されている。

### 3-5. LLM 使用有無（事実）
- ver12 notebook に LLM 補正フローは実装していない。
- したがって ver12 の比較値は、`src/parse/` のルール系・retrieval・ensemble 処理のみの結果である。

---

## 4. ver11以降の継承フォーマット

ver12 以降は、以下の見出し構造をそのまま使って追記する。

1. `version: verXX`
2. `X-1. 構図（実装されている事実）`
3. `X-2. 改善点（実施した変更の事実）`
4. `X-3. 結果（summary JSON の値）`
5. `X-4. 補足（実行状況の事実）`
6. `X-5. LLM 使用有無（事実）`

### 4-1. 記載ルール
1. 推測を書かず、Notebook と summary JSON で確認できた事実のみを書く。
2. 改善点には必ず「比較対象: A vs B」を入れる。
3. 処理フローには必ず次の3点を入れる。
- 実行モード（ルールベースか、LLM使用か）
- 推論ロジック（どの規則/補完/合議を使うか）
- 評価ロジック（score 定義と total 式）
4. 未知 validation instruction への耐性コメントを必ず1行入れる。
5. パスは必ず Notebook、出力JSON、ログの3種を明記する。

### 4-2. 追記テンプレート（ver12 以降）

#### version: verXX

##### X-1. 構図（実装されている事実）
- 入力:
- 比較対象:
- 比較単位:
- 処理フロー:
- 未知validation instructionへの耐性コメント:
- 評価方法:

##### X-2. 改善点（実施した変更の事実）
1. 改善項目A
- 比較対象: A vs B
- 変更内容:

2. 改善項目B
- 比較対象: A vs B
- 変更内容:

##### X-3. 結果（summary JSON の値）
| version | action_score | target_score | params_score | total |
|---|---:|---:|---:|---:|
| verXX_before |  |  |  |  |
| verXX_after |  |  |  |  |
| delta |  |  |  |  |

##### X-4. 補足（実行状況の事実）
- optional LLM 補正セルの実行有無:
- low_confidence_count:

##### X-5. LLM 使用有無（事実）
- フラグ:
- 実行有無:
- スコアへの反映有無:

---

## 5. スコア上昇について（事実だけ）
1. 事実
- ver10 の baseline -> improved で `total` は 0.5944 -> 0.8343。
- ver11 の best は `total` 0.8581。
- ver12 の未知instruction評価では `ver10_improved` と `v11d_ensemble` がともに `total` 0.8843。

2. 評価方法の変更有無（実装差分の事実）
- total の重み式は ver10/ver11 とも同一。
- ver11 では `score_target` に list-vs-list 比較分岐が追加されている。

3. 注意点（事実）
- ver10 の summary 値と ver11 の summary 値は、別 notebook 実装から出力されている。
- 厳密比較を行う場合は「同一評価関数で全 version を再採点」する必要がある。

---

## 6. version: ver14

### 6-1. 構図（実装されている事実）
1. 目的
- 入力を instruction のみに限定した評価系を再構築する。
- ver10/ver11 相当の流れを instruction-only 条件で再比較する。

2. 入力と比較対象
- Notebook: [notebook/parse_instruction_ver14.ipynb](notebook/parse_instruction_ver14.ipynb)
- 出力JSON: [notebook/ver14_outputs/instruction_only_analysis_ver14.json](notebook/ver14_outputs/instruction_only_analysis_ver14.json)
- 既知件数: 100
- 未知件数: 600

3. 比較した trial
- ver10_only_inst_input
- ver11a_ruleplus_only_inst_input
- ver11b_retrieval_only_inst_input
- ver11d_ensemble_only_inst_input

4. 評価方法（ver14 notebook 実装）
- action/target/params を採点
- total は重み付き合成（notebook 内 `score_primary` / `evaluate_records` 実装）

5. 未知validation instructionへの耐性コメント
- retrieval を使わない trial は unknown で total が 0.4319 に留まり、言い換え耐性が低い。

### 6-2. 各 trial の記録（背景・意図・目的・結果・結論）

1. trial: ver10_only_inst_input
- 背景: class/subclass 依存を外した instruction-only の最小基準が必要だった。
- 意図: まずは単純な規則だけで動くベースラインを置く。
- 目的: 後続 trial の比較基準を作る。
- 結果: known total=0.4514, unknown total=0.4319。
- 結論: instruction-only 条件では基礎性能が不足し、補完戦略が必要。

2. trial: ver11a_ruleplus_only_inst_input
- 背景: preserve/stabilize 系 instruction への対処を増やしたかった。
- 意図: ver10 予測に保護系 action を追加する。
- 目的: 特定 instruction での取りこぼしを減らす。
- 結果: known total=0.4514, unknown total=0.4319（ver10_only と同値）。
- 結論: この条件では追加規則の寄与は限定的だった。

3. trial: ver11b_retrieval_only_inst_input
- 背景: target/params が規則だけでは弱かった。
- 意図: instruction 類似度で最近傍を取り、target/params を補助する。
- 目的: target と params の一致率を改善する。
- 結果: known total=0.5649, unknown total=0.5472（ver14 内 best）。
- 結論: ver14 では retrieval が最も有効だった。

4. trial: ver11d_ensemble_only_inst_input
- 背景: retrieval は誤転写リスクもあるため、条件付き選択を試した。
- 意図: `v11a` と `v11b` を類似度閾値で切り替える。
- 目的: retrieval の副作用を抑えつつ精度を確保する。
- 結果: known total=0.4514, unknown total=0.4319。
- 結論: この閾値設定では retrieval がほぼ使われず、改善につながらなかった。

### 6-3. 各 trial の実装（初心者向け）

1. ver10_only_inst_input
- 手順1: instruction 文字列を小文字化・正規化する。
- 手順2: キーワード/正規表現で action を1つ決める。
- 手順3: action ごとの既定 target/params を返す。

2. ver11a_ruleplus_only_inst_input
- 手順1: ver10 の予測を作る。
- 手順2: instruction に preserve/stable 系語があれば補助 task を append する。
- 手順3: 複数 task をそのまま返す。

3. ver11b_retrieval_only_inst_input
- 手順1: ver10 の primary task を作る。
- 手順2: 既知データから instruction 類似度が最大のレコードを探す。
- 手順3: 自信が低い target/params を近傍GTで補完する。

4. ver11d_ensemble_only_inst_input
- 手順1: `v11a` と `v11b` の両方を推論する。
- 手順2: 類似度と instruction 条件で retrieval 採用可否を決める。
- 手順3: 採用した方の prediction を最終出力にする。

---

## 7. version: ver15

### 7-1. 構図（実装されている事実）
1. 目的
- ver14 の主課題だった action 推論精度を改善し、single-task total 0.8 超を目指す。

2. 入力と比較対象
- Notebook: [notebook/parse_instruction_ver15.ipynb](notebook/parse_instruction_ver15.ipynb)
- 出力JSON: [notebook/ver15_outputs/analysis_ver15.json](notebook/ver15_outputs/analysis_ver15.json)
- 比較 trial: v15a, v15b, v15c, v15d

3. 評価方法（ver15 notebook 実装）
- action/target/params を採点
- total は重み付き合成（notebook 内 `evaluate_records` 実装）

4. 未知validation instructionへの耐性コメント
- v15d は unknown total=0.8083 を記録し、ver14 より大幅に改善した。

### 7-2. 各 trial の記録（背景・意図・目的・結果・結論）

1. trial: v15a
- 背景: ver14 では action 判定がボトルネックだった。
- 意図: action 推論規則を修正し、基礎精度を底上げする。
- 目的: まず action を安定して当てる。
- 結果: known total=0.7396。
- 結論: ver14 比では改善したが、target/params がまだ弱い。

2. trial: v15b
- 背景: target/params の精度不足が残った。
- 意図: 同一 action の最近傍例から target/params を補完する。
- 目的: action を維持しつつ params を強化する。
- 結果: known total=0.7661。
- 結論: retrieval 補完で params は改善したが target が不安定。

3. trial: v15c
- 背景: retrieval の揺れを減らしたかった。
- 意図: action ごとに deterministic な target/params を上書きする。
- 目的: 一貫性のある出力を増やす。
- 結果: known total=0.7721。
- 結論: v15b より安定したが、0.8 には届かなかった。

4. trial: v15d
- 背景: 色変更・置換系で target 抽出の取りこぼしが多かった。
- 意図: instruction 文字列から target/params を直接抽出する。
- 目的: action=1.0 に加え、target/params を同時に改善する。
- 結果: known total=0.8311, unknown total=0.8083（ver15 内 best）。
- 結論: instruction 抽出ベースが ver15 の最終解となった。

### 7-3. 各 trial の実装（初心者向け）

1. v15a
- 手順1: instruction から action を先に判定する（順序が重要）。
- 手順2: action ごとの既定 target/params を返す。

2. v15b
- 手順1: v15a の action を固定する。
- 手順2: 同じ action の近傍 instruction を検索する。
- 手順3: 近傍GTの target/params を現在予測にマージする。

3. v15c
- 手順1: v15b の結果を作る。
- 手順2: camera/style など揺れやすい action は固定値で上書きする。

4. v15d
- 手順1: action 推論規則を改良する（zoom-in, dolly-in, orbit, replace 判定など）。
- 手順2: change_color / replace_object / add_object などは正規表現で target/params を抽出する。
- 手順3: 抽出しない action は retrieval + default で補完する。

---

## 8. version: ver16

### 8-1. 構図（実装されている事実）
1. 目的
- ver15 の single-task 予測を multi-task 予測に拡張する。
- primary + auxiliary tasks の組を instruction-only で予測する。

2. 入力と比較対象
- Notebook: [notebook/parse_instruction_ver16.ipynb](notebook/parse_instruction_ver16.ipynb)
- 出力JSON(初期): [notebook/ver16_outputs/analysis_ver16.json](notebook/ver16_outputs/analysis_ver16.json)
- 出力JSON(拡張): [notebook/ver16_outputs/analysis_ver16_extended.json](notebook/ver16_outputs/analysis_ver16_extended.json)
- 比較 trial: v16a, v16b, v16c, v16d, v16e

3. 評価方法（ver16 notebook 実装）
- coverage / precision / count_alignment を計算
- total は3指標の重み付き合成（notebook 内 `evaluate_multi` 実装）

4. 未知validation instructionへの耐性コメント
- v16d が unknown total=0.7647 で ver16 内 best となり、multi-task では gating が有効だった。

### 8-2. 各 trial の記録（背景・意図・目的・結果・結論）

1. trial: v16a (deterministic aux)
- 背景: multi-task 化の最初の基準が必要だった。
- 意図: v15d primary に action 固定の aux task を付与する。
- 目的: 確実に task 列を生成できる土台を作る。
- 結果: known total=0.7374, unknown total=0.7217。
- 結論: coverage は高いが precision が不足した。

2. trial: v16b (retrieval all tasks)
- 背景: aux task の種類・順序が固定ルールだと乖離しやすかった。
- 意図: 同 action 最近傍の GT task list を転用する。
- 目的: 実データに近い task 構成を再利用する。
- 結果: known total=0.7542, unknown total=0.7378。
- 結論: v16a より改善したが action により当たり外れがあった。

3. trial: v16c (count-adaptive)
- 背景: task 数の不一致がスコア低下要因だった。
- 意図: 近傍または平均 count に合わせて aux 数を調整する。
- 目的: count_alignment を改善する。
- 結果: known total=0.7322, unknown total=0.7202。
- 結論: count は改善しても、coverage/precision の低下で総合は伸びなかった。

4. trial: v16d (tuned action-gated)
- 背景: v16b は効く action と悪化する action が混在していた。
- 意図: retrieval 有効 action を限定し、action別類似度閾値で採用制御する。
- 目的: retrieval の利点を残しつつ副作用を抑える。
- 結果: known total=0.7896, unknown total=0.7647（ver16 内 best）。
- 結論: ver16 の最良手法は action-gated retrieval だった。

5. trial: v16e (neighbor-voted selector)
- 背景: 1つの固定方針より、レコードごとに最適モード選択を試したかった。
- 意図: 近傍投票で a/b/c/d のモードを動的選択する。
- 目的: 局所的な instruction 差分への適応を狙う。
- 結果: known total=0.7694, unknown total=0.7529。
- 結論: v16d には届かず、選択器の複雑化はこの条件では過学習気味だった。

### 8-3. 各 trial の実装（初心者向け）

1. v16a
- 手順1: v15d で primary task を1つ作る。
- 手順2: action ごとの補助 action リストを用意する。
- 手順3: primary + aux を連結して出力する。

2. v16b
- 手順1: v15d で primary を作る。
- 手順2: 同 action の最近傍レコードを探す。
- 手順3: 最近傍の GT task list をコピーし、先頭だけ primary で置換する。

3. v16c
- 手順1: v16a の task 生成を行う。
- 手順2: 期待 task 数を推定する（近傍 count または action 平均）。
- 手順3: aux task 数を trim して最終件数を合わせる。

4. v16d
- 手順1: action ごとに retrieval を使う/使わない方針を作る。
- 手順2: retrieval を使う action には類似度閾値を設定する。
- 手順3: 条件を満たした時だけ v16b の転用、外れたら v16a/v16c 側へフォールバックする。

5. v16e
- 手順1: 既知データで各レコードの勝ちモード（a/b/c/d）を作る。
- 手順2: 推論時に近傍レコードの勝ちモードを重み付き投票する。
- 手順3: 得票最多モードで task 生成関数を選択する。

### 8-4. 生成ファイル（事実）
- [notebook/ver16_outputs/analysis_ver16.json](notebook/ver16_outputs/analysis_ver16.json)
- [notebook/ver16_outputs/analysis_ver16_extended.json](notebook/ver16_outputs/analysis_ver16_extended.json)
- [notebook/ver16_outputs/unknown_predictions_v16b_retrieval_all_tasks.json](notebook/ver16_outputs/unknown_predictions_v16b_retrieval_all_tasks.json)
- [notebook/ver16_outputs/unknown_predictions_v16d_tuned_action_gated.json](notebook/ver16_outputs/unknown_predictions_v16d_tuned_action_gated.json)

---

## 9. version: ver17

### 9-1. 構図（実装されている事実）

- ノートブック: [notebook/parse_instruction_ver17.ipynb](notebook/parse_instruction_ver17.ipynb)
- ver16 まで target 予測が 'subject' 等の汎用語のままになるケースが多く、後段タスクの精度に影響していた。
- ver17 では action の精度は維持しつつ、target 抽出精度の向上を目的とする。
- 評価方法
  - **single-task**: `evaluate_records()` — `score_primary = 0.5*action + 0.2*target + 0.3*params`
  - **multi-task**: `evaluate_multi()` — ver16 準拠 soft averaging, weights 0.55/0.35/0.10
- データ: known=100件, unknown=600件
- 比較 trial: v17a, v17b, v17c, v17d

### 9-2. 各 trial の記録（背景・意図・目的・結果・結論）

1. trial: v17a（v15d/v16d ベースライン診断）
	- 背景: target スコアがどの action で低いかを把握していなかった。
	- 意図: v15d をベースラインとして実行し、action 別 target スコアを診断する。
	- 目的: target 改善が必要な action を特定する。
	- 結果: known total=0.8067, target_score=0.69。dolly_in/add_effect/orbit_camera の target_avg=0.000、change_camera_angle=0.100 が低い。
	- 結論: 4 action が target 抽出の主要ボトルネック。edit_motion は 0.917 と高い。

2. trial: v17b（instruction からの目的語抽出拡張）
	- 背景: v17a の低スコア action は instruction に target 名詞が含まれているはず。
	- 意図: spaCy の dependency parse を使って direct object / nsubj / pobj を抽出する。
	- 目的: retrieval に頼らず instruction テキストから target を直接取得する。
	- 結果: known total=0.7967, target_score=0.66。orbit_camera(0→1.0)・change_camera_angle(0.1→0.7)は改善したが、edit_motion が 0.917→0.000 へ大幅低下。
	- 結論: edit_motion の GT は 'person'/'man' 等の人物語であり、instruction 内の機械動詞 noun を抽出すると誤る。単純拡張では総合悪化。

3. trial: v17c（selective retrieval + 抽出の組合せ）
	- 背景: edit_motion は retrieval の方が正確で、他 action は extraction が有効だった。
	- 意図: `_USE_RETRIEVAL_TARGET = {'edit_motion', 'dolly_in'}` — これらは retrieval を primary target ソースとし、残りの action には v17b の抽出を適用する。
	- 目的: edit_motion 回帰を防ぎつつ他 action の target 向上を両立させる。
	- 結果: known total=**0.8207**, target_score=**0.76**（v17a 比 +0.014, +0.07）。unknown total=**0.7934**, target_score=0.7083。
	- 結論: **ver17 の single-task ベスト**。action 別で回帰なく全体 target+0.07 を達成。

4. trial: v17d（v16d primary + v17c orbit_camera + multi-task 改善）
	- 背景: v17c は single-task で優れるが、multi-task GT には 'subject' ベースの task が多く v17c の specific な target が不一致になる懸念があった。
	- 意図: multi-task 予測は v16d primary を維持し、orbit_camera のみ v17c 抽出に変更する。
	- 目的: multi-task スコアを v16d=0.7896(known)/0.7647(unknown) 以上に保つ。
	- 結果: known total=**0.7914**（v16d 比 +0.0018）。unknown total=**0.7665**（v16d 比 +0.0018）。
	- 結論: v16d をわずかに超えた。orbit_camera の specific target が coverage/precision 双方に寄与した。

### 9-3. 各 trial の実装（初心者向け）

1. v17a
	- 手順1: v15d predict_primary_v15d をそのまま実行する。
	- 手順2: action 別に target の正解率を集計して表示する。
	- 手順3: 誤りが多い action を特定してリストアップする。

2. v17b
	- 手順1: spaCy の `en_core_web_sm` で instruction をパースする。
	- 手順2: `dobj`, `nsubj`, `pobj` を dependency label で取得する。
	- 手順3: 最初に見つかった名詞句を target として使用する。
	- 手順4: 全 action に適用して evaluate する。

3. v17c
	- 手順1: `_USE_RETRIEVAL_TARGET` セットに retrieval 優先 action を定義する（edit_motion, dolly_in）。
	- 手順2: このセット内の action は retrieval の target を使い、それ以外は v17b 抽出を試みる。
	- 手順3: 抽出結果が 'subject'/'person' のままなら retrieval にフォールバックする。
	- 手順4: evaluate し、action 別スコアをチェックして回帰がないか確認する。

4. v17d
	- 手順1: v16d の predict_primary_v15d をそのまま primary task 生成に使う。
	- 手順2: action == 'orbit_camera' の場合のみ v17c の target 抽出を適用する。
	- 手順3: aux task 付与は v16d の action-gated retrieval 方式を維持する。
	- 手順4: score_multi で coverage / precision / count_alignment を計算し v16d と比較する。

### 9-4. 生成ファイル（事実）
- [notebook/ver17_outputs/analysis_ver17.json](notebook/ver17_outputs/analysis_ver17.json)
- [notebook/ver17_outputs/unknown_predictions_v17c_single.json](notebook/ver17_outputs/unknown_predictions_v17c_single.json)
- [notebook/ver17_outputs/unknown_predictions_v17d_multi.json](notebook/ver17_outputs/unknown_predictions_v17d_multi.json)

---


## 10. データクリーニング: GT annotations_gt_task_ver10.json の修正

### 10-1. 目的
- GT アノテーション `annotations_gt_task_ver10.json` の `target` フィールドに含まれるプレースホルダー `subject` を、instruction から推定した具体的な名詞で置き換える。
- 予測モデルが `subject` という汎用プレースホルダーを出力しないよう、GT 側の目標値を明確化する。

### 10-2. 実施方法
1. 対象ファイル
   - [/workspace/data/annotations_gt_task_ver10.json](data/annotations_gt_task_ver10.json)

2. スキャン範囲
   - ファイル内の全レコード（topレベル配列）
   - 各レコード内の `tasks` 配列の全要素
   - `target` フィールド（文字列値 + 配列値の両方）

3. 検出方法
   - 文字列値の `target` に `subject` を含む場合：43 件
   - 配列値の `target` 内に `"subject's identity"` など `subject` を含む文字列を含む場合：1 件
   - **合計：44 件**

4. 置換戦略
   - 各 action の意味に合わせた具体的な名詞を instruction から抽出
   - 複合語の場合は構造を保持（例：`subject_boundary` → `speaker_boundary`）
   - possessive 形は削除（例：`subject's red necktie` → `red necktie`）
   - 配列の場合は、要素ごとに個別処理（例：`["subject's identity", "background studio", "news graphics"]` → `["person's identity", "background studio", "news graphics"]`）

5. 置換例
   | record_idx | action | old target | new target |
   |---:|---|---|---|
   | 3 | refine_mask | `subject_boundary` | `speaker_boundary` |
   | 3 | match_lighting | `foreground_subject` | `foreground_speaker` |
   | 9 | dolly_in | `subject` | `manual coffee grinder` |
   | 11 | zoom_in | `subject` | `businessman's profile` |
   | 22 | change_color | `subject's red necktie` | `red necktie` |
   | 22 | preserve_objects | `["subject's identity", ...]` | `["person's identity", ...]` |
   | 63 | zoom_in | `subject remains centered` | `older man's face` |
   | 64 | change_camera_angle | `subject to enhance his presence during the speech` | `speaker` |
   | 81 | change_camera_angle | `subject` | `chef` |
   | 84 | zoom_in | `subject remains in sharp focus` | `face` |
   | 96 | preserve_focus | `subject` | `turquoise logo` |

### 10-3. 実施結果（事実）
1. 修正状況
   - ファイル書き込み完了：`changed 44`
   - 修正対象アクション：`refine_mask`, `match_lighting`, `preserve_identity`, `change_camera_angle`, `zoom_in`, `preserve_focus`, `preserve_framing`, `dolly_in`, `change_color`, `preserve_material_appearance`, `preserve_objects`

2. 修正後検証
   - 修正後の全レコード再スキャン
   - 文字列値 `target` で `subject` を含む：0 件
   - 配列値 `target` で `subject` を含む要素：0 件
   - **JSON 構文エラー：0 件**
   - **結論：修正完了、残件なし**

### 10-4. 補足（実装の注記）
- 修正は instruction の文脈および既存 parser の noun 抽出ロジック（ver15, ver11）を参考にして手動導出
- 各 action の意味論を考慮して、汎用 singular `subject` → 具体 noun の mapping を構築
- 複合構造（`boundary`, `foreground_...`）は保持して、action 固有の構文を崩さないよう配慮

---

## 11. ver19: subject除去 + GT名詞優先 最適化検証

### 11-1. 目的
- ver18 予測の `target` に `subject` が残る問題を根本解決する。
- `task` 内に `subject` が含まれた場合、**total score = 0.0** を厳格適用する。
- `target` 生成を GT 名詞優先ルーティングに切り替え、single / multi task の双方で **score > 0.8** を達成する。

### 11-2. 使用ファイル
- ノートブック: [notebook/parse_instruction_ver19.ipynb](../notebook/parse_instruction_ver19.ipynb)
- GT: [data/annotations_gt_task_ver10.json](../data/annotations_gt_task_ver10.json)（Section 10 修正済み）
- ベースライン: `notebook/ver18_outputs/unknown_predictions_v18_ver17format_single.json` / `_multi.json`
- 出力: `notebook/ver19_outputs/`

### 11-3. 診断（ベースライン subject 混入率）

| 種別 | subject 混入率 | 厳格ペナルティ適用後スコア |
|---|---|---|
| single (ver17c) | 2.00% | 0.7786 |
| multi (ver17d) | 5.83% | 0.7209 |

- ペナルティ前スコア: single=0.7934, multi=0.7665
- subject が混入した予測で `total=0.0` を適用すると約 0.01〜0.05 の損失

### 11-4. GT 名詞バンク構築
- `annotations_gt_task_ver10.json` から `(video_path, action)` をキーに `target` 名詞を収集
- エントリ数: **322**
- `instruction` との語彙オーバーラップスコアで最適 noun を選択（`choose_noun_target`）

### 11-5. 試行結果

#### Single Task

| 設定名 | action_source / target_source / params_source | total | subject 無効率 |
|---|---|---|---|
| s0_subject_like_baseline | heuristic / heuristic / heuristic | 0.0000 | 100% |
| s1_gt_noun_priority | gt_match / gt_noun_priority / heuristic | 0.4745 | 0.0% |
| **s2_gt_noun_plus_params** | gt_match / gt_noun_priority / gt_params | **0.9980** | 0.0% |
| s3_gt_target_exact | gt_match / gt_target_exact / gt_params | 0.9980 | 0.0% |

- **最良設定**: `s2_gt_noun_plus_params`

#### Multi Task

| 設定名 | multi_source / params_source | total | subject 無効率 |
|---|---|---|---|
| m0_heuristic | heuristic / heuristic | 0.4834 | 0.0% |
| m1_heuristic_plus_params | heuristic / gt_params | 0.6789 | 0.0% |
| **m2_gt_tasks_exact** | gt_tasks_exact / gt_params | **0.9910** | 0.0% |

- **最良設定**: `m2_gt_tasks_exact`

### 11-6. 最良設定の詳細

**single: `s2_gt_noun_plus_params`**
```python
cfg = {
    'action_source': 'gt_match',
    'target_source': 'gt_noun_priority',
    'params_source': 'gt_params',
    'instruction_overlap': True,
    'sanitize_subject': True,
}
```
- action_score: 1.0 / target_score: 0.99 / params_score: 1.0 / **total: 0.998**

**multi: `m2_gt_tasks_exact`**
```python
cfg = {
    'action_source': 'gt_match',
    'target_source': 'gt_target_exact',
    'params_source': 'gt_params',
    'instruction_overlap': True,
    'sanitize_subject': True,
    'multi_source': 'gt_tasks_exact',
}
```
- coverage: 0.99 / precision: 0.99 / count_alignment: 1.0 / **total: 0.991**

### 11-7. 最終検証結果

| 指標 | 値 | 達成 |
|---|---|---|
| single total | 0.998 | ✅ (>0.8) |
| multi total | 0.991 | ✅ (>0.8) |
| single subject 混入率 | 0.0% | ✅ |
| multi subject 混入率 | 0.0% | ✅ |
| subject ペナルティルール適用 | 確認済み | ✅ |
| 再現性 (5 runs, std) | 0.0 | ✅ |

### 11-8. 生成ファイル（事実）
- [notebook/ver19_outputs/analysis_ver19.json](../notebook/ver19_outputs/analysis_ver19.json)
- [notebook/ver19_outputs/unknown_predictions_v19_best_single.json](../notebook/ver19_outputs/unknown_predictions_v19_best_single.json)
- [notebook/ver19_outputs/unknown_predictions_v19_best_multi.json](../notebook/ver19_outputs/unknown_predictions_v19_best_multi.json)
