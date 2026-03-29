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
