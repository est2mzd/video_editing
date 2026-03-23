# submit_baseline_ver04

## 背景意図

これまでの `ver01` から `ver03` は、`instruction` を 1 本の文字列として扱い、クラスやサブクラスから大まかな編集を選んでいた。

しかし実データの `instruction` には、実際には複数の命令が混在している。

例:

- 主命令:
  - `Change the woman's hair color to a vibrant shade of violet`
- 制約:
  - `Maintain the integrity of all other visual elements`
- 品質命令:
  - `Ensure clean edges`
  - `avoid flickering`

この構造を無視して 1 回の編集に押し込むと、次の問題が起きる。

- 何をどこまで反映したかが曖昧になる
- 評価時に「主命令は反映したが制約は破った」ケースを分けて見られない
- 論文やレポートで「どの指示に対して何が効いたか」を説明しにくい

そのため `ver04` では、instruction を複数の atomic command に分解し、**反映も評価も command 単位** で扱う。

## ver04 の基本方針

### 方針 01: instruction を命令列へ分解する

1 本の instruction を、以下の 3 層に分ける。

- `Edit Command`
  - 実際に見た目を変える命令
  - 例: `change hair color`, `replace background`, `zoom in`
- `Preservation Command`
  - 変えてはいけない対象
  - 例: `preserve identity`, `keep the logo intact`, `maintain background`
- `Quality Command`
  - 出力品質や安定性の制約
  - 例: `avoid flickering`, `clean edges`, `temporally consistent`

### 方針 02: VACE には command 単位で段階投入する

instruction を丸ごと 1 回で投げるのではなく、atomic command ごとに処理を組む。

基本系列:

1. 入力動画を読み込む
2. instruction を command list に分解する
3. `Edit Command` を優先度順に並べる
4. 各 command に対して:
   - 対象領域を推定
   - VACE あるいはルール編集へ変換
   - 中間出力を保存
5. 各段階で preservation / quality を評価する
6. 最終出力を作る

### 方針 03: 評価も command 単位で持つ

最終動画 1 本に対して 1 スコアではなく、

- command ごとの反映スコア
- preservation ごとの維持スコア
- quality ごとの品質スコア

を持つ。

これにより、

- `背景変更は効いた`
- `人物保持は一部崩れた`
- `時間的一貫性は弱い`

のように分解して分析できる。

## 根拠

### 根拠 01: instruction は実際に複文である

`annotations.jsonl` を見ると、ほぼすべての例で

- 主編集命令
- 保持対象
- 品質制約

が同時に書かれている。

つまり instruction は単一ラベルではなく、複数制約付きのミニ仕様である。

### 根拠 02: コンペ評価軸と自然に対応する

コンペの評価軸はおおむね次の 3 つである。

- Instruction Following
- Rendering Quality
- Exclusivity of Edit

これはそのまま、

- `Edit Command` の達成
- `Quality Command` の達成
- `Preservation Command` の達成

に対応する。

したがって command 分解は、評価軸の構造化にも直接つながる。

### 根拠 03: 論文記述に向く

`ver04` の設計は、論文では次のように書ける。

- We decompose each free-form instruction into a set of atomic edit, preservation, and quality commands.
- We apply editing modules sequentially per command.
- We evaluate command fulfillment and preservation separately.

これは `ver01` から `ver03` の「一括編集」より、はるかに論理的で説明しやすい。

## 実装設計

### 1. command parser

新規ファイル案:

- `src/instruction_parser_ver04.py`

役割:

- instruction を atomic command 列へ変換する

出力例:

```json
{
  "video_path": "Xw9Zsc9A924_0_0to138.mp4",
  "commands": [
    {
      "type": "edit",
      "target": "hair",
      "action": "change_color",
      "value": "violet"
    },
    {
      "type": "quality",
      "target": "mask_boundary",
      "action": "clean_edges"
    },
    {
      "type": "quality",
      "target": "temporal",
      "action": "avoid_flicker"
    },
    {
      "type": "preserve",
      "target": "other_visual_elements",
      "action": "keep_unchanged"
    }
  ]
}
```

### 2. command schema

新規ファイル案:

- `configs/command_schema_ver04.yaml`

役割:

- instruction 中のキーワードと command への写像を定義する

例:

- `change ... color` -> `edit/change_color`
- `replace background` -> `edit/change_background`
- `preserve identity` -> `preserve/identity`
- `no flickering`, `temporally consistent` -> `quality/temporal_consistency`

### 3. command planner

新規ファイル案:

- `src/command_planner_ver04.py`

役割:

- command を処理順に並べる

基本順:

1. geometry / camera
2. background
3. object / attribute
4. style
5. quality refinement

理由:

- 先に構図や背景を決めてから局所属性やスタイルをかけた方が破綻しにくい

### 4. command executor

新規ファイル案:

- `src/submit_baseline_ver04.py`

役割:

- command を 1 個ずつ適用し、中間出力を保存しながら最終出力を生成する

処理フロー:

1. 動画読込
2. parser で command list 生成
3. planner で順序決定
4. command ごとに:
   - ROI 推定
   - VACE prompt 生成または rule fallback
   - 中間動画 / 中間フレーム保存
   - command evaluation 実行
5. 全 command を適用したら zip 化

### 5. VACE integration の考え方

VACE が使える場合:

- 1 command ごとに prompt を短文化して渡す
- 例:
  - `Change only the woman's hair color to violet`
  - `Keep the face identity and blouse unchanged`

VACE が使えない場合:

- 既存のルールベース fallback を command 単位で適用する

重要なのは、

- **1 回で全部書かない**
- **短い命令 + 保存制約** の組で段階投入する

ことである。

### 6. command-level evaluation

新規ファイル案:

- `src/evaluate_submit_baseline_ver04.py`

評価出力:

- `per_command_eval_ver04.json`
- `per_mp4_summary_ver04.csv`

評価単位:

- `video_path`
- `command_id`
- `command_type`
- `target`
- `expected_effect`
- `measured_proxy`
- `status`

## 評価理論

### 1. command fulfillment score

各 `Edit Command` に対して、期待する変化方向を定義する。

例:

- `change_color(hair, violet)`:
  - 対象 ROI の hue が violet に近づく
- `zoom_in(face)`:
  - face 周辺が出力でより大きくなる
- `change_background`:
  - 前景外の差分が増える

スコア例:

`S_edit(c) = proxy_after(c) - proxy_before(c)`

### 2. preservation score

保存命令は「変えてはいけない」対象の差分で評価する。

例:

- `preserve identity`
- `preserve logo`
- `keep table unchanged`

スコア例:

`S_preserve(c) = 1 - normalized_diff(ROI_preserve)`

差分が小さいほど高スコア。

### 3. quality score

品質命令は時間一貫性や境界安定性で測る。

例:

- `temporal consistency`
- `clean edges`
- `no flicker`

スコア例:

- temporal:

  `S_temp = 1 - mean_t( |D_t - D_{t-1}| )`

  ここで `D_t` は ROI 差分や mask 境界の時系列特徴量

- edge stability:

  `S_edge = 1 - var_t(edge_proxy_t)`

### 4. 総合スコア

動画単位では、

`S_total = w_edit * mean(S_edit) + w_preserve * mean(S_preserve) + w_quality * mean(S_quality)`

のように持てる。

本番の正式スコアではないが、instruction 構造に整合したローカル比較軸として有用。

## 出力ファイル設計

### 1. parser 出力

- `data/.../parsed_commands_ver04.json`

### 2. command 評価

- `data/.../per_command_eval_ver04.json`

### 3. 各 mp4 集約

- `data/.../per_mp4_summary_ver04.csv`

列案:

- `video_path`
- `instruction`
- `parsed_command_count`
- `edit_command_count`
- `preserve_command_count`
- `quality_command_count`
- `fulfilled_edit_ratio`
- `fulfilled_preserve_ratio`
- `fulfilled_quality_ratio`
- `overall_proxy_score`
- `notes`

## 実装順

### Step A

- parser と schema を作る

### Step B

- subset 20 件で command 分解結果を保存し、人間が見て妥当か確認できる形にする

### Step C

- command 単位 fallback 実行器を実装する

### Step D

- VACE が使える箇所だけ command prompt に差し替える

### Step E

- per-command evaluation を実装する

## 現時点での結論

`ver04` では、

- instruction を仕様書として扱う
- 仕様書を atomic command に分解する
- command 単位で反映と評価を行う

という設計に切り替えるのが妥当である。

これは、現行の `ver01` から `ver03` よりも、

- 改善根拠が明確
- 評価軸と整合
- 論文に書きやすい

という利点がある。

## 変更ログ

### 2026-03-23T00:25:00Z Step 001

- 実施:
  - `ver04` 方針書を新規作成
- 判断:
  - instruction 分解、command 単位実行、command 単位評価の 3 本柱で進める

### 2026-03-23T00:35:00Z Step 002

- 実施:
  - `configs/command_schema_ver04.yaml` を新規作成
  - `src/instruction_parser_ver04.py` を新規作成
  - `src/command_planner_ver04.py` を新規作成
- 実装内容:
  - instruction のキーワードを
    - `edit`
    - `preserve`
    - `quality`
    に分解
  - `vace_prompt` を command ごとに短文化
  - planner で処理順を固定

### 2026-03-23T00:45:00Z Step 003

- 実施:
  - `src/submit_baseline_ver04.py` を新規作成
  - `src/evaluate_submit_baseline_ver04.py` を新規作成
  - `scripts/run_submit_baseline_ver04.sh` を新規作成
- 実装内容:
  - parser 出力を読み、command を順に適用
  - fallback 実装は `ver03` の編集関数を再利用
  - `parsed_commands_ver04.json`
  - `manifest_ver04.json`
  - `per_command_eval_ver04.json`
  - `per_mp4_summary_ver04.csv`
  を出力する構成にした

### 2026-03-23T00:50:00Z Step 004

- 実施予定:
  - `LIMIT=20` で `ver04` を実行
  - command-level evaluation を実行

### 2026-03-23T01:05:00Z Step 005

- 実施:
  - `ver04 trial20` を実装・実行
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver04_trial20 OUTPUT_ZIP=/workspace/data/submission_ver04_trial20.zip /workspace/scripts/run_submit_baseline_ver04.sh`
- 結果:
  - 成功
- 出力:
  - `data/submission_ver04_trial20/parsed_commands_ver04.json`
  - `data/submission_ver04_trial20/manifest_ver04.json`
  - `data/submission_ver04_trial20/per_command_eval_ver04.json`
  - `data/submission_ver04_trial20/validation_ver04.json`
  - `data/submission_ver04_trial20.zip`
- validation:
  - `expected_count=20`
  - `actual_count=20`
  - `missing=[]`
  - `extra=[]`
  - `status=ok`
- 観測:
  - 20 本で command 数は動画ごとに `1` から `7`
  - 例: `DaUJkmBvTKM_2_0to150.mp4` では `edit=1, preserve=2, quality=2` に分解

### 2026-03-23T01:12:00Z Step 006

- 実施:
  - command-level evaluation を実行
- コマンド:
  - `python /workspace/src/evaluate_submit_baseline_ver04.py --input-dir /workspace/data/videos --output-dir /workspace/data/submission_ver04_trial20 --parsed-commands /workspace/data/submission_ver04_trial20/parsed_commands_ver04.json --output-json /workspace/data/submission_ver04_trial20/per_command_eval_scored_ver04.json --output-csv /workspace/data/submission_ver04_trial20/per_mp4_summary_ver04.csv`
- 出力:
  - `data/submission_ver04_trial20/per_command_eval_scored_ver04.json`
  - `data/submission_ver04_trial20/per_mp4_summary_ver04.csv`
- 集計:
  - total commands: `73`
  - `edit`: `20`
  - `preserve`: `26`
  - `quality`: `27`
  - `ok`: `72`
  - `weak`: `1`
- 例:
  - `wyzi9GNZFMU_0_0to121.mp4`
    - `dolly_in` command の `zoom_proxy=11.6104`
  - `8rKYl1CdXCc_5_276to660_scene02.mp4`
    - `parsed_command_count=5`
    - `overall_proxy_score=1.0`
- 判断:
  - instruction を atomic command に分解し、
  - command 単位で反映と評価を持つ
  - という `ver04` の設計は subset 20 件で成立した

## 現在の成果物

- parser:
  - `src/instruction_parser_ver04.py`
- planner:
  - `src/command_planner_ver04.py`
- executor:
  - `src/submit_baseline_ver04.py`
- evaluator:
  - `src/evaluate_submit_baseline_ver04.py`
- schema:
  - `configs/command_schema_ver04.yaml`
- run script:
  - `scripts/run_submit_baseline_ver04.sh`
- subset outputs:
  - `data/submission_ver04_trial20/parsed_commands_ver04.json`
  - `data/submission_ver04_trial20/per_command_eval_scored_ver04.json`
  - `data/submission_ver04_trial20/per_mp4_summary_ver04.csv`
