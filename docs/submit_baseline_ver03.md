# submit_baseline_ver03

## 背景意図

`ver02` では `Background Change` と `Color adjustment` の局所整合は改善したが、以下は依然として弱い。

- `Instance Motion Editing / Human motion`
- `Quantity Editing / Increase`
- `Instance Editing / Instance Removal`

`ver03` では、この 3 系統に絞って改善する。方針は「全体を壊さず、局所 ROI に対して時系列一貫な軽量編集を入れる」ことである。

## 改善仮説

### 仮説 01

`Human motion` はほぼ identity なので、評価上ほぼ無変更に見えている可能性が高い。

- 改善:
  - 顔 ROI に対して、時間に沿って口角・目元・顔下部へ軽い変形や明度変化を入れる
  - 手振り系は上半身 ROI に対して局所 warp を加える

### 仮説 02

`Quantity Increase` は複製パッチが単調で、増殖として弱い。

- 改善:
  - 複数パッチ候補を取り、位置・縮尺・ブレンドをパラメータ化する
  - 背景側へ配置して「追加された感」を上げる

### 仮説 03

`Instance Removal` は変化量が小さく、除去に見えにくい。

- 改善:
  - 中心以外の候補領域に対し、周辺色での塗り戻しとぼかしを組み合わせる
  - テキスト由来で白背景や単色背景が示唆される場合は、より積極的に平滑化する

## 評価手法

### 1. 位置づけ

ここで使っている評価は、コンペ本番の VBench / Human Evaluation ではない。ローカルで試行錯誤を回すための **proxy evaluation** である。

目的は次の 2 つ。

- 編集が「まったく変わっていない」状態を避ける
- 変更が「意図した領域」に寄っているかを粗く確認する

### 2. 入力データ

- `annotations.jsonl`
  - `video_path`
  - `selected_class`
  - `selected_subclass`
  - `instruction`
- 入力動画
- 出力動画

### 3. 計算方法

`src/evaluate_submit_baseline_ver03.py` では、各動画の `first / middle / last` の代表フレームを取り、主に最終フレーム同士の差分から proxy metric を計算する。

#### 共通指標

- `mean_abs_diff`
  - 入力と出力の平均絶対差分
  - 低すぎる場合は「ほぼ未編集」の疑い
- `center_diff`
  - 中央 60% 領域の差分
- `border_diff`
  - 周辺領域の差分

#### クラス依存指標

- `zoom_proxy`
  - 入力最終フレームに仮想ズームをかけた参照画像を作り、出力最終フレームとの距離差で評価
  - 正の値が大きいほど「ズームらしさ」が強い
- `bg_change_ratio`
  - `border_diff / center_diff`
  - 背景変更では、中心より周辺の変化が大きい方が望ましい
- `target_hue_score`
  - 指示文から抽出した目標色に対し、中心領域の hue がどれだけ近づいたか
  - `Color adjustment` の proxy

### 4. validation

`validation_ver01.json` では、内容の良し悪しではなく、提出仕様の整合だけを見る。

- 件数一致
- ファイル名一致
- frame 数一致

### 5. 解釈上の注意

- proxy metric は本番スコアそのものではない
- ただし、無変更・過剰変更・変更領域のズレを早く見つけるには有効
- 本ログでは、proxy の改善を「次のパラメータ変更の根拠」として使う

## 変更ログ

### 2026-03-22T03:30:00Z Step 001

- 実施:
  - `ver03` ログを新規作成
  - `ver02` の残課題を 3 クラスに絞った
- 判断:
  - `ver03` は「弱いクラスへ集中投資する版」として進める

### 2026-03-22T03:38:00Z Step 002

- 実施:
  - `ver02` を複製して `ver03` 系ファイルを作成
- 追加/更新ファイル:
  - `src/submit_baseline_ver03.py`
  - `src/evaluate_submit_baseline_ver03.py`
  - `scripts/run_submit_baseline_ver03.sh`
  - `configs/submit_baseline_ver03.yaml`
- 判断:
  - `ver03` は `ver02 exp02` を土台にして、弱点クラスだけ上書きする

### 2026-03-22T03:46:00Z Step 003

- 実施:
  - `Human motion` / `Increase` / `Instance Removal` 向けの新処理を追加
- 修正内容:
  - `Human motion`
    - smile 系: 顔 ROI の口元・下顔面に局所変形と色変化を追加
    - wave 系: 上半身左側 ROI に局所 affine warp を追加
  - `Quantity Increase`
    - anchor 起点の source patch を複数位置へ配置する方式に変更
  - `Instance Removal`
    - 背景ヒントに応じて blur + white/flat fill を行う除去処理を追加
- 背景意図:
  - `ver02` ではこれらのクラスがほぼ identity か、変化量不足だった

### 2026-03-22T03:52:00Z Step 004

- 実施:
  - `ver03 trial20` の初回実行
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver03_trial20 OUTPUT_ZIP=/workspace/data/submission_ver03_trial20.zip /workspace/scripts/run_submit_baseline_ver03.sh`
- 結果:
  - 失敗
- エラー:
  - `apply_human_motion_ver03()` 内で `w` 未定義
- 修正:
  - `frame.shape[:2]` から `h, w` を関数先頭で取得するよう修正
- 判断:
  - 実装バグなので即修正し、同条件で再試行する

### 2026-03-22T04:05:00Z Step 005

- 実施:
  - `ver03 trial20` を再実行し、`ver02c` と比較
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver03_trial20 OUTPUT_ZIP=/workspace/data/submission_ver03_trial20.zip /workspace/scripts/run_submit_baseline_ver03.sh`
  - `python /workspace/src/evaluate_submit_baseline_ver03.py --output-dir /workspace/data/submission_ver03_trial20 --report /workspace/data/submission_ver03_trial20/eval_report_ver03.json --limit 20`
- validation:
  - `expected_count=20`
  - `actual_count=20`
  - `frame_mismatches=[]`
  - `status=ok`
- 比較所見:
  - `Human motion::mean_abs_diff`
    - `ver02c=1.1172`
    - `ver03=1.1396`
  - `Instance Removal::mean_abs_diff`
    - `ver02c=2.2552`
    - `ver03=4.6958`
  - `Increase::mean_abs_diff`
    - `ver02c=3.2582`
    - `ver03=2.8928`
  - `Background Change` / `Color adjustment` / `Zoom` 系はほぼ維持
- 判断:
  - `Human motion` と `Instance Removal` は改善
  - `Increase` は悪化したため、次はこのクラスだけ再調整する

### 2026-03-22T04:10:00Z Step 006

- 実施:
  - `Quantity Increase` を強める再調整
- 修正内容:
  - patch 複製の配置数を増加
  - 各 patch の scale を拡大
  - より背景寄りに分散配置
- 意図:
  - 「追加された個体/物体」の存在感を上げる

### 2026-03-22T04:18:00Z Step 007

- 実施:
  - `Increase` 再調整後の `ver03 trial20b` を実行し比較
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver03_trial20b OUTPUT_ZIP=/workspace/data/submission_ver03_trial20b.zip /workspace/scripts/run_submit_baseline_ver03.sh`
  - `python /workspace/src/evaluate_submit_baseline_ver03.py --output-dir /workspace/data/submission_ver03_trial20b --report /workspace/data/submission_ver03_trial20b/eval_report_ver03.json --limit 20`
- validation:
  - `expected_count=20`
  - `actual_count=20`
  - `frame_mismatches=[]`
  - `status=ok`
- 比較所見:
  - `Human motion::mean_abs_diff`
    - `ver02c=1.1172`
    - `ver03a=1.1396`
    - `ver03b=1.1396`
  - `Increase::mean_abs_diff`
    - `ver02c=3.2582`
    - `ver03a=2.8928`
    - `ver03b=3.2295`
  - `Instance Removal::mean_abs_diff`
    - `ver02c=2.2552`
    - `ver03a=4.6958`
    - `ver03b=4.6958`
- 判断:
  - `ver03b` は `Human motion` と `Instance Removal` の改善を維持
  - `Increase` も `ver02c` に近い水準まで回復
  - 現時点の `ver03` は `trial20b` 相当を採用する

### 2026-03-22T04:28:00Z Step 008

- 実施:
  - 評価手法を文書化
  - 各 mp4 ごとの `instruction` と結果をまとめる CSV を生成
- コマンド:
  - `python /workspace/src/summarize_submit_baseline_ver03.py --annotations /workspace/data/annotations.jsonl --manifest /workspace/data/submission_ver03_trial20b/manifest_ver01.json --eval-report /workspace/data/submission_ver03_trial20b/eval_report_ver03.json --validation /workspace/data/submission_ver03_trial20b/validation_ver01.json --output-csv /workspace/data/submission_ver03_trial20b/per_mp4_summary_ver03.csv`
- 出力:
  - `data/submission_ver03_trial20b/per_mp4_summary_ver03.csv`
- 収録内容:
  - `video_path`
  - `selected_class`
  - `selected_subclass`
  - `instruction`
  - `manifest_status`
  - `codec`
  - `input_frames`
  - `output_frames`
  - proxy metrics
  - `summary_comment`
- 判断:
  - 現状把握用には、この CSV を見れば「各 mp4 に何を指示され、どういう proxy 結果だったか」を確認できる
