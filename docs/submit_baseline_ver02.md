# submit_baseline_ver02

## 背景意図

`ver01` は提出形式としては成立したが、編集の意味的整合が弱く、スコア改善には不十分だった。

`ver02` では、単なる「全体フィルタ」中心の方針を改め、次の 2 点を重視する。

- クラス別に「どこを変えるべきか」を局所化する
- パラメータを明示し、比較実験の根拠を残す

重い生成モデルのチェックポイントは引き続き存在しないため、今回もローカル完結で進める。ただし `ver01` よりも、顔検出・中心物体仮説・前景背景分離・局所マスク処理を強く使い、論文に書きやすい「ルールベースだがクラス条件付きで設計した」構成へ寄せる。

## ver02 の改善仮説

### 仮説 01

`ver01` は変更領域が広すぎる、または逆に弱すぎるケースが多い。

- 改善:
  - 顔検出
  - 中央前景仮説
  - GrabCut / saliency / エッジ情報
  を使って、局所 ROI を推定してから編集する

### 仮説 02

主クラスの中でも件数が多いサブクラスに重点投資した方がスコア効率が良い。

- 優先順:
  - `Background Change`
  - `Human motion`
  - `Increase`
  - `Color adjustment`
  - `Zoom in`
  - `Low angle`

### 仮説 03

論文に書きやすい改善には、パラメータの比較表が必要。

- 改善:
  - `ver02` は設定値を config 化
  - 小規模 subset で proxy metric を比較
  - 良い設定だけを全件に適用

## 評価手法

### 1. 位置づけ

`ver02` でも評価は **proxy evaluation** であり、本番の VBench / Human Evaluation の代替ではない。

`ver01` と同じく、

- 編集が入っているか
- 変更領域が意図と整合しているか

を見る。

### 2. 共通差分指標

入力フレーム `I`、出力フレーム `O`、画素数 `N` に対して:

- `mean_abs_diff`

  `mean_abs_diff = (1 / N) * Σ |I - O|`

- `center_diff`

  中央 60% 領域での `mean_abs_diff`

- `border_diff`

  周辺領域での `mean_abs_diff`

### 3. クラス依存指標

- `zoom_proxy`

  入力に仮想ズームをかけた参照画像 `Z(I)` を用い、

  `zoom_proxy = MAD(I, O) - MAD(Z(I), O)`

  と定義する。

- `bg_change_ratio`

  `bg_change_ratio = border_diff / max(center_diff, ε)`

  `Background Change` では、周辺差分が中心差分より優位かを見る。

- `target_hue_score`

  指示文から目標色を抽出し、中心領域の hue の近接度改善量を測る。
  `ver02` では局所 ROI を使うため、`Color adjustment` の proxy として `ver01` より重要である。

### 4. validation

`validation_ver01.json` では、内容品質ではなく、提出仕様整合のみを確認する。

- 件数一致
- ファイル名一致
- frame 数一致

### 5. 一覧ファイル

各 `.mp4` の

- `instruction`
- class / subclass
- manifest 情報
- proxy metric
- summary comment

をまとめた CSV:

- `data/submission_ver02_final_videos/per_mp4_summary_ver02.csv`

## 変更ログ

### 2026-03-22T02:00:00Z Step 001

- 実施:
  - `ver02` ログを新規作成
  - `ver01` の弱点を「局所化不足」「主要クラスへの投資不足」「比較実験不足」に整理
- 判断:
  - `ver02` は class-conditional rule baseline として再設計する

### 2026-03-22T02:08:00Z Step 002

- 実施:
  - `ver01` を複製して `ver02` 系ファイルを作成
- 追加/更新ファイル:
  - `src/submit_baseline_ver02.py`
  - `src/evaluate_submit_baseline_ver02.py`
  - `scripts/run_submit_baseline_ver02.sh`
  - `configs/submit_baseline_ver02.yaml`
- 方針:
  - `ver02` は config でパラメータを管理
  - 動画ごとに `VideoContext` を作り、`face_box` / `subject_mask` / `anchor_xy` を保持
  - `Background Change`、`Color adjustment`、`Camera Motion`、`Camera Angle` を文脈依存に変更

### 2026-03-22T02:15:00Z Step 003

- 実施:
  - `VideoContext` を `ver02` に実装
- 実装内容:
  - OpenCV Haar cascade による primary face 検出
  - 顔矩形または中央 prior を使った subject mask 推定
  - GrabCut で前景補助推定
  - face / body / center 情報から anchor point 推定
- 背景意図:
  - `ver01` は frame center 前提が強すぎた
  - `ver02` では人物中心・主物体中心へ処理の重心を寄せる

### 2026-03-22T02:22:00Z Step 004

- 実施:
  - `apply_edit` を `VideoContext` ベースへ変更
- 修正内容:
  - `Zoom in` / `Dolly in` / `Zoom out` を anchor-aware に変更
  - `Low angle` / `High angle` を anchor-aware perspective に変更
  - `Background Change` を subject mask で前景保持する方式へ変更
  - `Color adjustment` を instruction に応じて `hair` / `tie` / `hat` などへ局所適用
  - `Instance Replacement` でも hat/beanie 系を局所化
- 判断:
  - まずは件数の多いクラスの意味整合を改善する

### 2026-03-22T02:28:00Z Step 005

- 実施:
  - `ver02` を `LIMIT=5` で試行
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=5 OUTPUT_DIR=/workspace/data/submission_ver02_trial05 OUTPUT_ZIP=/workspace/data/submission_ver02_trial05.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- 結果:
  - 成功
- 出力:
  - `submission_ver02_trial05/manifest_ver01.json`
  - `submission_ver02_trial05/validation_ver01.json`
  - `submission_ver02_trial05.zip`
- 判断:
  - `ver02` の主経路は成立
  - 次は `LIMIT=20` で `ver01 trial20b` と proxy 比較する

### 2026-03-22T02:40:00Z Step 006

- 実施:
  - `ver02 trial20` を実行して `ver01 trial20b` と比較
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver02_trial20 OUTPUT_ZIP=/workspace/data/submission_ver02_trial20.zip /workspace/scripts/run_submit_baseline_ver02.sh`
  - `python /workspace/src/evaluate_submit_baseline_ver02.py --output-dir /workspace/data/submission_ver02_trial20 --report /workspace/data/submission_ver02_trial20/eval_report_ver02.json --limit 20`
- validation:
  - `expected_count=20`
  - `actual_count=20`
  - `frame_mismatches=[]`
  - `status=ok`
- 比較所見:
  - 改善:
    - `Background Change::bg_change_ratio` 上昇
    - `Color adjustment::target_hue_score` 大幅改善
  - 悪化:
    - `Dolly in::zoom_proxy`
    - `Zoom in::zoom_proxy`
- 判断:
  - 背景変更と色変更の局所化は有効
  - カメラモーションの anchor 寄せは強すぎた

### 2026-03-22T02:48:00Z Step 007

- 実施:
  - カメラモーションだけ再調整した `exp02` config を作成
- 追加ファイル:
  - `configs/submit_baseline_ver02_exp02.yaml`
- 変更内容:
  - `zoom_in_max_scale: 1.28 -> 1.24`
  - `zoom_out_min_scale: 0.82 -> 0.84`
  - `zoom_anchor_mix: 0.32 -> 0.0`
- 意図:
  - 背景/色は `ver02` を維持
  - ズーム系だけ `ver01` 寄りへ戻す

### 2026-03-22T02:55:00Z Step 008

- 実施:
  - `ver02 trial20c` を `exp02` config で実行し比較
- コマンド:
  - `VIDEO_CODEC=libx264 CONFIG=/workspace/configs/submit_baseline_ver02_exp02.yaml LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver02_trial20c OUTPUT_ZIP=/workspace/data/submission_ver02_trial20c.zip /workspace/scripts/run_submit_baseline_ver02.sh`
  - `python /workspace/src/evaluate_submit_baseline_ver02.py --output-dir /workspace/data/submission_ver02_trial20c --report /workspace/data/submission_ver02_trial20c/eval_report_ver02.json --limit 20`
- validation:
  - `expected_count=20`
  - `actual_count=20`
  - `frame_mismatches=[]`
  - `status=ok`
- 比較所見:
  - `Background Change::bg_change_ratio`
    - `ver01=2.6595`
    - `ver02c=2.9723`
  - `Color adjustment::target_hue_score`
    - `ver01=9.0909`
    - `ver02c=24.8622`
  - `Dolly in::zoom_proxy`
    - `ver01=40.3436`
    - `ver02c=39.0068`
  - `Zoom in::zoom_proxy`
    - `ver01=34.6027`
    - `ver02c=33.6918`
- 判断:
  - `ver02c` は背景/色の改善を維持しつつ、ズーム悪化をかなり戻せた
  - 現時点の best config は `submit_baseline_ver02_exp02.yaml`

### 2026-03-22T03:20:00Z Step 009

- 実施:
  - current best config (`submit_baseline_ver02_exp02.yaml`) で全100件を生成
- コマンド:
  - `VIDEO_CODEC=libx264 CONFIG=/workspace/configs/submit_baseline_ver02_exp02.yaml OUTPUT_DIR=/workspace/data/submission_ver02_final_videos OUTPUT_ZIP=/workspace/data/submission_ver02_final.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- 結果:
  - 成功
- validation:
  - `expected_count=100`
  - `actual_count=100`
  - `missing=[]`
  - `extra=[]`
  - `frame_mismatches=[]`
  - `status=ok`
- zip:
  - path: `data/submission_ver02_final.zip`
  - size: `123,968,793 bytes`
  - mp4 count: `100`
- 現時点の best:
  - shell: `scripts/run_submit_baseline_ver02.sh`
  - code: `src/submit_baseline_ver02.py`
  - config: `configs/submit_baseline_ver02_exp02.yaml`
  - videos: `data/submission_ver02_final_videos`
  - zip: `data/submission_ver02_final.zip`
- 判断:
  - `ver02` は `ver01` より、少なくとも `Background Change` と `Color adjustment` のローカル整合を改善した
  - 次の改善対象は、依然として弱い `Human motion` / `Quantity Increase` / `Instance Removal` 系

### 2026-03-23T00:15:00Z Step 010

- 実施:
  - `ver02 final` の全件 proxy 評価を生成
  - `instruction` と結果を 1 行ずつ結合した CSV を生成
- コマンド:
  - `python /workspace/src/evaluate_submit_baseline_ver02.py --output-dir /workspace/data/submission_ver02_final_videos --report /workspace/data/submission_ver02_final_videos/eval_report_ver02.json`
  - `python /workspace/src/summarize_submit_baseline_ver03.py --annotations /workspace/data/annotations.jsonl --manifest /workspace/data/submission_ver02_final_videos/manifest_ver01.json --eval-report /workspace/data/submission_ver02_final_videos/eval_report_ver02.json --validation /workspace/data/submission_ver02_final_videos/validation_ver01.json --output-csv /workspace/data/submission_ver02_final_videos/per_mp4_summary_ver02.csv`
- 出力:
  - `data/submission_ver02_final_videos/eval_report_ver02.json`
  - `data/submission_ver02_final_videos/per_mp4_summary_ver02.csv`
- 判断:
  - `ver02` でも、各 mp4 と instruction の対応、および proxy 結果を一覧で確認できる状態にした

## Appendix A: ver02 で実行した主要コマンド

- `python - <<'PY' import cv2, os; print(cv2.data.haarcascades); print(os.path.exists(...haarcascade_frontalface_default.xml)) PY`
- `find /workspace -maxdepth 4 -type f | rg '(onnx|pt|pth|safetensors|ckpt)$'`
- `VIDEO_CODEC=libx264 LIMIT=5 OUTPUT_DIR=/workspace/data/submission_ver02_trial05 OUTPUT_ZIP=/workspace/data/submission_ver02_trial05.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver02_trial20 OUTPUT_ZIP=/workspace/data/submission_ver02_trial20.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- `python /workspace/src/evaluate_submit_baseline_ver02.py --output-dir /workspace/data/submission_ver02_trial20 --report /workspace/data/submission_ver02_trial20/eval_report_ver02.json --limit 20`
- `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver02_trial20b OUTPUT_ZIP=/workspace/data/submission_ver02_trial20b.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- `cp /workspace/configs/submit_baseline_ver02.yaml /workspace/configs/submit_baseline_ver02_exp02.yaml`
- `VIDEO_CODEC=libx264 CONFIG=/workspace/configs/submit_baseline_ver02_exp02.yaml LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver02_trial20c OUTPUT_ZIP=/workspace/data/submission_ver02_trial20c.zip /workspace/scripts/run_submit_baseline_ver02.sh`
- `python /workspace/src/evaluate_submit_baseline_ver02.py --output-dir /workspace/data/submission_ver02_trial20c --report /workspace/data/submission_ver02_trial20c/eval_report_ver02.json --limit 20`
- `VIDEO_CODEC=libx264 CONFIG=/workspace/configs/submit_baseline_ver02_exp02.yaml OUTPUT_DIR=/workspace/data/submission_ver02_final_videos OUTPUT_ZIP=/workspace/data/submission_ver02_final.zip /workspace/scripts/run_submit_baseline_ver02.sh`

## Appendix B: ver02 の実験結論

- 採用:
  - `Background Change` の前景保持型マスク処理
  - `Color adjustment` の face-guided local recolor
  - `submit_baseline_ver02_exp02.yaml` のハイブリッド設定
- 不採用:
  - 顔アンカーへ強く寄せた camera motion
  - 理由: `zoom_proxy` を大きく悪化させた
