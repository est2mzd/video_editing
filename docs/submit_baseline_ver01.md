# submit_baseline_ver01

## 背景意図

この作業の目的は、`/workspace/docs/Overview.md`、`/workspace/data/annotations.jsonl`、`/workspace/data/videos` に存在するローカル情報だけを根拠に、コンペ提出物を再現可能な形で作ることです。

今回は「まず必ず提出できること」を最優先にしつつ、`annotations.jsonl` の `selected_class` / `selected_subclass` / `instruction` を使って、完全な identity copy ではなく、少なくとも一部クラスでは指示に沿ったルールベース編集を入れる方針を取ります。

重い生成モデルについては、現状ワークスペース内に VACE 本体チェックポイントが見当たらないため、妄想で「使える前提」にせず、実在するローカル資産だけで完結する OpenCV / ffmpeg ベースラインを先に成立させます。

## 仕様から抽出した固定制約

- 情報源: `/workspace/docs/Overview.md`
- 出力動画のフレーム数は入力動画と厳密に一致させる
- 出力動画のアスペクト比は入力動画を保持する
- 解像度は最低 480p
- 720p 以上が推奨だが、入力を不必要に壊さない方針を優先する

## 実装方針 ver01

- 入力: `annotations.jsonl` の 100 件と `data/videos/*.mp4`
- 出力: 提出用ディレクトリと zip
- 編集方針:
  - `Camera Motion Editing`: ズームや疑似ドリーをルールベースで実装
  - `Camera Angle Editing`: 軽い perspective warp で low/high angle を疑似再現
  - `Style Editing`: 全体フィルタでサブクラスごとの見た目変換
  - それ以外: 無理に壊さず identity を基本にし、必要なら軽微な全体処理のみ
- 検証方針:
  - 件数一致
  - 各動画で frame 数一致
  - 解像度・アスペクト比維持

## 評価手法

### 1. 位置づけ

`ver01` の評価は、コンペ本番の評価器ではなく、ローカルでの改善判断に使う **proxy evaluation** である。

目的は次の 2 点。

- 編集が本当に入ったか
- 変更が意図した領域に寄っているか

### 2. 共通指標

入力フレームを `I`、出力フレームを `O`、画素数を `N` とすると、代表フレーム上で次を計算する。

- `mean_abs_diff`

  `mean_abs_diff = (1 / N) * Σ |I - O|`

  低すぎる場合は、未編集に近いとみなす。

- `center_diff`

  画像中央 60% 領域に限定した `mean_abs_diff`

- `border_diff`

  中央以外の周辺領域に限定した `mean_abs_diff`

### 3. クラス依存指標

- `zoom_proxy`

  入力画像に仮想ズーム変換をかけた参照画像 `Z(I)` を作り、

  `zoom_proxy = MAD(I, O) - MAD(Z(I), O)`

  で定義する。値が大きいほど、出力が「ズーム後の見え方」に近い。

- `bg_change_ratio`

  `bg_change_ratio = border_diff / max(center_diff, ε)`

  背景変更では、人物中心より周辺背景の差分が大きい方が望ましい。

- `target_hue_score`

  指示文から目標色を抽出し、中心領域の hue がその色へどれだけ近づいたかを比較する。  
  `Color adjustment` の局所的な効き具合を見る proxy である。

### 4. validation

`validation_ver01.json` は内容評価ではなく、提出仕様整合だけを見る。

- 件数一致
- ファイル名一致
- frame 数一致

### 5. 一覧ファイル

各 `.mp4` について、

- `video_path`
- `selected_class`
- `selected_subclass`
- `instruction`
- manifest 情報
- proxy metric
- summary comment

を 1 行でまとめた CSV:

- `data/submission_ver01_final_videos/per_mp4_summary_ver01.csv`

## 変更ログ

### 2026-03-22T00:00:00Z Step 001

- 実施:
  - 新規ログ `docs/submit_baseline_ver01.md` を作成
  - `Overview.md` から仕様制約を抽出
  - `annotations.jsonl` の構造確認
- 判断:
  - まずはローカル完結のルールベース提出導線を作る
  - 生成モデル前提の実装はチェックポイント不在のため後回し

### 2026-03-22T00:10:00Z Step 002

- 実施:
  - `src/submit_baseline_ver01.py` を新規作成
  - `scripts/run_submit_baseline_ver01.sh` を新規作成
- 実装内容:
  - `annotations.jsonl` を読み、100件を順番に処理する独立 CLI を追加
  - `Camera Motion Editing`、`Camera Angle Editing`、`Style Editing`、`Visual Effect Editing`、`Attribute Editing/Color adjustment` に対して、OpenCV ベースのルール編集を実装
  - 各動画で入力 frame 数、fps、解像度を取得し、同じ frame 数・同じ解像度で出力する処理を追加
  - 出力 mp4 群を zip 化し、`manifest_ver01.json` を保存するようにした
- 判断:
  - 既存の未完成エントリポイントは触らず、`_ver01` 付きの独立導線で試行錯誤する

### 2026-03-22T00:14:00Z Step 003

- 実施:
  - `LIMIT=2` で `scripts/run_submit_baseline_ver01.sh` を試行
- コマンド:
  - `LIMIT=2 OUTPUT_DIR=/workspace/data/submission_ver01_trial02 OUTPUT_ZIP=/workspace/data/submission_ver01_trial02.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- 結果:
  - 成功
- 出力:
  - `data/submission_ver01_trial02/wyzi9GNZFMU_0_0to121.mp4`
  - `data/submission_ver01_trial02/8rKYl1CdXCc_5_276to660_scene02.mp4`
  - `data/submission_ver01_trial02/manifest_ver01.json`
  - `data/submission_ver01_trial02.zip`
- 判断:
  - 最低限の読み込み、編集、mp4 出力、zip 化の主経路は通った
  - 次は対象件数を増やして、スタイル系サブクラスや ffmpeg 再圧縮まわりの落ち方を確認する

### 2026-03-22T00:16:00Z Step 004

- 実施:
  - OpenCV 実装可否を事前確認
- 結果:
  - `cv2.xphoto.oilPainting` はこの環境では未提供
- 修正:
  - `src/submit_baseline_ver01.py` の `Oil painting` 処理を、`xphoto` が無い場合でも動くフォールバック実装に変更
- 判断:
  - 全件実行前に既知の API 非互換を先に潰した

### 2026-03-22T00:28:00Z Step 005

- 実施:
  - `LIMIT=20` で分岐の多いサブクラスをまとめて実行
- コマンド:
  - `LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver01_trial20 OUTPUT_ZIP=/workspace/data/submission_ver01_trial20.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- 結果:
  - 成功
- 確認できた適用対象:
  - `Dolly in`
  - `Background Change`
  - `Color adjustment`
  - `Ukiyo-e`
  - `Zoom in`
  - `Cyberpunk`
  - `Decoration effect`
  - `Pixel`
  - それ以外の非対応クラスは安全側で通過
- 判断:
  - 主要なクラス分岐を跨いでも処理が落ちないことを確認
  - 次は全100件を本番出力ディレクトリへ生成し、提出 zip と整合検証を行う

### 2026-03-22T00:40:00Z Step 006

- 追加要望:
  - RTX3090 24GB を使って高速化する
  - 出力した `.mp4` と `annotation` がマッチしない場合は、コードやパラメータを修正して再試行する
- 実施:
  - `nvidia-smi` で GPU を確認
  - `ffmpeg -encoders` で `h264_nvenc` を確認
  - 途中出力ディレクトリを点検し、全件完了前なので `annotation` 100件に対して出力 66件時点で未完了であることを確認
- 結果:
  - GPU: `NVIDIA GeForce RTX 3090, 24576 MiB`
  - `ffmpeg` の `h264_nvenc` 利用可
- 修正:
  - `src/submit_baseline_ver01.py` を GPU エンコード対応に変更
  - `validation_ver01.json` を追加し、件数・ファイル名・frame 数の自動照合を実装
  - `h264_nvenc` 失敗時は自動で `libx264` にフォールバックするように変更
- 判断:
  - 今後はユーザー確認なしで GPU 優先、失敗時 CPU フォールバックで継続する

### 2026-03-22T00:48:00Z Step 007

- 実施:
  - GPU 対応版を `LIMIT=5` で再試行
- コマンド:
  - `LIMIT=5 OUTPUT_DIR=/workspace/data/submission_ver01_gpu_trial05 OUTPUT_ZIP=/workspace/data/submission_ver01_gpu_trial05.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- 結果:
  - 成功
- 観測:
  - `h264_nvenc` は各動画で `Broken pipe` となり、実運用では安定しない
  - 自動フォールバックした `libx264` では正常完走
  - `validation_ver01.json` は `expected_count=5`, `actual_count=5`, `missing=[]`, `extra=[]`, `frame_mismatches=[]`, `status=ok`
- 修正:
  - 1本目で GPU 失敗時、残り動画も `libx264` に切り替えるロジックを追加
- 判断:
  - 「GPU優先、失敗時CPU継続」の要件は満たした
  - 本番は `libx264` 固定の方が全体時間を読みやすい

### 2026-03-22T00:55:00Z Step 008

- 実施:
  - `src/evaluate_submit_baseline_ver01.py` を新規作成
  - `submission_ver01_trial20` に対して proxy 評価を実施
- コマンド:
  - `python /workspace/src/evaluate_submit_baseline_ver01.py --output-dir /workspace/data/submission_ver01_trial20 --report /workspace/data/submission_ver01_trial20/eval_report_ver01.json --limit 20`
- 目的:
  - 目視の代わりに、入力と出力の差分から「効いている編集」と「弱い編集」を機械的に分類する
- 主な所見:
  - `Dolly in` / `Zoom in`: `zoom_proxy` が十分に正
  - `Background Change`: `bg_change_ratio` が高く、背景側差分が中心より大きい
  - `Cyberpunk` / `Ukiyo-e`: スタイル変化量は十分
  - `Pixel`: 変化量が弱い
  - `Decoration effect`: エッジ強化が弱い
  - `Instance Replacement` / `Quantity Increase` は変化量が小さい
- 判断:
  - 次は弱いクラスだけ効果を強める

### 2026-03-22T01:02:00Z Step 009

- 実施:
  - `src/submit_baseline_ver01.py` のルール編集を補強
- 修正内容:
  - `Color adjustment` の violet / purple 変換を強化
  - `Pixel` のブロックサイズと量子化を強めて 16-bit 風を明確化
  - `Decoration effect` の glow を強化
  - `Quantity Increase` に対して、中央近傍パッチを複製する簡易増殖処理を追加
  - `Instance Replacement` に対して、指示文中の色語を使った色置換を追加
  - `Instance Insertion` の `microphone` 指示に対して、底面影付きの簡易マイク描画を追加
- 判断:
  - 支持的な proxy を持つクラスは維持し、弱いクラスだけを重点補強した

### 2026-03-22T01:05:00Z Step 010

- 実施中:
  - 強化後の `ver01` を `libx264` 固定で `LIMIT=20` 再試行中
- コマンド:
  - `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver01_trial20b OUTPUT_ZIP=/workspace/data/submission_ver01_trial20b.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- 途中経過:
  - `0` から `13` までは正常出力を確認
- 判断:
  - 完走後に再度 proxy 評価を行い、改善度を見てから全100件へ進む

### 2026-03-22T01:12:00Z Step 011

- 実施:
  - `submission_ver01_trial20b` を完走
  - `validation_ver01.json` を確認
  - `eval_report_ver01.json` を旧 `trial20` と比較
- 結果:
  - validation は `expected_count=20`, `actual_count=20`, `missing=[]`, `extra=[]`, `frame_mismatches=[]`, `status=ok`
- 比較所見:
  - `Color adjustment::target_hue_score` が改善
  - `Decoration effect` の差分量が大幅に増加
  - `Pixel::style_energy` が改善
  - `Increase` / `Instance Insertion` / `Instance Replacement` の差分量も改善
  - `Zoom in` / `Dolly in` / `Background Change` の既存強みは概ね維持
- 判断:
  - 強化後パラメータの方がベースラインとして妥当
  - 最新版で全100件の最終生成へ進む

### 2026-03-22T01:14:00Z Step 012

- 実施中:
  - 最新 `ver01` で全100件の最終生成を開始
- コマンド:
  - `VIDEO_CODEC=libx264 OUTPUT_DIR=/workspace/data/submission_ver01_final_videos OUTPUT_ZIP=/workspace/data/submission_ver01_final.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- 方針:
  - GPU は安定しなかったため `libx264` 固定
  - 生成完了後に `validation_ver01.json` と zip 内容を確認する

### 2026-03-22T01:35:00Z Step 013

- 実施:
  - 全100件の最終生成が完了
  - `validation_ver01.json` を確認
  - zip の内容件数を確認
  - `eval_report_ver01.json` を全件で保存
- 結果:
  - `validation_ver01.json`
    - `expected_count=100`
    - `actual_count=100`
    - `missing=[]`
    - `extra=[]`
    - `frame_mismatches=[]`
    - `status=ok`
  - zip:
    - path: `data/submission_ver01_final.zip`
    - size: `121,017,435 bytes`
    - mp4 count: `100`
  - full proxy report:
    - `data/submission_ver01_final_videos/eval_report_ver01.json`
- 最終成果物:
  - 動画フォルダ: `data/submission_ver01_final_videos`
  - 提出 zip: `data/submission_ver01_final.zip`
  - 実行スクリプト: `scripts/run_submit_baseline_ver01.sh`
  - 本体コード: `src/submit_baseline_ver01.py`
  - 評価コード: `src/evaluate_submit_baseline_ver01.py`
- 判断:
  - 現時点のローカル情報だけで再現可能な提出ベースラインとして成立
  - 今後さらに改善する場合は、`eval_report_ver01.json` を見ながら弱いサブクラスに追加ルールを積む

### 2026-03-23T00:10:00Z Step 014

- 実施:
  - `ver01 final` の `instruction` と結果を 1 行ずつ結合した CSV を生成
- コマンド:
  - `python /workspace/src/summarize_submit_baseline_ver03.py --annotations /workspace/data/annotations.jsonl --manifest /workspace/data/submission_ver01_final_videos/manifest_ver01.json --eval-report /workspace/data/submission_ver01_final_videos/eval_report_ver01.json --validation /workspace/data/submission_ver01_final_videos/validation_ver01.json --output-csv /workspace/data/submission_ver01_final_videos/per_mp4_summary_ver01.csv`
- 出力:
  - `data/submission_ver01_final_videos/per_mp4_summary_ver01.csv`
- 判断:
  - `ver01` でも各 mp4 と instruction の対応を一覧で追える状態にした

## Appendix A: 実行コマンド詳細

### A-01 仕様・入力確認

- `sed -n '1,240p' /workspace/docs/Overview.md`
- `sed -n '1,120p' /workspace/scripts/run.sh`
- `python - <<'PY' ... /workspace/data/annotations.jsonl の先頭5件を json 表示 ... PY`
- `find /workspace/data/videos -maxdepth 2 -type f | sed -n '1,40p'`
- `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /workspace/data/videos/Xw9Zsc9A924_0_0to138.mp4`

### A-02 リポジトリ・既存資産確認

- `rg --files /workspace | sed -n '1,240p'`
- `find /workspace -maxdepth 3 -type f \( -name 'requirements*.txt' -o -name 'pyproject.toml' -o -name 'environment*.yml' -o -name 'setup.py' \) -print`
- `sed -n '1,240p' /workspace/src/utils/vace_executor.py`
- `sed -n '1,240p' /workspace/src/utils/metadata_loader.py`
- `sed -n '1,240p' /workspace/scripts/run_vace_id0.sh`
- `find /workspace -maxdepth 4 -type f | rg '(ckpt|checkpoint|safetensors|bin|pth|pt)$'`

### A-03 ver01 初期試行

- `chmod +x /workspace/scripts/run_submit_baseline_ver01.sh /workspace/src/submit_baseline_ver01.py`
- `LIMIT=2 OUTPUT_DIR=/workspace/data/submission_ver01_trial02 OUTPUT_ZIP=/workspace/data/submission_ver01_trial02.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- `python - <<'PY' import cv2; print('has_xphoto', hasattr(cv2, 'xphoto')); print('has_oilPainting', hasattr(getattr(cv2, 'xphoto', None), 'oilPainting')) PY`
- `LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver01_trial20 OUTPUT_ZIP=/workspace/data/submission_ver01_trial20.zip /workspace/scripts/run_submit_baseline_ver01.sh`

### A-04 GPU / codec 試行

- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`
- `ffmpeg -hide_banner -encoders | rg nvenc`
- `python - <<'PY' ... annotations 100件と /workspace/data/submission_ver01_videos の出力件数比較 ... PY`
- `LIMIT=5 OUTPUT_DIR=/workspace/data/submission_ver01_gpu_trial05 OUTPUT_ZIP=/workspace/data/submission_ver01_gpu_trial05.zip /workspace/scripts/run_submit_baseline_ver01.sh`

### A-05 proxy 評価とパラメータ改善

- `python /workspace/src/evaluate_submit_baseline_ver01.py --output-dir /workspace/data/submission_ver01_trial20 --report /workspace/data/submission_ver01_trial20/eval_report_ver01.json --limit 20`
- `VIDEO_CODEC=libx264 LIMIT=20 OUTPUT_DIR=/workspace/data/submission_ver01_trial20b OUTPUT_ZIP=/workspace/data/submission_ver01_trial20b.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- `python /workspace/src/evaluate_submit_baseline_ver01.py --output-dir /workspace/data/submission_ver01_trial20b --report /workspace/data/submission_ver01_trial20b/eval_report_ver01.json --limit 20`
- `python - <<'PY' ... trial20 と trial20b の eval_report_ver01.json を比較 ... PY`

### A-06 最終生成と検証

- `VIDEO_CODEC=libx264 OUTPUT_DIR=/workspace/data/submission_ver01_final_videos OUTPUT_ZIP=/workspace/data/submission_ver01_final.zip /workspace/scripts/run_submit_baseline_ver01.sh`
- `cat /workspace/data/submission_ver01_final_videos/validation_ver01.json`
- `python - <<'PY' ... /workspace/data/submission_ver01_final.zip の存在・サイズ・mp4件数を確認 ... PY`
- `python /workspace/src/evaluate_submit_baseline_ver01.py --output-dir /workspace/data/submission_ver01_final_videos --report /workspace/data/submission_ver01_final_videos/eval_report_ver01.json`

## Appendix B: 途中観測の要点

### B-01 GPU 関連

- `nvidia-smi` では `NVIDIA GeForce RTX 3090, 24576 MiB, 590.48.01`
- `ffmpeg` は `h264_nvenc` を列挙
- ただし `ver01` の rawvideo pipe 運用では `h264_nvenc` が `Broken pipe` で不安定
- そのため最終版は `libx264` 固定を採用

### B-02 validation の役割

- `validation_ver01.json` は、対象 annotation に対して:
  - 出力件数一致
  - ファイル名集合一致
  - `ffprobe` ベースの frame 数一致
- 条件を満たさない場合は成功扱いにしない

### B-03 proxy 評価で見た改善点

- 強化前から有効:
  - `Zoom in`
  - `Dolly in`
  - `Background Change`
  - `Cyberpunk`
  - `Ukiyo-e`
- 強化後に改善:
  - `Color adjustment::target_hue_score`
  - `Decoration effect` の差分量
  - `Pixel::style_energy`
  - `Quantity Increase`
  - `Instance Insertion`
  - `Instance Replacement`

## Appendix C: ver01 で触ったファイル

- 新規作成:
  - `/workspace/docs/submit_baseline_ver01.md`
  - `/workspace/src/submit_baseline_ver01.py`
  - `/workspace/scripts/run_submit_baseline_ver01.sh`
  - `/workspace/src/evaluate_submit_baseline_ver01.py`
- 逐次更新:
  - `/workspace/docs/submit_baseline_ver01.md`
  - `/workspace/src/submit_baseline_ver01.py`
