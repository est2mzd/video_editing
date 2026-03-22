# Baseline Trial Log

## Trial 001 - System Python で parquet 読み込み確認
- Time: 2026-03-22
- Command: `/usr/bin/python3` で `data/default/train/0000.parquet` を読み込み
- Result: 失敗
- Error: `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`
- Action: `pyarrow` 導入を実施

## Trial 002 - parquet エンジン導入
- Time: 2026-03-22
- Command: `bash scripts/download_pyarrow_system.sh`
- Result: 成功
- Installed: `pyarrow-23.0.1`
- Next: parquet の列構造を再確認し、`scripts/run.sh` 実行経路に接続

## Trial 003 - parquet スキーマ確認
- Time: 2026-03-22
- Command: `/usr/bin/python3` で `data/default/train/0000.parquet` の先頭レコード確認
- Result: 成功
- Findings:
	- rows: `100`
	- columns: `['video']`
	- `video` は dict で `bytes` と `path` を保持

## Trial 004 - データ配置確認
- Time: 2026-03-22
- Command: `find data -maxdepth 4 -type f`
- Result: `data/default/train/0000.parquet` のみ存在
- Findings:
	- prompt を直接保持する別ファイルは現時点で見当たらない
	- まずは固定promptでベースラインを通し、提出形式確認を優先する

## Trial 005 - parquet から入力動画を復元
- Time: 2026-03-22
- Command: `data/default/train/0000.parquet` の1行目 `video.bytes` を `data/work/input_0000.mp4` に書き出し
- Result: 成功
- Output:
	- file: `data/work/input_0000.mp4`
	- size: `1,835,090 bytes`
	- original path(meta): `-UcSvp9UkJk_16_0to152.mp4`

## Trial 006 - run.sh 実行 (初回)
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/work/input_0000.mp4 "Make the scene more rainy"`
- Result: 失敗
- Error:
	- `RuntimeError: Model not found: third_party/VACE/checkpoints/model.pth`
- Analysis:
	- `configs/base.yaml` の `model.model_path` が実在しない
	- 次は `third_party/VACE` 内の実在重み/推論entryを探索して config を合わせる

## Trial 007 - VACE依存不足への暫定フォールバック実装
- Time: 2026-03-22
- Change:
	- `src/model/vace_wrapper.py` に `identity_fallback` を追加
	- `configs/base.yaml` で `model.identity_fallback: true` を有効化
- Intent:
	- 重み未配置でも `run.sh` を最後まで通し、提出可能な最小ベースラインを確保

## Trial 008 - run.sh 実行 (フォールバック有効)
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/work/input_0000.mp4 "Make the scene more rainy"`
- Result: 成功
- Output:
	- `[RESULT]` が返却
	- `logs/<exp_id>/output.mp4` 生成
- Note:
	- 警告: `Model not found: third_party/VACE/checkpoints/model.pth, fallback=identity`
	- 現在のベースラインは「入力動画をそのまま出力」

## Trial 009 - 作業ルール更新（随時ログ追記）
- Time: 2026-03-22
- Request: 試行錯誤は都度 `docs/baseline_trial_log.md` に追記
- Action: 以後は「コマンド実行・修正・判定」ごとに即時記録

## Trial 010 - .venv 環境セットアップと依存導入
- Time: 2026-03-22
- Command:
	- `configure_python_environment` 実行
	- `install_python_packages(["pyyaml", "pandas", "pyarrow", "opencv-python-headless"])`
- Result: 成功
- Purpose:
	- parquet 読み書きと動画制約チェックを `.venv` 側で安定実行するため

## Trial 011 - parquet スキーマ再確認（.venv）
- Time: 2026-03-22
- Command: `/workspace/.venv/bin/python` で `data/default/train/0000.parquet` を検査
- Result: 成功
- Findings:
	- rows: `100`
	- columns: `['video']`
	- `video` は `{'bytes': <bytes>, 'path': <str>}`
	- 先頭 `video.path`: `-UcSvp9UkJk_16_0to152.mp4`

## Trial 012 - run.sh parquet対応の初回テスト（1行）
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_0000.parquet`
- Env: `LIMIT_ROWS=1`, `PYTHON_BIN=/workspace/.venv/bin/python`
- Result: 中断 (`KeyboardInterrupt`)
- Findings:
	- フォールバック時でも `ExperimentRunner` がフレーム分解/リサイズを実行して遅い
	- ベースライン提出目的には「bytes直コピー」高速パスが必要

## Trial 013 - 本番Python方針の更新
- Time: 2026-03-22
- Input from user:
	- `docker/vace/Dockerfile` と `docker/vace/install_vace_library.sh` を使い
	- 本番環境は `/usr/bin/python3` に準備済み
- Action:
	- 以後の実行検証は `/usr/bin/python3` を優先
	- `scripts/run.sh` も本番Pythonに合わせて調整する

## Trial 014 - run.sh を本番Python優先に修正
- Time: 2026-03-22
- File: `scripts/run.sh`
- Change:
	- `PYTHON_BIN` デフォルトを `/usr/bin/python3` に変更
	- `identity_direct_copy` の判定結果を `[INFO]` 出力
- Purpose:
	- Docker本番環境との整合性確保
	- フォールバック高速パスが有効かを実行ログで確認可能にする

## Trial 015 - 本番Python実行で parquet エンジン不足
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_0000.parquet`
- Runtime: `/usr/bin/python3`
- Result: 失敗
- Error:
	- `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`
- Observation:
	- `identity_direct_copy=True` 判定自体は期待通り
	- 失敗点は parquet 読み込み前提ライブラリのみ

## Trial 016 - 本番Pythonへ pyarrow 導入
- Time: 2026-03-22
- Command: `bash scripts/download_pyarrow_system.sh`
- Result: 成功
- Installed: `pyarrow-23.0.1`

## Trial 017 - run.sh parquet 1行テスト（本番Python）
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_0000.parquet`
- Runtime: `/usr/bin/python3`
- Result: 成功
- Output:
	- `data/work/submission_0000.parquet` 生成
	- rows: `100`（`LIMIT_ROWS=1` で先頭1行のみ処理、残りは入力保持）
- Log:
	- `identity_direct_copy=True`
	- `row=0 fast-copy`

## Trial 018 - run.sh 全100行の提出parquet生成（本番Python）
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_full_20260322.parquet`
- Runtime: `/usr/bin/python3`
- Result: 成功
- Output:
	- `data/work/submission_full_20260322.parquet`
	- rows: `100`
- Log:
	- `identity_direct_copy=True`
	- `row=0..99 fast-copy`

## Trial 019 - 生成提出parquetの整合確認
- Time: 2026-03-22
- Command: `/usr/bin/python3` で `data/work/submission_full_20260322.parquet` を検査
- Result: 成功
- Findings:
	- file exists, size: `170,331,186 bytes`
	- columns: `['video']`
	- rows: `100`
	- sample row 0/50/99 で `video.path` と `video.bytes` が正常

## Trial 020 - VACE実編集化に向けた現状調査
- Time: 2026-03-22
- Scope:
	- `configs/base.yaml`
	- `src/model/vace_wrapper.py`
	- `third_party/VACE/*`
- Findings:
	- 既存wrapper想定の `third_party/VACE/inference.py` は存在しない
	- 実CLIは `vace/vace_wan_inference.py` / `vace/vace_pipeline.py`
	- `third_party/VACE/checkpoints/` ディレクトリは未作成
- Decision:
	- 現時点では identity ベースラインを提出可能形で維持
	- 実編集化は重み配置後に wrapper/entry を正式対応する

## Trial 021 - 提出要件更新（zip + <id>.mp4）
- Time: 2026-03-22
- New requirement:
	- オンライン評価は失敗表示でも可
	- 提出は `edited videos` フォルダを zip 化して提出
	- 動画名は `<id>.mp4`
- Action:
	- `scripts/run.sh` を zip提出フローへ拡張
	- parquet出力は補助機能として残し、主導線を zip に切り替える

## Trial 022 - ローカルデータ列の再確認（id/prompt有無）
- Time: 2026-03-22
- Command: `/usr/bin/python3` で `data/default/train/0000.parquet` の列型確認
- Result: 成功
- Findings:
	- columns: `['video']` のみ
	- `id` / `prompt` 列はローカルデータには存在しない
	- zip命名は `video.path` の stem を id 代替で採用

## Trial 023 - run.sh を zip提出対応に拡張
- Time: 2026-03-22
- File: `scripts/run.sh`
- Change:
	- 第4引数を `OUTPUT_TARGET` に変更（`.zip` / `.parquet` / ディレクトリ運用）
	- parquet入力時に各行の動画を `<id>.mp4` で `videos_dir` に保存
	- id列が無い場合は `video.path` stem、それも無い場合は連番を使用
	- `.zip` 指定時は mp4 一式を zip 化して提出物を生成

## Trial 024 - zip提出物の全件生成
- Time: 2026-03-22
- Command: `bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_videos_20260322.zip`
- Runtime: `/usr/bin/python3`
- Result: 成功
- Output:
	- zip: `data/work/submission_videos_20260322.zip`
	- videos dir: `data/work/submission_videos_20260322`
	- 動画数: `100`

## Trial 025 - zip提出物の形式検証
- Time: 2026-03-22
- Command: `/usr/bin/python3` で zip 内容を検査
- Result: 成功
- Findings:
	- zip exists, size: `170,282,998 bytes`
	- zip_files: `100`
	- すべて `.mp4`
	- 先頭/末尾ファイル名は `video.path` 由来の id 命名になっている

## Trial 026 - GIVE Eval ローカル同梱有無の確認
- Time: 2026-03-22
- Command: ワークスペース内で `GIVE Eval` / `give_eval` 関連を探索
- Result: 明示的な評価ツール本体は未検出
- Note:
	- 現在ワークスペースには提出zip生成導線のみを保持
	- GIVE Eval はコンペページの Files から別途取得前提

## Trial 027 - `run.sh` の処理フロー整理（現行）
- Time: 2026-03-22
- Scope: `scripts/run.sh`
- Flow:
	1. 引数を解釈
		- 第1引数: config (`CONFIG_PATH`)
		- 第2引数: 入力 (`INPUT_SOURCE`, mp4 または parquet)
		- 第3引数: prompt (`PROMPT`)
		- 第4引数: 出力ターゲット (`OUTPUT_TARGET`, `.zip` / `.parquet`)
	2. Python実行環境を決定
		- 既定値は `/usr/bin/python3`（`PYTHON_BIN` で上書き可）
	3. YAML config を読み込み、`identity_direct_copy` を判定
		- `identity_fallback=true` かつ `model_path` 不在なら高速コピー経路
	4. `INPUT_SOURCE` が parquet の場合
		- parquet 読み込み（`video` 列必須）
		- 行ごとに `video.bytes` / `video.path` を取得
		- 出力名 `row_id` は `id` 列優先、無ければ `video.path` の stem
		- 生成方式:
			- 高速コピー有効時: `out_bytes = in_bytes`
			- それ以外: `ExperimentRunner` で推論して `output.mp4` を bytes 化
		- 各行を `<row_id>.mp4` で `videos_dir` に保存
		- `OUTPUT_TARGET` 拡張子に応じて最終成果物を生成
			- `.zip`: mp4群を zip 化
			- `.parquet`: `video` 列へ bytes/path を再格納して parquet 化
			- その他: `videos_dir` のみ生成
	5. `INPUT_SOURCE` が mp4 の場合
		- `ExperimentRunner.run(input_video, prompt)` を1本実行
	6. 終了時に `[INFO] Experiment finished` を出力

## Trial 028 - 今回作成した submission の中身整理
- Time: 2026-03-22
- Artifact:
	- zip: `data/work/submission_videos_20260322.zip`
	- folder: `data/work/submission_videos_20260322`
- Validation:
	- zip exists: `True`
	- zip size: `170,282,998 bytes`
	- files in zip: `100`
	- 全ファイル拡張子: `.mp4`
- Filename samples:
	- first5:
		- `-UcSvp9UkJk_16_0to152.mp4`
		- `0Th8ieWYQa0_78_0to142.mp4`
		- `1s9DER1bpm0_10_0to213.mp4`
		- `2Bb8PvQcGiI_3_0to136.mp4`
		- `2LB9h6ZM7Mc_6_0to173.mp4`
	- last5:
		- `xNLcquBRwvc_13_0to140.mp4`
		- `yHPualcu7wE_44_0to203.mp4`
		- `ytCeLiZ_85k_4_0to136.mp4`
		- `zUofaGtC3mY_68_0to190.mp4`
		- `zm3KV2vomiQ_23_0to160.mp4`

## Trial 029 - instruction 不在時の提出ファイル作成方法
- Time: 2026-03-22
- Question: instruction が入力に無い場合、どう作成したか
- Input reality:
	- `data/default/train/0000.parquet` は `video` 列のみ（`bytes`, `path`）
	- `instruction` / `prompt` / `id` 列はローカル入力には存在しない
- Generation policy in `scripts/run.sh`:
	1. prompt は CLI 第3引数（今回は `"Make the scene more rainy"`）を全行共通で使用
	2. ただし `model.identity_fallback=true` かつ `model_path` 不在のため `identity_direct_copy=True`
	3. その結果、各行は推論を回さず `video.bytes` をそのまま出力動画 bytes として利用
	4. 出力ファイル名は `id` 列が無いので `video.path` の stem を採用して `<id>.mp4` 化
	5. 100 本の mp4 をフォルダ化し、zip に固めて提出物を作成
- Practical meaning:
	- 今回の提出物は「instruction 追従編集」ではなく「入力動画の整形提出（identity baseline）」
	- 提出形式チェックを優先して通した段階

## Trial 030 - model_path 置換の着手
- Time: 2026-03-22
- Request:
	- `model_path` のモデルをダウンロードして置き換える
- Action plan:
	- `third_party/VACE/README.md` で公式ダウンロード先を確認
	- 実際に重みを取得
	- `configs/base.yaml` の `model_path` を実在ファイルへ更新

## Trial 031 - VACEモデルの実ダウンロード
- Time: 2026-03-22
- Source: Hugging Face `ali-vilab/VACE-Wan2.1-1.3B-Preview`
- Command: `bash scripts/download_vace_wan21_13b_preview.sh`
- Result: 成功
- Output dir: `third_party/VACE/models/VACE-Wan2.1-1.3B-Preview`
- Download size(log): 約 `19.0GB`

## Trial 032 - model_path の置換
- Time: 2026-03-22
- File: `configs/base.yaml`
- Change:
	- `model.model_path` を `./third_party/VACE/models/VACE-Wan2.1-1.3B-Preview` に更新
- Purpose:
	- `model_path` を実在する重み配置先へ置換

## Trial 033 - model_path 適用で submission 再作成を開始
- Time: 2026-03-22
- Request:
	- ダウンロード済みモデルを使って submission を作り直す
	- instruction を明示する
- Change:
	- `scripts/run.sh` に `instruction.txt` / `instruction_manifest.csv` 出力を追加
	- parquet各行で `instruction -> prompt -> CLI既定` の順で instruction を解決
	- 先頭行から `MAX_MODEL_ROWS` 件は `vace/vace_wan_inference.py` でモデル推論を試行
	- 失敗時は提出壊れ防止のため fallback copy を維持

---

## 補遺: 会話コンテキスト引き継ぎ（2026-03-22）

### 技術スタック（確認済み）

| 項目 | 値 |
|---|---|
| Python runtime | `/usr/bin/python3`（システム Python, Docker本番環境） |
| PyTorch | 2.5.1+cu124 |
| CUDA | 利用可能（`torch.cuda.is_available()=True`） |
| GPU | NVIDIA GeForce RTX 3090 |
| huggingface_hub | 1.7.2（システム Python にインストール済み） |

### VACEモデル（ダウンロード済み）

- HuggingFace リポジトリ: `ali-vilab/VACE-Wan2.1-1.3B-Preview`
- ローカルパス: `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview`
- 主要ファイル:
  - `diffusion_pytorch_model.safetensors`
  - `Wan2.1_VAE.pth`
  - `models_t5_umt5-xxl-enc-bf16.pth`
  - `config.json`
  - `google/` ディレクトリ
- 合計サイズ: 約 19GB

### VACE CLI 引数（確認済み）

推論スクリプト: `third_party/VACE/vace/vace_wan_inference.py`
実行時 cwd: `third_party/VACE`（必須: 相対インポート依存）

確認済み引数（`argparse` 定義より）:

| 引数 | 型 | デフォルト | メモ |
|---|---|---|---|
| `--ckpt_dir` | str | — | モデルディレクトリ （必須） |
| `--src_video` | str | — | 入力動画パス |
| `--prompt` | str | `""` | 編集指示テキスト |
| `--base_seed` | int | 42 | 乱数シード |
| `--sample_steps` | int | 25 | デノイジングステップ数 |
| `--save_dir` | str | `./results` | 出力ディレクトリ |
| `--save_file` | str | `out_video.mp4` | 出力ファイル名（`save_dir` 配下） |
| `--model_name` | str | `vace-1.3B` | モデル識別子 |
| `--size` | str | `480p` | 出力解像度 |

### `scripts/run.sh` の実推論パス（現行コード）

`can_try_direct_vace` が `True` の時、行ごとに以下のサブプロセスを実行:

```python
cmd = [
    "/usr/bin/python3",
    "vace/vace_wan_inference.py",
    "--ckpt_dir", "/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview",
    "--src_video", "data/work/parquet_tmp/input_0000.mp4",
    "--prompt", "Make the scene more rainy",
    "--base_seed", "42",
    "--sample_steps", "25",
    "--save_dir", "data/work/parquet_tmp/vace_run_0000",
    "--save_file", "data/work/parquet_tmp/model_out_0000.mp4",
]
# cwd = /workspace/third_party/VACE
```

### フラグ判定ロジック（configs/base.yaml の現状から）

```
identity_fallback = true  (base.yaml)
model_path        = ./third_party/VACE/models/VACE-Wan2.1-1.3B-Preview  (実在)

→ identity_direct_copy  = identity_fallback AND NOT model_path.exists()
                        = true AND NOT true  = False
→ can_try_direct_vace   = NOT identity_direct_copy AND ckpt_dir.exists() AND script.exists()
                        = true AND true AND true       = True
```

つまり現在 `can_try_direct_vace=True` なので **モデルが実際に呼ばれる**。

### `MAX_MODEL_ROWS` 環境変数

- デフォルト: `1`（テスト用に1行のみ推論）
- 全件推論には `MAX_MODEL_ROWS=100` を指定

### 出力ファイル

| ファイル | 説明 |
|---|---|
| `<videos_dir>/instruction.txt` | 全行共通プロンプトのテキスト保存 |
| `<videos_dir>/instruction_manifest.csv` | 列: `row_index, row_id, instruction, mode, status` |
| `<videos_dir>/<id>.mp4` | 各行の出力動画（`video.path` の stem を id 代替） |
| `<OUTPUT_TARGET>.zip` | 提出用 zip（mp4 一式） |

### 次のアクション（Trial 034 以降）

1. **Trial 034 — VACE推論 1行テスト**

   ```bash
   cd /workspace
   MAX_MODEL_ROWS=1 bash scripts/run.sh \
     configs/base.yaml \
     data/default/train/0000.parquet \
     "Make the scene more rainy" \
     data/work/submission_vace_test.zip
   ```

   期待結果:
   - row=0: `model_direct_vace` モードで `vace_wan_inference.py` 実行
   - row=1〜99: `identity_copy` フォールバック
   - `data/work/submission_vace_test.zip` に 100本の mp4

2. **Trial 035 — 全100行 VACE推論**（Trial 034 成功後）

   ```bash
   MAX_MODEL_ROWS=100 bash scripts/run.sh \
     configs/base.yaml \
     data/default/train/0000.parquet \
     "Make the scene more rainy" \
     data/work/submission_vace_20260322.zip
   ```

3. **Trial 036 — `instruction_manifest.csv` の内容確認**

### 既知のリスク

- GPU VRAM 不足（RTX 3090=24GB, モデル ~7GB 推定→通常は問題ない見込み）
- `third_party/VACE` の Python インポートパスが cwd 依存であるため、`cwd=third_party/VACE` 必須
- 推論エラー時は `fallback_copy` が自動発動し zip 提出が壊れないよう設計済み

## Trial 034 - 現状スナップショット追記
- Time: 2026-03-22
- Scope:
	- `configs/base.yaml`
	- `scripts/run.sh`
	- `docs/baseline_trial_log.md`
- Verified current state:
	- `model.model_path` は `./third_party/VACE/models/VACE-Wan2.1-1.3B-Preview` を指している
	- `model.identity_fallback` は **まだ `true` のまま** だが、`model_path` が実在するため `identity_direct_copy=False` になる設計
	- `scripts/run.sh` には `instruction.txt` と `instruction_manifest.csv` の出力処理が入っている
	- `scripts/run.sh` には `vace/vace_wan_inference.py` を直接呼ぶ `model_direct_vace` 経路が入っている
	- `MAX_MODEL_ROWS` の既定値は `1` で、まず1行だけ実推論を試す挙動になっている
- Current interpretation:
	- コードと設定の整備は完了しており、**次にやるべきことは実推論の実行確認**
	- 直近で確認できる提出物 `data/work/submission_videos_20260322.zip` は、会話経緯上は identity baseline 系の成果物
	- 実VACEモデルを使った再生成 submission の成功記録は、まだこのログには未反映
- Pending:
	- `MAX_MODEL_ROWS=1` での単発推論テスト
	- 成功時の `instruction_manifest.csv` の `mode/status` 確認
	- 全100件 submission の再生成

## Trial 035 - Fallback 免除と複数 instruction 試行体制準備
- Time: 2026-03-22
- 要件整理:
	- Overview より: Instruction Following / Rendering Quality / Exclusivity of Edit の3軸評価
	- UserGuide より: 複数 seed × 複数記述的 prompt での試行が標準
	- 現状の問題: `wan` import 失敗で全行 fallback_copy に陥っている
- 対策方針:
	1. `wan` パッケージをシステム Python に導入（requirements/inference確認予定）
	2. `vace_wan_inference.py` のシステム実行テスト（単独スクリプト）
	3. `scripts/run.sh` 枠で複数 instruction パラメータ対応化
	4. `instruction_manifest.csv` に instruction + seed を明記
- 目標:
	- 各行へ異なる instruction/seed の組み合わせを割り当て
	- 全100行から 5〜10 行程度で異なる指示条件を実推論（"Make the scene more rainy" / "Change the weather" / "Intensify emotions" など）
	- fallback_copy なしの実推論結果を得る

## Trial 036 - 本番環境セットアップ確認と試行錯誤打ち切り
- Time: 2026-03-22
- 発見:
	- Dockerfile と install_vace_library.sh で Wan2.1 は既に `/usr/bin/python3` に `-e install` されていた
	- 無駄な行動:
		- `.venv` に Wan を install （勝手に環境を作った）
		- system Python に重複 install 試行（既に完了済み）
		- test_vace_direct.py で `/workspace/.venv/bin/python` を指定（本来は `/usr/bin/python3`）
- 根本原因の再査:
	- 前回の失敗: `ModuleNotFoundError: No module named 'wan'`
	- 原因は「Wan がインストールされていない」のではなく「subprocess の cwd/path 解決」の可能性
- 反省:
	- Dockerfile の検証なしに環境構築をしてはいけない
	- 本番環境が完備されているなら、試験は本番環境を完全に信頼して実行すること

## Trial 037 - 本番 Python で direct VACE 単行テスト（再実行）
- Time: 2026-03-22
- **環境方針確定**:
	- 本番環境: `/usr/bin/python3` **のみ使用**
	- `.venv` は削除予定（Docker setup で既に Wan2.1 等がシステム Python に完備）
	- 今後新しい仮想環境を作らない
- **scripts/run.sh 既定値確認**:
	- `PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}` → 既に本番 Python が既定値
	- 外部環境変数 `PYTHON_BIN` による上書き可能だが、通常は本番 Python で固定
- **不要ファイル**:
	- `test_vace_direct.py` は試行錯誤時の遺物（削除予定）
- **実行コマンド** (ユーザーが実行すること):
	```bash
	cd /workspace
	MAX_MODEL_ROWS=1 bash scripts/run.sh configs/base.yaml data/default/train/0000.parquet "Make the scene more rainy" data/work/submission_vace_trial037.zip
	```
- **期待結果**:
	- row=0: `model_direct_vace` モード で推論実行 (wan import は本番 Python で利用可能)
	- row=1〜99: `identity_copy` フォールバック
	- `data/work/submission_vace_trial037.zip` 生成
	- `data/work/submission_vace_trial037/instruction_manifest.csv` に `model_direct_vace` と `ok` status が記録される
- **次:結果確認と overview-aligned instruction 群での複数実験**

---

## 補遺: Instruction 設計 (Overview 準拠)

### 評価軸（Overview より）:
1. **Instruction Following** — 編集指示の意図をどれだけ反映できたか
2. **Rendering Quality** — 時間的一貫性と視覚的リアリティ
3. **Exclusivity of Edit** — 指定部分のみを編集、不本意な変更がないか

### 複数 instruction 群 (Wan2.1 UserGuide に従い、記述的 prompt):
- `"Make the scene more rainy"` (現行) — weather/atmosphere 変更
- `"Change the lighting to warm golden hour"` — lighting control
- `"Intensify the emotional tone with dramatic music reflecting in the atmosphere"` — emotional emphasis
- `"Shift the color palette toward cooler tones"` — color grading
- `"Add subtle motion blur to emphasize speed and movement"` — motion control
- `"Enhance detail and sharpness in the foreground while keeping background consistent"` — focus/depth

### テスト運用案:
- **Phase 1 (Trial 037-038)**: 1 行テスト → 確認 → 5 行テスト
- **Phase 2 (Trial 039-040)**: instruction 1 種 × seed 5 種 で 5 行を実推論
- **Phase 3 (Trial 041+)**: instruction 6 種 × seed 1 種 で 6 行を実推論 → 評価対象 100 行 → 最終 submission

### 補足:
- seed: 42 固定だが、`--base_seed` で変更可能
- `sample_steps`: 25 が既定（品質/速度のバランス）
- Wan2.1 ドキュメント: "try again with a different seed" が失敗時の標準アプローチ

## Trial 038 - 現時点サマリー（できたこと / できていないこと / 次にやること）
- Time: 2026-03-22

### 1) できたこと
- `configs/base.yaml` の `model_path` を実在する VACE モデルへ更新済み
	- 取得元: Hugging Face `ali-vilab/VACE-Wan2.1-1.3B-Preview`
	- 保存先: `third_party/VACE/models/VACE-Wan2.1-1.3B-Preview`
	- ダウンロード方法: `bash scripts/download_vace_wan21_13b_preview.sh`（Trial 031 で実施）
	- 背景意図: ダミーパス依存を排除し、実推論へ遷移するため
- `scripts/run.sh` で parquet 入力から zip 提出（`<id>.mp4`）を作る経路は実装済み
	- ただし重要: これは「提出フォーマットを満たす」ための経路であり、編集品質を担保するものではない
	- 背景意図: 形式不備で即失格になるリスクを先に潰すため
- `instruction.txt` / `instruction_manifest.csv` の出力実装済み
	- 明記: 入力 parquet (`data/default/train/0000.parquet`) には `instruction` / `prompt` / `id` 列が存在しない
	- 現在は CLI 第3引数（例: `"Make the scene more rainy"`）を全行共通 instruction として記録している
	- 背景意図: instruction 不在データでも、実行条件を追跡可能にするため
- 本番実行ポリシーを `/usr/bin/python3` 固定に統一
	- 背景意図: Docker 本番環境との差分を消し、再現性を上げるため

### 2) できていないこと
- `fallback_copy` を使わない安定した direct VACE 成功ログが不足
	- 背景意図: VACE を実際に通していない動画は、編集タスクとしての検証価値が低いため
- 「VACE未適用の入力zip化」から脱却できていない
	- 明記: 入力 bytes をそのまま zip 化した提出は、ユーザー指摘の通り本質的価値が低い
	- 背景意図: ベースラインの意義は「VACEで編集した結果」を出すことにあるため
- 複数 instruction の比較実験（Instruction Following / Rendering Quality / Exclusivity）を未実施
	- 背景意図: Overview の評価軸に沿った改善ループが未成立のため

### 3) 次にやること（優先順）
1. `MAX_MODEL_ROWS=1` で単行 direct VACE 実行し、`instruction_manifest.csv` の `mode` が `model_direct_vace` かつ `status=ok` を確認
	- 背景意図: まず「VACEを本当に使えている」事実を1件で確定するため
2. `MAX_MODEL_ROWS=5` で 5 行を実推論し、`fallback_copy` が出る行を特定
	- 背景意図: 失敗行を分離し、全件実行前に原因を潰すため
3. 複数 instruction（少なくとも 3 種）で比較実験し、見た目と破綻率を記録
	- 背景意図: 形式提出ではなく、編集品質を改善するため
4. 条件確定後に全件再生成し、「VACE実適用版 submission」を最終提出候補にする
	- 背景意図: コンペの本旨（テキスト条件付き編集）に合致した提出物へ収束させるため

## Trial 039 - run.sh の実動作と fallback 件数の明示（質問への直接回答）
- Time: 2026-03-22

### Q1. run.sh は VACE を実行しているか？
- 回答: **VACEを実行する分岐はある**。
- 実際の分岐:
	1. `identity_direct_copy=True` の場合: 入力bytesをそのままコピー
	2. それ以外で `can_try_direct_vace=True` かつ `model_try_count < MAX_MODEL_ROWS` の場合: `vace/vace_wan_inference.py` を実行
	3. 上記以外: `ExperimentRunner`（pipeline_runner）を実行
	4. いずれかが例外終了した場合: `fallback_copy`（入力bytesコピー）
- 背景意図: ジョブ全体を止めずに提出形式を維持するため

### Q2. 今回は VACE 実編集になったか？
- 回答: **なっていない（確認できる範囲では全件 fallback）**。
- 証拠: `data/work/submission_videos_20260322_modelretry/instruction_manifest.csv`
	- `model_direct_vace+fallback_copy`: 1件
	- `pipeline_runner+fallback_copy`: 70件
	- `non-fallback`（`status=ok` かつ `mode` に `fallback_copy` なし）: 0件

### Q3. 100件入力に対する内訳（明示）
- 入力行数（dataset）: 100件
- manifest に記録された処理行: 71件
- fallback_copy 件数: 71件
- 非fallback 件数: 0件
- 未記録（処理中断/未到達）: 29件
- 背景意図: 「100件すべて成功した」と誤解しないよう、実処理済み件数を分離して記録するため

### Q4. エラーが出なかったように見える理由
- 回答: 行単位のエラーを `except` で吸収して `fallback_copy` 継続する設計のため、ジョブ全体の終了コードが成功に見えることがある。
- 背景意図: 提出ファイルを壊さないためだが、編集品質の検証には不利。

### 次の是正方針
1. `fallback_copy` が1件でも出たら実行全体を失敗扱いにする（fail-fast モード追加）
2. `MAX_MODEL_ROWS=1` で `mode=model_direct_vace` かつ `status=ok` をまず1件確定
3. その後 5件→100件へ段階拡大

## Trial 040 - run.sh を要件準拠に改修（複数 instruction + fallback禁止 + videos入力対応）
- Time: 2026-03-22
- Change (scripts/run.sh):
	- `INPUT_SOURCE` がディレクトリ（`data/default/train/videos`）の時に `*.mp4` を処理する分岐を追加
	- 複数 instruction を既定セットで定義し、行番号で round-robin 割当
	- `STRICT_NO_FALLBACK` を導入し、既定値 `1`（失敗時は即停止）
	- `fallback_copy` を strict時は禁止（例外を再送出）
- 背景意図:
	- ユーザー要件「VACEで100本処理」「fallback禁止」「エラーを見える化」を満たすため

## Trial 041 - strict実行 100本（v1）失敗
- Time: 2026-03-22
- Command:
	- `bash scripts/run.sh configs/base.yaml data/default/train/videos "Use instruction set" data/work/submission_vace_from_videos.zip`
- Result: 失敗（即停止）
- Error:
	- `FileNotFoundError: third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth`
- Analysis:
	- `cwd=third_party/VACE` で実行しているため、`--ckpt_dir` の相対パス解決が二重化
- Action:
	- `vace_ckpt_dir = Path(config["model"]["model_path"]).resolve()` に修正

## Trial 042 - strict実行 100本（v2）失敗
- Time: 2026-03-22
- Command:
	- `bash scripts/run.sh configs/base.yaml data/default/train/videos "Use instruction set" data/work/submission_vace_from_videos_v2.zip`
- Result: 失敗（即停止）
- Error:
	- `RuntimeError: Error reading data/default/train/videos/0.mp4`
- Analysis:
	- `--src_video` / `--save_dir` / `--save_file` も相対パスで、`cwd=third_party/VACE` 起点で参照失敗
- Action:
	- 上記3引数を `.resolve()` で絶対パス化

## Trial 043 - strict実行 1本（v4）成功
- Time: 2026-03-22
- Command:
	- `timeout 900 bash scripts/run.sh configs/base.yaml data/default/train/videos "Use instruction set" data/work/submission_vace_from_videos_v4.zip 1 1`
	- ※ 第5引数=`LIMIT_ROWS_ARG=1`, 第6引数=`STRICT_NO_FALLBACK_ARG=1`
- Result: 成功
- Evidence:
	- `data/work/submission_vace_from_videos_v4/instruction_manifest.csv`: 2行（header + 1行）
	- row0: `mode=model_direct_vace`, `status=ok`
	- zip生成: `data/work/submission_vace_from_videos_v4.zip`
- 背景意図:
	- 100本投入前に、strictモードで1本成功を先に確定するため

## Trial 044 - strict実行 100本（v5）開始状態
- Time: 2026-03-22
- Command:
	- `bash scripts/run.sh configs/base.yaml data/default/train/videos "Use instruction set" data/work/submission_vace_from_videos_v5.zip 100 1`
- Current status (記録時点):
	- `data/work/submission_vace_from_videos_v5/instruction_manifest.csv`: 0行
	- 生成済みmp4: 0本
	- `instruction.txt` は生成済み
- Note:
	- 長時間実行中のため、この時点では完了/失敗の最終判定未記録

## 運用ルール（この時点以降）
- 実行ステップごとに `docs/baseline_trial_log.md` へ即時追記する
- 記録項目: `Command / Result / Error / Action / 背景意図`

## Trial 045 - v4 を「成功」と判定した基準の明記
- Time: 2026-03-22
- Question:
	- `ver4` はどういう基準で成功としたか
	- `ver4` フォルダで入出力比較がしにくい件

### 成功判定の基準（v4時点）
1. `run.sh` が strict モード（fallback禁止）で `exit code 0` で終了
2. `instruction_manifest.csv` に `mode=model_direct_vace` かつ `status=ok` が記録
3. 出力動画 `0.mp4` が生成される

### 実測（v4）
- 出力フォルダ: `data/work/submission_vace_from_videos_v4`
	- 存在ファイル: `0.mp4`, `instruction.txt`, `instruction_manifest.csv`
- zip: `data/work/submission_vace_from_videos_v4.zip`
	- 含有ファイル: `0.mp4`（1本）
- サイズ比較:
	- 入力: `data/default/train/videos/0.mp4` = `1,835,090` bytes
	- 出力: `data/work/submission_vace_from_videos_v4/0.mp4` = `3,131,549` bytes
	- 結果: 出力サイズが入力と異なるため、少なくとも単純コピーではない

### 「比較しにくい」点について
- 指摘は正しい。
- 現実装は `v4` 出力フォルダ内に **入力動画のコピーを保存していない**。
- そのため、同じフォルダ内で `input vs output` を並べて確認できない。

### 次の改善アクション
- `run.sh` を改修し、デバッグ用に以下を保存する:
	- `videos_dir/_inputs/<id>.mp4`（入力コピー）
	- `videos_dir/<id>.mp4`（出力）
- 背景意図:
	- 目視比較と差分検証を即時に行えるようにし、成功判定をより厳密化するため

## Trial 046 - ver4 出力はコピーか？の検証（ハッシュ/メタデータ）
- Time: 2026-03-22
- Question:
	- `data/work/submission_vace_from_videos_v4/0.mp4` は入力の単純コピーか

### 検証方法
1. `sha256sum` によるバイト一致確認
2. OpenCV で動画メタデータ（frame/fps/width/height）比較
3. ファイルサイズ比較

### 検証結果
- SHA256:
	- input `data/default/train/videos/0.mp4`
		- `7cfaa78a40c176aad0f0cbaf6447937fe3da9eddd5f2c7a58c09deac528ec00c`
	- output `data/work/submission_vace_from_videos_v4/0.mp4`
		- `3e8609b6c99574bf73f2c609f6f47e5f2c1159db221d5f313c604f4c68805895`
	- 判定: **不一致**（バイト単位で別ファイル）

- 動画メタデータ:
	- input: `125 frames`, `25.0 fps`, `1920x1080`
	- output: `81 frames`, `16.0 fps`, `832x480`
	- 判定: **フレーム数/FPS/解像度が変化**

- ファイルサイズ:
	- input: `1,835,090 bytes`
	- output: `3,131,549 bytes`
	- 判定: **単純コピーではない**

### 解釈
- `ver4` の `0.mp4` は「入力そのままコピー」ではない。
- ただし、出力仕様が入力から変わっている（frame数・fps・解像度）ため、コンペ要件の「同フレーム数維持」観点では要注意。

### 背景意図
- 「見た目が似ている」主観ではなく、再現可能な客観指標（ハッシュ/メタデータ）で成否を判定するため。

## Trial 047 - Overview.md基準で見た ver4 出力の「あるべき姿」
- Time: 2026-03-22
- Source:
	- `docs/Overview.md` の Task Definition / Output Specifications

### 1) コンペ仕様から導く「あるべき姿」
1. Instruction Following（指示追従）
	- テキスト指示の意味を動画編集結果に反映していること
2. Rendering Quality（画質・時間一貫性）
	- 時系列で破綻（ちらつき・ドリフト）が少なく、自然な見た目であること
3. Exclusivity of Edit（編集の局所性）
	- 指示対象以外を不要に改変しないこと
4. Frame制約（必須）
	- **出力フレーム数は入力と厳密に同じ**
5. Resolution制約
	- 最低480p以上（推奨720p以上）
6. Aspect Ratio制約（必須）
	- 入力と同じアスペクト比を維持（歪み/不適切なクロップは減点）

### 2) ver4 実測との対応
- 入力 `0.mp4`: `125 frames`, `25.0 fps`, `1920x1080`
- 出力 `ver4/0.mp4`: `81 frames`, `16.0 fps`, `832x480`

評価:
- Frame制約: **未達**（125 → 81）
- Resolution制約: **形式上は達成**（480p系）
- Aspect Ratio制約: **達成**（16:9 を維持）
- Instruction Following / Rendering Quality / Exclusivity:
	- 仕様上は重要評価項目だが、現時点ログだけでは定量未評価（目視/評価指標の別途確認が必要）

### 3) ver4 に対する仕様準拠ゴール（明文化）
- Goal-A: 入出力フレーム数を一致させる（最優先）
- Goal-B: 入力アスペクト比を維持したまま、480p以上で安定出力
- Goal-C: 指示差分が視認でき、かつ不要改変が少ない出力にする

### 4) 次アクション（仕様準拠のため）
1. run後に自動検証を追加:
	- `input_frames == output_frames` を必須チェック
	- `input_aspect_ratio == output_aspect_ratio` を必須チェック
2. 不一致時は strict 失敗にする（提出不可扱い）
3. 条件を満たす推論設定（size/frame周り）へ調整し再実行

### 背景意図
- 「動いたかどうか」ではなく「コンペ仕様を満たすかどうか」で成否を判定するため。

## Trial 048 - 画像サイズ・framerate を変えない編集は GPU メモリ的に難しいか（考察）
- Time: 2026-03-22
- Question:
	- 画像サイズや framerate を変えずに VACE 編集したいが、GPU メモリ上の制約が主因で難しいのか

### 結論（現時点）
- **「難しい可能性が高い」が、完全に不可能ではない**。
- 現在の ver4 実測（`1920x1080@25fps` 入力 → `832x480@16fps` 出力）は、
	- モデル側の既定解像度（`--size 480p`）
	- 時間方向の内部サンプリング/再エンコード
	の影響が大きく、結果として frame/fps が変化している可能性が高い。

### なぜメモリが厳しくなるか
1. 動画拡散モデルのメモリは、概ね「空間解像度 × フレーム数 × 中間特徴量数」に比例して増える。
2. 1080p を 480p に対して面積比で見ると、
	- $1920\times1080 / (832\times480) \approx 5.2$
	- 空間だけで約5倍以上の負荷増が起きうる。
3. さらに 25fps の長尺をそのまま処理すると、時間方向のトークン/潜在表現が増え、VRAM を圧迫する。
4. RTX 3090 (24GB) では、1.3B クラスでも「1080p・元fps維持・十分なstep数」を同時に満たす設定は厳しくなりやすい。

### ただし「メモリだけ」が原因とは限らない
- `vace_wan_inference.py` の既定値が `--size 480p` なので、設定起因で解像度が下がる。
- 出力動画の fps は、推論後の保存実装（writer設定）や内部フレーム抽出方針の影響でも変化する。
- つまり、現象は
	- メモリ制約
	- 推論スクリプトの既定仕様
	の複合で起きている可能性が高い。

### 実務的な対応方針（優先順）
1. まず「仕様必須項目」を自動検証で固定
	- `input_frames == output_frames`
	- `input_aspect == output_aspect`
	- 不一致時は strict failure
2. 推論設定の見直し
	- `--size` を可能な限り上げる（VRAM監視しつつ段階的に）
	- `sample_steps` や処理バッチ相当を調整し、メモリ余裕を作る
3. どうしても元fps/元解像度で直接推論できない場合の二段構え
	- 推論は許容解像度で実施
	- その後、後処理で「フレーム数一致」「fps一致」「解像度一致」に再整形
	- ただし後処理は画質/時間一貫性に副作用があり、最終評価で要検証

### 今回の判断
- 「GPUメモリ上、難しいのか？」への回答は **Yes 寄り**。
- ただし、現段階では
	- 推論器の既定値依存（`480p`）
	- 書き出し仕様依存（`16fps`化）
	も混在しているため、純粋に VRAM だけに帰因はできない。
- 次の実験は「設定要因」と「メモリ要因」を分離して記録する。

### 背景意図
- frame/size が変わる原因を「モデル性能不足」と短絡せず、
	設定・実装・GPU制約を切り分けて、仕様準拠の改善ループを作るため。

## Trial 049 - 現在使っているモデル設定の変数一覧（説明・根拠・結果）
- Time: 2026-03-22
- Question:
	- 現在の実行で使っているモデル設定変数は何か
	- それぞれの説明と根拠（どこで効くか）を明示したい
	- その結果、出力がどうなったかも記録したい

### 1) 変数一覧（現在値 / 説明 / 根拠）

| 変数 | 現在値 | 説明 | 根拠（利用箇所） |
|---|---|---|---|
| `model.vace_repo` | `./third_party/VACE` | direct VACE 実行時の作業ディレクトリ基準 | `scripts/run.sh` で `cwd=str(vace_repo)` として `subprocess.run(cmd, cwd=str(vace_repo))` |
| `model.model_path` | `./third_party/VACE/models/VACE-Wan2.1-1.3B-Preview` | 利用するチェックポイントの場所 | `scripts/run.sh` で `vace_ckpt_dir=Path(config["model"]["model_path"]).resolve()`→`--ckpt_dir` に渡す |
| `model.identity_fallback` | `true` | モデル未配置時に入力コピーへ切替える安全弁 | `scripts/run.sh` の `identity_direct_copy = identity_fallback and not model_path.exists()` |
| `inference.seed` | `42` | 乱数シード（再現性） | `scripts/run.sh` で `--base_seed` に渡す |
| `inference.steps` | `25` | サンプリングステップ数（品質/速度） | `scripts/run.sh` で `--sample_steps` に渡す |
| `MAX_MODEL_ROWS` (env) | 直近 strict 実行は `1` | 何行を direct VACE で試すか | `scripts/run.sh` で `max_model_rows = int(os.getenv("MAX_MODEL_ROWS", "1"))` |
| `STRICT_NO_FALLBACK_ARG` (CLI第6引数) | 直近 strict 実行は `1` | 失敗時に fallback せず即停止 | `scripts/run.sh` の `strict_no_fallback = (strict_arg == "1")` と `except` 分岐 |

### 2) VACE 側の「未指定でも効く既定値」

| 変数（VACE引数） | 現在の実効値 | 説明 | 根拠（利用箇所） |
|---|---|---|---|
| `--model_name` | `vace-1.3B` | 使用モデル種別 | `vace_wan_inference.py` の `argparse` 既定値 |
| `--size` | `480p` | 出力解像度プリセット | `vace_wan_inference.py` の `argparse` 既定値（run.sh から未上書き） |
| `--frame_num` | `81` | サンプル対象フレーム数 | `vace_wan_inference.py` の `argparse`/`validate_args` |
| `--base_seed` | `42` | seed | `run.sh` が上書きして渡す |
| `--sample_steps` | `25` | step数 | `run.sh` が上書きして渡す |

補足:
- `model.entry_script` は `base.yaml` に存在するが、現行 `scripts/run.sh` の direct VACE 経路では参照していない（実際は `vace/vace_wan_inference.py` を固定呼び出し）。

### 3) その結果どうなったか（実測）
1. direct VACE 自体は起動・完走できる状態になった。
	- 根拠: Trial 043 で `mode=model_direct_vace`, `status=ok` を記録。
2. ただし出力仕様は入力と一致していない。
	- 入力: `1920x1080`, `25fps`, `125 frames`
	- 出力(ver4): `832x480`, `16fps`, `81 frames`
	- 根拠: Trial 046 のハッシュ/メタデータ比較。
3. 解釈:
	- 現在の設定群では「VACE編集の実行」は達成。
	- 一方で「入力と同一 frame/fps/解像度維持」は未達。
	- 既定 `--size=480p` と `--frame_num=81` が結果に強く影響している可能性が高い。

### 背景意図
- 変数の実効値と結果を 1 箇所に固定し、
	次の調整（frame一致・fps一致・解像度一致）を根拠付きで進めるため。

## Trial 050 - Frame数一致を死守する戦略（metadata.csv 活用）
- Time: 2026-03-22
- Question:
	- VACE は frame 数の一致が絶対条件
	- これを死守するにはどうしたらよいか
	- metadata.csv をどう活用すべきか

### 現状の問題
- 入力フレーム数: 125（例: 0.mp4)
- 出力フレーム数: 81（VACE 既定 `--frame_num=81`）
- **原因**: `scripts/run.sh` から `--frame_num` を指定していないため、VACE側の既定値が使われている

### metadata.csv の構成（確認済み）
```
video_path, frame, fps, duration, width, height
/workspace/data/default/train/videos/0.mp4, 125, 25.0, 5.0, 1920, 1080
/workspace/data/default/train/videos/1.mp4, 125, 25.0, 5.0, 1920, 1080
/workspace/data/default/train/videos/2.mp4, 120, 23.976, 5.005005005005005, 1920, 1080
```
- 各行に frame 情報が記録されている
- フレーム数は 120, 125, 150 など複数バリエーション

### Frame一致を死守する戦略

#### 戦略1: 事前にメタデータから frame 数を読み取り、--frame_num として VACE に渡す

**実装方針**:

1. `scripts/run.sh` で、ディレクトリ入力時に `metadata.csv` を並行読み込み
2. 各 mp4 ループで対応する frame 数を辞書/df にマップ
3. VACE 実行コマンドに `--frame_num={input_frame_count}` を追加

**実装例**:
```python
# metadata.csv の読み込み
metadata_df = pd.read_csv("data/default/train/videos/metadata.csv")
frame_map = dict(zip(metadata_df['video_path'], metadata_df['frame']))

# ループ内で利用
for src_mp4 in video_files:
    input_frame_count = frame_map.get(str(src_mp4.resolve()), 81)  # デフォルト81
    cmd = [
		"/usr/bin/python3",
		"vace/vace_wan_inference.py",
		"--ckpt_dir", "/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview",
		"--src_video", str(src_mp4.resolve()),
		"--prompt", "Make the scene more rainy",
		"--base_seed", "42",
		"--sample_steps", "25",
        "--frame_num", str(input_frame_count),  # ← 追加：入力フレーム数を指定
		"--save_dir", "/workspace/data/work/parquet_tmp/vace_run_0000",
		"--save_file", "/workspace/data/work/parquet_tmp/model_out_0000.mp4",
    ]
```

**期待される効果**:
- 出力フレーム数が入力と一致するようになる
- Trial 046 の問題（125 → 81）が解決予定

#### 戦略2: 後処理で frame 数を再調整する（バックアップ）

万が一モデル内部で フレーム数が変更されてしまう場合の対策:

1. 推論後の出力 mp4 を OpenCV で読み込み
2. 実際のフレーム数を計測
3. 不一致なら「フレーム補間」または「フレームカット」で入力フレーム数に調整
4. ただし「画質/時間一貫性が劣化する可能性あり」なため、まず戦略1を優先

**実装例（超簡略）**:
```python
import cv2
cap = cv2.VideoCapture(output_mp4)
actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if actual_frames != input_frame_count:
	raise RuntimeError(f"frame mismatch: expected={input_frame_count}, got={actual_frames}")
```

### 次のアクション（優先順）

1. **Trial 051**: `scripts/run.sh` を改修して `--frame_num` を動的に指定
	- metadata.csv の読み込み実装
	- video_path ← frame 数のマッピング実装
	- VACE 呼び出し時に `--frame_num={計算値}` を追加

2. **Trial 052**: 改修後に v6 として 1本テスト実行
	- 出力フレーム数が入力と一致するか確認
	- `md` で結果を記録

3. **Trial 053**: frame一致確認後、5本テスト
	- 複数フレーム値（120/125/150）でも動作確認

4. **Trial 054**: 全件実行 → コンペ仕様適合の判定

### 背景意図
- 「フレーム数の一致」は競技仕様の必須要件（Overview より）
- これまで既定値任せだったため意図的に制御する
- metadata.csv は既に入力に同梱されているため、これを活用すればコスト低い

## Trial 051 - 代表的 instruction CSV を作成・毎回使いまわす
- Time: 2026-03-22
- Objective:
	- 変化を視認しやすい instruction を 6 種類定義
	- 各 video_id に対応する instruction を CSV で固定化
	- 毎回同じ instruction セットで実験を繰り返す（再現性向上）

### 実装済み

**ファイル**: `/workspace/data/default/train/videos/test_instructions.csv`

**構成**:
- 列: `video_id, instruction`
- 行数: 101 行（header + 100 data）

**6 カテゴリ（round-robin 配分）**:

| ID | Category | Count | Example Instruction |
|---|---|---|---|
| 0 | Background Change | 17 | "Change the background to a vibrant sunset cityscape while keeping the main subject unchanged." |
| 1 | Color Change | 17 | "Shift the entire color palette to warm golden tones with enhanced saturation." |
| 2 | Count Change | 17 | "Increase the count of objects in the scene by adding three more similar items to the foreground." |
| 3 | Environmental Change | 17 | "Add heavy fog and mist to create an atmospheric environmental effect." |
| 4 | Instance Insertion | 16 | "Insert a large glowing orb floating in the center of the scene." |
| 5 | Style Change | 16 | "Transform the video style to look like an oil painting with visible brush strokes." |

**分布**（round-robin）:
- video_id % 6 == 0: Background Change
- video_id % 6 == 1: Color Change
- video_id % 6 == 2: Count Change
- video_id % 6 == 3: Environmental Change
- video_id % 6 == 4: Instance Insertion
- video_id % 6 == 5: Style Change

### 次の改修内容（Trial 052 以降）

`scripts/run.sh` を以下のように改修：

1. **metadata.csv の読み込み**
	- frame 数マッピングを取得

2. **test_instructions.csv の読み込み**
	- video_id → instruction マッピングを取得

3. **VACE 呼び出し時に `--frame_num` を動的指定**
	```python
	# metadata から入力フレーム数を取得
	input_frame_count = frame_map.get(src_mp4_str, 81)
	
	# test_instructions.csv から instruction を取得
	row_prompt = instruction_map.get(int(video_id), default_prompt)
	
	# VACE コマンドに --frame_num を追加
	cmd = [
	    "/usr/bin/python3",
	    "vace/vace_wan_inference.py",
	    "--ckpt_dir", "/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview",
	    "--src_video", str(src_mp4.resolve()),
	    "--prompt", row_prompt,
	    "--base_seed", "42",
	    "--sample_steps", "25",
	    "--frame_num", str(input_frame_count),  # ← 明示的指定
	    "--save_dir", str(row_save_dir.resolve()),
	    "--save_file", str(row_out.resolve()),
	]
	```

4. **実行結果 manifest に記録**
	- `row_index, row_id, instruction, input_frame_count, mode, status`

### 背景意図
- instruction の一貫性により、異なる seed や step 数での比較が意味を持つ
- frame 数を入力に合わせることで「仕様準拠」を強制
- メタデータ駆動型にすることで、今後の再利用・比較が容易

## Trial 052 - scripts/run.sh を改修 (metadata + test_instructions.csv 活用)
- Time: 2026-03-22
- Objective:
	- metadata.csv から frame 数を読み込んで `--frame_num` を動的指定
	- test_instructions.csv から instruction を読み込んで活用
	- manifest に `input_frame_count` カラムを追加

### 実装済み

**改修対象**: `scripts/run.sh` ディレクトリ入力セクション（`if input_source.is_dir():`）

**追加処理**:

1. **metadata.csv 読み込み**
	```python
	metadata_csv = input_source / "metadata.csv"
	if metadata_csv.exists():
	    meta_df = pd.read_csv(metadata_csv)
	    frame_map = dict(zip(meta_df['video_path'].astype(str), meta_df['frame'].astype(int)))
	```

2. **test_instructions.csv 読み込み**
	```python
	test_instructions_csv = input_source / "test_instructions.csv"
	if test_instructions_csv.exists():
	    instr_df = pd.read_csv(test_instructions_csv)
	    instruction_map = dict(zip(instr_df['video_id'].astype(int), instr_df['instruction'].astype(str)))
	```

3. **ループ内で動的参照**
	```python
	input_frame_count = frame_map.get(src_mp4_str, 81)
	
	if instruction_map and int(row_id) in instruction_map:
	    row_prompt = instruction_map[int(row_id)]
	else:
	    row_prompt = resolve_prompt(i, default_prompt)
	```

4. **VACE コマンドに `--frame_num` を追加**
	```python
	cmd = [
	    "/usr/bin/python3",
	    "vace/vace_wan_inference.py",
	    "--ckpt_dir", str(vace_ckpt_dir.resolve()),
	    "--src_video", str(src_mp4.resolve()),
	    "--prompt", row_prompt,
	    "--base_seed", str(seed),
	    "--sample_steps", str(steps),
	    "--frame_num", str(input_frame_count),  # ← 明示的指定
	    "--save_dir", str(row_save_dir.resolve()),
	    "--save_file", str(row_out.resolve()),
	]
	```

5. **manifest CSV に frame 情報を記録**
	- header: `["row_index", "row_id", "instruction", "input_frame_count", "mode", "status"]`
	- 各行: `[i, row_id, row_prompt, input_frame_count, "model_direct_vace", "ok"]`

### ログ出力例
```
[INFO] Loaded metadata.csv: 100 entries
[INFO] Loaded test_instructions.csv: 100 entries
[INFO] row=0 id=0 frame=125 model_direct_vace ok
```

### 背景意図
- frame 数の明示を VACE に伝えることで、出力フレーム数が入力一致する可能性が高まる
- instruction の一貫性で、コンペ仕様の「Instruction Following」「Exclusivity of Edit」が検証可能になる
- metadata 駆動型により、異なる condition での比較が再現可能

## Trial 053 - 改修後の 1 本テスト (v7)
- Time: 2026-03-22
- Objective:
	- 改修した `scripts/run.sh` で実行
	- metadata.csv から読み込んだ frame 数が VACE に渡されるか確認
	- test_instructions.csv から読み込んだ instruction が使われるか確認
	- 出力フレーム数が入力と一致するか確認（最優先）

### 実行コマンド（ユーザーが実行予定）
```bash
cd /workspace
bash scripts/run.sh \
  configs/base.yaml \
  data/default/train/videos \
  "Use instruction set" \
  data/work/submission_vace_from_videos_v7.zip \
  1 \
  1
```

### 期待される動作
1. metadata.csv から 100 entries を読み込み
2. test_instructions.csv から 100 entries を読み込み
3. video 0 に対して：
	- input_frame_count = 125（metadata.csv より）
	- row_prompt = "Change the background to a vibrant sunset cityscape while keeping the main subject unchanged." （test_instructions.csv より）
	- `--frame_num 125` を VACE に渡す
4. 出力フレーム数が 125 であることを確認

### 検証項目
1. **manifest.csv に input_frame_count が記録されるか**
	```
	row_index,row_id,instruction,input_frame_count,mode,status
	0,0,"Change the background to a vibrant sunset cityscape while keeping the main subject unchanged.",125,model_direct_vace,ok
	```
2. **出力フレーム数が入力と一致するか**
	```bash
	ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,avg_frame_rate -of default=noprint_wrappers=1:nokey=1:nokey=1 data/work/submission_vace_from_videos_v7/0.mp4
	# 期待: nb_frames=125
	```
3. **出力 mp4 のハッシュが Trial 043（v4）と異なるか**
	- v4: 旧デフォルト 480p/81frames で出力
	- v7: 新 125frames で出力 → ハッシュ変わるはず

### 次のアクション（成功時）
- Trial 054: 複数フレーム値（120/125/150）で動作確認
- Trial 055: 全件 (100 videos) 実行
- Trial 056: コンペ仕様要件チェック（frame/aspect/resolution）

### 背景意図
- frame 指定が実際に有効なのかを 1 本で確認
- metadata/instruction CSV 統合の動作を検証
- frame 一致まで確保できれば、仕様準拠への最大課題が解決

## Trial 054 - scripts/run.sh のモジュール化と簡潔化
- Time: 2026-03-22
- Objective:
	- run.sh に埋め込まれた ~300行の Python コードを src/ に切り出す
	- run.sh を「パス解決 + 引数パース → モジュール委譲 → 結果出力」に簡潔化
	- 機能の再利用性向上と可読性改善

### 実装内容

#### 1) 新規モジュール作成

**src/utils/metadata_loader.py**
- `MetadataLoader.load_frame_metadata()` - metadata.csv 読み込み
- `MetadataLoader.load_instructions()` - test_instructions.csv 読み込み
- `MetadataLoader.get_frame_count()` - frame 数取得
- `MetadataLoader.get_instruction()` - instruction 取得

**src/utils/vace_executor.py**
- `VaceExecutor.__init__()` - VACE環境初期化
- `VaceExecutor.is_available()` - VACE実行可能確認
- `VaceExecutor.execute()` - VACE推論実行（結構化結果返却）

**src/pipeline/video_processor.py**
- `VideoProcessor.__init__()` - 初期化
- `VideoProcessor.process_directory()` - ディレクトリ入力処理

#### 2) scripts/run.sh の簡潔化

**改修前**: ~350行
- config読み込み、csv読み込み、VACE呼び出し、マニフェスト生成など全て inline

**改修後**: ~60行
- パス解決 + 引数パース
- `VaceExecutor`, `VideoProcessor` を呼び出し
- 結果出力

**改修後のrun.sh構造**:
```bash
# 1. 引数パース＆パス解決
ENTRY_PY=${1:-src/run_experiment.py}
CONFIG_PATH=${2:-configs/base.yaml}
shift 2

# 2. Pythonへ委譲（残り引数を透過）
exec ${PYTHON_BIN:-/usr/bin/python3} "$ENTRY_PY" "$CONFIG_PATH" "$@"
```

**補足（Python側の実処理入口）**:
```python
# src/run_experiment.py
if input_path.is_dir():
	result = run_directory(
		config=config,
		input_dir=input_path,
		output_target=output_path,
		default_prompt=args.prompt,
		python_bin=args.python_bin,
		limit_rows=args.rows,
		strict_no_fallback=strict_mode,
	)
elif input_path.suffix.lower() == ".mp4":
	result = run_single_mp4(
		config=config,
		input_file=input_path,
		prompt=args.prompt,
	)
elif input_path.suffix.lower() == ".parquet":
	result = run_parquet(
		config=config,
		input_file=input_path,
		output_target=output_path,
		default_prompt=args.prompt,
		python_bin=args.python_bin,
		strict_no_fallback=strict_mode,
	)
```

### 今後の改修予定（Trial 055以降）

- **Parquet処理の モジュール化**
  - `ParquetProcessor` を作成
  - `process_parquet_input()` メソッド

- **設定管理の統一**
  - `ConfigLoader` クラス作成
  - 既定値の一元管理

- **エラーハンドリング の統一**
  - `VaceError`, `ProcessorError` 例外クラス
  - 共通ログ形式

### 背景意図
- コード行数削減（~300行 → ~60行）で可読性向上
- 各機能が独立モジュール化（再利用・テスト容易）
- run.sh は「指揮者」に徹する（詳細ロジックは隠蔽）

## Trial 055 - run.sh 引数設計の再整理と結果ファイルの情報フロー明文化
- Time: 2026-03-22
- Objective:
	- run.sh を「トップPython + config を受け取る薄いラッパー」に統一
	- 変更の意図と、結果ファイルが生成されるまでの情報の流れを明文化

### 変更の意図
1. run.sh の責務を最小化する
	- 方針: 実装ロジックは Python 側に集約し、run.sh は実行委譲のみ
	- 期待効果: シェル修正起因の不具合を減らし、運用手順を単純化
2. 引数仕様を一般的な形に揃える
	- 第1引数: 実行するトップPython
	- 第2引数: configファイル
	- 第3引数以降: Pythonエントリへ透過的に引き渡し
3. エントリポイントの階層を整理する
	- `src/cli/main.py` のような孤立階層を廃止
	- `src/run_experiment.py` を唯一の実行入口に統合

### 現在の実行入口
- run.sh: `scripts/run.sh`
	- `ENTRY_PY=${1:-src/run_experiment.py}`
	- `CONFIG_PATH=${2:-configs/base.yaml}`
	- `exec ${PYTHON_BIN:-/usr/bin/python3} "$ENTRY_PY" "$CONFIG_PATH" "$@"`
- Pythonエントリ: `src/run_experiment.py`
	- inputタイプで分岐（directory / mp4 / parquet）
	- directory処理は `VideoProcessor.process_directory()` へ委譲

### 結果ファイルの情報フロー（directory入力）
1. Input
	- 動画群: `data/default/train/videos/*.mp4`
	- メタ情報: `data/default/train/videos/metadata.csv`
	- 指示集合: `data/default/train/videos/test_instructions.csv`
2. run.sh
	- `scripts/run.sh <entry_py> <config> <input_dir> <prompt> <output> <rows> <strict>`
	- run.sh は Python 実行のみを担当
3. Pythonエントリ
	- `src/run_experiment.py` が config 読み込み
	- directory入力を検出して `VideoProcessor` を起動
4. メタデータ解決
	- `MetadataLoader.load_frame_metadata()` で video_path -> frame 数を作成
	- `MetadataLoader.load_instructions()` で video_id -> instruction を作成
5. 動画ごとの推論
	- 各 `row_id.mp4` に対して
		- 入力frame数を `--frame_num` として VACE に渡す
		- instruction を prompt として VACE に渡す
	- 実行本体は `VaceExecutor.execute()`
6. 中間・成果物生成
	- 中間: `data/work/parquet_tmp/model_out_XXXX.mp4`
	- 成果動画: `<videos_dir>/<row_id>.mp4`
	- マニフェスト: `<videos_dir>/instruction_manifest.csv`
	- 指示記録: `<videos_dir>/instruction.txt`
	- 目標出力が zip の場合: `<output_target>.zip` に mp4 群を圧縮
7. 最終返却
	- `status`, `output_path`, `videos_dir`, `manifest_path`, `success_count`, `failed_count`

### 追跡しやすくなった点
- run.sh 変更時に影響範囲が限定される（実行委譲のみ）
- 入力情報（metadata / instructions）から出力成果物までの経路が1本化された
- `instruction_manifest.csv` を見れば、各行の instruction / frame_num / 成否を遡及できる

## Trial 056 - 実験記録保存方式を再設計（train/eval/infer + exp_summary）
- Time: 2026-03-22
- Objective:
	- 実験管理不能の状態を解消するため、保存先と索引を標準化
	- 実験ごとに `logs/{train|eval|infer}/{exp_name}_{time}` を必ず生成
	- `logs/exp_summary` に `exp_name_time`, `exp_content` を毎回追記

### 仕様変更（要件反映）
1. config へ実験メタ情報を追加
	- `output.exp_name`: 実験名
	- `output.exp_content`: 実験背景（狙い）
2. 実験ディレクトリの命名規則
	- `logs/{stage}/{exp_name}_{YYYYmmdd_HHMMSS}`
	- stage は入力パスから `train/eval/infer` を推定（見つからなければ `infer`）
3. 追記型サマリ
	- `logs/exp_summary/exp_summary.csv` に1実行1行で追記
	- 記録項目: `timestamp, exp_id, exp_content, stage, status, input, output`

### 実装変更
- `configs/base.yaml`
	- `output.exp_name`
	- `output.exp_content`
- `src/run_experiment.py`
	- 実験コンテキスト生成（exp_id/stage/exp_dir）
	- `config_snapshot.yaml` を実験ディレクトリへ保存
	- `result.json` を実験ディレクトリへ保存
	- `logs/exp_summary/exp_summary.csv` への追記処理を追加
- `src/pipeline/video_processor.py`
	- `instruction_manifest.csv` と `instruction.txt` を実験ディレクトリへコピー
	- return に `experiment_dir` を追加

### 結果ファイルの新しい流れ
1. 設定入力
	- `configs/base.yaml` から `output.exp_name`, `output.exp_content`
2. 実行開始
	- `src/run_experiment.py` が `exp_name_time` を生成
	- `logs/{stage}/{exp_name_time}` を作成
3. 処理本体
	- `VideoProcessor` が動画処理・manifest生成
4. 実験記録の固定化
	- `logs/{stage}/{exp_name_time}/config_snapshot.yaml`
	- `logs/{stage}/{exp_name_time}/result.json`
	- `logs/{stage}/{exp_name_time}/instruction_manifest.csv`
	- `logs/{stage}/{exp_name_time}/instruction.txt`
5. 全体索引へ追記
	- `logs/exp_summary/exp_summary.csv` に `exp_name_time`, `exp_content` を追記

### 背景意図
- これまでの「出力物だけ散在する状態」を解消し、
  実験単位で設定・結果・説明を同一ディレクトリに固定するため。
- 後から「どの設定で、何を意図し、結果がどうだったか」を
  `exp_summary` から逆引きできるようにするため。

## Trial 057 - base + override YAML 読み込み方式へ変更
- Time: 2026-03-22
- Objective:
	- `base.yaml` を土台に固定し、実験ごとの差分は小さい override YAML だけで管理
	- 1ケース検証用の実験設定を差分ファイルとして追加

### 追加ファイル
- `configs/vace_test_only_1_case.yaml`
	- `output.exp_name: vace_test_only_1_case`
	- `output.exp_content: 1個の.mp4だけ VACEで処理をして、instruction通りの変化担っているか？ コンペ出力になっているか？ を確認する`

### 実装変更
- `src/run_experiment.py`
	- `configs/base.yaml` を常に先に読み込み
	- 第1引数 `config` を override YAML として後から再読み込み
	- `deep_merge_dict()` で再帰的に上書き統合
	- `config==configs/base.yaml` の場合は base のみ利用

### 使い方
```bash
bash scripts/run.sh \
	src/run_experiment.py \
	configs/vace_test_only_1_case.yaml \
	data/default/train/videos \
	"Use instruction set" \
	data/work/submission_vace_test_only_1_case.zip \
	1 \
	1
```

### 背景意図
- 実験管理を「巨大単一yaml更新」から「base + 差分yaml」に分離し、
	実験ごとの差分追跡と再現を容易にするため。

## Trial 058 - id=0 単体実行の動画メタデータ計測（frame/width/fps）
- Time: 2026-03-22
- Target experiment dir:
	- `logs/train/vace_test_only_1_case_20260322_085052`
- Purpose:
	- 今回の1ケース実行結果について、入力/出力の動画仕様を数値で記録する

### 実行結果ファイル
- 出力動画:
	- `logs/train/vace_test_only_1_case_20260322_085052/videos/0.mp4`
- 出力zip:
	- `logs/train/vace_test_only_1_case_20260322_085052/submission_vace_id0.zip`
- 実行マニフェスト:
	- `logs/train/vace_test_only_1_case_20260322_085052/instruction_manifest.csv`
	- 記録: `row_id=0`, `input_frame_count=125`, `mode=model_direct_vace`, `status=ok`

### 動画メタデータ比較（Input vs Output）
| 項目 | Input (`data/default/train/videos/0.mp4`) | Output (`logs/train/vace_test_only_1_case_20260322_085052/videos/0.mp4`) |
|---|---:|---:|
| size_bytes | 1,835,090 | 2,755,414 |
| frame_count | 125 | 81 |
| fps | 25.0 | 16.0 |
| width | 1920 | 832 |
| height | 1080 | 480 |
| duration_sec | 5.0 | 5.0625 |

### 事実ベースの判定
1. VACE実行自体は成功
	- `instruction_manifest.csv` 上で `model_direct_vace, ok`
2. ただし仕様観点では未達項目あり
	- Frame不一致: 125 -> 81
	- 解像度不一致: 1920x1080 -> 832x480
	- FPS不一致: 25.0 -> 16.0

### 背景意図
- 「実行できたか」だけでなく、コンペ要件（frame/size一致）観点で
  入出力差分を数値として残し、次の設定調整の根拠にするため。

## Trial 059 - Frame/解像度不一致の原因分析（実行なし・コード読解のみ）
- Time: 2026-03-22
- Constraint:
	- ユーザー指示により VACE は再実行しない
	- 既存コードと既存ログだけで原因を分析

### 観測済み事実（Trial 058）
- Input `0.mp4`: 125 frames, 25.0 fps, 1920x1080
- Output `0.mp4`: 81 frames, 16.0 fps, 832x480

### 1) FPS が 16 になる理由
- 根拠コード:
	- `third_party/VACE/vace/models/wan/configs/shared_config.py`
		- `wan_shared_cfg.sample_fps = 16`
	- `third_party/VACE/vace/vace_wan_inference.py`
		- `cache_video(tensor=video[None], save_file=save_file, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))`
- 解釈:
	- 書き出しFPSは入力fpsではなく、モデル設定 `sample_fps(=16)` で固定される。
	- そのため入力25fpsでも出力16fpsになる。

- 対応方法案:
	1. `sample_fps` を入力fpsに合わせる
		- `shared_config.py` の固定値依存をやめ、`src_video` のfpsを `cache_video(..., fps=input_fps, ...)` に渡す。
		- 変更先（最小）:
			- `third_party/VACE/vace/vace_wan_inference.py`
				- `main()` の `cache_video(..., fps=cfg.sample_fps, ...)` を `cache_video(..., fps=input_fps, ...)` へ変更
				- `prepare_source()` の戻り値 `fps`（現在は未使用）を受けて `cache_video` に渡す
			- 併せて必要なら `third_party/VACE/vace/models/wan/configs/shared_config.py` の `sample_fps=16` 固定を撤廃
	2. 後処理で fps を入力へ揃える
		- 推論後に ffmpeg で `fps=入力fps` へ再エンコード。
		- ただしフレーム補間/重複の副作用が出るため、最終手段。
		- 変更先（後処理方式）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` で `result["output_path"]` 取得後、保存前に fps 補正処理を挿入
	3. strict検証を追加
		- `output_fps == input_fps` を自動チェックし、不一致は失敗扱いにする。
		- 変更先（検証追加）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` の `out_mp4_path.write_bytes(result["output_path"].read_bytes())` 直後に fps 比較を追加し、strict時は `RuntimeError`
			- 既存の共通チェックを使うなら `src/eval/constraints.py` の `ConstraintChecker.check()` を呼び出す分岐を追加

### 2) 解像度が 832x480 になる理由
- 根拠コード:
	- `third_party/VACE/vace/vace_wan_inference.py`
		- `--size` 既定値は `480p`
	- `third_party/VACE/vace/models/wan/configs/__init__.py`
		- `SIZE_CONFIGS['480p'] = (480, 832)`
	- `third_party/VACE/vace/models/wan/wan_vace.py`
		- `prepare_source(src_video, src_mask, src_ref_images, args.frame_num, SIZE_CONFIGS[args.size], device)`
		- `set_area(480*832)` で処理面積を固定
- 解釈:
	- 現行実装は入力解像度を保持せず、`--size` で指定した面積へ再サンプリングする。
	- 既定 `480p` を明示上書きしていないため、832x480出力になる。

- 対応方法案:
	1. `--size` を入力アスペクトに対応する上位サイズへ切替
		- 例: 16:9入力なら `720p` または `(1280, 720)` 相当を明示指定。
		- 変更先（最小）:
			- `src/utils/vace_executor.py`
				- `execute()` の `cmd` に `--size` を追加（例: `--size 720p`）
			- `configs/base.yaml`
				- 例: `inference.size: 720p` を追加し、`vace_executor.py` から参照
	2. 入力解像度保持の後段復元
		- 推論は対応サイズで実施し、最後に入力 `width/height` へ復元。
		- ただし画質劣化リスクあり（補間依存）。
		- 変更先（後処理方式）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` で入力動画サイズ取得 -> 出力保存前に `width/height` 復元処理を追加
	3. strict検証を追加
		- `output_width==input_width` かつ `output_height==input_height` を必須化。
		- 変更先（検証追加）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` でサイズ検証を追加し、不一致を失敗扱い
			- もしくは `src/eval/constraints.py` の `ConstraintChecker.check()` 呼び出しを統合

### 3) frame_num を渡しても 81 になる理由（重要）
- 根拠コード:
	- 呼び出し側 `src/utils/vace_executor.py`
		- `--frame_num` は渡している
	- ただし `third_party/VACE/vace/models/wan/wan_vace.py::prepare_source()`
		- `sub_src_video is not None` の経路では `self.vid_proc.load_video(sub_src_video)` を使用
		- この経路は `num_frames` を使わない
	- `third_party/VACE/vace/models/utils/preprocessor.py::VaceVideoProcessor`
		- `load_video()` 内で `seq_len`・`max_area`・`downsample`・`keep_last` に基づきフレーム数を再計算
		- フレーム数は `of=(of-1)*df+1`（4n+1系列）で決まる
- 解釈:
	- 現行VACE実装では、入力動画ありの経路で実効フレーム数は前処理器が決める。
	- そのため `--frame_num=125` を指定しても、前処理側で 81 に再サンプリングされうる。

- 対応方法案:
	1. `prepare_source()` 側で `num_frames` を強制反映
		- `load_video()` 後に `num_frames` に合わせて frame index を再サンプリングする処理を追加。
		- 変更先（モデル内部を直す本命）:
			- `third_party/VACE/vace/models/wan/wan_vace.py`
				- `prepare_source()` の `self.vid_proc.load_video(sub_src_video)` 直後で `num_frames` への再サンプリングを追加
	2. `VaceVideoProcessor` に目標フレーム数引数を追加
		- `seq_len/max_area` 主導だけでなく、`target_frame_num` を優先する分岐を実装。
		- 変更先（前処理器を直す案）:
			- `third_party/VACE/vace/models/utils/preprocessor.py`
				- `load_video()` / `load_video_batch()` に `target_frame_num` を追加し、`frame_ids` 生成ロジックへ反映
			- `third_party/VACE/vace/models/wan/wan_vace.py`
				- `prepare_source()` から `target_frame_num=num_frames` を渡す
	3. 後処理で frame 数を一致化
		- 出力 `nb_frames` を測定し、不足/超過を補間・間引きで一致させる。
		- 変更先（後処理方式）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` で出力動画の実フレーム数を確認し、不一致なら補間/間引き処理
	4. strict検証を追加
		- `output_nb_frames == metadata.frame` を必須化し、不一致は失敗扱い。
		- 変更先（検証追加）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` で `input_frame_count` と出力 `nb_frames` 比較を追加
			- `src/utils/vace_executor.py`
				- ログに `requested_frame_num` を出して差分追跡しやすくする

### 4) 「size_bytes が大きくなっているのは型ミスか？」について
- 結論:
	- **型ミス（int/float32）が主因である可能性は低い**。
- 根拠コード:
	- `third_party/Wan2.1/wan/utils/utils.py::cache_video()`
		- テンソルを `clamp -> normalize -> (x*255).type(torch.uint8)` で明示的に `uint8` 化
		- `imageio` + `libx264` + `quality=8` で再エンコード
- bitrate の説明:
	- `bitrate` は「1秒あたりに割り当てるデータ量（bits/sec）」。
	- 動画サイズは概ね `size_bytes ≈ bitrate_bps × duration_sec / 8` で決まる。
	- つまり、解像度やfpsが下がっていても、エンコーダが高いbitrateを使えば `size_bytes` は増えうる。
- 解釈:
	- 出力バイトサイズは「解像度だけ」で決まらず、
	  エンコード設定（codec/quality/CRF相当/GOP/内容変化量）で大きく変動する。
	- とくに `cache_video()` は `quality=8` 固定で `libx264` に渡しており、
	  入力動画より高い実効bitrateで符号化されると、低解像度でもサイズ増加が起こる。
	- 低解像度でも再エンコード条件によって入力より大きくなることはあり得る。
	- したがって、今回の `size_bytes` 増加は codec/quality/内容差の影響として説明可能。

- 対応方法案:
	1. bitrate/品質制御パラメータを外出しする
		- `quality` 固定値をやめ、実験設定から `bitrate` または `crf` を渡せるようにする。
		- 変更先（本命）:
			- `third_party/Wan2.1/wan/utils/utils.py`
				- `cache_video()` に `bitrate` / `ffmpeg_params` / `quality` 引数を追加
				- `imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=quality, ffmpeg_params=ffmpeg_params)` に可変パラメータを反映
			- `third_party/VACE/vace/vace_wan_inference.py`
				- `cache_video(tensor=video[None], save_file=save_file, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1), quality=quality, ffmpeg_params=ffmpeg_params)` 呼び出しに上記引数を受け渡す
	2. 実行設定（YAML）から制御する
		- 実験ごとに bitrate 方針を切替できるようにする。
		- 変更先（設定連携）:
			- `configs/base.yaml`
				- 例: `inference.video_encode.quality` / `inference.video_encode.bitrate` を追加
			- `src/utils/vace_executor.py`
				- CLI引数として `vace_wan_inference.py` へ渡す（対応引数追加時）
	3. 後処理でサイズを収束させる（暫定）
		- 推論後に ffmpeg 再エンコードで `-b:v` / `-maxrate` を指定し、サイズ上振れを抑える。
		- 変更先（後処理方式）:
			- `src/pipeline/video_processor.py`
				- `process_directory()` で出力保存前に再エンコード処理を挿入
	4. strict検証を追加
		- `output_bit_rate` と `size_bytes` の閾値を設け、逸脱時は失敗扱い。
		- 変更先（検証追加）:
			- `src/pipeline/video_processor.py`
				- 生成直後に `ffprobe` または OpenCV で bitrate/size を検証
			- `src/eval/constraints.py`
				- `ConstraintChecker.check()` に bitrate/size ルールを追加

### 5) 現時点の一次結論
1. Frame不一致の主因:
	- VACE前処理器（`VaceVideoProcessor`）による再サンプリング
2. 解像度不一致の主因:
	- `--size` 既定 `480p`（832x480）
3. FPS不一致の主因:
	- `sample_fps=16` 固定書き出し
4. サイズ増加の主因:
	- 再エンコード（libx264, quality=8）と内容依存ビットレート
5. 型（int/float32）:
	- 変換は明示されており、今回現象の主因とは考えにくい