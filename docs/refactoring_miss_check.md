# refactoring_miss_check

## 対象と判定基準

- 検索対象は `/workspace/src` と `/workspace/notebook`。
- 「backup に間違って移動された可能性が高い」とした条件は次の 2 点です。
	- `src` または `notebook` 側で、`src.xxx` / `parse.xxx` として参照されている。
	- 期待される本来の配置先に実ファイルがなく、実体が `src/backup` にしかない。
- `src/postprocess` や `src/utils/video_utility.py` のように main 側に実体があるものは、backup 誤移動とは分けて扱う。

## 現状の要約

- `src` 直下には実行対象の `.py` がなく、`src/*.py` は 0 件だった。
- その一方で、`src/__pycache__` には `submit_baseline_ver05`, `run_experiment`, `instruction_parser_ver04` など、main 側にあった前提の `.pyc` が残っている。
- `src/parse`, `src/data`, `src/pipeline`, `src/model`, `src/preprocess` は、main 側に実装がないか、実質空に近い。
- 同名または対応実装は `src/backup` 側にまとまって存在する。
- notebook 側では、parse 系ノートブックと VACE 系ノートブックが、この欠落した main 側パスを前提に import している。

## 確度が高い誤移動

### 1. parse パッケージ一式

parse 系ノートブックは `parse.*` または `src.parse.*` を import しているが、main 側の `src/parse` には実装がない。実体は `src/backup/parse` に集まっている。

根拠:

- [notebook/parse_instruction/parse_instruction_ver12.ipynb#L50](../notebook/parse_instruction/parse_instruction_ver12.ipynb#L50)
- [notebook/parse_instruction/parse_instruction_ver12.ipynb#L51](../notebook/parse_instruction/parse_instruction_ver12.ipynb#L51)
- [notebook/parse_instruction/parse_instruction_ver12.ipynb#L52](../notebook/parse_instruction/parse_instruction_ver12.ipynb#L52)
- [notebook/parse_instruction/parse_instruction_ver13.ipynb#L54](../notebook/parse_instruction/parse_instruction_ver13.ipynb#L54)
- [notebook/parse_instruction/parse_instruction_ver13.ipynb#L55](../notebook/parse_instruction/parse_instruction_ver13.ipynb#L55)
- [notebook/parse_instruction/parse_instruction_ver13.ipynb#L56](../notebook/parse_instruction/parse_instruction_ver13.ipynb#L56)
- [notebook/parse_instruction/parse_instruction_ver20.ipynb#L94](../notebook/parse_instruction/parse_instruction_ver20.ipynb#L94)

backup 側にしかない実体:

- [src/backup/parse/data_loading.py](../src/backup/parse/data_loading.py)
- [src/backup/parse/evaluation.py](../src/backup/parse/evaluation.py)
- [src/backup/parse/features.py](../src/backup/parse/features.py)
- [src/backup/parse/models.py](../src/backup/parse/models.py)
- [src/backup/parse/runner.py](../src/backup/parse/runner.py)
- [src/backup/parse/instruction_parser_ver19.py](../src/backup/parse/instruction_parser_ver19.py)
- [src/backup/parse/model_ver10_baseline.py](../src/backup/parse/model_ver10_baseline.py)
- [src/backup/parse/model_ver10_improved.py](../src/backup/parse/model_ver10_improved.py)
- [src/backup/parse/model_v11a_ruleplus.py](../src/backup/parse/model_v11a_ruleplus.py)
- [src/backup/parse/model_v11b_retrieval.py](../src/backup/parse/model_v11b_retrieval.py)
- [src/backup/parse/model_v11d_ensemble.py](../src/backup/parse/model_v11d_ensemble.py)

見立て:

- `parse_instruction_ver12` から `ver19` までは `parse.*` を前提にしている。
- `parse_instruction_ver20` は `src.parse.instruction_parser_ver19` を前提にしている。
- つまり notebook 側の参照規約が途中で `parse.*` から `src.parse.*` に寄っているが、どちらにせよ main 側の `src/parse` に実体がないため、backup へ移したまま参照更新が止まっている可能性が高い。

### 2. submit_baseline_ver05.py

`parse_instruction_ver20` は `src.submit_baseline_ver05` を直接 import しているが、main 側の `src` 直下に `.py` はなく、実体は backup 側にしかない。

根拠:

- [notebook/parse_instruction/parse_instruction_ver20.ipynb#L266](../notebook/parse_instruction/parse_instruction_ver20.ipynb#L266)
- [notebook/parse_instruction/parse_instruction_ver20.ipynb#L322](../notebook/parse_instruction/parse_instruction_ver20.ipynb#L322)
- [src/backup/submit_baseline_ver05.py](../src/backup/submit_baseline_ver05.py)

見立て:

- notebook から直接使われているため、これは backup 誤移動の優先度が高い。

### 3. VACE 系 utility モジュール

VACE 系ノートブックは `src.utils.vace_executor` と `src.utils.vace_edit_assets` を import しているが、main 側の `src/utils` には `video_utility.py` しかない。実体は backup 側にのみ存在する。

根拠:

- [notebook/test/vace_pipeline_ver03.ipynb#L74](../notebook/test/vace_pipeline_ver03.ipynb#L74)
- [notebook/test/vace_pipeline_ver04.ipynb#L74](../notebook/test/vace_pipeline_ver04.ipynb#L74)
- [notebook/test/vace_pipeline_ver04.ipynb#L297](../notebook/test/vace_pipeline_ver04.ipynb#L297)
- [notebook/test/vace_pipeline_ver04.ipynb#L1696](../notebook/test/vace_pipeline_ver04.ipynb#L1696)

backup 側にしかない実体:

- [src/backup/utils/vace_executor.py](../src/backup/utils/vace_executor.py)
- [src/backup/utils/vace_edit_assets.py](../src/backup/utils/vace_edit_assets.py)
- [src/backup/utils/io_video.py](../src/backup/utils/io_video.py)
- [src/backup/utils/metadata_loader.py](../src/backup/utils/metadata_loader.py)

見立て:

- notebook からの直接参照があるため、`vace_executor.py` と `vace_edit_assets.py` は誤移動候補として強い。
- `io_video.py` と `metadata_loader.py` は notebook からは直接 import されていないが、後述の pipeline 依存で必要になる。

### 4. runner / pipeline / model / eval / data / preprocess の依存チェーン

`src/backup` 配下の実行スクリプト群は、`src.data.*`, `src.pipeline.*`, `src.model.*`, `src.eval.*`, `src.preprocess.*` を main 側パスとして import している。しかし main 側には対応実装がない。依存一式が backup 側に押し込まれている状態に見える。

根拠:

- [src/backup/runner.py#L15](../src/backup/runner.py#L15)
- [src/backup/runner.py#L21](../src/backup/runner.py#L21)
- [src/backup/runner.py#L24](../src/backup/runner.py#L24)
- [src/backup/runner.py#L27](../src/backup/runner.py#L27)
- [src/backup/runner.py#L30](../src/backup/runner.py#L30)
- [src/backup/runner.py#L31](../src/backup/runner.py#L31)
- [src/backup/pipeline/run_experiment.py#L15](../src/backup/pipeline/run_experiment.py#L15)
- [src/backup/pipeline/run_experiment.py#L16](../src/backup/pipeline/run_experiment.py#L16)
- [src/backup/pipeline/run_experiment.py#L17](../src/backup/pipeline/run_experiment.py#L17)
- [src/backup/pipeline/run_experiment.py#L18](../src/backup/pipeline/run_experiment.py#L18)
- [src/backup/pipeline/run_experiment.py#L19](../src/backup/pipeline/run_experiment.py#L19)
- [src/backup/pipeline/video_processor.py#L15](../src/backup/pipeline/video_processor.py#L15)
- [src/backup/pipeline/video_processor.py#L16](../src/backup/pipeline/video_processor.py#L16)
- [src/backup/run_experiment.py#L17](../src/backup/run_experiment.py#L17)
- [src/backup/run_experiment.py#L18](../src/backup/run_experiment.py#L18)
- [src/backup/run_experiment.py#L195](../src/backup/run_experiment.py#L195)

backup 側にしかない実体:

- data
	- [src/backup/data/video_io.py](../src/backup/data/video_io.py)
	- [src/backup/data/frame_manager.py](../src/backup/data/frame_manager.py)
- preprocess
	- [src/backup/preprocess/resize.py](../src/backup/preprocess/resize.py)
- model
	- [src/backup/model/vace_wrapper.py](../src/backup/model/vace_wrapper.py)
- eval
	- [src/backup/eval/evaluator.py](../src/backup/eval/evaluator.py)
	- [src/backup/eval/constraints.py](../src/backup/eval/constraints.py)
- pipeline
	- [src/backup/pipeline/video_processor.py](../src/backup/pipeline/video_processor.py)
	- [src/backup/pipeline/run_experiment.py](../src/backup/pipeline/run_experiment.py)

見立て:

- この系統は notebook から直接叩かれている数は parse 系より少ないが、src 内のコード自体が main 側パスを前提に書かれており、復元漏れのまとまりとして扱うのが自然。

### 5. baseline ver03 / ver04 系のトップレベルスクリプト

`src/backup/submit_baseline_ver04.py` は、`src.command_planner_ver04`, `src.instruction_parser_ver04`, `src.submit_baseline_ver03` を main 側として import している。main 側 `src/*.py` は存在せず、対応実体は backup 側にある。

根拠:

- [src/backup/submit_baseline_ver04.py#L15](../src/backup/submit_baseline_ver04.py#L15)
- [src/backup/submit_baseline_ver04.py#L16](../src/backup/submit_baseline_ver04.py#L16)
- [src/backup/submit_baseline_ver04.py#L17](../src/backup/submit_baseline_ver04.py#L17)
- [src/backup/submit_baseline_ver04.py#L118](../src/backup/submit_baseline_ver04.py#L118)
- [src/backup/utils/vace_edit_assets.py#L32](../src/backup/utils/vace_edit_assets.py#L32)

backup 側にしかない実体:

- [src/backup/command_planner_ver04.py](../src/backup/command_planner_ver04.py)
- [src/backup/instruction_parser_ver04.py](../src/backup/instruction_parser_ver04.py)
- [src/backup/submit_baseline_ver03.py](../src/backup/submit_baseline_ver03.py)
- [src/backup/submit_baseline_ver04.py](../src/backup/submit_baseline_ver04.py)

見立て:

- notebook からの直接 import は見つからなかったが、src 側の依存として閉じているため、main から backup へ寄せられたままになっている候補。

## 保留扱い

次のファイルは backup 側にしかないが、今回の検索範囲では「main 側参照が今も残っている」証拠が弱い。単なる退避、旧版保存、未使用コードの可能性もある。

- [src/backup/build_atomic_vace_eval_assets.py](../src/backup/build_atomic_vace_eval_assets.py)
- [src/backup/build_instruction_catalog.py](../src/backup/build_instruction_catalog.py)
- [src/backup/download_wan_weights.py](../src/backup/download_wan_weights.py)
- [src/backup/download_wan_weights_2.py](../src/backup/download_wan_weights_2.py)
- [src/backup/evaluate_submit_baseline_ver01.py](../src/backup/evaluate_submit_baseline_ver01.py)
- [src/backup/evaluate_submit_baseline_ver02.py](../src/backup/evaluate_submit_baseline_ver02.py)
- [src/backup/evaluate_submit_baseline_ver03.py](../src/backup/evaluate_submit_baseline_ver03.py)
- [src/backup/evaluate_submit_baseline_ver04.py](../src/backup/evaluate_submit_baseline_ver04.py)
- [src/backup/export_parquet_videos.py](../src/backup/export_parquet_videos.py)
- [src/backup/submit_baseline_ver01.py](../src/backup/submit_baseline_ver01.py)
- [src/backup/submit_baseline_ver02.py](../src/backup/submit_baseline_ver02.py)
- [src/backup/summarize_submit_baseline_ver03.py](../src/backup/summarize_submit_baseline_ver03.py)

補足:

- `src/__pycache__` に対応する `.pyc` が残っているものがあり、以前 main 側にあった痕跡は見える。
- ただし今回の基準では、`src` / `notebook` 内の現行 import で裏づけられるものを優先して「確度が高い誤移動」に分類した。

## backup 誤移動ではなく、別問題として切り分けるべき点

### 1. `posteprocess` タイポ

`postprocess` は main 側に存在しているため、これは backup への誤移動ではなく import パスの typo。

根拠:

- [src/postprocess/task_rules_ver05_functions.py](../src/postprocess/task_rules_ver05_functions.py)
- [notebook/task_rules_ver05_trial_06.ipynb#L51](../notebook/task_rules_ver05_trial_06.ipynb#L51)
- [src/backup/submit_baseline_ver05.py#L61](../src/backup/submit_baseline_ver05.py#L61)

見立て:

- `src.posteprocess` は単純な綴りミス。
- `src.postprocess` に実装があるので、restore 作業とは別に直すべき。

### 2. task_rules_ver05 系 notebook 群

次の notebook は `src.postprocess.task_rules_ver05_functions` と `src.utils.video_utility` を使っており、main 側の実体がある。

- [notebook/task_rules_ver05_trial_01.ipynb#L51](../notebook/task_rules_ver05_trial_01.ipynb#L51)
- [notebook/task_rules_ver05_trial_02.ipynb#L51](../notebook/task_rules_ver05_trial_02.ipynb#L51)
- [notebook/task_rules_ver05_trial_03.ipynb#L56](../notebook/task_rules_ver05_trial_03.ipynb#L56)
- [notebook/task_rules_ver05_trial_04.ipynb#L49](../notebook/task_rules_ver05_trial_04.ipynb#L49)
- [notebook/task_rules_ver05_trial_05.ipynb#L59](../notebook/task_rules_ver05_trial_05.ipynb#L59)
- [notebook/task_rules_ver05_trial_07.ipynb#L60](../notebook/task_rules_ver05_trial_07.ipynb#L60)
- [notebook/task_rules_ver05_trial_08_1.ipynb#L57](../notebook/task_rules_ver05_trial_08_1.ipynb#L57)
- [notebook/task_rules_ver05_trial_08_2.ipynb#L57](../notebook/task_rules_ver05_trial_08_2.ipynb#L57)
- [src/postprocess/task_rules_ver05_functions.py](../src/postprocess/task_rules_ver05_functions.py)
- [src/utils/video_utility.py](../src/utils/video_utility.py)

見立て:

- ここは main 側の現行配置が正しく、backup 誤移動の対象ではない。

## 現在の整理結果

優先度高:

1. `src/parse/*` 一式
2. `src/submit_baseline_ver05.py`
3. `src/utils/vace_executor.py`
4. `src/utils/vace_edit_assets.py`

次点:

1. `src/data/*`
2. `src/preprocess/resize.py`
3. `src/model/vace_wrapper.py`
4. `src/eval/*`
5. `src/pipeline/*`
6. `src/submit_baseline_ver03.py`
7. `src/submit_baseline_ver04.py`
8. `src/command_planner_ver04.py`
9. `src/instruction_parser_ver04.py`

別問題:

1. `src.posteprocess` typo の修正

## 追記: 復元対応の実施結果

backup ディレクトリは削除せず、そのまま残した状態で、`src/backup` 配下のファイルを対応する `src` 配下へ複製して戻した。

実施方針:

- `src/backup/foo/bar.py` を `src/foo/bar.py` へ復元した。
- 既存ファイルの上書きはしていない。
- 調査時点で main 側に欠けていた `.py` / `.txt` を中心に復元した。

主な復元結果:

- top-level scripts
	- [src/submit_baseline_ver01.py](../src/submit_baseline_ver01.py)
	- [src/submit_baseline_ver02.py](../src/submit_baseline_ver02.py)
	- [src/submit_baseline_ver03.py](../src/submit_baseline_ver03.py)
	- [src/submit_baseline_ver04.py](../src/submit_baseline_ver04.py)
	- [src/submit_baseline_ver05.py](../src/submit_baseline_ver05.py)
	- [src/run_experiment.py](../src/run_experiment.py)
	- [src/runner.py](../src/runner.py)
	- [src/command_planner_ver04.py](../src/command_planner_ver04.py)
	- [src/instruction_parser_ver04.py](../src/instruction_parser_ver04.py)
- parse package
	- [src/parse/__init__.py](../src/parse/__init__.py)
	- [src/parse/data_loading.py](../src/parse/data_loading.py)
	- [src/parse/evaluation.py](../src/parse/evaluation.py)
	- [src/parse/features.py](../src/parse/features.py)
	- [src/parse/instruction_parser_ver19.py](../src/parse/instruction_parser_ver19.py)
	- [src/parse/models.py](../src/parse/models.py)
	- [src/parse/runner.py](../src/parse/runner.py)
	- [src/parse/model_ver10_baseline.py](../src/parse/model_ver10_baseline.py)
	- [src/parse/model_ver10_improved.py](../src/parse/model_ver10_improved.py)
	- [src/parse/model_v11a_ruleplus.py](../src/parse/model_v11a_ruleplus.py)
	- [src/parse/model_v11b_retrieval.py](../src/parse/model_v11b_retrieval.py)
	- [src/parse/model_v11d_ensemble.py](../src/parse/model_v11d_ensemble.py)
- VACE / pipeline dependencies
	- [src/utils/vace_executor.py](../src/utils/vace_executor.py)
	- [src/utils/vace_edit_assets.py](../src/utils/vace_edit_assets.py)
	- [src/utils/io_video.py](../src/utils/io_video.py)
	- [src/utils/metadata_loader.py](../src/utils/metadata_loader.py)
	- [src/data/video_io.py](../src/data/video_io.py)
	- [src/data/frame_manager.py](../src/data/frame_manager.py)
	- [src/preprocess/resize.py](../src/preprocess/resize.py)
	- [src/model/vace_wrapper.py](../src/model/vace_wrapper.py)
	- [src/eval/evaluator.py](../src/eval/evaluator.py)
	- [src/eval/constraints.py](../src/eval/constraints.py)
	- [src/pipeline/video_processor.py](../src/pipeline/video_processor.py)
	- [src/pipeline/run_experiment.py](../src/pipeline/run_experiment.py)

復元後の確認:

- 次の主要ファイルでは、少なくともエディタ診断上のエラーは出ていない。
	- [src/submit_baseline_ver05.py](../src/submit_baseline_ver05.py)
	- [src/parse/instruction_parser_ver19.py](../src/parse/instruction_parser_ver19.py)
	- [src/utils/vace_executor.py](../src/utils/vace_executor.py)
	- [src/pipeline/run_experiment.py](../src/pipeline/run_experiment.py)
	- [src/runner.py](../src/runner.py)

復元後も残る課題:

- [src/submit_baseline_ver05.py](../src/submit_baseline_ver05.py) と notebook の一部には `src.posteprocess` という typo が残っている可能性がある。
- 今回は restore のみ実施し、import typo や runtime 動作確認まではまだ反映していない。
- backup と main の二重管理状態になっているため、今後はどちらを正とするか決めて整理が必要。

現時点の整理:

- notebook / src から参照されていた main 側の欠落ファイルは、概ね `src` 配下へ戻せた。
- `src/backup` もそのまま残しているため、比較や差分確認は引き続き可能。

