# Weight File Audit

このファイルは、`/workspace/src` と `/workspace/notebook` について、単なるファイル名検索ではなく、以下を分けて確認した結果をまとめる。

- 現在の非 `backup` コードが実際に参照するパス
- notebook のうち提出チューニングに近い経路と、単発検証用の古い経路
- 実体ファイルが今どこにあるか
- Hugging Face モデルがローカル固定パスではなく cache を使うかどうか

## 結論

前回の「GroundingDINO と SAM が `src` / `notebook` で共通利用されている」というまとめは不正確だった。

正しくは以下。

- `SAM` は、現行の非 `backup` コードでも `/workspace/weights/sam_vit_h_4b8939.pth` を参照している。
- `GroundingDINO` は、現行の非 `backup` コードで参照していたのは `third_party` 側だった。
- ただし実ファイルは `/workspace/weights/groundingdino_swint_ogc.pth` にも存在するため、提出経路で `weights` を優先するよう修正した。
- `runwayml/stable-diffusion-v1-5` のような Hugging Face モデルは、コード上はモデル ID 参照であり、ローカル固定ファイル参照ではない。
- notebook には古い単発検証用コードが多数残っており、そこには `third_party/GroundingDINO/weights/...` の直書きがまだある。

## Verified File Placement

実体確認済みの配置:

| Artifact | Verified path | Exists |
| --- | --- | --- |
| GroundingDINO checkpoint | `/workspace/weights/groundingdino_swint_ogc.pth` | Yes |
| GroundingDINO checkpoint (legacy copy) | `/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth` | Yes |
| SAM checkpoint | `/workspace/weights/sam_vit_h_4b8939.pth` | Yes |
| Stable Diffusion cache root | `/workspace/.cache/huggingface` | Yes |
| LoRA directory | `/workspace/weights/lora` | Yes |
| VACE repo | `/workspace/third_party/VACE` | Yes |
| RAFT repo | `/workspace/third_party/RAFT` | Yes |
| XMem repo | `/workspace/third_party/XMem` | Yes |

## Active Non-Backup Code In `/workspace/src`

`/workspace/src` の非 `backup` Python ファイルはかなり少なく、実体として確認できたのは主に以下。

- `src/postprocess/task_rules_ver05_functions.py`
- `src/postprocess/apply_style_ver2.py`
- `src/postprocess/apply_style_ver3.py`
- `src/postprocess/apply_style_ver4.py`
- `src/postprocess/apply_style_ver5.py`
- `src/test/test_lora.py`
- `src/test/test_stable_diffusion.py`

このため、提出に近い現行コードを確認する場合、`backup` を混ぜると誤判定しやすい。

## What The Current Code Actually Uses

### 1. GroundingDINO

現行の中心コード:

- `src/postprocess/task_rules_ver05_functions.py`

確認できた事実:

- GroundingDINO の config は引き続き `/workspace/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` を使う。
- checkpoint は、監査時点では `third_party` を直参照していた。
- ただし実ファイルは `/workspace/weights/groundingdino_swint_ogc.pth` にある。
- このズレを解消するため、`src/postprocess/task_rules_ver05_functions.py` を修正し、今は以下の優先順で解決する。
	- `/workspace/weights/groundingdino_swint_ogc.pth`
	- `/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth`

つまり現時点の現行 `src` コードは、`weights` を優先して使う。

### 2. SAM

現行の中心コード:

- `src/postprocess/task_rules_ver05_functions.py`

確認できた事実:

- `SAM` は最初から `/workspace/weights/sam_vit_h_4b8939.pth` を参照している。
- `third_party` 側の SAM checkpoint 参照は、現行の非 `backup` `src` には見つからなかった。

つまり現時点の現行 `src` コードは、`SAM` を `weights` から使う。

### 3. Stable Diffusion / LoRA

対象ファイル:

- `src/postprocess/apply_style_ver2.py`
- `src/postprocess/apply_style_ver3.py`
- `src/postprocess/apply_style_ver4.py`
- `src/postprocess/apply_style_ver5.py`
- `src/test/test_lora.py`
- `src/test/test_stable_diffusion.py`

確認できた事実:

- ベースモデルは `runwayml/stable-diffusion-v1-5` を `from_pretrained(...)` で読む。
- これはローカル固定パスではなく Hugging Face モデル ID。
- 実行時のキャッシュ先は `/workspace/.cache/huggingface`。
- notebook 出力にも `/workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/...` からの load が残っている。
- LoRA は `/workspace/weights/lora` の `anime.safetensors` を読む。

つまり Stable Diffusion 本体は `weights/` ではなく `.cache/huggingface`、LoRA だけが `weights/`。

### 4. RAFT

対象ファイル:

- `src/postprocess/task_rules_ver05_functions.py`

確認できた事実:

- `RAFT` は `/workspace/third_party/RAFT/models/raft-things.pth` を参照する。
- 現行の非 `backup` `src` で `weights/` 側の RAFT 参照は見つからない。

### 5. XMem

対象ファイル:

- `src/postprocess/task_rules_ver05_functions.py`

確認できた事実:

- 以下の候補パス探索になっている。
	- `params["xmem_model_path"]`
	- `/workspace/third_party/XMem/saves/XMem.pth`
	- `/workspace/weights/XMem.pth`
	- `/workspace/weights/xmem.pth`
	- `/workspace/models/XMem.pth`

つまり XMem はまだ配置が統一されていない。

## Notebook Classification

### A. 提出チューニングに近い notebook

以下は `src.postprocess.task_rules_ver05_functions` を import している。

- `notebook/task_rules_ver05_trial_01.ipynb`
- `notebook/task_rules_ver05_trial_02.ipynb`
- `notebook/task_rules_ver05_trial_03.ipynb`
- `notebook/task_rules_ver05_trial_04.ipynb`
- `notebook/task_rules_ver05_trial_05.ipynb`
- `notebook/task_rules_ver05_trial_07.ipynb`
- `notebook/task_rules_ver05_trial_08_1.ipynb`
- `notebook/task_rules_ver05_trial_08_2.ipynb`

これらは notebook 自身が GroundingDINO / SAM の checkpoint を直接持つのではなく、`src/postprocess/task_rules_ver05_functions.py` の実装に従う。

したがって、これらの notebook 経路で効く GroundingDINO / SAM の参照先は、現時点では以下。

- GroundingDINO checkpoint: `/workspace/weights/groundingdino_swint_ogc.pth` 優先
- SAM checkpoint: `/workspace/weights/sam_vit_h_4b8939.pth`

### B. 単発検証・古い直書き notebook / script

以下には、GroundingDINO の古い `third_party` 直書きが残っている。

- `notebook/test/task_02_background.py`
- `notebook/test/task_02_background.ipynb`
- `notebook/test/task_02_background copy.ipynb`
- `notebook/test/task_01_zoom_in.ipynb`
- `notebook/parse_instruction/grounding_dino_sam_02.py`
- `notebook/parse_instruction/grounding_dino_sam_02.ipynb`

これらは「現在の提出経路」と同一視しない方がよい。

## Backup And Missing Entrypoints

重要事項:

- `scripts/run_submit_baseline_ver03.sh` は `/workspace/src/submit_baseline_ver03.py` を呼ぶが、現 workspace にはそのファイルが存在しない。
- `scripts/run_submit_baseline_ver04.sh` も `/workspace/src/submit_baseline_ver04.py` を呼ぶが、同様に存在しない。
- 実体として残っているのは `src/backup/submit_baseline_ver03.py` など `backup` 側。

したがって、現 workspace では `scripts/run_submit_baseline_ver03.sh` / `ver04.sh` をそのまま提出経路と見なすのは危険。

## Practical Verdict

提出前に最低限押さえるべき点は以下。

1. GroundingDINO は「もう `third_party` を使っていない」とは言い切れなかった。
	 監査時点では現行 `src` が `third_party` を見ていたため、`weights` 優先に修正した。

2. SAM は現行 `src` でも `weights` を使っている。

3. Hugging Face モデルは `weights/` ではなく `/workspace/.cache/huggingface` に自動展開される。

4. `third_party` 直書きが残っている notebook はあるが、それは主に古い単発検証コード。

5. 提出スクリプト本体の欠損があるため、competition 用の実行経路を固定するなら、どの entrypoint を採用するかを先に決める必要がある。

## Recommended Cleanup Order

1. GroundingDINO / SAM / XMem の path 解決を `src` 側の共通 helper に統一する。
2. `third_party` 直書きの notebook は「旧検証用」と明記するか、`weights` 優先に修正する。
3. 提出に使う entrypoint を 1 本に固定し、存在しない `src/submit_baseline_ver03.py` / `ver04.py` 参照を解消する。
