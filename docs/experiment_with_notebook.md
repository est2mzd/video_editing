# Experiment With Notebook

このファイルには、`/workspace/notebook/vace_pipeline_ver03.ipynb` に対して notebook セルだけで実施した試行錯誤を時系列で記録する。

## Rules

- 実行と検証は notebook のセルで行う
- `subprocess` を使った検証は増やさない
- 各試行で、仮説・実行セルの要点・結果・次の判断を残す

## Log Template

### YYYY-MM-DD HH:MM UTC | Trial N

- Hypothesis:
- Notebook cell:
- Result:
- Next action:

## Existing Notes

### 2026-03-23 07:08 UTC | Initial inspection

- Hypothesis: `subprocess` で外部 Python を起動していることが notebook 実行失敗の主要因になっている可能性が高い。
- Notebook cell: 未実施。ここまでの確認は構造把握のみ。
- Result: `vace_pipeline_ver03.ipynb` には `subprocess` を使う診断セルと Easy Experiment 実行セルが存在し、`src/utils/vace_executor.py` も subprocess 前提だった。
- Next action: 次回以降は notebook セルから直接 import / 実行できる形に寄せ、試行結果をこのファイルへ都度追記する。
### 2026-03-23 07:14 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight failed: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 304: OS call failed or operation not supported on this OS
- Next action: 失敗要因に応じて path か import 順序を修正する。
### 2026-03-23 07:15 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight failed: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 304: OS call failed or operation not supported on this OS
- Next action: 失敗要因に応じて path か import 順序を修正する。
### 2026-03-23 07:15 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight failed: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 304: OS call failed or operation not supported on this OS
- Next action: 失敗要因に応じて path か import 順序を修正する。
### 2026-03-23 07:17 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight failed: No module named 'models.wan'
- Next action: 失敗要因に応じて path か import 順序を修正する。
### 2026-03-23 07:17 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 07:18 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Failed: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 304: OS call failed or operation not supported on this OS
- Next action: エラー内容に応じて notebook セルと in-process 実行ラッパを修正する。
### 2026-03-23 07:19 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Failed: VACE execution timeout (5s)
- Next action: エラー内容に応じて notebook セルと in-process 実行ラッパを修正する。
### 2026-03-23 07:20 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 07:20 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Failed: VACE execution timeout (5s)
- Next action: エラー内容に応じて notebook セルと in-process 実行ラッパを修正する。
### 2026-03-23 07:32 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Skipped: CUDA is not available in the notebook kernel. checks={'python': '/usr/bin/python', 'cuda_available': False, 'device_count': 0, 'wan_file': '/workspace/third_party/Wan2.1/wan/__init__.py', 'inference_module': '/workspace/third_party/VACE/vace/vace_wan_inference.py'}
- Next action: CUDA が見える notebook カーネルで再実行する。
### 2026-03-23 07:32 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 07:33 UTC | Trial 3

- Hypothesis: Batch execution should stay inside the notebook kernel and avoid hard errors when runtime requirements are missing.
- Notebook cell: Cell 13: batch run
- Result: status_counts={'skipped': 5}
- Next action: Inspect skipped / failed rows and continue with ad-hoc and easy experiment cells.
### 2026-03-23 07:33 UTC | Trial 4

- Hypothesis: Ad-hoc execution should degrade to skipped instead of raising when notebook runtime is not ready for generation.
- Notebook cell: Cell 23: ad-hoc run
- Result: Skipped: CUDA is not available in the notebook kernel. checks={'python': '/usr/bin/python', 'cuda_available': False, 'device_count': 0, 'wan_file': '/workspace/third_party/Wan2.1/wan/__init__.py', 'inference_module': '/workspace/third_party/VACE/vace/vace_wan_inference.py'}
- Next action: Retry on a CUDA-visible notebook kernel or continue with environment diagnostics.
### 2026-03-23 07:33 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 07:33 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Skipped: CUDA is not available in the notebook kernel. checks={'python': '/usr/bin/python', 'cuda_available': False, 'device_count': 0, 'wan_file': '/workspace/third_party/Wan2.1/wan/__init__.py', 'inference_module': '/workspace/third_party/VACE/vace/vace_wan_inference.py'}
- Next action: CUDA が見える notebook カーネルで再実行する。
### 2026-03-23 09:11 UTC | Trial 3

- Hypothesis: Batch execution should stay inside the notebook kernel and avoid hard errors when runtime requirements are missing.
- Notebook cell: Cell 13: batch run
- Result: status_counts={'error': 5}
- Next action: Inspect skipped / failed rows and continue with ad-hoc and easy experiment cells.
### 2026-03-23 09:14 UTC | Trial 4

- Hypothesis: Ad-hoc execution should degrade to skipped instead of raising when notebook runtime is not ready for generation.
- Notebook cell: Cell 23: ad-hoc run
- Result: Failed: VACE execution timeout (180s)
- Next action: Inspect stderr and adjust the in-process execution path.
### 2026-03-23 09:14 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 09:17 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Failed: VACE execution timeout (180s)
- Next action: エラー内容に応じて notebook セルと in-process 実行ラッパを修正する。
### 2026-03-23 10:04 UTC | Trial 3

- Hypothesis: Batch execution should stay inside the notebook kernel and avoid hard errors when runtime requirements are missing.
- Notebook cell: Cell 13: batch run
- Result: status_counts={'ok': 5}
- Next action: Inspect skipped / failed rows and continue with ad-hoc and easy experiment cells.
### 2026-03-23 10:11 UTC | Trial 4

- Hypothesis: Ad-hoc execution should degrade to skipped instead of raising when notebook runtime is not ready for generation.
- Notebook cell: Cell 23: ad-hoc run
- Result: Succeeded. output_path=/workspace/logs/notebooks/vace_pipeline_ver03/20260323_092527/case_9999_seed_42/output.mp4
- Next action: Inspect the generated video preview.
### 2026-03-23 10:11 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
### 2026-03-23 10:14 UTC | Trial 2

- Hypothesis: Easy Experiment も notebook カーネル内の in-process 実行へ切り替えれば subprocess なしで完走できる。
- Notebook cell: Cell 27: easy experiment run
- Result: Failed: VACE execution timeout (180s)
- Next action: エラー内容に応じて notebook セルと in-process 実行ラッパを修正する。
### 2026-03-23 13:06 UTC | Trial 4

- Hypothesis: Ad-hoc execution should degrade to skipped instead of raising when notebook runtime is not ready for generation.
- Notebook cell: Cell 23: ad-hoc run
- Result: Succeeded. output_path=/workspace/logs/notebooks/vace_pipeline_ver03/20260323_121119/case_9999_seed_42/output.mp4
- Next action: Inspect the generated video preview.
### 2026-03-23 13:06 UTC | Trial 1

- Hypothesis: subprocess を使わず notebook カーネル内の path と import を整えれば preflight が通る。
- Notebook cell: Cell 24: in-kernel preflight
- Result: In-kernel preflight succeeded.
- Next action: Easy Experiment も同じカーネル内実行へ切り替えて試す。
