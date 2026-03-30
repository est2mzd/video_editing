# preprocess/task_rules_ver05_functions.py

`/workspace/src/preprocess/task_rules_ver05_functions.py` は、`task_rules_ver05.json` ベースで 1 action = 1 関数を提供するチューニング用ユーティリティです。

## 目的
- 設計者が action ごとの関数実装と `params` を変更し、
- 1動画ずつ before/after を比較しながら調整する。

## 主な API
- `read_task_rules(file_path_json, video_id=0)`
- `run_action(action, video_path_in, video_path_out, params=None)`
- `ACTION_FUNCTIONS` (action 名 -> 関数)

## 関数シグネチャ
各 action 関数は次の形で統一されています。

```python
def zoom_in(video_path_in, video_path_out, params):
    ...
```

同様に `zoom_out`, `dolly_in`, `change_color`, `apply_style` など、
`task_rules_ver05.json` の action 一覧に対応する関数を定義しています。

## 使い方（最小例）

```python
from pathlib import Path
from src.preprocess.task_rules_ver05_functions import read_task_rules, run_action

rules = read_task_rules('/workspace/logs/submit/submission_ver05_json/task_rules_ver05.json')
params = dict(rules['actions']['zoom_in'].get('params', {}))
params['max_scale'] = 1.35

run_action(
    action='zoom_in',
    video_path_in='/workspace/data/videos/sample.mp4',
    video_path_out='/workspace/logs/submit/submission_ver05_tuning/sample_zoom_in.mp4',
    params=params,
)
```

## 実装方針
- 1機能1関数を維持しつつ、動画I/Oは共通ヘルパーで管理
- 出力の frame 数/解像度は入力に合わせる
- 未実装系 action は identity（pass-through）で安全に処理

## Notebook 連携
- `/workspace/notebook/check_submission_ver05.ipynb` から import して利用可能
- `VIDEO_INDEX`, `ACTION_NAME`, `PARAMS_OVERRIDE` を変更して反復チューニングできます
