# Notebook Analysis and src/ Comparison

## 概要
4つのnotebookを正確に分析し、src/postprocess/ との差異を記録する。
gt_unit_test.py がnotebook相当の出力を出すためのパラメータ・呼び出しフローを確認した。

---

## 共通インフラ
全3notebook（dolly_in_02, add_object_05, apply_style_03）は以下の `run_action_core` を使用する。

```python
def run_action_core(video_path_in, video_path_out, action, params_override=None, ...):
    frames, fps, width, height = load_video(video_path_in)
    rule = dict(rules.get(action, {}))        # task_rules_ver05.json から取得
    method = str(rule.get('method', 'identity'))
    params = dict(rule.get('params', {}))      # rules の base params
    if params_override:
        params.update(params_override)         # override で上書き
    out_frames = task_rule_funcs.run_method(
        method=method,
        frames=frames,
        params=params,
        instruction=instruction,
        logger=logger,
    )
```

### task_rules_ver05.json の各アクション
| action       | method                    | base_params                              |
|--------------|---------------------------|------------------------------------------|
| dolly_in     | progressive_crop_resize   | `{"start_scale": 1.0, "end_scale": 1.3}` |
| zoom_in      | crop_resize               | `{"max_scale": 1.3}`                     |
| add_object   | identity                  | `{}`                                     |
| apply_style  | stylize                   | `{}`                                     |

### task_rule_funcs の場所
- dolly_in_02.ipynb: `from backup import task_rules_ver05_functions` → `/workspace/src/backup/task_rules_ver05_functions.py`
- add_object_05.ipynb / apply_style_03.ipynb: `from src.postprocess import task_rules_ver05_functions`
  （現在は refactored されて src/postprocess/ には存在しないため、src/backup に統一して使用する）

---

## dolly_in_02.ipynb

### 対象
- video: `wyzi9GNZFMU_0_0to121.mp4`
- annotation[0], tasks[0]
- instruction: "Apply a smooth dolly in effect toward the man's face..."
- tasks[0].target: "man's face"
- tasks[0].constraints: ["smooth_motion"]
- tasks[0].params: `{"motion_type": "dolly_in", "start_framing": "medium_shot", "end_framing": "close_up"}`

### notebookが構築するparams（run_action_coreに渡す）
```python
params = dict(first_task.get('params', {}))
# = {"motion_type": "dolly_in", "start_framing": "medium_shot", "end_framing": "close_up"}
params['_action'] = 'dolly_in'
params['_constraints'] = ['smooth_motion']
params['action'] = 'dolly_in'
params['end_scale'] = 0.5        # ★ 重要: rulesの1.3を上書き
params['constraints'] = ['smooth_motion']
params['target'] = "man's face"
params['video_path'] = 'wyzi9GNZFMU_0_0to121.mp4'
```

### 実際のパラメータ（rules base + override合成後）
```python
{
    "start_scale": 1.0,       # from rules
    "end_scale": 0.5,         # override ★
    "motion_type": "dolly_in",
    "start_framing": "medium_shot",
    "end_framing": "close_up",
    "_action": "dolly_in",
    "action": "dolly_in",
    "_constraints": ["smooth_motion"],
    "constraints": ["smooth_motion"],
    "target": "man's face",
    "video_path": "wyzi9GNZFMU_0_0to121.mp4",
}
```

### コール経路
`run_method(method='progressive_crop_resize', ...)` → `stable_zoom_in(frames, params, logger)` → action=='dolly_in' → `_stable_object_zoom_in(frames, params, instruction, logger)`

### _stable_object_zoom_in の動作
- end_scale=0.5 (<=1.0) → `object_end_scale = 1/0.5 = 2.0`
- `scales = linspace(1.0, 2.0, N)` (foreground scale up)
- 各フレーム: DINO+SAM でtargetを検出してマスク作成 → マスク前景を scale 倍に拡大合成

### src/postprocess/camera_ops.py との差異
- `camera_ops.stable_object_zoom_in` と `backup._stable_object_zoom_in` は IDENTICAL なロジック
- 差異なし（関数名が違うだけ）

### gt_unit_test.py での修正点
- **end_scale=0.5 を params に追加** する必要がある
- target="man's face" は既にScenarioに定義済み

---

## zoom_in_01.ipynb

### 対象
- video: `_pQAUwy0yWs_0_119to277.mp4`
- instruction: "zoom in on face"（notebook固定値）

### 実装
zoom_in_01.ipynb は `run_method` を使わず、完全に独自アルゴリズムを実装している。

```python
zoom_factor = 1.0           # ★ 固定値
TEXT_PROMPT = "face . person ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# 初回フレームのみDINO検出
boxes, _, _ = predict(model, transform_image(frames[0]), "face . person .", ...)
target_box = boxes[0]
cx = (x1+x2)/2, cy = (y1+y2)/2, bw = x2-x1, bh = y2-y1
bbox_long = max(bw, bh)

# スケール算出
target_scale = (bbox_long / max(w,h)) / zoom_factor
scales = np.linspace(1.0, target_scale, T_total)

# 各フレーム: cropして元サイズにresize
crop_w = int(w * scale), crop_h = int(h * scale)
x1 = cx - crop_w/2  (clamp)
cropped = frame[y1:y2, x1:x2]
resized = cv2.resize(cropped, (w,h))
```

### backup stable_zoom_in との対応
backupの `stable_zoom_in` は同じアルゴリズムで、パラメータが以下の場合に同一結果になる:
- `zoom_factor=1.0` を params に追加
- `text_prompt="face . person ."` (backup default と同じ)
- action に "dolly_in" が含まれない（dolly_inブランチに入らないため）

### src/postprocess/camera_ops.py との差異
camera_ops の `stable_zoom_in` は build_detection_prompts でプロンプトを組み立てるが、
zoom_factor は同じロジックで計算する。
**params["zoom_factor"]=1.0** を渡せば同等の結果になる。

### gt_unit_test.py での修正点
- zoom_in params に **`"zoom_factor": 1.0`** を追加

---

## add_object_05.ipynb

### 対象
- video: `1s9DER1bpm0_10_0to213.mp4`
- tasks[0].target: `"rhino_and_buffalo"` (annotation GT)
- tasks[0].params: `{"count": 2, "position": ["background", "mid-ground"], "spatial_distribution": "background", "density": "dense"}`

### notebookが構築するparams（最終ver9実行時）
```python
params = dict(first_task.get('params', {}))
# = {"count": 2, "position": [...], "spatial_distribution": "background", "density": "dense"}
params['_action'] = 'add_object'
params['_constraints'] = []
params['action'] = 'add_object'
params['constraints'] = []
params['target'] = "buffalo"          # ★ 強制上書き（GT target を無視）
params['add_object_version'] = 'ver9' # ★
params['temporal_smooth_alpha'] = 0.7 # ★
params['xmem_mem_every'] = 5          # ★
```

### コール経路
`run_method(method='identity', params={'action':'add_object', ...})` → action=='add_object' check → `add_object_frames(frames, params, instruction, logger)` → version='ver9' → `add_object_frames_ver9(...)`

### add_object_frames_ver9 の動作
- DINO で "buffalo ." を検出してSAMでマスク化
- オブジェクトをコピー・シフトして合成（centroid-based placement + EMA-smoothed center）

### 現在のgt_unit_test.py との差異
- target="buffalo" は Scenario に設定されているが実際にparamに追加されているか要確認
- **temporal_smooth_alpha=0.7**, **xmem_mem_every=5** が不足
- plan_map の params にこれらを追加する必要がある

---

## apply_style_03.ipynb

### 対象
- annotation[5] (tasks[0].action=='apply_style' の最初のエントリ)
- video: `94msufYZzaQ_26_0to273.mp4`
- tasks[0].params: `{"style": "ukiyo-e"}` (GT値)
- instruction: "Transform the entire video into a traditional Japanese Ukiyo-e..."

### notebookが構築するparams
```python
params = dict(first_task.get('params', {}))
# = {"style": "ukiyo-e"}  ← GT値
params['_action'] = 'apply_style'
params['_constraints'] = []
params['action'] = 'apply_style'
params['constraints'] = []
params['style'] = 'oil_painting'   # ★ STYLE定数で上書き（GT 'ukiyo-e' を無視）
params['target'] = 'full_frame'    # tasks[0].target
params['video_path'] = TARGET_VIDEO
```

### コール経路
`run_method(method='stylize', params={'action':'apply_style', 'style':'oil_painting', ...})` → method=='stylize' → `stylize(frames, params)` → `apply_style_frames(frames, 'oil_painting')`

### src/postprocess/style_ops.py との差異
- backup の `stylize` と src の `stylize` は同一ロジック（`apply_style_frames` を呼ぶ）
- 差異なし

### gt_unit_test.py での修正点
- style='oil_painting' は既に Scenario params_override に設定済み
- 問題なし（現在のコードで動作するはず）

---

## 修正サマリー

| case         | 問題の原因                                    | 修正内容                                                  |
|--------------|----------------------------------------------|----------------------------------------------------------|
| dolly_in     | end_scale未設定でmax_scale=1.3が使われている  | plan_map params に `"end_scale": 0.5` を追加             |
| zoom_in      | zoom_factor未設定でmax_scale=1.3から計算される | plan_map params に `"zoom_factor": 1.0` を追加           |
| add_object   | temporal_smooth_alpha, xmem_mem_every 未設定   | plan_map params にこれらを追加。targetが正しく渡されているか確認 |
| apply_style  | 問題なし（既にstyle='oil_painting'設定済み）   | なし                                                     |

## 10フレームストライド
全動画をフレーム全体で処理するが、10フレームおきにサンプリングする:
```python
frames = frames[::10]
```

## 呼び出し方法の統一案
notebookと同じ `backup.task_rules_ver05_functions.run_method(method=, ...)` を使う方が確実。
ただし camera_ops の実装と backup は等価なので dispatcher 経由でも可（パラメータが正しければ）。

---

## 参照ファイル
- `/workspace/src/backup/task_rules_ver05_functions.py` — 正規リファレンス実装
- `/workspace/logs/submit/submission_ver05_json/task_rules_ver05.json` — method/base_params
- `/workspace/data/annotations_gt_task_ver10.json` — GT annotation
- `/workspace/src/test/gt_unit_test.py` — テストスクリプト
- `/workspace/src/postprocess/camera_ops.py` — zoom/dolly実装
- `/workspace/src/postprocess/style_ops.py` — スタイル実装
- `/workspace/src/postprocess/add_object_service.py` — addオブジェクト実装
