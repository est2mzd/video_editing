# zoom_in_01.ipynb と src/postprocess の対応分析

対象Notebook:
- [notebook/zoom_in_01.ipynb](notebook/zoom_in_01.ipynb)

確認した結論:
- Notebookの「DINOで初回bbox取得 + OpenCV crop/resizeでzoom in」に最も近い実装は、
	[src/postprocess/camera_ops.py](src/postprocess/camera_ops.py) の `stable_zoom_in`。
- ただし、Notebook側コードは実質「DINO + OpenCV」で、処理本体では SAM を使っていない。
- `src/postprocess` 側で「DINO + SAM + OpenCV による zoom-in 相当」に最も近いのは、
	同ファイルの `stable_object_zoom_in`（`dolly_in` 分岐で使用）。

## 1. Notebookの処理フロー（要約）

[notebook/zoom_in_01.ipynb](notebook/zoom_in_01.ipynb) の主要ロジック:
- GroundingDINO をロード
- 1フレーム目で bbox を取得
- bbox中心とスケール列を作る
- 各フレームで crop → resize（OpenCV）

この流れは、`stable_zoom_in` の通常分岐と一致する。

## 2. 同等実装がある場所

### 2-1. ルーティング入口
- [src/postprocess/dispatcher.py](src/postprocess/dispatcher.py)

`run_method` 内で以下が `stable_zoom_in` にルーティングされる:
- `method == "crop_resize"`
- `method == "progressive_crop_resize"`

つまり、task rule の method が上記なら Notebookの zoom-in 系と同じ系統に入る。

### 2-2. zoom-in本体（DINO + OpenCV）
- [src/postprocess/camera_ops.py](src/postprocess/camera_ops.py)

`stable_zoom_in` が Notebookとほぼ同じ責務を持つ:
- `detect_primary_box(...)` で初回 bbox を取得
- `np.linspace` でスケール列を作る
- 各フレームで crop 範囲計算 → `cv2.resize` で元解像度に戻す

Notebookの `# 3 初回bbox取得` と `# 5 各フレーム処理` に対応。

### 2-3. DINO推論実装
- [src/postprocess/detectors.py](src/postprocess/detectors.py)
- [src/postprocess/model_registry.py](src/postprocess/model_registry.py)

対応関係:
- `load_grounding_dino_model` でモデル初期化
- `detect_all_boxes` / `detect_primary_box` で bbox 推論

Notebookで手書きしている `load_model(...)` + `predict(...)` 部分に対応。

## 3. SAM を使う zoom-in 相当の場所

### 3-1. DINO + SAM + OpenCV を併用する実装
- [src/postprocess/camera_ops.py](src/postprocess/camera_ops.py)

`stable_object_zoom_in` は以下を実施:
- `resolve_target_union_box`（DINO）で対象候補
- `get_sam_mask_from_box`（SAM）で対象マスク
- `compose_scaled_mask_foreground`（OpenCV）で前景のみ拡大合成

このため、Notebookの説明「dino + sam + opencv2」により近いのは `stable_object_zoom_in`。

### 3-2. ただし呼ばれる条件が異なる
- `stable_zoom_in` 内で、`action == "dolly_in"` または `motion_type == "dolly_in"` のときに
	`stable_object_zoom_in` 分岐へ入る。
- 通常の `zoom_in` (`crop_resize`) は SAM なしの crop/resize 分岐。

## 4. 追加メモ（差分）

- Notebookの実コードは、表示上は SAM 言及があっても zoom-in 本体では SAM未使用。
- `src/postprocess` は2系統を持つ:
	- `stable_zoom_in`: DINO + OpenCV（Notebook本体と同等）
	- `stable_object_zoom_in`: DINO + SAM + OpenCV（より高度な前景ベース拡大）

## 5. まとめ

質問「どこで同じ実装がされているか」への回答:
- 第一候補（同等）:
	- [src/postprocess/camera_ops.py](src/postprocess/camera_ops.py) の `stable_zoom_in`
	- [src/postprocess/dispatcher.py](src/postprocess/dispatcher.py) の `crop_resize` / `progressive_crop_resize` 分岐
	- [src/postprocess/detectors.py](src/postprocess/detectors.py) の DINO bbox取得
- SAM込みで近い実装:
	- [src/postprocess/camera_ops.py](src/postprocess/camera_ops.py) の `stable_object_zoom_in`

