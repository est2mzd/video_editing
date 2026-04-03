# task_rules_ver05_functions.py 分割リファクタリング案

対象: [src/postprocess/task_rules_ver05_functions.py](src/postprocess/task_rules_ver05_functions.py)

## 背景

- 現状は単一ファイルに 60 関数以上が集約され、責務が混在している。
- 主な混在要素:
	- ルール実行ディスパッチ
	- 背景/色/カメラ系の軽量 OpenCV 処理
	- GroundingDINO / SAM / RAFT / XMem のモデルロード
	- add_object の複数バージョン実装（ver1-9）
	- マスク・bbox・時系列安定化ユーティリティ
- 変更影響範囲が広く、レビュー・テスト・デバッグコストが高い。

## 分割方針

- 方針1: 「処理の種類」で分割する（効果処理、検出、追跡、合成、dispatch）。
- 方針2: まず import 互換を維持する（既存の `run_method` シグネチャを変えない）。
- 方針3: 大型領域（add_object ver群）を最優先で隔離する。
- 方針4: グローバルモデル変数を専用モジュールへ集約する。

## 提案ディレクトリ構成

推奨ファイル構成:

- [src/postprocess/__init__.py](src/postprocess/__init__.py)
- [src/postprocess/dispatcher.py](src/postprocess/dispatcher.py)
- [src/postprocess/progress.py](src/postprocess/progress.py)
- [src/postprocess/color_utils.py](src/postprocess/color_utils.py)
- [src/postprocess/mask_ops.py](src/postprocess/mask_ops.py)
- [src/postprocess/camera_ops.py](src/postprocess/camera_ops.py)
- [src/postprocess/background_ops.py](src/postprocess/background_ops.py)
- [src/postprocess/style_ops.py](src/postprocess/style_ops.py)
- [src/postprocess/model_registry.py](src/postprocess/model_registry.py)
- [src/postprocess/detectors.py](src/postprocess/detectors.py)
- [src/postprocess/trackers.py](src/postprocess/trackers.py)
- [src/postprocess/add_object_versions.py](src/postprocess/add_object_versions.py)
- [src/postprocess/add_object_service.py](src/postprocess/add_object_service.py)
- [src/postprocess/types.py](src/postprocess/types.py)

## 関数マッピング案

### 1) progress.py

- `_resolve_video_name_for_progress`
- `_iter_frames_with_progress`

### 2) color_utils.py

- `extract_target_color`
- `target_color_bgr`

### 3) mask_ops.py

- `estimate_foreground_mask`
- `_mask_area`
- `_mask_iou`
- `_keep_largest_component`
- `_refine_mask`
- `_mask_to_box`
- `_clip_box`
- `_expand_box`
- `_fuse_masks_adaptive`
- `_build_fg_mask_from_boxes`
- `_derive_dynamic_box_from_masks`
- `_inpaint_masked_background`
- `_warp_mask_with_flow`
- `_temporal_stabilize_mask`

### 4) model_registry.py

- グローバルモデル変数群
	- `GROUNDING_DINO_MODEL`
	- `GROUNDING_DINO_TRANSFORMS`
	- `SAM_PREDICTOR`
	- `RAFT_MODEL`
	- `RAFT_DEVICE`
	- `XMEM_NETWORK`
	- `XMEM_DEVICE`
	- `XMEM_IMAGE_TO_TORCH`
	- `XMEMInferenceCore`
- `_resolve_grounding_dino_checkpoint_path`
- `_find_xmem_model_path`
- `load_grounding_dino_model`
- `load_sam_predictor`
- `load_raft_model`
- `load_xmem_model`
- `_make_xmem_processor`

### 5) detectors.py

- `get_sam_mask_from_box`
- `_detect_all_boxes`
- `detect_primary_box`
- `_split_target_keywords`
- `_resolve_target_union_box`

### 6) trackers.py

- `_estimate_optical_flow`
- `_xmem_predict_mask`
- `_track_mask_with_xmem_or_ostrack`

### 7) camera_ops.py

- `_compose_scaled_mask_foreground`
- `_stable_object_zoom_in`
- `stable_zoom_in`
- `zoom_out`
- `perspective_warp`
- `horizontal_shift`

### 8) background_ops.py

- `change_background_color`
- `replace_background`
- `inpaint`

### 9) style_ops.py

- `stylize`
- `blur_or_brightness`
- `sharpness`
- `histogram_match`
- `identity`

### 10) add_object_versions.py

- `_resolve_add_object_prompts`
- `_compose_shifted_add_object`
- `add_object_frames_ver1` 〜 `add_object_frames_ver9`

### 11) add_object_service.py

- `add_object_frames`（version switch）

### 12) dispatcher.py

- `run_method` のみ保持
- 依存先は各 *_ops / add_object_service へ委譲

## 依存ルール（循環参照回避）

- `dispatcher.py` は他モジュールを呼ぶだけ（逆依存禁止）。
- `add_object_versions.py` は `detectors.py` / `mask_ops.py` / `trackers.py` / `progress.py` に依存可。
- `detectors.py` と `trackers.py` は `model_registry.py` に依存可。
- `model_registry.py` は他業務ロジックに依存しない。

## 段階的移行プラン

### Phase 0: 安全網

- 最低限の回帰テスト観点を固定:
	- `segment_and_replace` 実行で動画フレーム数が維持される
	- `crop_resize` / `progressive_crop_resize` が実行できる
	- `add_object` が最低 1 バージョン実行できる

### Phase 1: 純粋ユーティリティ分離（低リスク）

- `progress.py`, `color_utils.py`, `mask_ops.py` を先に分離。
- 既存ファイル側は import して再エクスポート可能にして差分を小さくする。

### Phase 2: モデルロード分離（中リスク）

- `model_registry.py` へ移動。
- グローバルキャッシュ初期化順序の差異に注意。

### Phase 3: 実処理分離（中リスク）

- `camera_ops.py`, `background_ops.py`, `style_ops.py` を分離。

### Phase 4: add_object 群分離（高リスク）

- `add_object_versions.py` と `add_object_service.py` を分割。
- まず ver1, ver2 を移し、次に ver3-9 を段階移行。

### Phase 5: dispatcher の薄型化

- 旧ファイル [src/postprocess/task_rules_ver05_functions.py](src/postprocess/task_rules_ver05_functions.py) は互換ラッパーのみ残す。
- 最終的に `run_method` を [src/postprocess/dispatcher.py](src/postprocess/dispatcher.py) に一本化。

## 最小互換戦略

- 当面は [src/postprocess/task_rules_ver05_functions.py](src/postprocess/task_rules_ver05_functions.py) に以下のみ残す:
	- 既存 import 互換のための再エクスポート
	- `run_method` へのフォワード
- 呼び出し側の修正を不要化しつつ内部だけ分割する。

## リスクと対策

- リスク: グローバルモデルの二重初期化
	- 対策: `model_registry.py` に初期化責務を一元化
- リスク: add_object ver間の暗黙依存
	- 対策: 共通処理を `mask_ops.py` / `detectors.py` へ抽出
- リスク: import 循環
	- 対策: 依存方向を dispatcher 起点の一方向に固定

## 受け入れ条件

- 既存ノートブックから `run_method` 呼び出しが壊れない。
- `method="segment_and_replace"` の処理結果が分割前後で同等。
- `add_object` の既定バージョン実行が可能。
- 主要処理で例外率が上がらない。

## まず着手する具体的タスク（推奨順）

- 1. [src/postprocess/task_rules_ver05](src/postprocess/task_rules_ver05) を作成
- 2. `progress.py` / `color_utils.py` / `mask_ops.py` を移動
- 3. 旧ファイルから新モジュールを import して動作確認
- 4. `background_ops.py` と `dispatcher.py` を先に分離（`replace_background` を早期安定化）
- 5. 最後に `add_object_versions.py` を分離
