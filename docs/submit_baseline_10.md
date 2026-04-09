
# 1. 現状
- 提出用のファイルを整えている

# 2. 処理の流れ
- Instruction から action, target を取得する
    - python : /workspace/src/parse/instruction_parser_v3_rulebase_trial020_singlefile.py
    - 提出用 instruction : /workspace/data/annotations.jsonl
    - video dir : /workspace/data/videos

- action, target, instruction を使って、使用する関数を決める
    - python : /workspace/src/postprocess/dispatcher.py
    - 現在は、 action, target, instruction のみを関数に渡している
    - 出力先 : /workspace/logs/submit/

# 3. 対応すべきこと
- cp /workspace/src/postprocess/dispatcher.py　/workspace/src/postprocess/dispatcher_v2.py
- dispatcher_v2.py に 以下の変更を反映する
    1. 各関数 の 入力が　action, target, instruction を、何らかの形で読み込めるようにする
    2. 以下の関数ファイルが使えるように、既存の分岐の関数を入れ替える
        2.1. /workspace/src/postprocess/add_effect.py
        2.2. /workspace/src/postprocess/add_object.py
        2.3. /workspace/src/postprocess/apply_style_ver5.py
            - これは、apply_style_video_foreground_background() をつかう
        2.4. /workspace/src/postprocess/change_color.py
        2.5. /workspace/src/postprocess/dolly_in.py
        2.6. /workspace/src/postprocess/replace_background.py
            - 前景と背景を分けるために、 入力の foreground_prompt='man .', を次のように作成する
                - /workspace/weights/yolo/yolov8m.pt を使って person がいる場合 "person ."
                - person がいない場合、/workspace/notebook/yolo_01.ipynb　の震度を参考に前景クラスを決めて "クラス名 ." とする
        2.7. /workspace/src/postprocess/zoom_in.py

- 実行ファイルを作成する
    - cp /workspace/src/run_video_editor.py /workspace/src/run_video_editor_v2.py
    - cp /workspace/configs/base_config.yaml /workspace/configs/base_config_v2.yaml
    - run_video_editor_v2.py と base_config_v2.yaml を修正し、上記の動画修正ができるようにする

- 全ファイルを処理し、提出ファイルを作成する
    - コンペの仕様: /workspace/docs/Overview.md
    - 要求1: 成功するまで試行錯誤する過程を、この .md に記載. １回試行 -> 目的、結果を記載
    - 要求2: 動画は全フレーム処理する
    - 要求3: 各動画に対して、instructionの分析結果(action, target, その他)を *.logに記載する
    - 権限: ファイルの作成読み込み、コマンド実行の権限を与える

# 4. 試行錯誤ログ

## 試行 1
- 目的: run_video_editor_v2.py を全件実行して初期動作を確認
- 実行: /usr/bin/python /workspace/src/run_video_editor_v2.py
- 結果: 起動直後に ImportError（dispatcher_v2.py の相対 import）で停止
- 対応: dispatcher_v2.py の import を `from utils.video_utility import ...` に修正

## 試行 2
- 目的: A の解釈に合わせ、2.1-2.7 だけ差し替え、その他分岐は既存関数維持を確認
- 実行: dispatcher_v2.py を修正し、非対象分岐（zoom_out/change_camera_angle/orbit_camera/remove_object）を既存関数へ復帰
- 結果: 反映完了
- 対応: 続けて分岐単体テストを作成

## 試行 3
- 目的: B の要件に従い、1動画に対して 2.1-2.7 を1分岐ずつ実行してエラー有無を確認
- 実行: /usr/bin/python /workspace/src/test/test_dispatcher_v2_single_video.py --row-index 0 --max-frames 8
- 結果: zoom_in 実行時に GroundingDINO 重みパス不一致（/workspace/weights/groundingdino_swint_ogc.pth 不在）を検出
- 対応: /workspace/src/postprocess/zoom_in.py の CHECKPOINT を /workspace/weights/groundingdino/groundingdino_swint_ogc.pth に修正

## 試行 4
- 目的: 修正後に分岐単体テストを再実行し、2.1-2.7 全分岐のエラーなしを確認
- 実行: PYTHONPATH=/workspace:/workspace/src /usr/bin/python /workspace/src/test/test_dispatcher_v2_single_video.py --row-index 0 --max-frames 8
- 結果: 成功。report: /workspace/logs/test/dispatcher_v2_single_video/20260409_154733/report.json
- 補足: 2.1-2.7 すべて success=true、出力フレーム数は入力と一致（8/8）

## 試行 5（進行中）
- 目的: C の要件に従い、全件（100件）を全フレームで実行
- 実行: PYTHONPATH=/workspace:/workspace/src /usr/bin/python /workspace/src/run_video_editor_v2.py
- 結果: 進行中（開始時点で rows=100、出力先は /workspace/logs/submit/all_20260409_155035）
- 補足: 各動画ごとの instruction 分析ログは output_dir/instruction_logs/*.log に保存する実装済み

## 試行 6（次試行向け調整）
- 目的: dolly_in の拡大率不足を改善し、change_color は instruction に「徐々に」がない場合は即時変更にする
- 実装:
    - dispatcher_v2.py の dolly_in デフォルト拡大率を `1.25 -> 1.85` に変更
    - dispatcher_v2.py の change_color で instruction 文面に `gradual/gradually/徐々に/だんだん/少しずつ` がある場合のみ gradual=True
    - change_color.py の `run_change_color_gradual_pipeline()` に `gradual` 引数を追加し、gradual=False 時は全フレームで即時反映（progress=1.0）
- 結果: コード反映完了（未実行）