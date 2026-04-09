
# 1. 現状
- 提出用のファイルを整えている

# 2. 処理の流れ
- Instruction から action, target を取得する
    - python : /workspace/src/parse/instruction_parser_v3_rulebase_trial020_singlefile.py
    - 提出用 instruction : /workspace/data/annotations.jsonl
    - video dir : /workspace/data/videos

- action, target, instruction, video frameの1frame目 を使って、使用する関数を決める
    - python : /workspace/src/postprocess/dispatcher.py
    - 現在は、 action, target, instruction のみを関数に渡している
    - 出力先 : /workspace/logs/submit/

# 3. 対応すべきこと
- cp /workspace/src/postprocess/dispatcher.py　/workspace/src/postprocess/dispatcher_v2.py
- kak
