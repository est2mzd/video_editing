
## 1. 全体アーキテクチャ

```
run.sh
  ↓
run_experiment.py
  ↓
ExperimentRunner.run()

  for each video:

    ① fps取得
    ② tmp_dir生成（video単位）
    ③ extract_frames
    ④ resize（必要なら）
    ⑤ VACE実行
    ⑥ restore（元サイズ）
    ⑦ encode_video（fps維持）
    ⑧ constraintチェック
    ⑨ evaluation
    ⑩ log保存
```

## 2. ディレクトリ構成（修正版）
```
project/
├── src/
│   ├── run_experiment.py     # エントリポイント
│   ├── runner.py             # 全体制御クラス
│   │
│   ├── data/
│   │   ├── video_io.py       # 動画⇄フレーム
│   │   └── frame_manager.py  # tmp_dir管理（重要）
│   │
│   ├── preprocess/
│   │   └── resize.py         # resize/restore
│   │
│   ├── model/
│   │   └── vace_wrapper.py   # VACE実行
│   │
│   ├── eval/
│   │   ├── evaluator.py      # スコア計算
│   │   └── constraints.py    # 提出条件チェック
│   │
│   ├── utils/
│   │   ├── config.py         # config読込
│   │   └── logger.py         # ログ
│
├── configs/
│   └── base.yaml
│
├── scripts/
│   ├── run.sh
│   └── run_multi_seed.sh
│
├── data/
│   ├── input_videos/
│   ├── output_videos/
│   ├── tmp_frames/           # run_id単位で分離
│
├── logs/
├── tensorboard/
├── third_party/
│   └── VACE/
```
