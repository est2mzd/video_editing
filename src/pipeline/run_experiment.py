# coding: utf-8  # 背景意図: 日本語コメント対応

"""  # 背景意図: モジュール責務を明示
run_experiment.py
背景意図:
    - 全パイプラインを統合して1回の実験を実行する
    - config駆動で再現性のある実験を可能にする
    - ログを残してワークショップ資料に繋げる
"""  # 背景意図: ドキュメント終了

import json  # 背景意図: config/ログ保存
from pathlib import Path  # 背景意図: パス操作
from datetime import datetime  # 背景意図: 実験ID生成

from src.utils.io_video import VideoIO  # 背景意図: 動画I/O
from src.preprocess.resize import ResizeManager  # 背景意図: リサイズ
from src.model.vace_wrapper import VACEWrapper  # 背景意図: モデル実行
from src.eval.evaluator import Evaluator  # 背景意図: ローカル評価
from src.eval.constraints import ConstraintChecker  # 背景意図: 提出チェック


class ExperimentRunner:  # 背景意図: 実験全体を管理
    """  # 背景意図: クラス責務明示
    1つの実験（動画1本＋prompt1つ）を実行する。
    """  # 背景意図: ドキュメント終了

    def __init__(self, config: dict):  # 背景意図: config注入
        self.config = config  # 背景意図: 全処理で参照

        # --- モジュール初期化 ---  # 背景意図: 再利用性確保
        self.video_io = VideoIO()  # 背景意図: 動画処理
        self.resizer = ResizeManager(config["preprocess"]["long_side"])  # 背景意図: サイズ統一
        self.model = VACEWrapper(config)  # 背景意図: モデル実行
        self.evaluator = Evaluator()  # 背景意図: スコア計算
        self.checker = ConstraintChecker()  # 背景意図: 提出制約チェック

    def run(self, input_video: str, prompt: str) -> dict:  # 背景意図: 実験実行
        """1回の実験を実行する。"""  # 背景意図: 役割明示

        # =========================
        # 実験ID生成
        # =========================

        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # 背景意図: 一意なID
        base_dir = Path(self.config["output"]["base_dir"]) / exp_id  # 背景意図: 出力ルート

        # --- ディレクトリ構成 ---  # 背景意図: 可視化・デバッグ容易化
        frames_in_dir = base_dir / "frames_in"
        frames_resized_dir = base_dir / "frames_resized"
        frames_out_dir = base_dir / "frames_out"
        frames_restored_dir = base_dir / "frames_restored"
        video_out_path = base_dir / "output.mp4"
        log_path = base_dir / "log.json"

        for d in [frames_in_dir, frames_resized_dir, frames_out_dir, frames_restored_dir]:
            d.mkdir(parents=True, exist_ok=True)  # 背景意図: 必要ディレクトリ作成

        # =========================
        # 1. 動画 → フレーム
        # =========================

        meta_video = self.video_io.video_to_frames(  # 背景意図: 入力分解
            input_video,
            frames_in_dir,
        )

        # =========================
        # 2. リサイズ
        # =========================

        resize_meta = self.resizer.resize_dir(  # 背景意図: 推論サイズへ変換
            frames_in_dir,
            frames_resized_dir,
        )

        # =========================
        # 3. モデル推論
        # =========================

        self.model.run(  # 背景意図: VACE実行
            input_dir=str(frames_resized_dir),
            output_dir=str(frames_out_dir),
            prompt=prompt,
        )

        # =========================
        # 4. 元サイズ復元
        # =========================

        self.resizer.restore_dir(  # 背景意図: 提出条件に戻す
            frames_out_dir,
            frames_restored_dir,
            resize_meta,
        )

        # =========================
        # 5. フレーム → 動画
        # =========================

        self.video_io.frames_to_video(  # 背景意図: 動画再構築
            frames_restored_dir,
            video_out_path,
            fps=meta_video["fps"],  # 背景意図: fps維持
        )

        # =========================
        # 6. 制約チェック
        # =========================

        ok, reason = self.checker.check(  # 背景意図: 提出可否判定
            input_video,
            str(video_out_path),
        )

        if not ok:  # 背景意図: NGなら即終了
            raise RuntimeError(f"Constraint failed: {reason}")  # 背景意図: 明示的エラー

        # =========================
        # 7. ローカル評価
        # =========================

        scores = self.evaluator.evaluate(  # 背景意図: スコア算出
            frames_in_dir,
            frames_restored_dir,
        )

        # =========================
        # 8. ログ保存
        # =========================

        log = {  # 背景意図: 実験再現性確保
            "exp_id": exp_id,
            "input_video": input_video,
            "prompt": prompt,
            "scores": scores,
            "config": self.config,
        }

        with open(log_path, "w") as f:  # 背景意図: JSON保存
            json.dump(log, f, indent=2)

        return log  # 背景意図: 上位で利用可能