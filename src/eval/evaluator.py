# coding: utf-8  # 背景意図: 日本語コメント対応

"""  # 背景意図: モジュール責務を明示
evaluator.py
背景意図:
    - 出力動画の品質をローカルで定量評価する
    - seedやpromptの比較を可能にする
    - コンペ提出前のスコア指標として使う
"""  # 背景意図: ドキュメント終了

from pathlib import Path  # 背景意図: パス操作
from typing import Dict, List  # 背景意図: 型ヒント

import cv2  # 背景意図: 画像読み込み
import numpy as np  # 背景意図: 数値計算


class Evaluator:  # 背景意図: 評価処理をカプセル化
    """  # 背景意図: クラス責務明示
    フレーム単位で動画品質を評価する。
    軽量・高速を優先。
    """  # 背景意図: ドキュメント終了

    def evaluate(self, input_dir: str, output_dir: str) -> Dict:  # 背景意図: メイン評価関数
        """評価を実行し、スコア辞書を返す。"""  # 背景意図: 役割明示

        # --- パス処理 ---  # 背景意図: OS非依存
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # --- フレーム取得 ---  # 背景意図: 順序保証
        in_frames = sorted(input_dir.glob("*.png"))
        out_frames = sorted(output_dir.glob("*.png"))

        # --- 基本チェック ---  # 背景意図: 不整合防止
        if len(in_frames) == 0 or len(out_frames) == 0:
            raise RuntimeError("Frames are empty.")

        if len(in_frames) != len(out_frames):
            raise RuntimeError("Frame count mismatch.")

        # =========================
        # 指標1: Temporal Consistency
        # =========================

        temporal_score = self._temporal_consistency(out_frames)  # 背景意図: フリッカー評価

        # =========================
        # 指標2: 内容変化量（編集量）
        # =========================

        edit_score = self._edit_magnitude(in_frames, out_frames)  # 背景意図: 変化量評価

        # =========================
        # 指標3: ノイズ/破綻検出
        # =========================

        stability_score = self._stability(out_frames)  # 背景意図: ノイズ検出

        # =========================
        # 総合スコア
        # =========================

        total = (  # 背景意図: 重み付き合成
            0.4 * temporal_score +
            0.3 * edit_score +
            0.3 * stability_score
        )

        return {  # 背景意図: 上位でログ・比較しやすい形式
            "temporal": float(temporal_score),
            "edit": float(edit_score),
            "stability": float(stability_score),
            "total": float(total),
        }

    # =========================
    # 各評価関数
    # =========================

    def _temporal_consistency(self, frames: List[Path]) -> float:  # 背景意図: フリッカー検出
        """フレーム間の変化量を評価（小さいほど安定）。"""  # 背景意図: 役割明示

        diffs = []  # 背景意図: フレーム差分を蓄積

        for i in range(1, len(frames)):
            img1 = cv2.imread(str(frames[i - 1]))  # 背景意図: 前フレーム
            img2 = cv2.imread(str(frames[i]))  # 背景意図: 次フレーム

            if img1 is None or img2 is None:
                continue  # 背景意図: 読み込み失敗はスキップ

            diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))  # 背景意図: L1差分
            diffs.append(diff)

        if len(diffs) == 0:
            return 0.0  # 背景意図: 異常時

        score = 1.0 / (1.0 + np.mean(diffs))  # 背景意図: 小さい差分ほど高スコア
        return float(score)

    def _edit_magnitude(self, in_frames: List[Path], out_frames: List[Path]) -> float:  # 背景意図: 編集量評価
        """入力と出力の差分を評価（適度な変化を評価）。"""  # 背景意図: 役割明示

        diffs = []  # 背景意図: 差分蓄積

        for p_in, p_out in zip(in_frames, out_frames):
            img_in = cv2.imread(str(p_in))  # 背景意図: 入力画像
            img_out = cv2.imread(str(p_out))  # 背景意図: 出力画像

            if img_in is None or img_out is None:
                continue  # 背景意図: 読み込み失敗回避

            diff = np.mean(np.abs(img_in.astype(np.float32) - img_out.astype(np.float32)))  # 背景意図: 差分
            diffs.append(diff)

        if len(diffs) == 0:
            return 0.0  # 背景意図: 異常時

        # 背景意図:
        #   - 変化がゼロ → 編集できてない
        #   - 変化が大きすぎ → 崩壊
        #   → 中間を評価

        mean_diff = np.mean(diffs)
        score = np.exp(-abs(mean_diff - 20.0) / 20.0)  # 背景意図: ヒューリスティック中心値20

        return float(score)

    def _stability(self, frames: List[Path]) -> float:  # 背景意図: ノイズ検出
        """高周波ノイズを検出する。"""  # 背景意図: 役割明示

        scores = []  # 背景意図: スコア蓄積

        for p in frames:
            img = cv2.imread(str(p))  # 背景意図: 画像読み込み
            if img is None:
                continue  # 背景意図: 読み込み失敗回避

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 背景意図: グレースケール化
            lap = cv2.Laplacian(gray, cv2.CV_32F)  # 背景意図: 高周波成分抽出
            variance = lap.var()  # 背景意図: 分散 = ノイズ量指標

            scores.append(variance)

        if len(scores) == 0:
            return 0.0  # 背景意図: 異常時

        # 背景意図:
        #   - 分散が高すぎ → ノイズ過多
        #   - 低すぎ → ぼやけ
        #   → 中庸を評価

        mean_var = np.mean(scores)
        score = np.exp(-abs(mean_var - 100.0) / 100.0)  # 背景意図: 中心値100（経験値）

        return float(score)