# coding: utf-8  # 背景意図: 日本語コメントを安全に扱う

"""  # 背景意図: モジュール責務を明示
constraints.py
背景意図:
    - コンペ提出条件を満たしているか検証する
    - NG動画を提出前に確実に弾く
    - silent failure（気づかない不具合）を防ぐ
"""  # 背景意図: ドキュメント終了

import cv2  # 背景意図: 動画メタ情報取得
from typing import Tuple  # 背景意図: 戻り値の型明示


class ConstraintChecker:  # 背景意図: 提出条件チェックをカプセル化
    """  # 背景意図: クラス責務明示
    入力動画と出力動画を比較し、
    コンペ要件違反を検出する。
    """  # 背景意図: ドキュメント終了

    def check(self, input_video: str, output_video: str) -> Tuple[bool, str]:  # 背景意図: 判定関数
        """制約チェックを実行する。"""  # 背景意図: 役割明示

        # --- 動画読み込み ---  # 背景意図: 基本情報取得
        cap_in = cv2.VideoCapture(input_video)  # 背景意図: 入力動画
        cap_out = cv2.VideoCapture(output_video)  # 背景意図: 出力動画

        if not cap_in.isOpened():  # 背景意図: 入力動画の異常検知
            return False, "input_video_open_failed"  # 背景意図: 原因を返す

        if not cap_out.isOpened():  # 背景意図: 出力動画の異常検知
            return False, "output_video_open_failed"  # 背景意図: 原因を返す

        # =========================
        # 1. フレーム数チェック
        # =========================

        frame_count_in = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))  # 背景意図: 入力フレーム数
        frame_count_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))  # 背景意図: 出力フレーム数

        if frame_count_in != frame_count_out:  # 背景意図: フレーム欠落・増加を検知
            return False, f"frame_count_mismatch: {frame_count_in} vs {frame_count_out}"  # 背景意図: 差分明示

        # =========================
        # 2. 解像度チェック
        # =========================

        width_in = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))  # 背景意図: 入力幅
        height_in = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 背景意図: 入力高さ

        width_out = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))  # 背景意図: 出力幅
        height_out = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 背景意図: 出力高さ

        if width_in != width_out or height_in != height_out:  # 背景意図: サイズ一致必須
            return False, f"resolution_mismatch: {(width_in, height_in)} vs {(width_out, height_out)}"  # 背景意図: 差分明示

        # =========================
        # 3. FPSチェック
        # =========================

        fps_in = cap_in.get(cv2.CAP_PROP_FPS)  # 背景意図: 入力fps
        fps_out = cap_out.get(cv2.CAP_PROP_FPS)  # 背景意図: 出力fps

        # 許容誤差（float誤差対策）
        if abs(fps_in - fps_out) > 0.1:  # 背景意図: 微小誤差は許容
            return False, f"fps_mismatch: {fps_in} vs {fps_out}"  # 背景意図: 差分明示

        # =========================
        # 4. フレーム読み取り検証
        # =========================

        # 背景意図:
        #   - 動画は存在しても途中で壊れているケースがある
        #   - 先頭数フレームを実際に読んで確認

        for _ in range(3):  # 背景意図: 軽量チェック（全フレームは重い）
            ret_in, _ = cap_in.read()  # 背景意図: 入力読み込み
            ret_out, _ = cap_out.read()  # 背景意図: 出力読み込み

            if not ret_in or not ret_out:  # 背景意図: 読み込み失敗検知
                return False, "frame_read_error"  # 背景意図: 動画破損

        # --- リソース解放 ---  # 背景意図: メモリリーク防止
        cap_in.release()
        cap_out.release()

        # =========================
        # OK
        # =========================

        return True, "ok"  # 背景意図: 全条件クリア