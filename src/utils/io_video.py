# coding: utf-8  # 背景意図: 日本語コメント対応

"""  # 背景意図: モジュール責務を明示
io_video.py
背景意図:
    - 動画とフレームの相互変換を担当
    - フレーム順序・fpsを厳密に維持
    - コンペ提出条件を満たす動画を生成
"""  # 背景意図: ドキュメント終了

import cv2  # 背景意図: 動画入出力
from pathlib import Path  # 背景意図: OS非依存パス
from typing import Dict  # 背景意図: メタ情報返却


class VideoIO:  # 背景意図: 動画処理をカプセル化
    """  # 背景意図: クラス責務明示
    動画 → フレーム
    フレーム → 動画
    を統一的に扱う。
    """  # 背景意図: ドキュメント終了

    # =========================
    # 動画 → フレーム
    # =========================

    def video_to_frames(self, video_path: str, output_dir: str) -> Dict:  # 背景意図: 分解処理
        """動画をフレーム列に変換する。"""  # 背景意図: 役割明示

        video_path = str(video_path)  # 背景意図: cv2互換
        output_dir = Path(output_dir)  # 背景意図: Path化
        output_dir.mkdir(parents=True, exist_ok=True)  # 背景意図: 出力先作成

        cap = cv2.VideoCapture(video_path)  # 背景意図: 動画読み込み

        if not cap.isOpened():  # 背景意図: 読み込み失敗検知
            raise RuntimeError(f"Failed to open video: {video_path}")  # 背景意図: 即停止

        # --- メタ情報取得 ---  # 背景意図: 復元用
        fps = cap.get(cv2.CAP_PROP_FPS)  # 背景意図: fps保持
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 背景意図: 幅
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 背景意図: 高さ

        frame_idx = 0  # 背景意図: フレーム番号

        while True:  # 背景意図: 全フレーム処理
            ret, frame = cap.read()  # 背景意図: 1フレーム取得

            if not ret:  # 背景意図: 終端検知
                break

            # --- ファイル名設計 ---  # 背景意図: 順序保証
            filename = f"{frame_idx:06d}.png"  # 背景意図: ゼロ埋めで安定ソート
            out_path = output_dir / filename  # 背景意図: 出力パス

            cv2.imwrite(str(out_path), frame)  # 背景意図: フレーム保存

            frame_idx += 1  # 背景意図: インクリメント

        cap.release()  # 背景意図: リソース解放

        if frame_idx == 0:  # 背景意図: 空動画検知
            raise RuntimeError("No frames extracted.")  # 背景意図: 即停止

        return {  # 背景意図: 後段で必要な情報
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": frame_idx,
        }

    # =========================
    # フレーム → 動画
    # =========================

    def frames_to_video(self, input_dir: str, output_path: str, fps: float) -> None:  # 背景意図: 再構築
        """フレーム列を動画に変換する。"""  # 背景意図: 役割明示

        input_dir = Path(input_dir)  # 背景意図: Path化
        frames = sorted(input_dir.glob("*.png"))  # 背景意図: 順序保証

        if len(frames) == 0:  # 背景意図: 空入力検知
            raise RuntimeError("No frames found.")  # 背景意図: 即停止

        # --- 最初のフレームからサイズ取得 ---  # 背景意図: 動画サイズ決定
        first = cv2.imread(str(frames[0]))  # 背景意図: 先頭画像
        if first is None:  # 背景意図: 読み込み失敗
            raise RuntimeError("Failed to read first frame.")  # 背景意図: 即停止

        height, width = first.shape[:2]  # 背景意図: サイズ取得

        # --- コーデック設定 ---  # 背景意図: 汎用性重視
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 背景意図: 多環境で動く

        writer = cv2.VideoWriter(  # 背景意図: 動画書き込み初期化
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():  # 背景意図: 初期化失敗検知
            raise RuntimeError("Failed to open VideoWriter.")  # 背景意図: 即停止

        # --- フレーム書き込み ---  # 背景意図: 順序維持
        for p in frames:
            img = cv2.imread(str(p))  # 背景意図: フレーム読み込み

            if img is None:  # 背景意図: 読み込み失敗
                raise RuntimeError(f"Failed to read frame: {p}")  # 背景意図: 即停止

            # --- サイズチェック ---  # 背景意図: 不整合防止
            if img.shape[0] != height or img.shape[1] != width:
                raise RuntimeError("Frame size mismatch.")  # 背景意図: 提出NG防止

            writer.write(img)  # 背景意図: フレーム書き込み

        writer.release()  # 背景意図: リソース解放