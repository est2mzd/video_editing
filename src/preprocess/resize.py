# coding: utf-8  # 背景意図: 日本語コメントを安全に扱う

"""  # 背景意図: モジュールの責務を明示
resize.py
背景意図:
    - 推論前に画像サイズを統一してOOMや不安定性を防ぐ
    - アスペクト比を維持して歪みを防ぐ（評価に直結）
    - 後段で元サイズへ正確に復元するためのメタ情報を保持する
"""  # 背景意図: ドキュメント終了

from pathlib import Path  # 背景意図: OS非依存でパス操作
from typing import Dict, List, Tuple  # 背景意図: 型ヒントで可読性向上

import cv2  # 背景意図: 高速な画像リサイズに使用


class ResizeManager:  # 背景意図: リサイズ処理を一元管理
    """  # 背景意図: クラス責務を明確化
    ディレクトリ単位でリサイズ/復元を行う。
    各フレームの元サイズをメタとして保持する。
    """  # 背景意図: ドキュメント終了

    def __init__(self, long_side: int):  # 背景意図: 推論サイズ基準（長辺）
        self.long_side = int(long_side)  # 背景意図: 数値として固定

    # =========================
    # 公開API
    # =========================

    def resize_dir(self, input_dir: str, output_dir: str) -> Dict:  # 背景意図: ディレクトリ単位で処理
        """入力ディレクトリの全画像をリサイズし、メタを返す。"""  # 背景意図: 役割明示
        in_dir = Path(input_dir)  # 背景意図: Path化
        out_dir = Path(output_dir)  # 背景意図: Path化
        out_dir.mkdir(parents=True, exist_ok=True)  # 背景意図: 出力先を事前作成

        frame_paths = self._sorted_pngs(in_dir)  # 背景意図: 順序保証（%06d.png前提）
        if len(frame_paths) == 0:  # 背景意図: 空入力は異常
            raise RuntimeError(f"No frames in: {input_dir}")  # 背景意図: 上位で停止

        meta: Dict[str, Tuple[int, int]] = {}  # 背景意図: {filename: (orig_h, orig_w)} を保持

        for p in frame_paths:  # 背景意図: 各フレームを順に処理
            img = cv2.imread(str(p))  # 背景意図: 画像読み込み
            if img is None:  # 背景意図: 読み込み失敗検知
                raise RuntimeError(f"Failed to read image: {p}")  # 背景意図: 明確な例外

            resized, orig_hw = self._resize_keep_aspect(img)  # 背景意図: 歪みなしリサイズ
            meta[p.name] = orig_hw  # 背景意図: 復元用に元サイズを保存

            out_path = out_dir / p.name  # 背景意図: 同名で保存（順序維持）
            cv2.imwrite(str(out_path), resized)  # 背景意図: 書き出し

        return {  # 背景意図: 後段で使う情報をまとめて返す
            "frames": [p.name for p in frame_paths],  # 背景意図: 順序を明示保持
            "orig_sizes": meta,  # 背景意図: 各フレームの元サイズ
        }

    def restore_dir(self, input_dir: str, output_dir: str, meta: Dict) -> None:  # 背景意図: 元サイズ復元
        """リサイズ済みフレームを元サイズへ復元する。"""  # 背景意図: 役割明示
        in_dir = Path(input_dir)  # 背景意図: Path化
        out_dir = Path(output_dir)  # 背景意図: Path化
        out_dir.mkdir(parents=True, exist_ok=True)  # 背景意図: 出力先作成

        frame_names: List[str] = meta["frames"]  # 背景意図: 正しい順序で復元
        orig_sizes: Dict[str, Tuple[int, int]] = meta["orig_sizes"]  # 背景意図: 元サイズ参照

        for name in frame_names:  # 背景意図: 順序通りに処理
            in_path = in_dir / name  # 背景意図: 入力パス
            if not in_path.exists():  # 背景意図: 欠落フレーム検知
                raise RuntimeError(f"Missing frame: {in_path}")  # 背景意図: 上位で停止

            img = cv2.imread(str(in_path))  # 背景意図: 画像読み込み
            if img is None:  # 背景意図: 読み込み失敗検知
                raise RuntimeError(f"Failed to read image: {in_path}")  # 背景意図: 明確な例外

            orig_h, orig_w = orig_sizes[name]  # 背景意図: 元サイズ取得
            restored = cv2.resize(  # 背景意図: 元サイズへ戻す
                img,
                (orig_w, orig_h),  # 背景意図: cv2は(w, h)順
                interpolation=cv2.INTER_LINEAR,  # 背景意図: 拡大時に自然
            )

            out_path = out_dir / name  # 背景意図: 同名で保存
            cv2.imwrite(str(out_path), restored)  # 背景意図: 書き出し

    # =========================
    # 内部ユーティリティ
    # =========================

    def _resize_keep_aspect(self, img):  # 背景意図: 歪みなしリサイズの中核
        """アスペクト比を維持して長辺基準でリサイズする。"""  # 背景意図: 役割明示
        h, w = img.shape[:2]  # 背景意図: 元サイズ取得

        scale = self.long_side / max(h, w)  # 背景意図: 長辺を基準にスケール計算
        new_w = max(2, int(round(w * scale)))  # 背景意図: 0/1px防止
        new_h = max(2, int(round(h * scale)))  # 背景意図: 0/1px防止

        resized = cv2.resize(  # 背景意図: 実リサイズ
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,  # 背景意図: 縮小/拡大で最適化
        )

        return resized, (h, w)  # 背景意図: 元サイズを返して後で復元可能にする

    def _sorted_pngs(self, dir_path: Path) -> List[Path]:  # 背景意図: 順序保証
        """ゼロ埋めファイル名前提で安定ソートする。"""  # 背景意図: 役割明示
        paths = sorted(dir_path.glob("*.png"))  # 背景意図: ファイル名順でソート
        return paths  # 背景意図: 呼び出し元で利用