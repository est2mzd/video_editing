# coding: utf-8  # 背景意図: 日本語コメントを安全に扱う

"""  # 背景意図: モジュールの責務を明示する
frame_manager.py
背景意図:
    - tmpディレクトリの衝突（動画間・run間）を防ぐ
    - 入力/リサイズ/VACE出力/復元の各ディレクトリを明確に分離する
    - ディレクトリ生成と削除を一元管理してバグを防ぐ
"""  # 背景意図: モジュールドキュメント終了

from pathlib import Path  # 背景意図: OS非依存で安全にパス操作する
import shutil  # 背景意図: ディレクトリ削除（cleanup）に使用


class FrameManager:  # 背景意図: フレーム用ディレクトリ管理をカプセル化
    """  # 背景意図: クラスの責務を明確化
    run_idとvideo_nameを組み合わせて、
    完全に独立したtmpディレクトリを生成・管理する。
    """  # 背景意図: ドキュメント終了

    def __init__(self, base_tmp_dir: str, run_id: str):  # 背景意図: ベースパスとrun識別子を受け取る
        self.base_tmp_dir = Path(base_tmp_dir)  # 背景意図: ルートtmpディレクトリ
        self.run_id = run_id  # 背景意図: run単位で分離するためのキー

        # --- run単位ディレクトリ ---  # 背景意図: runごとに完全分離
        self.run_dir = self.base_tmp_dir / run_id  # 背景意図: tmp_frames/run_id/
        self.run_dir.mkdir(parents=True, exist_ok=True)  # 背景意図: 必ず存在させる

    # =========================
    # 内部ユーティリティ
    # =========================

    def _video_root(self, video_name: str) -> Path:  # 背景意図: 動画単位のルートを生成
        """動画ごとのルートディレクトリを返す。"""  # 背景意図: 役割明示
        root = self.run_dir / video_name  # 背景意図: tmp/run_id/video_name/
        root.mkdir(parents=True, exist_ok=True)  # 背景意図: 必ず生成
        return root  # 背景意図: 呼び出し元で利用

    def _ensure(self, path: Path) -> Path:  # 背景意図: ディレクトリ作成を共通化
        """ディレクトリを作成して返す。"""  # 背景意図: 役割明示
        path.mkdir(parents=True, exist_ok=True)  # 背景意図: 再実行でも壊れない
        return path  # 背景意図: チェーンで使えるよう返す

    # =========================
    # 公開API
    # =========================

    def get_input_dir(self, video_name: str) -> Path:  # 背景意図: 元フレーム格納先
        """入力フレームディレクトリを返す。"""  # 背景意図: 役割明示
        return self._ensure(self._video_root(video_name) / "input")  # 背景意図: input専用に分離

    def get_resized_dir(self, video_name: str) -> Path:  # 背景意図: リサイズ後フレーム
        """リサイズ後フレームディレクトリを返す。"""  # 背景意図: 役割明示
        return self._ensure(self._video_root(video_name) / "resized")  # 背景意図: 前処理専用

    def get_output_dir(self, video_name: str) -> Path:  # 背景意図: VACE出力
        """VACE出力フレームディレクトリを返す。"""  # 背景意図: 役割明示
        return self._ensure(self._video_root(video_name) / "vace_output")  # 背景意図: モデル出力専用

    def get_restored_dir(self, video_name: str) -> Path:  # 背景意図: 元サイズ復元後
        """復元後フレームディレクトリを返す。"""  # 背景意図: 役割明示
        return self._ensure(self._video_root(video_name) / "restored")  # 背景意図: 最終フレーム

    # =========================
    # クリーンアップ
    # =========================

    def cleanup(self, video_name: str) -> None:  # 背景意図: ディスク圧迫対策
        """動画単位でtmpディレクトリを削除する。"""  # 背景意図: 役割明示
        root = self.run_dir / video_name  # 背景意図: 削除対象パス
        if root.exists():  # 背景意図: 存在確認
            shutil.rmtree(root)  # 背景意図: 再帰削除で完全に消す

    def cleanup_all(self) -> None:  # 背景意図: run単位で一括削除
        """run単位でtmpを全削除する。"""  # 背景意図: 役割明示
        if self.run_dir.exists():  # 背景意図: 存在確認
            shutil.rmtree(self.run_dir)  # 背景意図: run全体削除