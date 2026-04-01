# coding: utf-8  # 背景意図: 日本語コメントを安全に扱う

"""  # 背景意図: モジュールの責務を明示する
video_io.py
背景意図:
    - 動画⇄フレーム変換の不具合（fpsズレ・順序ズレ・フレーム欠落）を防ぐ
    - ffmpegをsubprocessで明示的に実行して挙動を固定する
    - 失敗を早期検知して上位に伝播する
"""  # 背景意図: モジュールドキュメント終了

import subprocess  # 背景意図: ffmpegを外部コマンドとして実行するために使用
from pathlib import Path  # 背景意図: OS非依存でパス操作する
from typing import List  # 背景意図: 型ヒントで可読性を上げる

import cv2  # 背景意図: fps取得と簡易検証に使用


def get_video_fps(video_path: str) -> float:  # 背景意図: 元動画のfpsを正確に取得する
    """動画のfpsを取得する。"""  # 背景意図: 関数の役割を簡潔に明示
    cap = cv2.VideoCapture(video_path)  # 背景意図: OpenCVで動画メタ情報を取得
    if not cap.isOpened():  # 背景意図: ファイル不正やパスミスを早期検知
        raise RuntimeError(f"Failed to open video: {video_path}")  # 背景意図: 原因を明確にする
    fps = cap.get(cv2.CAP_PROP_FPS)  # 背景意図: フレームレートを取得
    cap.release()  # 背景意図: リソース解放
    if fps is None or fps <= 0:  # 背景意図: 異常値（0やNone）を弾く
        raise RuntimeError(f"Invalid FPS detected: {fps}")  # 背景意図: 下流でのズレを防ぐ
    return float(fps)  # 背景意図: 型を明確化して返す


def extract_frames(video_path: str, out_dir: str) -> None:  # 背景意図: 動画→フレーム変換（順序保証）
    """動画からフレームを抽出する（順序・フレーム数を維持）。"""  # 背景意図: 目的を明確にする
    out_path = Path(out_dir)  # 背景意図: Pathで安全に扱う
    out_path.mkdir(parents=True, exist_ok=True)  # 背景意図: 出力先を事前に作成

    cmd: List[str] = [  # 背景意図: ffmpegコマンドをリストで安全に構築
        "ffmpeg",  # 背景意図: 動画処理ツール
        "-y",  # 背景意図: 既存ファイルを上書き（再実行性確保）
        "-i", video_path,  # 背景意図: 入力動画指定
        "-vsync", "0",  # 背景意図: フレーム補間を禁止（フレーム数維持）
        str(out_path / "%06d.png"),  # 背景意図: ゼロ埋めで順序保証（000001.png…）
    ]  # 背景意図: コマンド定義終了

    _run_cmd(cmd)  # 背景意図: 共通の実行関数で失敗検知

    # --- 抽出結果の検証 ---  # 背景意図: silent failure防止
    num = _count_png(out_path)  # 背景意図: 生成フレーム数を確認
    if num == 0:  # 背景意図: 出力ゼロは異常
        raise RuntimeError("No frames extracted.")  # 背景意図: 上位で停止させる


def encode_video(frames_dir: str, out_path: str, fps: float) -> None:  # 背景意図: フレーム→動画（fps維持）
    """フレームから動画を生成する（元fpsを維持）。"""  # 背景意図: 目的を明確化
    frames_path = Path(frames_dir)  # 背景意図: パス操作を安全に
    if not frames_path.exists():  # 背景意図: 入力ディレクトリ存在チェック
        raise RuntimeError(f"Frames dir not found: {frames_dir}")  # 背景意図: 明確なエラー

    num = _count_png(frames_path)  # 背景意図: 入力フレーム数を確認
    if num == 0:  # 背景意図: フレームが無い場合は異常
        raise RuntimeError("No frames to encode.")  # 背景意図: 上位で停止

    cmd: List[str] = [  # 背景意図: ffmpegコマンド構築
        "ffmpeg",  # 背景意図: 動画エンコードツール
        "-y",  # 背景意図: 上書き許可
        "-framerate", str(fps),  # 背景意図: 入力fpsを明示（ここが重要）
        "-i", str(frames_path / "%06d.png"),  # 背景意図: 入力フレームパターン
        "-c:v", "libx264",  # 背景意図: H264で互換性確保
        "-pix_fmt", "yuv420p",  # 背景意図: 多くのプレイヤーで再生可能にする
        "-movflags", "+faststart",  # 背景意図: ストリーミング互換（先頭にメタデータ）
        out_path,  # 背景意図: 出力動画パス
    ]  # 背景意図: コマンド定義終了

    _run_cmd(cmd)  # 背景意図: 実行と失敗検知

    # --- 出力検証 ---  # 背景意図: 生成失敗の早期検知
    if not Path(out_path).exists():  # 背景意図: ファイル存在チェック
        raise RuntimeError(f"Failed to create video: {out_path}")  # 背景意図: 明確な例外


def _run_cmd(cmd: List[str]) -> None:  # 背景意図: 外部コマンド実行を一元化
    """subprocessでコマンドを実行し、失敗時に例外を出す。"""  # 背景意図: 共通処理の説明
    print("[CMD]", " ".join(cmd))  # 背景意図: デバッグ時に実行コマンドを確認できるようにする
    result = subprocess.run(  # 背景意図: 外部コマンド実行
        cmd,
        stdout=subprocess.PIPE,  # 背景意図: 標準出力を捕捉（必要に応じてログ化）
        stderr=subprocess.PIPE,  # 背景意図: エラー出力を捕捉
    )  # 背景意図: 実行終了

    if result.returncode != 0:  # 背景意図: 非ゼロは失敗
        raise RuntimeError(  # 背景意図: 上位に失敗を伝播
            f"Command failed:\n{' '.join(cmd)}\nSTDERR:\n{result.stderr.decode(errors='ignore')}"
        )  # 背景意図: 詳細エラーを含める


def _count_png(dir_path: Path) -> int:  # 背景意図: フレーム数確認を共通化
    """ディレクトリ内のpng枚数を数える。"""  # 背景意図: 関数説明
    return len(list(dir_path.glob("*.png")))  # 背景意図: 単純・確実にカウント