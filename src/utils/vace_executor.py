# coding: utf-8
"""
vace_executor.py
背景意図:
    - VACE推論の実行ロジックをカプセル化
    - subprocess管理、パス解決、エラーハンドリング
    - 実行結果の構造化返却
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Any


class VaceExecutor:
    """VACE推論実行を統一的に管理."""

    def __init__(
        self,
        vace_repo: Path,
        vace_ckpt_dir: Path,
        python_bin: str = "/usr/bin/python3",
    ):
        """初期化.

        Args:
            vace_repo: VACE リポジトリディレクトリ
            vace_ckpt_dir: チェックポイント格納ディレクトリ
            python_bin: Python実行バイナリパス
        """
        self.vace_repo = Path(vace_repo).resolve()
        self.vace_ckpt_dir = Path(vace_ckpt_dir).resolve()
        self.python_bin = python_bin
        self.script_path = (
            self.vace_repo / "vace" / "vace_wan_inference.py"
        ).resolve()

    def is_available(self) -> bool:
        """VACE実行可能か確認."""
        return self.vace_ckpt_dir.exists() and self.script_path.exists()

    def execute(
        self,
        input_video: Path,
        output_dir: Path,
        output_file: Path,
        prompt: str,
        frame_num: int,
        seed: int = 42,
        steps: int = 25,
        timeout_sec: int = 600,
    ) -> Dict[str, Any]:
        """VACE推論を実行.

        Args:
            input_video: 入力動画パス（絶対パス推奨）
            output_dir: 出力ディレクトリ
            output_file: 出力ファイルパス（output_dir配下のファイル名）
            prompt: 編集指示
            frame_num: 出力フレーム数
            seed: 乱数シード
            steps: サンプリングステップ数

        Returns:
            {"status": "ok"/"error", "output_path": Path, "error_msg": str}
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin,
            str(self.script_path),
            "--ckpt_dir", str(self.vace_ckpt_dir),
            "--src_video", str(input_video.resolve()),
            "--prompt", prompt,
            "--base_seed", str(seed),
            "--sample_steps", str(steps),
            "--frame_num", str(frame_num),
            "--save_dir", str(output_dir.resolve()),
            "--save_file", str(output_file.resolve()),
        ]

        try:
            started_at = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(self.vace_repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
            duration_sec = time.time() - started_at

            if result.returncode != 0 or not output_file.exists():
                err_msg = (result.stderr or "").strip()[:4000]
                message = (
                    f"VACE failed (rc={result.returncode}): {err_msg}"
                )
                return {
                    "status": "error",
                    "output_path": None,
                    "error_msg": message,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "cmd": cmd,
                    "duration_sec": duration_sec,
                    "timed_out": False,
                }

            return {
                "status": "ok",
                "output_path": output_file,
                "error_msg": None,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "cmd": cmd,
                "duration_sec": duration_sec,
                "timed_out": False,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output_path": None,
                "error_msg": f"VACE execution timeout ({timeout_sec}s)",
                "returncode": None,
                "stdout": None,
                "stderr": None,
                "cmd": cmd,
                "duration_sec": float(timeout_sec),
                "timed_out": True,
            }
        except Exception as e:
            return {
                "status": "error",
                "output_path": None,
                "error_msg": str(e),
                "returncode": None,
                "stdout": None,
                "stderr": None,
                "cmd": cmd,
                "duration_sec": None,
                "timed_out": False,
            }
