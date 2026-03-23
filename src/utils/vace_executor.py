# coding: utf-8
"""
vace_executor.py
背景意図:
    - VACE推論の実行ロジックをカプセル化
    - notebook / Python プロセス内からの直接実行をサポート
    - パス解決、環境設定、エラーハンドリング
    - 実行結果の構造化返却
"""

import contextlib
import importlib
import io
import os
import signal
import sys
import time
import traceback
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
        self.project_root = self.vace_repo.parent.parent.resolve()

    def required_python_paths(self) -> list[Path]:
        """VACE 実行に必要な import path 一覧."""
        candidates = [
            self.project_root,
            self.project_root / "third_party" / "VACE",
            self.project_root / "third_party" / "VACE" / "vace",
            self.project_root / "third_party" / "Wan2.1",
            self.project_root / "third_party" / "GroundingDINO",
        ]
        uniq: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            resolved = path.resolve()
            key = str(resolved)
            if key not in seen and resolved.exists():
                uniq.append(resolved)
                seen.add(key)
        return uniq

    def ensure_runtime_paths(self) -> list[str]:
        """現在プロセスの sys.path / PYTHONPATH に必要パスを追加."""
        added: list[str] = []
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_items = [p for p in current_pythonpath.split(os.pathsep) if p]
        for path in reversed(self.required_python_paths()):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                added.append(path_str)
            if path_str not in pythonpath_items:
                pythonpath_items.insert(0, path_str)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
        return added

    def preflight(self) -> Dict[str, Any]:
        """notebook カーネル内で import / CUDA 可視性を確認."""
        started_at = time.time()
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()
        try:
            self.ensure_runtime_paths()
            with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                import site
                import torch
                inference_module = self._load_inference_module()
                wan = importlib.import_module("wan")

                try:
                    cuda_available = torch.cuda.is_available()
                except Exception as cuda_error:
                    cuda_available = f"error: {cuda_error}"

                try:
                    device_count = torch.cuda.device_count()
                except Exception as cuda_error:
                    device_count = f"error: {cuda_error}"

                print("python=", sys.executable)
                print("user_site=", site.getusersitepackages())
                print("cuda_available=", cuda_available)
                print("device_count=", device_count)
                print("wan_file=", wan.__file__)
                print("inference_module=", inference_module.__file__)
            return {
                "status": "ok",
                "returncode": 0,
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "duration_sec": time.time() - started_at,
                "timed_out": False,
                "error_msg": None,
                "checks": {
                    "python": sys.executable,
                    "cuda_available": cuda_available,
                    "device_count": device_count,
                    "wan_file": wan.__file__,
                    "inference_module": inference_module.__file__,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "returncode": 1,
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue() + traceback.format_exc(),
                "duration_sec": time.time() - started_at,
                "timed_out": False,
                "error_msg": str(e),
                "checks": {},
            }

    @contextlib.contextmanager
    def _runtime_context(self):
        old_cwd = Path.cwd()
        old_env = os.environ.copy()
        old_sys_path = list(sys.path)
        try:
            self.ensure_runtime_paths()
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")
            os.chdir(self.vace_repo)
            yield
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)
            sys.path[:] = old_sys_path

    @contextlib.contextmanager
    def _safe_cuda_import_context(self):
        patched = False
        original_current_device = None
        try:
            import torch
            try:
                torch.cuda.current_device()
            except Exception:
                original_current_device = torch.cuda.current_device
                torch.cuda.current_device = lambda: 0
                patched = True
        except Exception:
            pass

        try:
            yield
        finally:
            if patched and original_current_device is not None:
                import torch
                torch.cuda.current_device = original_current_device

    def _load_inference_module(self):
        self.ensure_runtime_paths()
        with self._safe_cuda_import_context():
            return importlib.import_module("vace.vace_wan_inference")

    @contextlib.contextmanager
    def _time_limit(self, timeout_sec: int | None):
        if not timeout_sec or timeout_sec <= 0:
            yield
            return

        def _handle_timeout(signum, frame):
            raise TimeoutError(f"VACE execution timeout ({timeout_sec}s)")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(int(timeout_sec))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

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
            "inprocess",
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
        args = {
            "ckpt_dir": str(self.vace_ckpt_dir),
            "src_video": str(input_video.resolve()),
            "prompt": prompt,
            "base_seed": int(seed),
            "sample_steps": int(steps),
            "frame_num": int(frame_num),
            "save_dir": str(output_dir.resolve()),
            "save_file": str(output_file.resolve()),
            "model_name": "vace-1.3B",
            "size": "480p",
            "offload_model": None,
            "ulysses_size": 1,
            "ring_size": 1,
            "t5_fsdp": False,
            "t5_cpu": False,
            "dit_fsdp": False,
            "src_mask": None,
            "src_ref_images": None,
            "use_prompt_extend": "plain",
            "sample_solver": "unipc",
            "sample_shift": None,
            "sample_guide_scale": 5.0,
        }

        started_at = time.time()
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()
        try:
            with self._runtime_context():
                module = self._load_inference_module()
                with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                    with self._time_limit(timeout_sec):
                        module.main(args)

            duration_sec = time.time() - started_at
            stdout = stdout_io.getvalue()
            stderr = stderr_io.getvalue()

            if not output_file.exists():
                err_msg = stderr.strip()[:4000] or "output file was not created."
                return {
                    "status": "error",
                    "output_path": None,
                    "error_msg": f"VACE failed (rc=1): {err_msg}",
                    "returncode": 1,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cmd": cmd,
                    "duration_sec": duration_sec,
                    "timed_out": False,
                }

            return {
                "status": "ok",
                "output_path": output_file,
                "error_msg": None,
                "returncode": 0,
                "stdout": stdout,
                "stderr": stderr,
                "cmd": cmd,
                "duration_sec": duration_sec,
                "timed_out": False,
            }
        except TimeoutError:
            duration_sec = time.time() - started_at
            stdout = stdout_io.getvalue()
            stderr = stderr_io.getvalue() + traceback.format_exc()
            return {
                "status": "error",
                "output_path": None,
                "error_msg": f"VACE execution timeout ({timeout_sec}s)",
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "cmd": cmd,
                "duration_sec": duration_sec,
                "timed_out": True,
            }
        except Exception as e:
            duration_sec = time.time() - started_at
            stdout = stdout_io.getvalue()
            stderr = stderr_io.getvalue() + traceback.format_exc()
            message = str(e)
            return {
                "status": "error",
                "output_path": None,
                "error_msg": message,
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "cmd": cmd,
                "duration_sec": duration_sec,
                "timed_out": False,
            }
