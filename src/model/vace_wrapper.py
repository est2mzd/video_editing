# coding: utf-8  # 背景意図: 日本語コメント対応

"""  # 背景意図: モジュール責務を明示
vace_wrapper.py
背景意図:
    - 外部VACEリポジトリの実行をラップする
    - CLI依存をここに閉じ込める（他コードを汚さない）
    - モデル差し替え（PISCO等）を容易にする
    - 実行失敗を確実に検知する
"""  # 背景意図: ドキュメント終了

import subprocess  # 背景意図: 外部コマンド実行
import shutil
from pathlib import Path  # 背景意図: パス操作
from typing import Dict  # 背景意図: 型ヒント


class VACEWrapper:  # 背景意図: VACE実行をカプセル化
    """  # 背景意図: クラス責務を明示
    外部VACEリポジトリを呼び出して推論を行う。
    CLI構造の違いをこのクラス内に閉じ込める。
    """  # 背景意図: ドキュメント終了

    def __init__(self, config: Dict):  # 背景意図: configから必要情報を取得
        self.config = config  # 背景意図: 後続で参照

        # --- モデル設定 ---  # 背景意図: 外部依存をここに集中
        self.repo_dir = Path(config["model"]["vace_repo"])  # 背景意図: VACE repoの場所
        self.model_path = Path(config["model"]["model_path"])  # 背景意図: 重みファイル

        # --- 実行設定 ---  # 背景意図: 推論パラメータ
        self.device = config["runtime"]["device"]  # 背景意図: cuda/cpu
        self.seed = config["inference"].get("seed", 0)  # 背景意図: 再現性確保
        self.steps = config["inference"].get("steps", 25)  # 背景意図: 品質/速度トレードオフ

        # --- CLI設定（重要） ---  # 背景意図: repo差異を吸収する
        self.entry_script = config["model"].get(  # 背景意図: スクリプト名をconfigで可変にする
            "entry_script",
            "inference.py"  # デフォルト（要修正ポイント）
        )
        self.identity_fallback = bool(config["model"].get("identity_fallback", False))

        # --- 事前チェック ---  # 背景意図: 実行前に失敗を検知
        if not self.repo_dir.exists():  # 背景意図: repo存在確認
            raise RuntimeError(f"VACE repo not found: {self.repo_dir}")  # 背景意図: 明確なエラー

        if not self.model_path.exists():  # 背景意図: 重み存在確認
            if self.identity_fallback:
                print(f"[WARN] Model not found, fallback=identity: {self.model_path}")
            else:
                raise RuntimeError(f"Model not found: {self.model_path}")  # 背景意図: 明確なエラー

    # =========================
    # 公開API
    # =========================

    def run(self, input_dir: str, output_dir: str, prompt: str) -> None:  # 背景意図: 推論実行
        """VACE推論を実行する。"""  # 背景意図: 役割明示

        input_dir = Path(input_dir)  # 背景意図: Path化
        output_dir = Path(output_dir)  # 背景意図: Path化
        output_dir.mkdir(parents=True, exist_ok=True)  # 背景意図: 出力先確保

        # --- 入力チェック ---  # 背景意図: 空入力防止
        num_frames = len(list(input_dir.glob("*.png")))  # 背景意図: フレーム数確認
        if num_frames == 0:  # 背景意図: 異常検知
            raise RuntimeError("Input frames are empty.")  # 背景意図: 即停止

        # --- prompt整形 ---  # 背景意図: モデル安定化
        refined_prompt = self._build_prompt(prompt)  # 背景意図: 一貫した指示形式

        if self.identity_fallback and not self.model_path.exists():
            for src in sorted(input_dir.glob("*.png")):
                shutil.copy2(src, output_dir / src.name)
            return

        # --- コマンド構築 ---  # 背景意図: CLI依存をここに集約
        cmd = self._build_command(  # 背景意図: コマンド生成を分離
            input_dir=input_dir,
            output_dir=output_dir,
            prompt=refined_prompt,
        )

        # --- 実行 ---  # 背景意図: subprocessで安全に実行
        self._run_cmd(cmd)  # 背景意図: 失敗検知込み

        # --- 出力検証 ---  # 背景意図: silent failure防止
        out_frames = list(output_dir.glob("*.png"))  # 背景意図: 出力確認
        if len(out_frames) == 0:  # 背景意図: 生成失敗検知
            raise RuntimeError("VACE output is empty.")  # 背景意図: 即停止

    # =========================
    # 内部処理
    # =========================

    def _build_prompt(self, prompt: str) -> str:  # 背景意図: prompt統一
        """プロンプトを整形する。"""  # 背景意図: 役割明示
        return f"""Edit the video according to the instruction.
Only modify the target.
Keep background consistent.
Keep temporal consistency.

Instruction:
{prompt}
"""  # 背景意図: 安定した生成を促す

    def _build_command(self, input_dir: Path, output_dir: Path, prompt: str):  # 背景意図: CLI生成
        """実行コマンドを構築する。"""  # 背景意図: 役割明示

        # 注意:
        # 実際のVACE repoに合わせてここは必ず修正する
        return [
            "python", self.entry_script,  # 背景意図: エントリポイント
            "--input_dir", str(input_dir),  # 背景意図: 入力フレーム
            "--output_dir", str(output_dir),  # 背景意図: 出力先
            "--prompt", prompt,  # 背景意図: テキスト指示
            "--model_path", str(self.model_path),  # 背景意図: 重み
            "--seed", str(self.seed),  # 背景意図: 再現性
            "--steps", str(self.steps),  # 背景意図: 推論回数
            "--device", self.device,  # 背景意図: GPU指定
        ]

    def _run_cmd(self, cmd):  # 背景意図: subprocess共通処理
        """コマンドを実行して失敗を検知する。"""  # 背景意図: 役割明示

        print("[VACE CMD]", " ".join(cmd))  # 背景意図: デバッグ用ログ

        result = subprocess.run(  # 背景意図: 外部実行
            cmd,
            cwd=self.repo_dir,  # 背景意図: repo内で実行
            stdout=subprocess.PIPE,  # 背景意図: ログ取得
            stderr=subprocess.PIPE,  # 背景意図: エラー取得
        )

        if result.returncode != 0:  # 背景意図: 異常終了検知
            raise RuntimeError(  # 背景意図: 上位に伝播
                f"VACE failed:\nSTDERR:\n{result.stderr.decode(errors='ignore')}"
            )