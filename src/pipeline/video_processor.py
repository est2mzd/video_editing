# coding: utf-8
"""
video_processor.py
背景意図:
    - 高レベルビデオ処理ワークフロー
    - ディレクトリ入力 / parquet 入力の統一インターフェース
    - manifest 生成、zip出力等
"""

import csv
import zipfile
from pathlib import Path
from typing import Dict

from src.utils.metadata_loader import MetadataLoader
from src.utils.vace_executor import VaceExecutor


class VideoProcessor:
    """ビデオ処理ワークフロー統一化."""

    def __init__(
        self,
        config: Dict,
        vace_executor: VaceExecutor,
        strict_no_fallback: bool = True,
    ):
        """初期化.

        Args:
            config: YAML設定dict
            vace_executor: VACE実行エンジン
            strict_no_fallback: True時は失敗で即停止
        """
        self.config = config
        self.vace_executor = vace_executor
        self.strict_no_fallback = strict_no_fallback
        self.metadata_loader = MetadataLoader()

    def process_directory(
        self,
        input_dir: Path,
        output_target: Path,
        default_prompt: str,
        limit_rows: int = 0,
    ) -> Dict:
        """ディレクトリ内のmp4群を処理.

        Args:
            input_dir: mp4が格納されているディレクトリ
            output_target: 出力先（.zip または ディレクトリ）
            default_prompt: デフォルトプロンプト
            limit_rows: 0なら全件、>0なら指定件数

        Returns:
            処理結果dict
        """
        video_files = sorted(
            [p for p in input_dir.glob("*.mp4") if p.is_file()],
            key=lambda p: (
                (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)
            ),
        )

        if not video_files:
            raise RuntimeError(f"No mp4 files found in {input_dir}")

        target_rows = (
            len(video_files)
            if limit_rows <= 0
            else min(limit_rows, len(video_files))
        )
        print(f"[INFO] Processing {target_rows} / {len(video_files)} videos")

        # メタデータ読み込み
        frame_map = self.metadata_loader.load_frame_metadata(input_dir)
        instr_map = self.metadata_loader.load_instructions(input_dir)

        # 実験ディレクトリを唯一の出力先に統一
        exp_dir_str = str(
            self.config.get("runtime", {}).get("exp_dir", "")
        ).strip()
        if not exp_dir_str:
            raise RuntimeError(
                "runtime.exp_dir is required for output routing"
            )
        exp_dir = Path(exp_dir_str)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 出力先決定（実体は常に exp_dir 配下）
        output_is_zip = output_target.suffix.lower() == ".zip"
        videos_dir = exp_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # manifest / instruction.txt 準備
        manifest_path = exp_dir / "instruction_manifest.csv"
        instruction_txt = exp_dir / "instruction.txt"
        instruction_body = (
            "\n".join(instr_map.values()) + "\n"
            if instr_map
            else default_prompt + "\n"
        )
        instruction_txt.write_text(instruction_body)

        work_dir = exp_dir / "tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 処理実行
        success_count = 0
        failed_count = 0

        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(
                [
                    "row_index",
                    "row_id",
                    "instruction",
                    "input_frame_count",
                    "mode",
                    "status",
                ]
            )

            for i, src_mp4 in enumerate(video_files[:target_rows]):
                row_id = src_mp4.stem
                input_frame_count = self.metadata_loader.get_frame_count(
                    str(src_mp4.resolve()), frame_map
                )

                # instruction 決定
                if instr_map and int(row_id) in instr_map:
                    row_prompt = instr_map[int(row_id)]
                else:
                    row_prompt = default_prompt

                row_out = work_dir / f"model_out_{i:04d}.mp4"
                row_save_dir = work_dir / f"vace_run_{i:04d}"

                # VACE実行
                result = self.vace_executor.execute(
                    input_video=src_mp4,
                    output_dir=row_save_dir,
                    output_file=row_out,
                    prompt=row_prompt,
                    frame_num=input_frame_count,
                    seed=self.config.get("inference", {}).get("seed", 42),
                    steps=self.config.get("inference", {}).get("steps", 25),
                )

                if result["status"] != "ok":
                    if self.strict_no_fallback:
                        raise RuntimeError(result["error_msg"])
                    writer.writerow(
                        [
                            i,
                            row_id,
                            row_prompt,
                            input_frame_count,
                            "model_vace+fallback",
                            result["error_msg"],
                        ]
                    )
                    print(
                        f"[WARN] row={i} id={row_id} "
                        f"failed -> skip: {result['error_msg']}"
                    )
                    failed_count += 1
                    continue

                # 出力保存
                out_mp4_path = videos_dir / f"{row_id}.mp4"
                out_mp4_path.write_bytes(result["output_path"].read_bytes())
                writer.writerow(
                    [
                        i,
                        row_id,
                        row_prompt,
                        input_frame_count,
                        "model_direct_vace",
                        "ok",
                    ]
                )
                print(
                    f"[INFO] row={i} id={row_id} "
                    f"frame={input_frame_count} ok"
                )
                success_count += 1

        # ZIP作成（必要に応じて）
        final_output = videos_dir
        if output_is_zip:
            zip_path = exp_dir / output_target.name
            with zipfile.ZipFile(
                zip_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                for p in sorted(videos_dir.glob("*.mp4")):
                    zf.write(p, arcname=p.name)
            final_output = zip_path
            print(f"[INFO] Created submission zip: {zip_path}")

        return {
            "status": "ok" if failed_count == 0 else "partial",
            "output_path": str(final_output),
            "videos_dir": str(videos_dir),
            "manifest_path": str(manifest_path),
            "experiment_dir": str(
                self.config.get("runtime", {}).get("exp_dir", "")
            ),
            "success_count": success_count,
            "failed_count": failed_count,
        }
