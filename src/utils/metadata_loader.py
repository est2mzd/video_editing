# coding: utf-8
"""
metadata_loader.py
背景意図:
    - metadata.csv と test_instructions.csv 読み込み
    - video_id/path ↔ frame/instruction のマッピング生成
    - CSV未検出時の安全なデフォルト処理
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd


class MetadataLoader:
    """メタデータCSV読み込みとマッピング生成."""

    @staticmethod
    def load_frame_metadata(video_dir: Path) -> Dict[str, int]:
        """metadata.csv から video_path → frame数 のマッピングを作成.
        
        Args:
            video_dir: メタデータCSVが格納されているディレクトリ
            
        Returns:
            {video_path: frame_count} のdict
            ファイルが無い場合は空dict を返す
        """
        metadata_csv = video_dir / "metadata.csv"
        
        if not metadata_csv.exists():
            return {}
        
        try:
            df = pd.read_csv(metadata_csv)
            frame_map = dict(zip(df['video_path'].astype(str), df['frame'].astype(int)))
            print(f"[INFO] Loaded {len(frame_map)} frame entries from {metadata_csv}")
            return frame_map
        except Exception as e:
            print(f"[WARN] Failed to load metadata.csv: {e}")
            return {}

    @staticmethod
    def load_instructions(video_dir: Path) -> Dict[int, str]:
        """test_instructions.csv から video_id → instruction のマッピングを作成.
        
        Args:
            video_dir: 指示CSVが格納されているディレクトリ
            
        Returns:
            {video_id: instruction} のdict
            ファイルが無い場合は空dict を返す
        """
        instr_csv = video_dir / "test_instructions.csv"
        
        if not instr_csv.exists():
            return {}
        
        try:
            df = pd.read_csv(instr_csv)
            instr_map = dict(zip(df['video_id'].astype(int), df['instruction'].astype(str)))
            print(f"[INFO] Loaded {len(instr_map)} instruction entries from {instr_csv}")
            return instr_map
        except Exception as e:
            print(f"[WARN] Failed to load test_instructions.csv: {e}")
            return {}

    @staticmethod
    def get_frame_count(video_path: str, frame_map: Dict[str, int], default: int = 81) -> int:
        """video_path から frame 数を取得.
        
        Args:
            video_path: 動画ファイルパス（絶対パス推奨）
            frame_map: メタデータマッピング
            default: 未検出時のデフォルト値
            
        Returns:
            フレーム数
        """
        return frame_map.get(video_path, default)

    @staticmethod
    def get_instruction(video_id: int, instr_map: Dict[int, str], fallback: str = "") -> Optional[str]:
        """video_id から instruction を取得.
        
        Args:
            video_id: 動画ID
            instr_map: 指示マッピング
            fallback: 未検出時のフォールバック
            
        Returns:
            instruction 文字列（未検出時は fallback）
        """
        return instr_map.get(video_id, fallback)
