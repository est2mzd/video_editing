# coding: utf-8  # 背景意図: 日本語コメントを安全に扱う

"""  # 背景意図: モジュールの責務を最初に明示する
runner.py
背景意図:
    - 1 run（複数動画）を完全に実行する中核
    - 各モジュール（I/O, 前処理, 推論, 評価, 制約, ログ）を順序通りに接続
    - 失敗を早期検知し、再現可能な形でログを残す
"""  # 背景意図: モジュールドキュメント終了

from pathlib import Path  # 背景意図: OS非依存でパス操作する
from typing import List, Dict, Any  # 背景意図: 型ヒントで可読性と保守性を上げる

# --- Data/I-O ---  # 背景意図: 層ごとにimportをまとめて見通しを良くする
from src.data.video_io import (  # 背景意図: 動画⇄フレーム・fps取得をここに集約
    get_video_fps,
    extract_frames,
    encode_video,
)

from src.data.frame_manager import FrameManager  # 背景意図: tmp_dir衝突を防ぐ管理クラス

# --- Preprocess ---  # 背景意図: 前処理系の責務を分離
from src.preprocess.resize import ResizeManager  # 背景意図: サイズ統一と復元を一元管理

# --- Model ---  # 背景意図: モデル依存をラップして切り替え可能にする
from src.model.vace_wrapper import VACEWrapper  # 背景意図: VACE呼び出しを隔離

# --- Evaluation ---  # 背景意図: スコア算出と制約チェックを分離
from src.eval.evaluator import Evaluator  # 背景意図: ローカルスコア
from src.eval.constraints import ConstraintChecker  # 背景意図: 提出条件チェック

# --- Utils ---  # 背景意図: 共通ユーティリティ
from src.utils.logger import ExperimentLogger  # 背景意図: 再現性・分析のためのログ


class ExperimentRunner:  # 背景意図: 1 runの実行をカプセル化
    """  # 背景意図: クラスの責務を明確化
    1 run（複数動画）を実行する。
    各動画は独立に処理し、副作用を持たない。
    """  # 背景意図: ドキュメント終了

    def __init__(self, config: Dict[str, Any]):  # 背景意図: 設定を受け取り各コンポーネントを初期化
        self.config = config  # 背景意図: 全処理で参照するため保持

        # --- run識別 ---  # 背景意図: 全出力・tmpを一意に分離するためのキー
        self.run_id: str = config["runtime"]["run_id"]  # 背景意図: 上位で付与した一意IDを使用

        # --- ディレクトリ設定 ---  # 背景意図: 出力先を一元管理して混乱を防ぐ
        self.output_root = Path("data/output_videos") / self.run_id  # 背景意図: run単位で出力を分離
        self.output_root.mkdir(parents=True, exist_ok=True)  # 背景意図: 実行前に必ず存在させる

        # --- tmpフレーム管理 ---  # 背景意図: 動画ごとに分離して衝突を防ぐ
        base_tmp_dir = config["runtime"]["tmp_dir"]  # 背景意図: SSD等の高速領域を指定できるようにする
        self.frame_manager = FrameManager(base_tmp_dir=base_tmp_dir, run_id=self.run_id)  # 背景意図: run_idでネームスペース分離

        # --- 前処理 ---  # 背景意図: 推論サイズを統一しOOM/歪みを防ぐ
        long_side = config["preprocess"]["long_side"]  # 背景意図: configで変更可能にする
        self.resizer = ResizeManager(long_side=long_side)  # 背景意図: サイズ変換を一元管理

        # --- モデル ---  # 背景意図: 推論実行をラップして他モデルへ差し替え可能にする
        self.model = VACEWrapper(config=config)  # 背景意図: VACE依存をここに閉じ込める

        # --- 評価/制約 ---  # 背景意図: 提出前のフィルタとランキング
        self.evaluator = Evaluator()  # 背景意図: 軽量スコアで比較する
        self.checker = ConstraintChecker()  # 背景意図: コンペ条件違反を弾く

        # --- ログ ---  # 背景意図: 再現性と分析のために全結果を記録
        self.logger = ExperimentLogger(run_id=self.run_id)  # 背景意図: run単位でログを分離

    def run(self) -> None:  # 背景意図: 全動画を処理する公開API
        """  # 背景意図: 処理の全体像を明記
        configに定義された全動画を順に処理する。
        途中で失敗しても他動画は継続する（fail-soft）。
        """  # 背景意図: ドキュメント終了

        video_list: List[str] = self.config["data"]["input_videos"]  # 背景意図: 処理対象の動画一覧

        results: List[Dict[str, Any]] = []  # 背景意図: 最終的な集約結果（ログ用）

        for video_path in video_list:  # 背景意図: 各動画を独立に処理
            try:  # 背景意図: 1動画の失敗で全体が止まらないようにする
                res = self.process_video(video_path)  # 背景意図: 1動画の完全パイプライン
                results.append(res)  # 背景意図: 成功結果を収集
            except Exception as e:  # 背景意図: 想定外エラーも捕捉してログに残す
                err = {"video": video_path, "score": -1e9, "error": str(e)}  # 背景意図: 失敗を数値化して後で除外可能にする
                print(f"[ERROR] video={video_path} error={e}")  # 背景意図: 即時に原因を把握できるよう出力
                results.append(err)  # 背景意図: 失敗も結果として記録

        # --- 最終ログ ---  # 背景意図: run全体のサマリを保存
        self.logger.finalize(config=self.config, results=results)  # 背景意図: config込みで再現可能にする

    def process_video(self, video_path: str) -> Dict[str, Any]:  # 背景意図: 1動画の完全処理をカプセル化
        """  # 背景意図: ステップを明示
        1動画の処理フロー:
            1. fps取得
            2. tmpディレクトリ作成（動画単位）
            3. フレーム抽出
            4. リサイズ
            5. VACE推論
            6. サイズ復元
            7. 動画再構成（fps維持）
            8. 制約チェック
            9. 評価
            10. ログ
        """  # 背景意図: ドキュメント終了

        video_path = str(video_path)  # 背景意図: Path/str混在を避ける
        video_name = Path(video_path).stem  # 背景意図: 拡張子除去して識別名にする

        # --- 1. fps取得 ---  # 背景意図: 再エンコード時に元fpsを維持するため必須
        fps = get_video_fps(video_path)  # 背景意図: cv2で安全に取得

        # --- 2. tmpディレクトリ（動画単位） ---  # 背景意図: フレーム混線を防ぐ
        input_frames_dir = self.frame_manager.get_input_dir(video_name)  # 背景意図: 入力フレーム用
        resized_frames_dir = self.frame_manager.get_resized_dir(video_name)  # 背景意図: リサイズ後
        vace_out_dir = self.frame_manager.get_output_dir(video_name)  # 背景意図: VACE出力
        restored_frames_dir = self.frame_manager.get_restored_dir(video_name)  # 背景意図: 元サイズ復元後

        # --- 3. フレーム抽出 ---  # 背景意図: VACEはフレーム単位処理のため
        extract_frames(video_path, str(input_frames_dir))  # 背景意図: ffmpegで順序保持して抽出

        # --- 4. リサイズ ---  # 背景意図: OOM回避・入力統一
        resize_meta = self.resizer.resize_dir(  # 背景意図: 全フレームを同一ポリシーで変換
            input_dir=str(input_frames_dir),
            output_dir=str(resized_frames_dir),
        )

        # --- 5. VACE推論 ---  # 背景意図: モデル本体はここでのみ呼ぶ
        prompt = self.config["inference"]["prompt"]  # 背景意図: configから取得（実験可変）
        self.model.run(  # 背景意図: 外部repo依存をラップ
            input_dir=str(resized_frames_dir),
            output_dir=str(vace_out_dir),
            prompt=prompt,
        )

        # --- 6. サイズ復元 ---  # 背景意図: コンペ要件（元解像度一致）
        self.resizer.restore_dir(  # 背景意図: metaを使って正確に復元
            input_dir=str(vace_out_dir),
            output_dir=str(restored_frames_dir),
            meta=resize_meta,
        )

        # --- 7. 動画再構成 ---  # 背景意図: 元fpsを維持して出力
        output_video_path = self.output_root / f"{video_name}.mp4"  # 背景意図: run配下に保存
        encode_video(  # 背景意図: H264で互換性の高い動画を生成
            frames_dir=str(restored_frames_dir),
            out_path=str(output_video_path),
            fps=fps,
        )

        # --- 8. 制約チェック ---  # 背景意図: NG提出を防ぐ最重要ステップ
        ok, reason = self.checker.check(  # 背景意図: フレーム数・解像度一致を確認
            input_video=video_path,
            output_video=str(output_video_path),
        )
        if not ok:  # 背景意図: 条件違反は即除外
            raise RuntimeError(f"Constraint violation: {reason}")  # 背景意図: 失敗として上位に伝播

        # --- 9. 評価 ---  # 背景意図: 提出前に粗い品質フィルタを行う
        score = self.evaluator.evaluate_video(str(output_video_path))  # 背景意図: 数値化して比較可能にする

        # --- 10. ログ ---  # 背景意図: 後から分析・再現できるよう保存
        self.logger.log_video_result(  # 背景意図: 動画単位で記録
            video_path=video_path,
            output_path=str(output_video_path),
            score=score,
        )

        # --- 後片付け（任意） ---  # 背景意図: ディスク圧迫を防ぐ（必要に応じて有効化）
        # self.frame_manager.cleanup(video_name)  # 背景意図: tmp削除（デバッグ時は残す）

        return {  # 背景意図: 上位で集約するための結果フォーマット
            "video": video_path,
            "output": str(output_video_path),
            "score": float(score),
        }  # 背景意図: シリアライズしやすい型に揃える