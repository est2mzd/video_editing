#!/usr/bin/env python
# coding: utf-8

# # 背景
# あなたの目的：
# 
# 同じ結果を高速に出す
# 
# なら：
# 
# LLMを「生成」ではなく「分類器」として使う

# ## Parameter Settings

# In[1]:


DATA_ROOT_DIR = '/workspace/data'
VIDEOS_DIR = f'{DATA_ROOT_DIR}/videos'
annotation_path = f'{DATA_ROOT_DIR}/annotations.jsonl'


# In[2]:


# read annotation file as pandas dataframe
import json
import pandas as pd

with open(annotation_path, 'r') as f:
    annotations = [json.loads(line) for line in f]

df_annotations = pd.DataFrame(annotations) 

# 折り返さずに表示する設定
pd.set_option('display.max_colwidth', 200)

print(df_annotations.head())


# In[3]:


# df の selected_class のユニークな値を取得する
unique_classes = df_annotations['selected_class'].unique()
print(unique_classes)


# In[4]:


# df の selected_subclass のユニークな値を取得する
unique_subclasses = df_annotations['selected_subclass'].unique()
print(unique_subclasses)


# In[5]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


# ==========================================
# モデルロード
# ==========================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # 背景意図: 使用モデルを明示し、再現性を確保する

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  
# 背景意図: tokenizerをロードし、テキスト→トークン変換を行う

tokenizer.pad_token = tokenizer.eos_token  
# 背景意図: Mistralはpad_tokenを持たないため、batch処理でpaddingを可能にする

tokenizer.padding_side = "left"  
# 背景意図: decoder-onlyモデルではleft paddingにしないと生成が崩壊するため必須

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
# 背景意図:
# - float16: GPUメモリ削減と高速化
# - device_map="auto": GPUに自動配置し実行速度を最大化


# In[6]:


# ==========================================
# プロンプト生成
# ==========================================
def build_prompt(instruction):
    return f"""
You are an expert video editing parser.

Extract structured parameters from the instruction.

Return ONLY valid JSON.

Schema:
{{
"group": "camera | other",
"task": "zoom_in | zoom_out | low_angle | high_angle | arc_shot | unknown",
"target": "string",
"target_kind": "face | person | object | scene | unknown",
"motion_style": "smooth | gradual | steady | slow | none",
"keep_centered": true | false
}}

Instruction:
{instruction}

JSON:
"""
# 背景意図:
# - LLMに役割（parser）を明確に与える
# - 出力形式をJSONに固定し、後処理を容易にする
# - schemaを明示し、出力空間を制限する


# In[7]:


# ==========================================
# JSON抽出
# ==========================================
import re

def extract_json(text):
    # 背景意図:
    # LLM出力の中から「最初のJSONブロックだけ」を確実に抽出するため

    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    # 背景意図:
    # 最短一致でJSON候補をすべて取得（過剰取得を防ぐ）

    for m in matches:
        try:
            return json.loads(m)
            # 背景意図:
            # 正しくパースできた最初のJSONのみ採用
        except:
            continue

    return {"error": text}
    # 背景意図:
    # すべて失敗した場合のみエラー扱い



# In[8]:


# ==========================================
# batch推論
# ==========================================
def parse_with_llm_batch(instructions, batch_size=8):

    results = []  
    # 背景意図: 全instructionの結果を格納するリスト

    for i in range(0, len(instructions), batch_size):
        # 背景意図: 大量データを一括処理せず、メモリ効率のため分割する

        batch = instructions[i:i+batch_size]  
        # 背景意図: 現在処理するinstructionのサブセット

        prompts = [build_prompt(inst) for inst in batch]  
        # 背景意図: 各instructionをLLM入力用プロンプトに変換

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # 背景意図:
        # - padding=True: batch内で長さを揃える
        # - truncation=True: 異常に長い入力でOOMを防ぐ

        inputs = {k: v.to(model.device) for k, v in inputs.items()}  
        # 背景意図: GPUにデータを転送し、高速推論を実現

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in texts:
            parsed = extract_json(text)
            results.append(parsed)

    return results  
    # 背景意図: 全バッチの結果を統合して返却


# In[9]:


# ==========================================
# 実行
# ==========================================
instructions = df_annotations['instruction'].tolist()  
# 背景意図: DataFrameからinstruction列をリスト化し処理対象を準備

results = parse_with_llm_batch(instructions, batch_size=8)  
# 背景意図: batch処理により高速に全データを解析

num_total = len(results)  
# 背景意図: 全件数を把握

num_error = sum(1 for r in results if "error" in r)  
# 背景意図: JSONパース失敗件数を定量評価

print(f"Total: {num_total}")  
print(f"Error: {num_error}")  
# 背景意図: 処理品質（成功率）を確認するための指標

# ==========================================
# txt出力
# ==========================================
output_path = "parse_instruction_ver03_1.txt"  
# 背景意図: 出力ファイル名を固定し、再現性と管理性を確保

with open(output_path, "w", encoding="utf-8") as f:
    # 背景意図: UTF-8で保存し、日本語や特殊文字の文字化けを防ぐ

    for inst, res in zip(instructions, results):
        # 背景意図: 入力instructionと対応する結果をペアで扱う

        f.write(f"Instruction: {inst}\n")
        # 背景意図: 元データを記録し、後から検証可能にする

        f.write(f"Parsed: {res}\n")
        # 背景意図: LLM出力（またはエラー）をそのまま保存

        f.write("-" * 50 + "\n")
        # 背景意図: 各レコードの区切りを明確にする

print(f"Saved to {output_path}")
# 背景意図: 保存先を明示し、確認ミスを防ぐ


# # LLM解析結果の評価と改善方針
# 
# ## 結論
# 現状は「JSON形式」ではなく、「意味解析」が崩れている状態。
# 
# ---
# 
# ## 1. 実測結果
# 
# 全100件のうち：
# 
# - group = camera : 21件  
# - group = other  : 79件  
# 
# ---
# 
# ## 2. 評価
# 
# camera系instructionが多いデータに対して  
# otherが79%は不自然。
# 
# → 分類器として機能していない
# 
# ---
# 
# ## 3. 観測された問題
# 
# 以下の出力が多数：
# 
# - "unknown"
# - "other"
# - "task": "unknown"
# 
# ---
# 
# ## 4. 原因（コードと結果から確定）
# 
# ### ① 保守的な出力
# 
# - temperature = 0.0
# - do_sample = False
# 
# → 分からない場合は "other" に倒れる
# 
# ---
# 
# ### ② スキーマが弱い
# 
# ```json
# "group": "camera | other"
# ```
# 
# → camera以外すべて other
# 
# ---
# 
# ### ③ task定義が狭い
# 
# ```json
# "task": "zoom_in | zoom_out | low_angle | high_angle | arc_shot | unknown"
# ```
# 
# 実際のinstruction：
# 
# - pan
# - tilt
# - move forward
# - focus
# 
# → 表現できず unknown になる
# 
# ---
# 
# ## 5. 本質
# 
# これはモデルの問題ではない
# 
# → 問題は「分類設計」
# 
# ---
# 
# ## 6. 最小修正（プロンプトのみ）
# 
# ### 修正①
# 
# Before:
# "group": "camera | other"
# 
# After:
# "group": "camera | object | scene | other"
# 
# ---
# 
# ### 修正②
# 
# Before:
# task = 固定6種
# 
# After:
# task = 自由文字列
# 
# ---
# 
# ### 修正③（最重要）
# 
# 追加ルール：
# 
# If the instruction involves camera movement, you MUST set group="camera".
# 
# ---
# 
# ## 7. 期待される変化
# 
# - other 減少
# - camera 増加
# - unknown 減少
# 
# ---
# 
# ## 8. 結論
# 
# 問題はJSONではない  
# 問題は意味空間の設計
# 
# ---
# 
# ## 9. 次のアクション
# 
# プロンプトのみ修正して再実行
# 

# In[10]:


def build_prompt(instruction):
    # ==========================================
    # プロンプト構築（修正版）
    # ==========================================

    return f"""
You are an expert video editing parser.

Extract structured parameters from the instruction.

Return ONLY valid JSON.

Rules:
- If the instruction involves camera movement (zoom, pan, tilt, dolly, tracking, etc.), you MUST set "group" to "camera".
- Do NOT overuse "other".
- Always choose the most specific category possible.
- "task" should be a short descriptive phrase (NOT limited to predefined labels).

Schema:
{{
  "group": "camera | object | scene | other",
  "task": "string",
  "target": "string",
  "target_kind": "face | person | object | scene | unknown",
  "motion_style": "smooth | gradual | steady | slow | fast | none",
  "keep_centered": true | false
}}

Instruction:
{instruction}

JSON:
"""


# In[11]:


# ==========================================
# 実行
# ==========================================
instructions = df_annotations['instruction'].tolist()  
# 背景意図: DataFrameからinstruction列をリスト化し処理対象を準備

results = parse_with_llm_batch(instructions, batch_size=8)  
# 背景意図: batch処理により高速に全データを解析

num_total = len(results)  
# 背景意図: 全件数を把握

num_error = sum(1 for r in results if "error" in r)  
# 背景意図: JSONパース失敗件数を定量評価

print(f"Total: {num_total}")  
print(f"Error: {num_error}")  
# 背景意図: 処理品質（成功率）を確認するための指標

# ==========================================
# txt出力
# ==========================================
output_path = "parse_instruction_ver03_2.txt"  
# 背景意図: 出力ファイル名を固定し、再現性と管理性を確保

with open(output_path, "w", encoding="utf-8") as f:
    # 背景意図: UTF-8で保存し、日本語や特殊文字の文字化けを防ぐ

    for inst, res in zip(instructions, results):
        # 背景意図: 入力instructionと対応する結果をペアで扱う

        f.write(f"Instruction: {inst}\n")
        # 背景意図: 元データを記録し、後から検証可能にする

        f.write(f"Parsed: {res}\n")
        # 背景意図: LLM出力（またはエラー）をそのまま保存

        f.write("-" * 50 + "\n")
        # 背景意図: 各レコードの区切りを明確にする

print(f"Saved to {output_path}")
# 背景意図: 保存先を明示し、確認ミスを防ぐ


# # LLMプロンプト設計（後段タスク指向）
# 
# ## 結論
# groupは意味分類ではなく「後段処理ルーティング」に使う。
# 
# ---
# 
# ## 1. 設計方針
# 
# group = 実行エンジン（処理手段）
# 
# ---
# 
# ## 2. group定義
# 
# ```json
# "group": "opencv | vace | pisco | segmentation | tracking | unknown"
# ```
# 
# ---
# 
# ## 3. 各groupの役割
# 
# | group | 役割 |
# |------|------|
# | opencv | 軽量画像処理（blur, crop, color, overlay） |
# | vace | 生成系・動画編集（背景変更、生成、補完） |
# | pisco | 高度編集（構造的編集・推論系） |
# | segmentation | マスク生成（GroundingDINO / SAM） |
# | tracking | 時系列処理（追跡・安定化） |
# | unknown | fallback |
# 
# ---
# 
# ## 4. group決定ルール
# 
# - generative editing（背景変更など） → vace  
# - simple image processing → opencv  
# - mask / region selection → segmentation  
# - temporal consistency / tracking → tracking  
# - complex reasoning → pisco  
# - それ以外 → unknown  
# 
# ---
# 
# ## 5. 修正版プロンプト
# 
# ```python
# def build_prompt(instruction):
#     return f"""
# You are an expert video editing parser.
# 
# Your goal is to convert instructions into executable parameters.
# 
# Return ONLY valid JSON.
# 
# Rules:
# - Choose "group" based on the required processing method (NOT semantic category).
# - If the instruction requires generative editing (background change, scene replacement), use "vace".
# - If the instruction is simple image processing (crop, blur, brightness, resize), use "opencv".
# - If the instruction requires object mask or region selection, use "segmentation".
# - If the instruction involves temporal consistency or subject following, use "tracking".
# - If the instruction requires complex reasoning or advanced editing, use "pisco".
# - Use "unknown" ONLY if none apply.
# 
# Schema:
# {{
#   "group": "opencv | vace | pisco | segmentation | tracking | unknown",
#   "task": "string",
#   "target": "string",
#   "target_kind": "face | person | object | scene | unknown",
#   "motion_style": "smooth | gradual | steady | slow | fast | none",
#   "keep_centered": true | false
# }}
# 
# Instruction:
# {instruction}
# 
# JSON:
# """
# ```
# 
# ---
# 
# ## 6. 効果
# 
# - LLM → 実行系に直接マッピング可能  
# - if文のみで後段処理に接続  
# - unknown削減  
# 
# ---
# 
# ## 7. 次のステップ
# 
# 1. プロンプトを適用して再実行  
# 2. group分布を確認  
# 3. groupごとの処理分岐を実装  
# 
# 

# In[12]:


def build_prompt(instruction):
    return f"""
You are an expert video editing parser.

Your goal is to convert instructions into executable parameters.

Return ONLY valid JSON.

Rules:
- Choose "group" based on the required processing method (NOT semantic category).
- If the instruction requires generative editing (background change, scene replacement), use "vace".
- If the instruction is simple image processing (crop, blur, brightness, resize), use "opencv".
- If the instruction requires object mask or region selection, use "segmentation".
- If the instruction involves temporal consistency or subject following, use "tracking".
- If the instruction requires complex reasoning or advanced editing, use "pisco".
- Use "unknown" ONLY if none apply.

Schema:
{{
  "group": "opencv | vace | pisco | segmentation | tracking | unknown",
  "task": "string",
  "target": "string",
  "target_kind": "face | person | object | scene | unknown",
  "motion_style": "smooth | gradual | steady | slow | fast | none",
  "keep_centered": true | false
}}

Instruction:
{instruction}

JSON:
"""


# In[13]:


# ==========================================
# 実行
# ==========================================
instructions = df_annotations['instruction'].tolist()  
# 背景意図: DataFrameからinstruction列をリスト化し処理対象を準備

results = parse_with_llm_batch(instructions, batch_size=8)  
# 背景意図: batch処理により高速に全データを解析

num_total = len(results)  
# 背景意図: 全件数を把握

num_error = sum(1 for r in results if "error" in r)  
# 背景意図: JSONパース失敗件数を定量評価

print(f"Total: {num_total}")  
print(f"Error: {num_error}")  
# 背景意図: 処理品質（成功率）を確認するための指標

# ==========================================
# txt出力
# ==========================================
output_path = "parse_instruction_ver03_3.txt"  
# 背景意図: 出力ファイル名を固定し、再現性と管理性を確保

with open(output_path, "w", encoding="utf-8") as f:
    # 背景意図: UTF-8で保存し、日本語や特殊文字の文字化けを防ぐ

    for inst, res in zip(instructions, results):
        # 背景意図: 入力instructionと対応する結果をペアで扱う

        f.write(f"Instruction: {inst}\n")
        # 背景意図: 元データを記録し、後から検証可能にする

        f.write(f"Parsed: {res}\n")
        # 背景意図: LLM出力（またはエラー）をそのまま保存

        f.write("-" * 50 + "\n")
        # 背景意図: 各レコードの区切りを明確にする

print(f"Saved to {output_path}")
# 背景意図: 保存先を明示し、確認ミスを防ぐ


# # LLMルーティング結果の分析レポート
# 
# ## 結論
# 「ルーティング粒度」は向上しているが、「精度が上がった」とはまだ言えない。
# 
# ---
# 
# ## 1. 件数サマリ
# 
# - 総件数: 100
# - 正常出力: 94
# - error: 6
# 
# ---
# 
# ## 2. group内訳（正常94件）
# 
# - vace: 64
# - tracking: 13
# - pisco: 13
# - opencv: 4
# - segmentation: 0
# - unknown: 0
# 
# ---
# 
# ## 3. 改善点
# 
# - otherが消滅（前回は79件）
# - 全件が実行エンジンに割当
# - downstreamに直接渡せる構造になった
# 
# ---
# 
# ## 4. 問題点
# 
# ### 4.1 vaceへの偏り
# 
# - vace: 64%
# - tracking+pisco: 26%
# - opencv: 4%
# 
# → 生成系に寄りすぎ
# 
# ---
# 
# ### 4.2 camera系の誤分類
# 
# camera系instruction: 33件
# 
# - vace: 23
# - tracking: 8
# - error: 2
# 
# → 約70%がvaceに誤分類
# 
# ---
# 
# ### 4.3 segmentationが0件
# 
# - mask生成が一切出ていない
# - pipelineに乗らない
# 
# ---
# 
# ### 4.4 schema違反
# 
# - target_kind違反: 2件
# - motion_style違反: 6件
# 
# 合計: 8件
# 
# ---
# 
# ### 4.5 error分析（6件）
# 
# 原因:
# - schema外フィールド追加
# - ネスト構造崩壊
# - 型不一致
# 
# ---
# 
# ## 5. 評価まとめ
# 
# ### 改善された点
# - ルーティング可能になった
# - other依存が消えた
# - 出力率94%
# 
# ### 未解決
# - vace偏重
# - segmentation未使用
# - camera誤分類
# - schema崩れ
# 
# ---
# 
# ## 6. 結論
# 
# 問題はJSONではない  
# 問題は「ルーティング精度」
# 
# ---
# 
# ## 7. 次にやるべきこと
# 
# 1. vaceの使用条件を厳格化
# 2. camera → tracking/opencvに分離
# 3. segmentationを強制発火条件に入れる
# 4. schema制約を強化
# 
# ---
# 
# ## 8. 状態評価
# 
# - 構造: OK
# - 速度: OK
# - 実用性: 未到達
# 

# ---
# ---
# ---

# # LLM2による後処理改善設計
# 
# ## 結論
# LLM2（修正専用モデル）を導入することで、31〜37件の改善が可能。
# 今回の問題は生成ではなく「正規化・補正」であり、LLM2が有効。
# 
# ---
# 
# ## 1. 前提
# 
# - 総件数: 100
# - 正常: 94
# - error: 6
# 
# ---
# 
# ## 2. LLM2で改善可能な対象
# 
# | 種類 | 件数 | 適性 |
# |------|------|------|
# | camera誤分類 | 23 | ◎ |
# | schema違反 | 8 | ◎ |
# | error復元 | 6 | ○ |
# 
# 合計: 31〜37件
# 
# ---
# 
# ## 3. アーキテクチャ
# 
# LLM1（生成）
# ↓
# JSON（粗い）
# ↓
# LLM2（修正専用）
# ↓
# JSON（正規化済）
# ↓
# OpenCV / VACE / PISCO
# 
# ---
# 
# ## 4. LLM2の役割
# 
# - group再分類
# - task正規化
# - schema強制
# - JSON修復
# 
# ---
# 
# ## 5. LLM2プロンプト
# 
# ```python
# def build_refine_prompt(original_instruction, parsed_json):
#     return f"""
# You are a strict JSON refiner for video editing pipelines.
# 
# Your job is to FIX the given JSON so it is:
# 1. Correctly routed (group)
# 2. Schema-valid
# 3. Executable downstream
# 
# DO NOT add extra fields.
# DO NOT change meaning.
# 
# Rules:
# - If camera movement exists -> group = "tracking" or "opencv"
# - Do NOT overuse "vace"
# - Fix invalid enum values
# - Fix broken JSON if needed
# 
# Schema:
# {
#   "group": "opencv | vace | pisco | segmentation | tracking",
#   "task": "string",
#   "target": "string",
#   "target_kind": "face | person | object | scene | unknown",
#   "motion_style": "smooth | gradual | steady | slow | fast | none",
#   "keep_centered": true | false
# }
# 
# Instruction:
# {original_instruction}
# 
# Broken JSON:
# {parsed_json}
# 
# Return ONLY fixed JSON:
# """
# ```
# 
# ---
# 
# ## 6. 設計ポイント
# 
# ### ① 役割分離
# - LLM1: 解釈
# - LLM2: 修正
# 
# ---
# 
# ### ② 温度設定
# temperature = 0.0
# 
# 理由: 修正は決定的処理
# 
# ---
# 
# ### ③ 入力
# - instruction
# - JSON
# 
# 両方渡すことで文脈保持
# 
# ---
# 
# ## 7. 注意点
# 
# ### NG例
# - 同じプロンプトを使う -> 再び崩壊
# - 自由生成させる -> vace偏重に戻る
# 
# ---
# 
# ## 8. 効果
# 
# - vace偏重解消
# - schema違反修正
# - error削減
# 
# ---
# 
# ## 9. コスト
# 
# - 推論回数: 2倍
# - 精度: 大幅改善（90%以上想定）
# 
# ---
# 
# ## 10. 結論
# 
# LLM2は有効  
# 今回の問題は「2段構成が前提」の典型例
# 

# In[15]:


import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""
# ==========================================
# モデル（LLM2）
# ==========================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
"""


# In[16]:


# ==========================================
# LLM2プロンプト
# ==========================================
def build_refine_prompt(instruction, parsed_json):
    # 背景意図: LLM1の出力を「修正専用」で再解釈させる
    return f"""
You are a strict JSON refiner for video editing pipelines.

Fix the JSON so it is executable.

Rules:
- If camera movement exists -> group = "tracking" or "opencv"
- Do NOT overuse "vace"
- Fix invalid values
- Keep schema strictly

Schema:
{{
  "group": "opencv | vace | pisco | segmentation | tracking",
  "task": "string",
  "target": "string",
  "target_kind": "face | person | object | scene | unknown",
  "motion_style": "smooth | gradual | steady | slow | fast | none",
  "keep_centered": true | false
}}

Instruction:
{instruction}

Broken JSON:
{parsed_json}

Return ONLY JSON:
"""


# In[17]:


# ==========================================
# txt読み込み
# ==========================================
def load_txt(path):
    # 背景意図: Instruction / Parsed をペアで抽出
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("-" * 50)

    data = []

    for block in blocks:
        if "Instruction:" not in block:
            continue

        inst_match = re.search(r"Instruction:\s*(.*)", block)
        parsed_match = re.search(r"Parsed:\s*(.*)", block)

        if not inst_match or not parsed_match:
            continue

        inst = inst_match.group(1).strip()
        parsed = parsed_match.group(1).strip()

        data.append((inst, parsed))

    return data


# In[18]:


# ==========================================
# LLM2修正
# ==========================================
def refine_batch(data, batch_size=4):

    results = []

    for i in range(0, len(data), batch_size):

        batch = data[i:i+batch_size]

        prompts = [
            build_refine_prompt(inst, parsed)
            for inst, parsed in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in texts:
            try:
                json_str = text.split("JSON:")[-1].strip()
                parsed = json.loads(json_str)
            except:
                parsed = {"error": text}

            results.append(parsed)

    return results


# In[ ]:


# ==========================================
# 実行
# ==========================================
input_path = "/workspace/notebook/parse_instruction_ver03_3.txt"

data = load_txt(input_path)

refined = refine_batch(data, batch_size=4)

# ==========================================
# 保存
# ==========================================
output_path = "/workspace/notebook/parse_instruction_ver03_3_refined_1.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for (inst, _), res in zip(data, refined):
        f.write(f"Instruction: {inst}\n")
        f.write(f"Refined: {res}\n")
        f.write("-" * 50 + "\n")

print("Saved:", output_path)

# ==========================================
# 簡易評価
# ==========================================
num_total = len(refined)
num_error = sum(1 for r in refined if "error" in r)

print("Total:", num_total)
print("Error:", num_error)


# # ■ 全体結果（根拠あり）
# 
# ```
# 総数: 100件
# エラー: 0件
# ```
# 
# ---
# 
# # ■ group分布（件数ベース）
# 
# ```
# tracking      : 66件
# opencv        : 13件
# pisco         : 7件
# segmentation  : 6件
# unknown       : 8件
# ```
# 
# ---
# 
# # ■ 重要な事実（妄想なし）
# 
# ### ① 「other地獄」は解消されている
# - 以前: otherが多数
# - 今回: **unknown = 8件のみ（8%）**
# 
# 👉 明確に改善
# 
# ---
# 
# ### ② trackingに強く寄っている
# - **66% = tracking**
# 
# 👉 camera系 + motion系がほぼここに吸収されている
# 
# ---
# 
# ### ③ vace系（pisco含む）はまだ少ない
# - pisco: 7件
# - segmentation: 6件
# 
# 👉 合計でも **13件（13%）**
# 
# ---
# 
# ### ④ opencvは適度に分離されている
# - 13件
# 
# 👉 ルールベース処理候補として機能している
# 
# ---
# 
# # ■ 問題点（定量ベース）
# 
# ### 問題①: trackingが広すぎる
# ```
# 66件 = 2/3
# ```
# 
# 中身を見ると：
# - dolly
# - zoom
# - move
# - gesture
# - pose変更
# 
# 👉 **異なる処理が1グループに混在**
# 
# ---
# 
# ### 問題②: pisco/vaceに流れていない
# ```
# 本来 generative/edit 系が少ない
# ```
# 
# 👉 LLM2が「安全側（tracking）」に寄せている
# 
# ---
# 
# ### 問題③: unknown 8件はまだ改善余地あり
# ```
# 8% = 後段で救える余地
# ```
# 
# ---
# 
# # ■ 改善余地（件数ベース）
# 
# ## 改善可能な対象
# 
# ### ① tracking → 再分類余地
# ```
# 66件のうち少なくとも 30〜40件
# ```
# 
# 理由:
# - motionだけでなく
# - 「編集系（生成/変形）」が含まれている
# 
# ---
# 
# ### ② unknown → 救済
# ```
# 8件 → ほぼ全件修正可能
# ```
# 
# ---
# 
# ### ③ pisco/vace不足
# ```
# +10〜20件は振り分け可能
# ```
# 
# ---
# 
# # ■ 結論（数値ベース）
# 
# ```
# 改善余地合計: 約40〜60件
# ```
# 
# 内訳：
# - tracking再分類: 30〜40件
# - unknown修正: 8件
# - pisco/vace補正: 10〜20件（重複含む）
# 
# ---
# 
# # ■ 次にやるべきこと（重要）
# 
# ```
# LLM2をさらに分解する
# ```
# 
# 具体：
# 
# ### NG
# ```
# group = tracking
# ```
# 
# ### 必要
# ```
# group:
# - camera_motion
# - human_motion
# - object_edit
# - generative_edit（pisco/vace）
# - segmentation
# ```
# 
# ---
# 
# # ■ 本質
# 
# ```
# 今は「分類はできたが、粒度が粗い」
# ```
# 
# 👉 精度問題ではなく **設計問題**
# 
# ---
# 
# 必要なら次：
# 
# ```
# ・tracking内訳をさらに分解（何件が何タスクか）
# ・どのinstructionが誤分類か特定
# ```
# 

# ---
# ---
# ---

# # 2段目のプロンプトを修正して改善を試みる

# In[25]:


# ==========================================
# 2段目LLM用プロンプト
# ==========================================
def build_refine_prompt(original_instruction, parsed_json):
    """
    背景意図:
    - 1段目の出力を「作り直す」のではなく「補正」する
    - 1回目の2段目で安定していた JSON only の性質を維持する
    - tracking に寄りすぎる問題だけを最小限の修正で抑える
    - 長い説明や複雑な推論を避け、LLMを「変換器」として使う
    """

    return f"""
You are a strict JSON refiner for a video editing pipeline.

Your job is to FIX the given JSON so it is:
1. correctly routed
2. schema-valid
3. executable downstream

Do NOT add explanations.
Do NOT add notes.
Do NOT output multiple JSON objects.
Do NOT add extra fields.
Do NOT change the meaning unless necessary for correction.

Use ONLY one of these group values:
- camera_motion
- framing_control
- human_motion
- object_motion
- generative_edit
- segmentation
- opencv

Use these principles:
- Prefer a more specific group over a generic one.
- Use camera_motion for zoom, dolly, pan, tilt, arc shot, angle change.
- Use framing_control for keep centered, focus, crop, subject framing.
- Use human_motion for pose, gesture, facial expression, body motion.
- Use object_motion for moving or animating objects.
- Use generative_edit for style transfer, background replacement, object replacement, scene generation.
- Use segmentation for mask extraction, cutout, foreground/background separation.
- Use opencv for simple color/blur/filter/brightness style operations.

Schema:
{{
  "group": "camera_motion | framing_control | human_motion | object_motion | generative_edit | segmentation | opencv",
  "task": "string",
  "target": "string",
  "target_kind": "face | person | object | scene | unknown",
  "motion_style": "smooth | gradual | steady | slow | fast | none",
  "keep_centered": true | false
}}

Instruction:
{original_instruction}

Broken JSON:
{parsed_json}

Return ONLY valid JSON.
"""


# In[26]:


# ==========================================
# 実行
# ==========================================
input_path = "/workspace/notebook/parse_instruction_ver03_3.txt"

data = load_txt(input_path)

refined = refine_batch(data, batch_size=4)

# ==========================================
# 保存
# ==========================================
output_path = "/workspace/notebook/parse_instruction_ver03_3_refined_2.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for (inst, _), res in zip(data, refined):
        f.write(f"Instruction: {inst}\n")
        f.write(f"Refined: {res}\n")
        f.write("-" * 50 + "\n")

print("Saved:", output_path)

# ==========================================
# 簡易評価
# ==========================================
num_total = len(refined)
num_error = sum(1 for r in refined if "error" in r)

print("Total:", num_total)
print("Error:", num_error)


# ## 1. 今回（2段目改善後）
# - **総件数**: 100
# - **error**: 15件（=15%）
# 
# ### group分布
# - generative_edit: **30**
# - camera_motion: **24**
# - human_motion: **10**
# - segmentation: **9**
# - object_motion: **5**
# - opencv: **5**
# - framing_control: **1**
# - 不正（複数group混在）: **1**
# 
# ---
# 
# ## 2. 精度の観点（妄想なしで事実ベース）
# 
# ### 改善している点
# - groupが**明確に分散**
#   - camera_motion + generative_edit で **54%**
#   - segmentation / opencv / object_motion が**ちゃんと出ている**
# - 「other系に偏る問題」は解消されている（構造的改善）
# 
# → **ルーティング精度は上がっている**
# 
# ---
# 
# ### 悪化している点
# - error: **15%**
# - 不正group（複数値）: 1件
# 
# → **フォーマット安定性は悪化**
# 
# ---
# 
# ## 3. 結論
# 
# ```text
# 内容精度（routing）: 明確に向上
# フォーマット安定性: 悪化
# ```
# 
# ---
# 
# ## 4. 判断
# 
# 今回の結果は良い状態
# 
# 理由：
# - errorは後処理で潰せる
# - groupミスは後段で修正可能
# - しかし routingの誤りは後段で修復困難
# 
# ---
# 
# ## 5. 次の一手（重要）
# 
# やるべきはこれだけ：
# 
# ```text
# errorを減らす（プロンプトではなく後処理で）
# ```
# 
# 具体：
# - extract_json 強化
# - normalizeで fallback 強制
# - 複数group → 1つに丸める
# 
# ---
# 
# ## まとめ（短く）
# 
# ```text
# 精度は上がっている（OK）
# 壊れやすくなっている（修正対象）
# ```
# 
# 次は「error 15 → 0」にするフェーズ。
# 
