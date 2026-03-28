#!/usr/bin/env python
# coding: utf-8

# ## 前提条件
# - 設計方針は、instruction → 正規化意味表現 → 実行プラン → 編集実行 の4段に分けるのがよいです。
# - ver01 / ver02 は instruction の表現ゆれ、ver08 は instruction に対する GT task であり、<br>この組み合わせは「表現差に頑健な instruction parser」を作るのに適しています。

# ## 1. 全体方針
# - 最初から instruction から OpenCV / VACE / PISCO などの具体手段を直接出させるより、まず 中間表現 を作るべきです。
# - 理由は、ver08 の task はかなり有用ですが、target や params の粒度に揺れや壊れた例も一部あり、<br>
#   そのまま最終実行形式にすると downstream が不安定になるためです。
# - たとえば色変更で target が長文化している例や、replacement の category が対象全体ではなく hat に寄っている例が見られます。
# 
# - したがって、LLM の責務は次の2つに限定します。
#     1. instruction を読んで 意味を壊さずに構造化 する
#     2. 構造化結果から 後段が選択可能な編集意図 を出す
# 
# - 編集そのものの品質は後段の CV / video editing pipeline に寄せます。
# - この分離をしないと、LLM の気分で処理系の選択が揺れます。

# ## 2. 推奨アーキテクチャ
# ### A. Instruction Parser 層
# 
# 入力:
# - ver01 / ver02 の instruction
# - 必要なら video_path
# - 必要なら selected_class / selected_subclass 相当の補助情報
# 
# 出力:
# - Canonical Edit Spec（正規化中間表現）
# 
# ここでは、表現ゆれを吸収します。
# 
# 「Replace」「Swap」「Edit the video so that replace」のような言い換え差を同じ意味に畳みます。
# 
# ver01 / ver02 はまさにこの学習材料です。
# 
# ### B. Planner 層
# 
# 入力:
# - Canonical Edit Spec
# 
# 出力:
# - 実行可能な subtask 列
#     - 必要な検出対象
#     - 必要な補助処理
#     - 推奨 backend
# 
# ここで初めて、OpenCV / GroundingDINO+SAM / PISCO / LaMa / E2FGVI / RAFT / diffusion 系を選びます。
# 
# 各編集カテゴリごとの現実解は、添付の詳細版 md 群にかなり整理されています。
# 
# 背景変更は 
# - segmentation 中心、
# - 色変更は OpenCV 中心、
# - 数量増加は insertion 問題、
# - instance replacement は SAM + diffusion / PISCO 系、
# - motion は pose or tracking、
# - camera motion は zoom と dolly を分ける、
# という設計が妥当です。
# 
# ### C. Executor 層
# 
# 入力:
# - planner 出力
# - 動画フレーム列
# 
# 出力:
# - 編集済み動画
# 
# ここは LLM を使わず、決め打ちの実装にします。

# In[15]:


print("test")


# ## このノートブックの構成
# 
# instruction<br>
#     ↓<br>
# LLM（構造化出力）<br>
#    ↓<br>
# 正規化（軽微）<br>
#    ↓<br>
# GT比較<br>
#    ↓<br>
# スコア算出<br>
#    ↓<br>
# 改善（プロンプト or ルール）<br>
# 
# ### 1. 入力
# ```json
# {
#   "instruction": "...",
#   "gt": {gt_ver09 の1件}
# }
# ```
# 
# ### 2. LLM出力仕様（ここが最重要）
# 出力フォーマット（固定）
# ```json
# {
#   "action": "...",
#   "target": "...",
#   "constraints": [...],
#   "params": {...}
# }
# ```

# ### プロンプトの例
# ```json
# You are an instruction parser for video editing tasks.
# 
# Convert the given instruction into structured JSON.
# 
# Requirements:
# - Output must strictly follow the schema
# - Do not add extra keys
# - Keep it minimal and executable
# - Target must be specific but detectable
# - Constraints must be concise
# - Params must be minimal but sufficient
# 
# Schema:
# {
#   "action": string,
#   "target": string,
#   "constraints": string[],
#   "params": object
# }
# 
# Instruction:
# {instruction}
# ```

# ### 3. 正規化

# In[16]:


# 正規化
def normalize(pred):
    # 小文字化
    pred["target"] = pred["target"].lower()

    # constraints ソート
    pred["constraints"] = sorted(set(pred["constraints"]))

    # 数値補正
    if "count" in pred["params"]:
        pred["params"]["count"] = int(pred["params"]["count"])

    return pred


# ### 4. 評価ロジック
# スコアを分解
# ```json
# total = 100
# 
# action      : 30点（完全一致）
# target      : 30点（文字一致 or 部分一致）
# params      : 30点（key + value一致）
# constraints : 10点（集合一致）
# ```

# #### 実装例

# In[17]:


def score(pred, gt):
    s = 0

    # action
    if pred["action"] == gt["action"]:
        s += 30

    # target（部分一致OK）
    if pred["target"] in gt["target"] or gt["target"] in pred["target"]:
        s += 30

    # params
    match = 0
    for k in gt["params"]:
        if k in pred["params"] and pred["params"][k] == gt["params"][k]:
            match += 1
    if len(gt["params"]) > 0:
        s += 30 * (match / len(gt["params"]))

    # constraints
    inter = set(pred["constraints"]) & set(gt["constraints"])
    union = set(pred["constraints"]) | set(gt["constraints"])
    if len(union) > 0:
        s += 10 * (len(inter) / len(union))

    return s


# In[18]:


import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

gt_path = "/workspace/data/annotations_gt_task_ver09.json"
augment_path_ver01 = "/workspace/data/annotations_grouped_ver01.json"
augment_path_ver02 = "/workspace/data/annotations_grouped_ver02.json"

# =========================
# 1. モデルロード
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


# In[19]:


# =========================
# 2. プロンプト
# =========================
SYSTEM_PROMPT = """You are an instruction parser for video editing tasks.

Convert the given instruction into structured JSON.

Requirements:
- Output must strictly follow the schema
- Do not add extra keys
- Keep it minimal and executable
- Target must be specific but detectable
- Constraints must be concise
- Params must be minimal but sufficient

Schema:
{
  "action": string,
  "target": string,
  "constraints": string[],
  "params": object
}
"""

def build_prompt(instruction):
    return f"""<s>[INST] {SYSTEM_PROMPT}

Instruction:
{instruction}

Output JSON only:
[/INST]"""


# In[20]:


# =========================
# 3. LLM呼び出し
# =========================
def call_llm(instruction):
    prompt = build_prompt(instruction)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,   # 安定化
            temperature=0.0
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # JSON抽出
    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except:
        return {"action": "", "target": "", "constraints": [], "params": {}}

# =========================
# 4. 正規化
# =========================
def normalize(pred):
    if "target" in pred:
        pred["target"] = pred["target"].lower()

    if "constraints" in pred:
        pred["constraints"] = sorted(set(pred["constraints"]))

    if "params" in pred and "count" in pred["params"]:
        try:
            pred["params"]["count"] = int(pred["params"]["count"])
        except:
            pass

    return pred

# =========================
# 5. スコア
# =========================
def score(pred, gt):
    s = 0

    # action
    if pred.get("action") == gt.get("action"):
        s += 30

    # target（部分一致）
    if pred.get("target") and gt.get("target"):
        if pred["target"] in gt["target"] or gt["target"] in pred["target"]:
            s += 30

    # params
    match = 0
    gt_params = gt.get("params", {})
    pred_params = pred.get("params", {})

    for k in gt_params:
        if k in pred_params and pred_params[k] == gt_params[k]:
            match += 1

    if len(gt_params) > 0:
        s += 30 * (match / len(gt_params))

    # constraints
    pred_c = set(pred.get("constraints", []))
    gt_c = set(gt.get("constraints", []))

    if len(pred_c | gt_c) > 0:
        s += 10 * (len(pred_c & gt_c) / len(pred_c | gt_c))

    return s


# In[21]:


# =========================
# 6. データ読み込み
# =========================
with open(gt_path, "r") as f:
    dataset = json.load(f)

from pprint import pprint
pprint("dataset[0]:")
pprint(dataset[0])


# In[22]:


# =========================
# 7. 評価ループ
# =========================
import tqdm
results = []

for data in tqdm.tqdm(dataset):
    instruction = data["instruction"]

    if "tasks" not in data or len(data["tasks"]) == 0:
        continue

    gt = data["tasks"][0]   # ← ここが重要

    pred = call_llm(instruction)
    pred = normalize(pred)

    sc = score(pred, gt)
    
    print(f"score = {sc}")
    print(f"   gt = {gt}")
    print(f"   pred = {pred}")

    results.append(sc)

# =========================
# 8. 結果
# =========================
avg_score = sum(results) / len(results)

print("AVG SCORE:", avg_score)


# In[ ]:




