#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 背景
# ============================================================
# このコードの目的は、
#   instruction -> task planning JSON
# を、GT（annotations_gt_task_ver09.json）に強く制約された形で安定出力すること。
#
# 過去の試行錯誤では：
# - ver03:
#     自由生成に寄っており、意味は近くてもGTラベルと一致せず評価が崩れた
# - ver05:
#     GT候補を見せる設計には進化したが、
#     「候補から選ぶ」ではなく「候補を見てまだ生成する」状態が残っていた
#
# そのため今回は、
#   生成器ではなく「GT制約付き選択器」
# としてLLMを使う。
#
# 重要方針：
# 1. action は GT の候補からしか選ばせない
# 2. target は action に紐づく候補からしか選ばせない
# 3. params も action に紐づく候補例からしか選ばせない
# 4. 最後に後処理で GT 空間へ強制丸め込みする
#
# これにより、
#   「意味は合っているがラベルが違う」
# という評価崩壊を防ぐ。


# ============================================================
# 0. import
# ============================================================
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


# ============================================================
# 1. 設定
# ============================================================
# 背景意図:
# - ver05 のファイルパス運用を踏襲する
# - モデル差し替えしやすくする
# - ローカル実行前提にできるようにする
DATA_ROOT_DIR = "/workspace/data"
GT_PATH = f"{DATA_ROOT_DIR}/annotations_gt_task_ver09.json"

# 推奨:
# JSON安定性が比較的高い Qwen 系
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 既に手元にあるならこちらでも可
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

LOCAL_FILES_ONLY = False

# Retrievalで使う候補数
TOP_K_RETRIEVE = 12

# プロンプトに載せる few-shot 数
TOP_K_EXAMPLES = 5

# 出力長
MAX_NEW_TOKENS = 260

# 途中保存
OUTPUT_PATH = "/workspace/notebook/prediction_results_ver07.json"



# In[2]:


# ============================================================
# 2. 文字列正規化
# ============================================================
# 背景意図:
# - GT比較や retrieval の際に、
#   大文字小文字や記号差で不利にならないようにする
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 背景意図:
# - 埋め込みを使わず軽量に instruction 類似度を取る
# - retrieval と後処理丸め込みの両方で使う
def text_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0

    char_sim = SequenceMatcher(None, a_n, b_n).ratio()

    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

    # 背景意図:
    # - token overlap をやや重めにして意味近傍を優先
    return 0.45 * char_sim + 0.55 * jaccard


# ============================================================
# 3. GTの主タスク抽出
# ============================================================
# 背景意図:
# - GTは tasks 配列を持つが、今回の予測出力は単一 task
# - そのため GT 側も単一 task に縮約して比較する必要がある
# - tasks[0] を盲信せず、主編集 action を優先して抽出する
PRIMARY_ACTION_PRIORITY = [
    "replace_background",
    "replace_object",
    "add_object",
    "increase_amount",
    "change_color",
    "remove_object",
    "edit_motion",
    "edit_expression",
    "change_camera_angle",
    "zoom_in",
    "zoom_out",
    "dolly_in",
    "orbit_camera",
    "apply_style",
    "add_effect",
    # 以下は補助タスク
    "preserve_foreground",
    "preserve_objects",
    "preserve_identity",
    "preserve_focus",
    "preserve_framing",
    "preserve_layout",
    "preserve_material_appearance",
    "align_replacement",
    "match_appearance",
    "match_lighting",
    "match_background_camera_properties",
    "match_effect_lighting",
    "match_scene_interaction",
    "stabilize_instances",
    "stabilize_edit",
    "stabilize_motion",
    "stabilize_style",
    "stabilize_effect",
    "stabilize_composite",
    "stabilize_inpaint",
    "refine_mask",
    "blend_instances",
    "inpaint_background",
    "adjust_perspective",
    "track_effect",
    "enhance_style_details",
]

PRIMARY_ACTION_RANK = {a: i for i, a in enumerate(PRIMARY_ACTION_PRIORITY)}


# 背景意図:
# - 単一 task schema に合わせるための GT 縮約
def extract_primary_task(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tasks:
        return {
            "action": "",
            "target": "",
            "constraints": [],
            "params": {}
        }

    ranked = []
    for idx, t in enumerate(tasks):
        action = normalize_text(t.get("action", ""))
        rank = PRIMARY_ACTION_RANK.get(action, 9999)
        ranked.append((rank, idx, t))

    ranked.sort(key=lambda x: (x[0], x[1]))
    primary = ranked[0][2]

    return {
        "action": primary.get("action", ""),
        "target": primary.get("target", ""),
        "constraints": primary.get("constraints", []),
        "params": primary.get("params", {}),
    }


# ============================================================
# 4. GT候補辞書構築
# ============================================================
# 背景意図:
# - 「ゼロから作らない」ための中核
# - class/subclass/action/target/params の unique 値と関係を保持する
def freeze_value(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((k, freeze_value(val)) for k, val in v.items()))
    if isinstance(v, list):
        return tuple(freeze_value(x) for x in v)
    return v


def unfreeze_value(v: Any) -> Any:
    if isinstance(v, tuple):
        if len(v) > 0 and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in v):
            return {k: unfreeze_value(val) for k, val in v}
        return [unfreeze_value(x) for x in v]
    return v



# In[3]:


import json
from collections import defaultdict

# ==============================
# 1. 候補辞書構築
# ==============================
def build_candidate_dictionary(dataset):
    """
    背景：
    - LLMの自由生成を禁止するため、GTから候補空間を構築する
    - actionを中心に、target / constraints / paramsを紐付ける

    意図：
    - action → 他要素 の関係を保持（独立させない）
    - setで重複排除
    - hash不可型(list/dict)を安全に扱う
    """

    # ------------------------------
    # 初期化
    # ------------------------------
    action_set = set()

    action_to_targets = defaultdict(set)
    action_to_constraints = defaultdict(set)
    action_to_params = defaultdict(set)

    subclass_to_actions = defaultdict(set)

    # ------------------------------
    # dataset走査
    # ------------------------------
    for data in dataset:

        subclass = data.get("subclass")
        tasks = data.get("tasks", [])

        for task in tasks:
            act = task.get("action")
            tgt = task.get("target")
            cons = task.get("constraints", [])
            prm = task.get("params", {})

            # ------------------------------
            # action
            # ------------------------------
            if not act:
                continue

            action_set.add(act)

            if subclass:
                subclass_to_actions[subclass].add(act)

            # ------------------------------
            # target（stringなのでそのまま）
            # ------------------------------
            if tgt:
                if isinstance(tgt, list):
                    tgt = tuple(tgt)
                elif isinstance(tgt, str):
                    tgt = (tgt,)   # ←統一（重要）
                else:
                    tgt = (str(tgt),)

                action_to_targets[act].add(tgt)

            # ------------------------------
            # constraints（list → tuple）
            # ------------------------------
            if isinstance(cons, list):
                cons_tuple = tuple(cons)
            else:
                cons_tuple = (cons,)

            action_to_constraints[act].add(cons_tuple)

            # ------------------------------
            # params（dict → json string）
            # ------------------------------
            if isinstance(prm, dict):
                prm_str = json.dumps(prm, sort_keys=True)
            else:
                prm_str = str(prm)

            action_to_params[act].add(prm_str)

    # ------------------------------
    # set → list（後処理しやすくする）
    # ------------------------------
    candidate_dict = {
        "actions": sorted(list(action_set)),
        "action_to_targets": {
            k: sorted(list(v)) for k, v in action_to_targets.items()
        },
        "action_to_constraints": {
            k: [list(x) for x in v] for k, v in action_to_constraints.items()
        },
        "action_to_params": {
            k: [json.loads(x) for x in v] for k, v in action_to_params.items()
        },
        "subclass_to_actions": {
            k: sorted(list(v)) for k, v in subclass_to_actions.items()
        }
    }

    return candidate_dict


# In[4]:


# ============================================================
# 5. Retrieval
# ============================================================
# 背景意図:
# - 全GTを毎回見せると長すぎる
# - instruction に近い事例だけを見せることで、
#   候補空間を圧縮する
def retrieve_topk_examples(
    instruction: str,
    dataset: List[Dict[str, Any]],
    top_k: int = TOP_K_RETRIEVE
) -> List[Dict[str, Any]]:
    scored = []
    for item in dataset:
        s = text_similarity(instruction, item["instruction"])
        scored.append((s, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# 背景意図:
# - retrieval 上位からローカル候補集合を作る
# - これをLLMに渡すことで、自由生成をさらに抑える
def build_local_candidates(retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes = set()
    subclasses = set()
    actions = set()
    targets = set()
    constraints_pool = set()
    params_pool = set() #[]

    for item in retrieved:
        classes.add(item.get("class", ""))
        subclasses.add(item.get("subclass", ""))

        primary = extract_primary_task(item.get("tasks", []))
        act = primary.get("action", "")
        tgt = primary.get("target", "")

        if act:
            actions.add(act)
            
        if tgt:
            if isinstance(tgt, list):
                tgt = tuple(tgt)
            elif isinstance(tgt, str):
                tgt = (tgt,)
            else:
                tgt = (str(tgt),)

            targets.add(tgt)

        for c in primary.get("constraints", []):
            if isinstance(c, list):
                c = tuple(c)
            constraints_pool.add(c)

        prm = primary.get("params", {})
        if isinstance(prm, dict):
            prm = json.dumps(prm, sort_keys=True)

        params_pool.add(prm)

    return {
        "classes": sorted(classes),
        "subclasses": sorted(subclasses),
        "actions": sorted(actions),
        "targets": sorted(targets),
        "constraints_pool": sorted(constraints_pool),
        "params_pool": [json.loads(x) for x in params_pool]
    }


# ============================================================
# 6. few-shot 構築
# ============================================================
# 背景意図:
# - retrieval 上位の具体例を few-shot として見せる
# - ただし生成ではなく「どのような形で選ばれるか」を示すため
def build_fewshot_examples(retrieved: List[Dict[str, Any]], k: int = TOP_K_EXAMPLES) -> str:
    chunks = []
    for item in retrieved[:k]:
        primary = extract_primary_task(item.get("tasks", []))
        example = {
            "tasks": [{
                "action": primary["action"],
                "target": primary["target"],
                "constraints": primary["constraints"],
                "params": primary["params"]
            }]
        }

        chunks.append(
            f"""Instruction:
{item["instruction"]}

Output:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""
        )

    return "\n".join(chunks)



# In[5]:


# ============================================================
# 7. プロンプト構築
# ============================================================
# 背景意図:
# - ver05の "Prefer" では弱い
# - 今回は "MUST select" に変える
# - action/target/params を候補から選ばせることを明示
def build_prompt(instruction: str, retrieved: List[Dict[str, Any]], local_candidates: Dict[str, Any]) -> str:
    fewshots = build_fewshot_examples(retrieved)

    return f"""
You are a strict video editing task selector.

Your job is NOT to invent labels.
You MUST select values from the candidate lists derived from GT examples.

Input:
- One instruction only

Output:
- Exactly one task in JSON
- Output schema MUST be:
{{
  "tasks": [
    {{
      "action": "...",
      "target": "...",
      "constraints": ["..."],
      "params": {{}}
    }}
  ]
}}

Hard Rules:
1. Do NOT output multiple tasks.
2. Do NOT invent a new action.
3. Do NOT invent a new target if a candidate exists.
4. You MUST choose action from Candidate actions.
5. You MUST choose target from Candidate targets.
6. params should be selected in a minimal but valid way from Candidate params examples.
7. Output JSON only. No explanation.

Candidate classes:
{json.dumps(local_candidates["classes"], ensure_ascii=False)}

Candidate subclasses:
{json.dumps(local_candidates["subclasses"], ensure_ascii=False)}

Candidate actions:
{json.dumps(local_candidates["actions"], ensure_ascii=False)}

Candidate targets:
{json.dumps(local_candidates["targets"], ensure_ascii=False)}

Candidate constraints pool:
{json.dumps(local_candidates["constraints_pool"], ensure_ascii=False)}

Candidate params examples:
{json.dumps(local_candidates["params_pool"], ensure_ascii=False, indent=2)}

Reference examples:
{fewshots}

Now solve this instruction:

Instruction:
{instruction}
""".strip()


# ============================================================
# 8. モデルロード
# ============================================================
# 背景意図:
# - ver05の構造を踏襲しつつ、deprecated対策で dtype を使う
def load_model(model_name: str = MODEL_NAME, local_files_only: bool = LOCAL_FILES_ONLY):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only
    )

    return tokenizer, model


# ============================================================
# 9. JSON抽出
# ============================================================
# 背景意図:
# - LLM出力が前置き/後置きを含む場合がある
# - 最後のJSONブロックを抽出する
def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        return {}

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except Exception:
        return {}


# ============================================================
# 10. 候補への丸め込み
# ============================================================
# 背景意図:
# - LLMが候補外の近い文字列を出しても、
#   最終的にGT空間へ戻すための安全策
def nearest_choice(value: str, candidates: List[str]) -> str:
    if not candidates:
        return ""

    best = None
    best_score = -1.0
    for cand in candidates:
        s = text_similarity(value, cand)
        if s > best_score:
            best_score = s
            best = cand
    return best if best is not None else ""


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, str]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            out.update(flatten_json(v, new_prefix))
    else:
        out[prefix] = str(obj)
    return out


# 背景意図:
# - params は文字列類似度だけでなく構造も見る
# - action に紐づく params 候補例の中から最も近いものに丸める
def nearest_params(pred_params: Dict[str, Any], param_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not param_candidates:
        return {}

    pred_flat = flatten_json(pred_params)

    best = None
    best_score = -1.0

    for cand in param_candidates:
        cand_flat = flatten_json(cand)

        if not pred_flat and not cand_flat:
            return cand

        total = 0.0
        count = 0

        for ck, cv in cand_flat.items():
            local_best = 0.0
            for pk, pv in pred_flat.items():
                key_sim = text_similarity(ck, pk)
                val_sim = text_similarity(cv, pv)
                local_best = max(local_best, 0.5 * key_sim + 0.5 * val_sim)

            total += local_best
            count += 1

        score = total / max(1, count)

        if score > best_score:
            best_score = score
            best = cand

    return best if best is not None else {}


# 背景意図:
# - action が決まった後、その action に紐づく候補のみに target/params を制限する
def normalize_prediction_to_gt_space(
    pred: Dict[str, Any],
    local_candidates: Dict[str, Any],
    global_dict: Dict[str, Any]
) -> Dict[str, Any]:
    if "tasks" not in pred or not isinstance(pred["tasks"], list) or len(pred["tasks"]) == 0:
        pred = {"tasks": [{"action": "", "target": "", "constraints": [], "params": {}}]}

    task = pred["tasks"][0]
    raw_action = task.get("action", "")
    raw_target = task.get("target", "")
    raw_constraints = task.get("constraints", [])
    raw_params = task.get("params", {})

    # 1) action を retrieval候補へ丸める
    norm_action = nearest_choice(raw_action, local_candidates["actions"])

    # 2) target は action に紐づく候補へ丸める
    target_candidates = global_dict["action_to_targets"].get(norm_action, [])
    norm_target = nearest_choice(raw_target, target_candidates)

    # 3) constraints も action に紐づく候補集合から近いものを拾う
    constraint_candidates = global_dict["action_to_constraints"].get(norm_action, [])
    # 今回は最も近い constraint set を1つ選ぶ
    best_constraints = []
    best_c_score = -1.0
    for cand in constraint_candidates:
        total = 0.0
        if not cand and not raw_constraints:
            best_constraints = []
            best_c_score = 1.0
            break

        for c in cand:
            local_best = 0.0
            for rc in raw_constraints:
                local_best = max(local_best, text_similarity(c, rc))
            total += local_best

        score = total / max(1, len(cand))
        if score > best_c_score:
            best_c_score = score
            best_constraints = cand

    # 4) params も action に紐づく候補例から近いものへ丸める
    params_candidates = global_dict["action_to_params"].get(norm_action, [])
    norm_params = nearest_params(raw_params, params_candidates)

    return {
        "tasks": [{
            "action": norm_action,
            "target": norm_target,
            "constraints": best_constraints,
            "params": norm_params
        }]
    }


# ============================================================
# 11. LLM呼び出し
# ============================================================
# 背景意図:
# - ver05の retrieval / few-shot を流用
# - ただし生成後に必ず GT 空間へ丸める
def generate_task_json(
    instruction: str,
    dataset: List[Dict[str, Any]],
    global_dict: Dict[str, Any],
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, Any]:
    retrieved = retrieve_topk_examples(instruction, dataset, TOP_K_RETRIEVE)
    local_candidates = build_local_candidates(retrieved)
    prompt = build_prompt(instruction, retrieved, local_candidates)

    # 背景意図:
    # - chat template があるモデルはそれを使う
    # - ない場合は通常prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # 背景意図: 選択問題なので決定的にする
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_json_object(decoded)

    # schema 安全化
    if not pred:
        pred = {"tasks": [{"action": "", "target": "", "constraints": [], "params": {}}], "_raw": decoded}

    # GT空間に丸める
    pred = normalize_prediction_to_gt_space(pred, local_candidates, global_dict)

    return pred


# ============================================================
# 12. スコア計算（ver05流用）
# ============================================================
# 背景意図:
# - ver05の良い部分をそのまま流用する
# - 完全一致ではなく軽い表記ゆれ耐性を維持する
def score_constraints(pred_constraints: List[str], gt_constraints: List[str]) -> float:
    if not pred_constraints and not gt_constraints:
        return 1.0
    if not gt_constraints:
        return 1.0 if not pred_constraints else 0.5

    total = 0.0
    for g in gt_constraints:
        best = 0.0
        for p in pred_constraints:
            best = max(best, text_similarity(g, p))
        total += best

    return total / len(gt_constraints)


def score_params(pred_params: Dict[str, Any], gt_params: Dict[str, Any]) -> float:
    gt_flat = flatten_json(gt_params)
    pred_flat = flatten_json(pred_params)

    if not gt_flat and not pred_flat:
        return 1.0
    if not gt_flat:
        return 0.5

    total = 0.0
    for gk, gv in gt_flat.items():
        best = 0.0
        for pk, pv in pred_flat.items():
            key_sim = text_similarity(gk, pk)
            val_sim = text_similarity(gv, pv)
            best = max(best, 0.5 * key_sim + 0.5 * val_sim)
        total += best

    return total / len(gt_flat)


def score_prediction(pred: Dict[str, Any], gt_item: Dict[str, Any]) -> Dict[str, float]:
    gt_task = extract_primary_task(gt_item["tasks"])
    pred_task = pred["tasks"][0]

    scores = {
        "action": text_similarity(pred_task.get("action", ""), gt_task.get("action", "")),
        "target": text_similarity(pred_task.get("target", ""), gt_task.get("target", "")),
        "constraints": score_constraints(pred_task.get("constraints", []), gt_task.get("constraints", [])),
        "params": score_params(pred_task.get("params", {}), gt_task.get("params", {})),
    }

    # 背景意図:
    # - action / target を重め
    # - constraints は補助なので軽め
    scores["total"] = (
        0.35 * scores["action"] +
        0.30 * scores["target"] +
        0.15 * scores["constraints"] +
        0.20 * scores["params"]
    )

    return scores



# ## LLM多段化して改善する
# Stage1: action選択
# 
# Stage2: target / constraints
# 
# Stage3: params

# In[6]:


# ============================================================
# 11'. Multi-stage LLM（ver06拡張）
# ============================================================
# 背景：
# - ver06は1回の生成で全部決めていた
# - その結果、代表的なactionに潰れる問題があった
#
# 意図：
# - action → target → params を段階的に決定
# - 各ステージで候補空間を絞る


# ============================================================
# Stage1: action選択
# ============================================================
def build_prompt_stage1(instruction, local_candidates):
    return f"""
Select ONLY ONE action from the candidate list.

Instruction:
{instruction}

Candidate actions:
{json.dumps(local_candidates["actions"])}

Rules:
- MUST choose from candidate actions
- Output JSON only

Output:
{{"action": "..."}}
"""


def run_stage1_action(instruction, tokenizer, model, local_candidates):
    prompt = build_prompt_stage1(instruction, local_candidates)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_json_object(decoded)

    raw_action = pred.get("action", "")
    action = nearest_choice(raw_action, local_candidates["actions"])

    return action


# ============================================================
# Stage2: target + constraints
# ============================================================
def build_prompt_stage2(instruction, action, target_candidates, constraint_candidates):
    return f"""
Select target and constraints.

Instruction:
{instruction}

Action:
{action}

Candidate targets:
{json.dumps(target_candidates)}

Candidate constraints:
{json.dumps(constraint_candidates)}

Rules:
- MUST choose from candidates
- Output JSON only

Output:
{{
  "target": "...",
  "constraints": ["..."]
}}
"""


def run_stage2_target(
    instruction,
    action,
    global_dict,
    tokenizer,
    model
):
    target_candidates = global_dict["action_to_targets"].get(action, [])
    constraint_candidates = global_dict["action_to_constraints"].get(action, [])

    prompt = build_prompt_stage2(
        instruction,
        action,
        target_candidates,
        constraint_candidates
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_json_object(decoded)

    raw_target = pred.get("target", "")
    raw_constraints = pred.get("constraints", [])

    target = nearest_choice(raw_target, target_candidates)

    # constraints丸め
    best_constraints = []
    best_score = -1

    for cand in constraint_candidates:
        score = 0
        for c in cand:
            score += max([text_similarity(c, rc) for rc in raw_constraints] + [0])
        score /= max(1, len(cand))

        if score > best_score:
            best_score = score
            best_constraints = cand

    return target, best_constraints


# ============================================================
# Stage3: params
# ============================================================
def build_prompt_stage3(instruction, action, params_candidates):
    return f"""
Select params.

Instruction:
{instruction}

Action:
{action}

Candidate params:
{json.dumps(params_candidates, indent=2)}

Rules:
- Choose closest params
- Output JSON only

Output:
{{"params": {{}}}}
"""


def run_stage3_params(instruction, action, global_dict, tokenizer, model):
    params_candidates = global_dict["action_to_params"].get(action, [])

    prompt = build_prompt_stage3(instruction, action, params_candidates)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_json_object(decoded)

    raw_params = pred.get("params", {})

    params = nearest_params(raw_params, params_candidates)

    return params


# ============================================================
# Multi-stage wrapper（ver06置き換え）
# ============================================================
def generate_task_json_multistage(
    instruction,
    dataset,
    global_dict,
    tokenizer,
    model
):
    # ------------------------------
    # retrieval（ver06流用）
    # ------------------------------
    retrieved = retrieve_topk_examples(instruction, dataset, TOP_K_RETRIEVE)
    local_candidates = build_local_candidates(retrieved)

    # ------------------------------
    # Stage1: action
    # ------------------------------
    action = run_stage1_action(
        instruction,
        tokenizer,
        model,
        local_candidates
    )

    # ------------------------------
    # Stage2: target / constraints
    # ------------------------------
    target, constraints = run_stage2_target(
        instruction,
        action,
        global_dict,
        tokenizer,
        model
    )

    # ------------------------------
    # Stage3: params
    # ------------------------------
    params = run_stage3_params(
        instruction,
        action,
        global_dict,
        tokenizer,
        model
    )

    return {
        "tasks": [{
            "action": action,
            "target": target,
            "constraints": constraints,
            "params": params
        }]
    }


# In[7]:


# ============================================================
# 13. main
# ============================================================
# 背景意図:
# - tqdmで途中経過を見せる
# - ver05風に summary を出す
# - 結果を json 保存する

# ------------------------------
# 13.1 GT読み込み
# ------------------------------
with open(GT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

from pprint import pprint
print("dataset[0]:")
pprint(dataset[0])


# In[8]:


# ------------------------------
# 13.2 候補辞書構築
# ------------------------------
global_dict = build_candidate_dictionary(dataset)

# ------------------------------
# 13.3 モデルロード
# ------------------------------
tokenizer, model = load_model()

# ------------------------------
# 13.4 推論 + 評価
# ------------------------------
results = []

progress = tqdm(dataset, desc="infer+eval", total=len(dataset))

for idx, item in enumerate(progress, start=1):
    instruction = item["instruction"]

    pred = generate_task_json_multistage(
        instruction=instruction,
        dataset=dataset,
        global_dict=global_dict,
        tokenizer=tokenizer,
        model=model
    )

    score = score_prediction(pred, item)

    results.append({
        "video_path": item.get("video_path", ""),
        "instruction": instruction,
        "pred": pred,
        "score": score
    })

    progress.set_postfix({
        "idx": idx,
        "total": f"{score['total']:.3f}",
        "a": f"{score['action']:.3f}",
        "t": f"{score['target']:.3f}",
        "p": f"{score['params']:.3f}"
    })

# ------------------------------
# 13.5 summary
# ------------------------------
keys = ["action", "target", "constraints", "params", "total"]
summary = {}
for k in keys:
    summary[k] = sum(r["score"][k] for r in results) / len(results)

print("\n===== SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))

# ------------------------------
# 13.6 保存
# ------------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved to: {OUTPUT_PATH}")

