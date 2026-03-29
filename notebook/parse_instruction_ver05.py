#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 0. 設定
# ============================================================
GT_PATH = "/workspace/data/annotations_gt_task_ver09.json"

# 推奨
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 既に手元にあるならこちらでも可
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

LOCAL_FILES_ONLY = False
TOP_K_RETRIEVE = 12
TOP_K_EXAMPLES = 5
MAX_NEW_TOKENS = 300



# In[2]:


# ============================================================
# 1. 文字列正規化
# ============================================================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_similarity(a: str, b: str) -> float:
    """
    軽量な類似度。
    retrieval用途なので埋め込みでなくてもまず回る。
    """
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

    return 0.45 * char_sim + 0.55 * jaccard


# ============================================================
# 2. GTの主タスク抽出
#    今回のschemaは単一taskなので、GT側も主タスク1件に揃える
# ============================================================
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
    # 以下は補助
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


def extract_primary_task(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    tasks[0] を盲目的に使わない。
    主編集 action を優先して1件抽出する。
    """
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
# 3. GTから候補辞書を作る
#    「ゼロから生成しない」ための中核
# ============================================================
def freeze_value(v: Any) -> Any:
    """
    dict/list を hashable にするための補助
    """
    if isinstance(v, dict):
        return tuple(sorted((k, freeze_value(val)) for k, val in v.items()))
    if isinstance(v, list):
        return tuple(freeze_value(x) for x in v)
    return v


def unfreeze_value(v: Any) -> Any:
    if isinstance(v, tuple):
        # dict か list か判定
        if len(v) > 0 and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in v):
            return {k: unfreeze_value(val) for k, val in v}
        return [unfreeze_value(x) for x in v]
    return v


def build_candidate_dictionary(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    GT全体からUnique候補を作る。
    ただし独立に全部持つのではなく、class/subclass/action の関係も保持する。
    """
    classes = set()
    subclasses = set()

    class_to_subclasses = {}
    subclass_to_actions = {}
    action_to_targets = {}
    action_to_constraints = {}
    action_to_params = {}

    for item in dataset:
        cls = item.get("class", "")
        sub = item.get("subclass", "")
        primary = extract_primary_task(item.get("tasks", []))

        act = primary.get("action", "")
        tgt = primary.get("target", "")
        cons = tuple(primary.get("constraints", []))
        prm = freeze_value(primary.get("params", {}))

        classes.add(cls)
        subclasses.add(sub)

        class_to_subclasses.setdefault(cls, set()).add(sub)
        subclass_to_actions.setdefault(sub, set()).add(act)

        if act:
            action_to_targets.setdefault(act, set()).add(tgt)
            action_to_constraints.setdefault(act, set()).add(cons)
            action_to_params.setdefault(act, set()).add(prm)

    return {
        "classes": sorted(classes),
        "subclasses": sorted(subclasses),
        "class_to_subclasses": {k: sorted(v) for k, v in class_to_subclasses.items()},
        "subclass_to_actions": {k: sorted(v) for k, v in subclass_to_actions.items()},
        "action_to_targets": {k: sorted(v) for k, v in action_to_targets.items()},
        "action_to_constraints": {
            k: [list(x) for x in sorted(v, key=lambda z: str(z))]
            for k, v in action_to_constraints.items()
        },
        "action_to_params": {
            k: [unfreeze_value(x) for x in sorted(v, key=lambda z: str(z))]
            for k, v in action_to_params.items()
        }
    }


# ============================================================
# 4. retrieval
#    instructionに近いGT事例を探す
# ============================================================
def retrieve_topk_examples(instruction: str, dataset: List[Dict[str, Any]], top_k: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
    scored = []
    for item in dataset:
        s = text_similarity(instruction, item["instruction"])
        scored.append((s, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# ============================================================
# 5. 候補を絞る
#    全辞書を丸ごと見せると長すぎるので、retrieval上位から候補集合を作る
# ============================================================
def build_local_candidates(retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes = set()
    subclasses = set()
    actions = set()
    targets = set()
    constraints_pool = set()
    params_pool = []

    for item in retrieved:
        classes.add(item.get("class", ""))
        subclasses.add(item.get("subclass", ""))

        primary = extract_primary_task(item.get("tasks", []))
        act = primary.get("action", "")
        tgt = primary.get("target", "")

        if act:
            actions.add(act)
        if tgt:
            targets.add(tgt)

        for c in primary.get("constraints", []):
            constraints_pool.add(c)

        prm = primary.get("params", {})
        if prm not in params_pool:
            params_pool.append(prm)

    return {
        "classes": sorted(classes),
        "subclasses": sorted(subclasses),
        "actions": sorted(actions),
        "targets": sorted(targets),
        "constraints_pool": sorted(constraints_pool),
        "params_pool": params_pool
    }


# ============================================================
# 6. few-shot 例を作る
# ============================================================
def build_fewshot_examples(retrieved: List[Dict[str, Any]], k: int = TOP_K_EXAMPLES) -> str:
    chunks = []
    for item in retrieved[:k]:
        primary = extract_primary_task(item.get("tasks", []))
        chunks.append(
            f"""Instruction:
{item["instruction"]}

Output:
{json.dumps({
    "tasks": [{
        "action": primary["action"],
        "target": primary["target"],
        "constraints": primary["constraints"],
        "params": primary["params"]
    }]
}, ensure_ascii=False, indent=2)}
"""
        )
    return "\n".join(chunks)


# ============================================================
# 7. prompt
#    生成ではなく「候補から選択」を強制
# ============================================================
def build_prompt(instruction: str, retrieved: List[Dict[str, Any]], local_candidates: Dict[str, Any]) -> str:
    fewshots = build_fewshot_examples(retrieved)

    return f"""
You are a video editing task planner.

Your job is NOT to invent labels.
You must choose values from candidate lists derived from GT examples.

Input:
- One instruction only

Output:
- Exactly one task in JSON
- Output schema must be:
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

Rules:
1. Do NOT output multiple tasks.
2. Do NOT invent a new action.
3. Prefer values from the candidate lists below.
4. params must be chosen in a minimal but valid way from GT-style examples.
5. Output JSON only. No explanation.

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
def load_model(model_name: str = MODEL_NAME, local_files_only: bool = LOCAL_FILES_ONLY):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only
    )

    return tokenizer, model


# ============================================================
# 9. LLM呼び出し
# ============================================================
def generate_task_json(
    instruction: str,
    dataset: List[Dict[str, Any]],
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, Any]:
    retrieved = retrieve_topk_examples(instruction, dataset, TOP_K_RETRIEVE)
    local_candidates = build_local_candidates(retrieved)
    prompt = build_prompt(instruction, retrieved, local_candidates)

    # chat template が使えるモデルではこちらが安定
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt}
        ]
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
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 最後のJSONを抽出
    start = decoded.find("{")
    end = decoded.rfind("}") + 1
    if start == -1 or end == 0:
        return {"tasks": [{"action": "", "target": "", "constraints": [], "params": {}}], "_raw": decoded}

    json_str = decoded[start:end]

    try:
        pred = json.loads(json_str)
    except Exception:
        pred = {"tasks": [{"action": "", "target": "", "constraints": [], "params": {}}], "_raw": decoded}

    # schema安全化
    if "tasks" not in pred or not isinstance(pred["tasks"], list) or len(pred["tasks"]) == 0:
        pred = {"tasks": [{"action": "", "target": "", "constraints": [], "params": {}}], "_raw": decoded}

    task = pred["tasks"][0]
    pred["tasks"] = [{
        "action": task.get("action", ""),
        "target": task.get("target", ""),
        "constraints": task.get("constraints", []) if isinstance(task.get("constraints", []), list) else [],
        "params": task.get("params", {}) if isinstance(task.get("params", {}), dict) else {},
    }]

    return pred


# ============================================================
# 10. 評価
#     完全一致ではなく、validation向けに軽い表記ゆれ耐性を持たせる
# ============================================================
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
        "class": 0.0,      # 今回predにclass/subclassを出していないので参考用
        "subclass": 0.0,
        "action": text_similarity(pred_task.get("action", ""), gt_task.get("action", "")),
        "target": text_similarity(pred_task.get("target", ""), gt_task.get("target", "")),
        "constraints": score_constraints(pred_task.get("constraints", []), gt_task.get("constraints", [])),
        "params": score_params(pred_task.get("params", {}), gt_task.get("params", {})),
    }

    # 単一task schema前提の総合点
    scores["total"] = (
        0.35 * scores["action"] +
        0.30 * scores["target"] +
        0.15 * scores["constraints"] +
        0.20 * scores["params"]
    )

    return scores



# In[3]:


# ============================================================
# 11. 実行
# ============================================================
with open(GT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

tokenizer, model = load_model()

results = []

for idx, item in enumerate(dataset):
    instruction = item["instruction"]
    pred = generate_task_json(instruction, dataset, tokenizer, model)
    score = score_prediction(pred, item)

    results.append({
        "video_path": item.get("video_path", ""),
        "instruction": instruction,
        "pred": pred,
        "score": score
    })

    print(f"[{idx+1}/{len(dataset)}] total={score['total']:.4f} "
            f"action={score['action']:.4f} target={score['target']:.4f} "
            f"constraints={score['constraints']:.4f} params={score['params']:.4f}")

# 集計
keys = ["action", "target", "constraints", "params", "total"]
summary = {}
for k in keys:
    summary[k] = sum(r["score"][k] for r in results) / len(results)

print("\n===== SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))

with open("prediction_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "results": results
    }, f, ensure_ascii=False, indent=2)

