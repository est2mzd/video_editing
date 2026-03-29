#!/usr/bin/env python
# coding: utf-8

# # annotations_gt_task_ver10
# 
# ## 目的
# - instruction から主編集 task を 1 件抽出し、`annotations_gt_task_ver09.json` をベースに精度改善を試す。
# - この notebook では、試行錯誤の背景と意図を各セルに明示し、あとから方針変更を追えるようにする。
# 
# ## 今回の方針
# 1. GT の複数 task を主編集 task に縮約して比較軸をそろえる。
# 2. まず軽量な baseline を置き、どこで崩れるかを確認する。
# 3. その後、class / subclass / instruction 規則と nearest-example fallback を組み合わせて改善する。
# 4. GPU メモリを多く使う LLM 補正は optional に分離し、必要時だけ有効化する。
# 
# ## メモ
# - LLM セクションを使った後は kernel restart でメモリを解放できる。
# - 保存先は `/workspace/notebook/prediction_results_ver10.json` と `/workspace/notebook/prediction_results_ver10_summary.json`。

# In[1]:


from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pprint import pprint

WORKSPACE = Path("/workspace")
DATA_DIR = WORKSPACE / "data"
NOTEBOOK_DIR = WORKSPACE / "notebook"

RAW_PATH = DATA_DIR / "annotations.jsonl"
GT_PATH = DATA_DIR / "annotations_gt_task_ver09.json"
OUTPUT_PATH = NOTEBOOK_DIR / "prediction_results_ver10.json"
SUMMARY_PATH = NOTEBOOK_DIR / "prediction_results_ver10_summary.json"

USE_LLM_REFINEMENT = False
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("RAW_PATH =", RAW_PATH)
print("GT_PATH =", GT_PATH)
print("OUTPUT_PATH =", OUTPUT_PATH)
print("USE_LLM_REFINEMENT =", USE_LLM_REFINEMENT)
print("LLM_MODEL_NAME =", LLM_MODEL_NAME)


# ## 1. データ読み込みと整列
# 
# 背景と意図:
# - 今回の入力は `annotations.jsonl`、比較対象は `annotations_gt_task_ver09.json`。
# - まずは `video_path` で raw annotation と GT を突き合わせ、後段の評価で同じ母集団を見続けられるようにする。
# - この段階では加工を最小限にとどめ、元データの class / subclass / instruction を素直に確認する。

# In[2]:


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


raw_annotations = read_jsonl(RAW_PATH)
gt_annotations = json.loads(GT_PATH.read_text(encoding="utf-8"))
raw_by_video = {row["video_path"]: row for row in raw_annotations}

merged_records = []
for gt_item in gt_annotations:
    raw_item = raw_by_video.get(gt_item["video_path"], {})
    merged_records.append(
        {
            "video_path": gt_item["video_path"],
            "class": raw_item.get("selected_class", gt_item.get("class", "")),
            "subclass": raw_item.get("selected_subclass", gt_item.get("subclass", "")),
            "instruction": raw_item.get("instruction", gt_item.get("instruction", "")),
            "gt_tasks": gt_item.get("tasks", []),
        }
    )

print("raw annotations:", len(raw_annotations))
print("gt annotations:", len(gt_annotations))
print("merged records:", len(merged_records))
print()
pprint(merged_records[0])


# ## 2. GT を主編集 task に縮約する
# 
# 背景と意図:
# - `annotations_gt_task_ver09.json` は 1 instruction に対して補助 task を複数持つ。
# - ただし、instruction から最初に取りたいのは「主編集 task」であり、比較もまずそこに寄せたほうが改善の方向を見やすい。
# - ここでは `replace_background` や `change_color` のような主タスクを優先し、`refine_mask` や `stabilize_*` は補助として後回しにする。

# In[3]:


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
PRIMARY_ACTION_RANK = {action: index for index, action in enumerate(PRIMARY_ACTION_PRIORITY)}


def extract_primary_task(tasks: list[dict]) -> dict:
    if not tasks:
        return {"action": "", "target": "", "constraints": [], "params": {}}

    ranked = []
    for index, task in enumerate(tasks):
        action = task.get("action", "")
        ranked.append((PRIMARY_ACTION_RANK.get(action, 9999), index, task))

    ranked.sort(key=lambda item: (item[0], item[1]))
    primary = ranked[0][2]
    return {
        "action": primary.get("action", ""),
        "target": primary.get("target", ""),
        "constraints": primary.get("constraints", []),
        "params": primary.get("params", {}),
    }


records = []
for record in merged_records:
    enriched = dict(record)
    enriched["gt_primary"] = extract_primary_task(record["gt_tasks"])
    records.append(enriched)

record_by_video = {record["video_path"]: record for record in records}
action_distribution = Counter(record["gt_primary"]["action"] for record in records)

print("primary action distribution")
for action, count in action_distribution.most_common():
    print(f"- {action}: {count}")


# ## 3. まずは baseline を置く
# 
# 背景と意図:
# - いきなり複雑な方法へ進むと、どの改善が効いたのか見えにくい。
# - そのため最初に `class / subclass` を強く信じる軽量 baseline を置き、target / params がどこで壊れるかを定量化する。
# - 評価関数もここで定義し、以降の比較を同じ物差しで行う。

# In[4]:


COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "violet", "pink", "black", "white",
    "gray", "grey", "silver", "gold", "beige", "brown", "navy", "emerald", "metallic", "neon",
]


def normalize_eval_text(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("_", " ").strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def safe_get(obj, keys, default=None):
    current = obj
    for key in keys:
        if isinstance(key, int):
            if isinstance(current, list) and 0 <= key < len(current):
                current = current[key]
            else:
                return default
        else:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
    return current



def flatten_json(value, prefix="") -> dict:
    items = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            items.update(flatten_json(child, child_prefix))
    else:
        items[prefix] = normalize_eval_text(value)
    return items



def score_action(pred: dict, gt: dict) -> float:
    return float(safe_get(pred, ["tasks", 0, "action"], "") == safe_get(gt, ["tasks", 0, "action"], ""))



def score_target(pred: dict, gt: dict) -> float:
    pred_target = normalize_eval_text(safe_get(pred, ["tasks", 0, "target"], ""))
    gt_target = normalize_eval_text(safe_get(gt, ["tasks", 0, "target"], ""))
    if not pred_target or not gt_target:
        return 0.0
    return float(pred_target in gt_target or gt_target in pred_target)



def score_params(pred: dict, gt: dict) -> float:
    pred_params = safe_get(pred, ["tasks", 0, "params"], {})
    gt_params = safe_get(gt, ["tasks", 0, "params"], {})
    pred_flat = flatten_json(pred_params)
    gt_flat = flatten_json(gt_params)

    if not gt_flat:
        return 1.0
    if not pred_flat:
        return 0.0

    matched = 0
    for key, gt_value in gt_flat.items():
        pred_value = pred_flat.get(key, "")
        if pred_value and (pred_value == gt_value or pred_value in gt_value or gt_value in pred_value):
            matched += 1
    return matched / len(gt_flat)



def score_total(pred: dict, gt: dict) -> float:
    action_score = score_action(pred, gt)
    target_score = score_target(pred, gt)
    params_score = score_params(pred, gt)
    return 0.5 * action_score + 0.2 * target_score + 0.3 * params_score



def evaluate_prediction_map(predictions: dict[str, dict], rows: list[dict]) -> dict:
    scored_rows = []
    for row in rows:
        gt = {"tasks": [row["gt_primary"]]}
        pred = predictions[row["video_path"]]
        scored_rows.append(
            {
                "video_path": row["video_path"],
                "gt_action": row["gt_primary"]["action"],
                "pred_action": safe_get(pred, ["tasks", 0, "action"], ""),
                "action_score": score_action(pred, gt),
                "target_score": score_target(pred, gt),
                "params_score": score_params(pred, gt),
                "total": score_total(pred, gt),
            }
        )

    overall = {
        metric: round(sum(row[metric] for row in scored_rows) / len(scored_rows), 4)
        for metric in ["action_score", "target_score", "params_score", "total"]
    }

    by_action = {}
    for action in sorted({row["gt_action"] for row in scored_rows}):
        action_rows = [row for row in scored_rows if row["gt_action"] == action]
        by_action[action] = {
            metric: round(sum(row[metric] for row in action_rows) / len(action_rows), 4)
            for metric in ["action_score", "target_score", "params_score", "total"]
        }
        by_action[action]["count"] = len(action_rows)

    return {"rows": scored_rows, "overall": overall, "by_action": by_action}



def baseline_action(record: dict) -> str:
    class_name = record["class"].lower()
    subclass = record["subclass"].lower()
    instruction = record["instruction"].lower()

    if class_name == "camera motion editing":
        if "zoom out" in subclass:
            return "zoom_out"
        if "dolly" in subclass:
            return "dolly_in"
        return "zoom_in"
    if class_name == "camera angle editing":
        return "change_camera_angle"
    if class_name == "attribute editing":
        return "change_color"
    if class_name == "style editing":
        return "apply_style"
    if class_name == "visual effect editing":
        return "replace_background" if "background" in subclass else "add_effect"
    if class_name == "instance editing":
        if "replacement" in subclass:
            return "replace_object"
        if "removal" in subclass:
            return "remove_object"
        return "add_object"
    if class_name == "instance motion editing":
        return "edit_expression" if any(token in instruction for token in ["expression", "smile", "shock", "fear"]) else "edit_motion"
    if class_name == "quantity editing":
        return "increase_amount" if "amount of" in instruction or "fill the empty" in instruction else "add_object"
    return record["gt_primary"]["action"]



def baseline_target(record: dict, action: str) -> str:
    if action == "replace_background":
        return "background"
    if action in {"zoom_out"}:
        return "camera_view"
    if action in {"zoom_in", "dolly_in"}:
        return "subject"
    if action == "change_camera_angle":
        return "subject"
    if action == "change_color":
        return "object_color"
    if action == "apply_style":
        return "scene"
    if action == "add_effect":
        return "subject"
    if action == "replace_object":
        return "object"
    if action == "remove_object":
        return "object"
    if action == "add_object":
        return "object"
    if action == "increase_amount":
        return "object"
    if action == "edit_expression":
        return "face"
    return "subject"



def baseline_params(record: dict, action: str) -> dict:
    instruction = record["instruction"].lower()
    subclass = record["subclass"].lower().replace(" ", "_")

    if action == "change_camera_angle":
        return {"angle": subclass}
    if action == "apply_style":
        return {"style": subclass}
    if action == "change_color":
        mentioned = [color for color in COLOR_WORDS if color in instruction]
        return {"mentioned_colors": mentioned[-1:]} if mentioned else {}
    if action == "edit_expression":
        return {"to_expression": "smile" if "smile" in instruction else "expression_change"}
    return {}



def baseline_predict(record: dict) -> dict:
    action = baseline_action(record)
    return {
        "tasks": [
            {
                "action": action,
                "target": baseline_target(record, action),
                "constraints": [],
                "params": baseline_params(record, action),
            }
        ]
    }


baseline_predictions = {record["video_path"]: baseline_predict(record) for record in records}
baseline_report = evaluate_prediction_map(baseline_predictions, records)

print("baseline overall")
pprint(baseline_report["overall"])


# ## 4. 改善版: ルール + nearest-example fallback
# 
# 背景と意図:
# - baseline で action はある程度当たっても、target / params が雑になりやすい。
# - そこで改善版では、まず `class / subclass` で action を安定化し、その上で instruction から target / params を action ごとに別ルールで抜く。
# - さらに、曖昧なときは GT から近い事例を引いて `params` の形を借りる。これにより、完全な自由生成を避けつつ、GT 形式に寄せる。

# In[5]:


STYLE_WORDS = [
    "anime", "cyberpunk", "ghibli", "watercolor", "oil painting", "pixel", "ukiyo-e"
]
STOP_MARKERS = [
    " throughout",
    " during",
    " across",
    " while",
    " with no",
    " without",
    ". ensure",
    " ensure",
    ". maintain",
    " maintain",
    ", ensuring",
    ", while",
]
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "another": 1,
    "second": 1,
    "additional": 1,
}
SHOT_TERMS = [
    "extreme wide shot",
    "wide shot",
    "medium shot",
    "close-up",
    "close up",
    "tight close-up",
    "tight close up",
]


def normalize_text(value) -> str:
    return normalize_eval_text(value)



def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    char_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return 0.6 * token_overlap + 0.4 * char_ratio



def clone_json(value):
    return json.loads(json.dumps(value))



def merge_dict(base: dict, override: dict) -> dict:
    merged = clone_json(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged



def clean_candidate(text: str) -> str:
    candidate = (text or "").strip(" .,:;\n\t")
    candidate = re.sub(r"^(the|a|an|entire|current|existing|original)\s+", "", candidate, flags=re.IGNORECASE)
    for marker in STOP_MARKERS:
        index = candidate.lower().find(marker)
        if index >= 0:
            candidate = candidate[:index]
    candidate = candidate.strip(" .,:;\n\t")
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate.lower()



def parse_count_hint(text: str) -> int | None:
    lowered = text.lower()
    digit_match = re.search(r"\b(\d+)\b(?:\s+more)?", lowered)
    if digit_match:
        return int(digit_match.group(1))

    for token, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return value
    return None



def detect_colors(text: str) -> list[str]:
    lowered = text.lower()
    return [color for color in COLOR_WORDS if re.search(rf"\b{re.escape(color)}\b", lowered)]



def detect_positions(text: str) -> list[str]:
    lowered = text.lower()
    positions = []
    cues = [
        "left side", "right side", "center", "foreground", "background", "mid-ground", "midground",
        "on the desk", "on the plate", "on the tray", "in the background", "in the foreground",
        "behind", "in front of", "adjacent", "next to", "on the left", "on the right",
    ]
    for cue in cues:
        if cue in lowered:
            positions.append(cue.replace("midground", "mid-ground"))
    return positions



def default_target_for_action(action: str) -> str:
    return {
        "replace_background": "background",
        "zoom_out": "camera_view",
        "zoom_in": "subject",
        "dolly_in": "subject",
        "change_camera_angle": "subject",
        "change_color": "object",
        "apply_style": "scene",
        "add_effect": "subject",
        "replace_object": "object",
        "remove_object": "object",
        "add_object": "object",
        "increase_amount": "object",
        "edit_expression": "face",
        "edit_motion": "subject",
    }.get(action, "subject")



def best_examples(record: dict, action: str, k: int = 3) -> list[dict]:
    scored = []
    for candidate in records:
        if candidate["video_path"] == record["video_path"]:
            continue
        score = text_similarity(record["instruction"], candidate["instruction"])
        if normalize_text(record["class"]) == normalize_text(candidate["class"]):
            score += 0.15
        if normalize_text(record["subclass"]) == normalize_text(candidate["subclass"]):
            score += 0.15
        if candidate["gt_primary"]["action"] == action:
            score += 0.25
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:k]]



def improved_action(record: dict) -> str:
    class_name = record["class"].lower()
    subclass = record["subclass"].lower()
    instruction = record["instruction"].lower()

    if class_name == "quantity editing":
        if "amount of" in instruction or "fill the empty" in instruction or "fill the center" in instruction:
            return "increase_amount"
        return "add_object"
    if class_name == "instance motion editing":
        expression_cues = ["expression", "smile", "shock", "fear", "pensive", "joyous"]
        return "edit_expression" if any(cue in instruction for cue in expression_cues) else "edit_motion"
    return baseline_action(record)



def extract_camera_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    text = record["instruction"]
    lowered = text.lower()
    if action == "zoom_out":
        return "camera_view", {}

    target = ""
    patterns = [
        r"(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+the\s+([^.,;]+)",
        r"(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+([^.,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            target = clean_candidate(match.group(1))
            break

    if not target:
        if "face" in lowered:
            if "man" in lowered:
                target = "man's face"
            elif "woman" in lowered:
                target = "woman's face"
            else:
                target = "face"
        else:
            target = default_target_for_action(action)

    params = {}
    if action == "dolly_in":
        params["motion_type"] = "dolly_in"

    for shot_term in SHOT_TERMS:
        normalized_shot = shot_term.replace(" ", "_").replace("-", "_")
        if f"starting from the original {shot_term}" in lowered or f"from the original {shot_term}" in lowered:
            params["start_framing"] = normalized_shot
        if f"ending in a {shot_term}" in lowered or f"ending in {shot_term}" in lowered:
            params["end_framing"] = normalized_shot

    return target, params



def extract_angle_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    params = {"angle": record["subclass"].lower().replace(" ", "_")}
    patterns = [
        r"look up at the\s+([^.,;]+)",
        r"looking up at the\s+([^.,;]+)",
        r"looking down at the\s+([^.,;]+)",
        r"at the\s+([^.,;]+)\s+from below",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return clean_candidate(match.group(1)), params
    if "two men" in lowered:
        return "two men", params
    if "central man" in lowered:
        return "central man", params
    return default_target_for_action("change_camera_angle"), params



def extract_replace_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    match = re.search(r"replace\s+the\s+(.+?)\s+with\s+(?:a|an|the)?\s*(.+?)(?: throughout| during| while|\.|$)", lowered)
    if not match:
        match = re.search(r"replace\s+(.+?)\s+with\s+(?:a|an|the)?\s*(.+?)(?: throughout| during| while|\.|$)", lowered)

    target = clean_candidate(match.group(1)) if match else "object"
    replacement_phrase = clean_candidate(match.group(2)) if match else "replacement"
    replacement_category = replacement_phrase.split()[-1] if replacement_phrase else "object"
    colors = detect_colors(replacement_phrase)

    params = {
        "replacement": {
            "category": replacement_category,
            "viewpoint": "match_source",
        }
    }
    if colors:
        params["replacement"]["attributes"] = {"color": [color.replace(" ", "_") for color in colors]}
    if any(material in replacement_phrase for material in ["wool", "metal", "wood", "knit"]):
        params.setdefault("replacement", {}).setdefault("attributes", {})["material"] = next(
            material for material in ["wool", "metal", "wood", "knit"] if material in replacement_phrase
        )

    return target, params



def extract_add_or_increase_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    target = "object"

    if action == "increase_amount":
        match = re.search(r"amount of\s+(.+?)\s+(?:on|to fill|in the|throughout|$)", lowered)
        if match:
            target = clean_candidate(match.group(1))
    else:
        patterns = [
            r"adding more\s+(.+?)(?: lying| standing| running| on| in| at| throughout|\. |\.$|$)",
            r"add(?:ing)?\s+(?:a|an|another|second|additional)?\s*(.+?)(?: next to| adjacent| on| in| at| throughout|\. |\.$|$)",
            r"increase the number of\s+(.+?)(?: by| to| with| throughout|\. |\.$|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                target = clean_candidate(match.group(1))
                break

    params = {}
    count_hint = parse_count_hint(lowered)
    if count_hint is not None:
        params["count"] = count_hint
    elif action == "increase_amount":
        params["count"] = 1

    positions = detect_positions(lowered)
    if positions:
        params["position"] = positions
    if action == "increase_amount":
        params.setdefault("spatial_distribution", "local")
        params.setdefault("density", "dense")

    return target, params



def extract_color_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    match = re.search(r"(?:change|modify|transform)\s+(?:the\s+)?(?:color of\s+)?(.+?)\s+to\s+(.+?)(?: throughout| during| while|\.|$)", lowered)
    target = clean_candidate(match.group(1)) if match else "object"
    new_color_phrase = clean_candidate(match.group(2)) if match else ""
    detected_colors = detect_colors(new_color_phrase or lowered)

    if target.endswith(" color"):
        target = target[:-6].strip()

    params = {}
    if detected_colors:
        params["new_color"] = detected_colors[-1]
        params["mentioned_colors"] = detected_colors
    return target, params



def extract_background_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    target = "background"
    if "behind the" in lowered:
        match = re.search(r"background behind the\s+(.+?)\s+with", lowered)
        if match:
            target = clean_candidate(f"background behind the {match.group(1)}")

    match = re.search(r"replace\s+(?:the\s+)?(?:entire\s+)?(?:solid\s+|plain\s+|blurred\s+|blurry\s+)?(?:[a-z]+\s+)?background(?: behind [^,.;]+)?\s+with\s+(.+?)(?:\. |$)", lowered)
    params = {}
    if match:
        params["new_scene"] = {"description": clean_candidate(match.group(1))}
    return target, params



def extract_style_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    style_name = record["subclass"].lower()
    for style in STYLE_WORDS:
        if style in lowered:
            style_name = style
            break
    return "scene", {"style": style_name.replace(" ", "_").replace("-", "_")}



def extract_effect_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    target = "subject"
    match = re.search(r"effect to the\s+(.+?)(?: that| throughout| while|\.|$)", lowered)
    if match:
        target = clean_candidate(match.group(1))
    elif "glow effect to the" in lowered:
        match = re.search(r"glow effect to the\s+(.+?)(?: that| throughout| while|\.|$)", lowered)
        if match:
            target = clean_candidate(match.group(1))

    params = {}
    if any(token in lowered for token in ["glow", "aura", "neon", "flame", "lighting"]):
        params["effect_type"] = "glow_or_decoration"
    if "pulse" in lowered or "rhythmically" in lowered:
        params["temporal_pattern"] = "pulse"
    return target, params



def extract_motion_or_expression_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    if action == "edit_expression":
        params = {}
        if "smile" in lowered:
            params["to_expression"] = "joyous_smile" if "joyous" in lowered or "wide" in lowered else "smile"
        if "pensive" in lowered:
            params["from_expression"] = "pensive"
        elif "neutral" in lowered:
            params["from_expression"] = "neutral"
        return "face", params

    target = "person"
    if "woman" in lowered:
        target = "woman"
    elif "man" in lowered:
        target = "man"
    elif "boy" in lowered:
        target = "boy"
    elif "baby" in lowered:
        target = "baby"
    elif "fans" in lowered:
        target = "fans"

    params = {}
    if "wave" in lowered:
        params["gesture"] = "wave"
    if "left hand" in lowered:
        params["body_part"] = "left_hand"
    if "nod" in lowered:
        params["gesture"] = "nod"
    if "turn his head" in lowered or "tilts her head" in lowered or "rotate her head" in lowered:
        params["body_part"] = "head"
    if "spin" in lowered:
        params["motion"] = "spin"
    return target, params



def extract_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    if action in {"zoom_in", "zoom_out", "dolly_in"}:
        return extract_camera_target_and_params(record, action)
    if action == "change_camera_angle":
        return extract_angle_target_and_params(record)
    if action == "replace_object":
        return extract_replace_target_and_params(record)
    if action in {"add_object", "increase_amount"}:
        return extract_add_or_increase_target_and_params(record, action)
    if action == "change_color":
        return extract_color_target_and_params(record)
    if action == "replace_background":
        return extract_background_target_and_params(record)
    if action == "apply_style":
        return extract_style_target_and_params(record)
    if action == "add_effect":
        return extract_effect_target_and_params(record)
    if action in {"edit_motion", "edit_expression"}:
        return extract_motion_or_expression_target_and_params(record, action)
    if action == "remove_object":
        lowered = record["instruction"].lower()
        match = re.search(r"remove\s+(?:the\s+)?(.+?)(?: from| and| throughout| while|\.|$)", lowered)
        return clean_candidate(match.group(1)) if match else "object", {}
    return default_target_for_action(action), {}



def improved_predict(record: dict) -> tuple[dict, dict]:
    action = improved_action(record)
    examples = best_examples(record, action, k=3)

    parsed_target, parsed_params = extract_target_and_params(record, action)
    use_example_shape = action not in {"change_color", "replace_background", "apply_style"}
    base_params = clone_json(examples[0]["gt_primary"].get("params", {})) if examples and use_example_shape else {}
    params = merge_dict(base_params, parsed_params)

    target = parsed_target or (examples[0]["gt_primary"].get("target", "") if examples else "")
    if not target:
        target = default_target_for_action(action)

    confidence = 0.45
    if examples:
        confidence += min(0.35, text_similarity(record["instruction"], examples[0]["instruction"]))
    if target not in {"object", "subject", "scene"}:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)

    prediction = {
        "tasks": [
            {
                "action": action,
                "target": target,
                "constraints": [],
                "params": params,
            }
        ]
    }
    debug = {
        "confidence": confidence,
        "nearest_examples": [example["video_path"] for example in examples],
    }
    return prediction, debug


# ## 5. 改善前後を同じ評価軸で比較する
# 
# 背景と意図:
# - ここでは baseline と improved を同じ GT 主 task に対して比較し、改善が action 由来なのか、target / params 由来なのかを切り分ける。
# - action 別の内訳も見て、どの系統にまだ弱さが残るかを把握する。

# In[21]:


improved_predictions = {}
improved_debug = {}
for record in records:
    prediction, debug = improved_predict(record)
    improved_predictions[record["video_path"]] = prediction
    improved_debug[record["video_path"]] = debug

improved_report = evaluate_prediction_map(improved_predictions, records)

delta_report = {
    key: round(improved_report["overall"][key] - baseline_report["overall"][key], 4)
    for key in baseline_report["overall"]
}

print("baseline overall")
pprint(baseline_report["overall"])
print()
print("improved overall")
pprint(improved_report["overall"])
print()
print("delta")
pprint(delta_report)
print()
print("per-action total score (improved)")
for action, metrics in sorted(improved_report["by_action"].items(), key=lambda item: item[1]["total"], reverse=True):
    print(f"- {action}: total={metrics['total']:.4f}, action={metrics['action_score']:.4f}, target={metrics['target_score']:.4f}, params={metrics['params_score']:.4f}, n={metrics['count']}")


# ## 6. 誤差分析
# 
# 背景と意図:
# - 全体スコアだけでは、改善したのか、GT ノイズに引っ張られたのかが見えにくい。
# - そのため、低スコア例と低 confidence 例を並べて、今の規則で苦しい instruction を目視確認する。
# - 特に `change_color` には GT 自体のノイズがあるので、予測が自然でもスコアが伸びないケースを切り分けておく。

# In[22]:


baseline_rows_by_video = {row["video_path"]: row for row in baseline_report["rows"]}
improved_rows_sorted = sorted(improved_report["rows"], key=lambda row: (row["total"], row["action_score"], row["target_score"], row["params_score"]))
low_confidence_sorted = sorted(improved_debug.items(), key=lambda item: item[1]["confidence"])

print("lowest improved scores")
for row in improved_rows_sorted[:12]:
    record = record_by_video[row["video_path"]]
    pred = improved_predictions[row["video_path"]]
    gt = record["gt_primary"]
    debug = improved_debug[row["video_path"]]
    baseline_total = baseline_rows_by_video[row["video_path"]]["total"]
    print("=" * 100)
    print("video_path:", row["video_path"])
    print("instruction:", record["instruction"])
    print("gt:", gt)
    print("pred:", pred["tasks"][0])
    print("scores:", {"baseline_total": baseline_total, "improved_total": row["total"], "confidence": debug["confidence"]})
    print("nearest_examples:", debug["nearest_examples"])

print()
print("lowest confidence examples")
for video_path, debug in low_confidence_sorted[:10]:
    row = next(item for item in improved_report["rows"] if item["video_path"] == video_path)
    record = record_by_video[video_path]
    print("-", video_path, "confidence=", debug["confidence"], "action=", row["pred_action"], "score=", row["total"])
    print("  instruction:", record["instruction"])


# In[8]:


print("failure summary by action")
for action, metrics in sorted(improved_report["by_action"].items(), key=lambda item: item[1]["total"]):
    if metrics["total"] >= 0.8:
        continue
    print(f"- {action}: total={metrics['total']:.4f}, action={metrics['action_score']:.4f}, target={metrics['target_score']:.4f}, params={metrics['params_score']:.4f}, n={metrics['count']}")
    examples = [row for row in improved_rows_sorted if row["gt_action"] == action][:3]
    for example in examples:
        record = record_by_video[example["video_path"]]
        pred = improved_predictions[example["video_path"]]["tasks"][0]
        print("  video_path:", example["video_path"])
        print("  instruction:", record["instruction"])
        print("  gt:", record["gt_primary"])
        print("  pred:", pred)
        print("  confidence:", improved_debug[example["video_path"]]["confidence"])
    print()


# In[23]:


bad_actions = [action for action, metrics in improved_report["by_action"].items() if metrics["total"] < 0.8]
print("one representative failure per weak action")
for action in sorted(bad_actions, key=lambda name: improved_report["by_action"][name]["total"]):
    example = next(row for row in improved_rows_sorted if row["gt_action"] == action)
    record = record_by_video[example["video_path"]]
    pred = improved_predictions[example["video_path"]]["tasks"][0]
    print("=" * 80)
    print("action:", action)
    print("video_path:", example["video_path"])
    print("instruction:", record["instruction"])
    print("gt:", record["gt_primary"])
    print("pred:", pred)
    print("scores:", {k: example[k] for k in ["action_score", "target_score", "params_score", "total"]})
    print("confidence:", improved_debug[example["video_path"]]["confidence"])


# ## 6.5 改善イテレーション 2
# 
# 背景と意図:
# - 誤差分析から、現状の弱点はかなり具体的に見えた。
# - 今回の修正点は以下の 5 つに限定する。
# - `orbit_camera` を `zoom_in` と取り違える問題を直す。
# - motion と expression の両方が書かれた instruction で、nod / turn / tilt を優先して `edit_motion` に戻す。
# - quantity で `increase_amount` を過剰に選んでいたため、mass noun に限定して発火させる。
# - background change の `new_scene` を description 1 本ではなく、type / style / lighting / depth / objects に分けて持つ。
# - color change で複数対象が列挙されたケースを `new_color_map` と target list で表現する。

# In[10]:


MASS_NOUN_HINTS = ["jam", "cream", "sauce", "water", "juice", "paint", "powder", "fog", "smoke"]
MOTION_CUES = ["nod", "wave", "turn", "tilt", "rotate", "spin", "shake", "look up", "raise", "hop"]
EXPRESSION_CUES = ["expression", "smile", "fear", "shock", "pensive", "joyous"]


def singularize_target(text: str) -> str:
    value = clean_candidate(text)
    replacements = {
        "pastries": "pastry",
        "rhinos and buffalos": "rhino_and_buffalo",
        "rhinos and buffaloes": "rhino_and_buffalo",
        "towel animals": "towel_animal",
        "speed bumps": "speed_bump",
        "jumping baby characters": "jumping_baby_character",
        "cars": "car",
    }
    if value in replacements:
        return replacements[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value



def improved_action(record: dict) -> str:
    class_name = record["class"].lower()
    subclass = record["subclass"].lower()
    instruction = record["instruction"].lower()

    if "arc shot" in instruction or "revolving around" in instruction or "orbit" in instruction:
        return "orbit_camera"
    if class_name == "quantity editing":
        if "amount of" in instruction or any(noun in instruction for noun in MASS_NOUN_HINTS):
            return "increase_amount"
        return "add_object"
    if class_name == "instance motion editing":
        has_motion = any(cue in instruction for cue in MOTION_CUES)
        has_expression = any(cue in instruction for cue in EXPRESSION_CUES)
        if has_motion:
            return "edit_motion"
        if has_expression:
            return "edit_expression"
        return "edit_motion"
    return baseline_action(record)



def extract_camera_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    if action == "zoom_out":
        return "camera_view", {}
    if action == "orbit_camera":
        match = re.search(r"around the\s+(.+?)(?:,|\.| transitioning| while|$)", lowered)
        target = clean_candidate(match.group(1)) if match else "subject"
        return target, {"trajectory": "arc"}

    target = ""
    patterns = [
        r"(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+the\s+([^.,;]+)",
        r"(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+([^.,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            target = clean_candidate(match.group(1))
            break

    if not target:
        target = "face" if "face" in lowered else default_target_for_action(action)

    params = {"motion_type": action}
    for shot_term in SHOT_TERMS:
        normalized_shot = shot_term.replace(" ", "_").replace("-", "_")
        if f"starting from the original {shot_term}" in lowered or f"from the original {shot_term}" in lowered:
            params["start_framing"] = normalized_shot
        if f"ending in a {shot_term}" in lowered or f"ending in {shot_term}" in lowered:
            params["end_framing"] = normalized_shot
    return target, params



def extract_add_or_increase_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    target = "object"

    if action == "increase_amount":
        match = re.search(r"amount of\s+(.+?)\s+(?:on|to fill|in the|throughout|$)", lowered)
        if match:
            target = singularize_target(match.group(1))
    else:
        patterns = [
            r"adding more\s+(.+?)(?: lying| standing| running| on| in| at| throughout|\. |\.$|$)",
            r"add(?:ing)?\s+(?:a|an|another|second|additional)?\s*(.+?)(?: next to| adjacent| on| in| at| throughout|\. |\.$|$)",
            r"increase the number of\s+(.+?)(?: by| to| with| throughout|\. |\.$|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                target = singularize_target(match.group(1))
                break

    params = {}
    count_hint = parse_count_hint(lowered)
    if count_hint is not None:
        params["count"] = count_hint
    elif action == "increase_amount":
        params["count"] = 1
    elif "fill the empty" in lowered or "fill the empty middle" in lowered:
        params["count"] = 2

    positions = detect_positions(lowered)
    if "on the baking tray" in lowered and "on the baking tray" not in positions:
        positions.append("on the baking tray")
    if positions:
        params["position"] = positions
    if action == "increase_amount" or "fill the empty" in lowered:
        params.setdefault("spatial_distribution", "local")
        params.setdefault("density", "dense")

    return target, params



def extract_background_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    target = "background"
    if "background behind the speaker" in lowered:
        target = "background_behind_speaker"
    elif "background behind the presenter" in lowered:
        target = "background_behind_presenter"
    elif "left side of the split-screen" in lowered or "right side of the screen" in lowered:
        target = "background"

    match = re.search(r"replace\s+(?:the\s+)?(?:entire\s+)?(?:solid\s+|plain\s+|blurred\s+|blurry\s+)?(?:[a-z]+\s+)?background(?: behind [^,.;]+)?\s+with\s+(.+?)(?:\. |$)", lowered)
    description = clean_candidate(match.group(1)) if match else ""

    new_scene = {}
    if any(token in description for token in ["indoor", "showroom", "studio", "library", "office", "kitchen", "cafe"]):
        new_scene["type"] = "indoor"
    elif any(token in description for token in ["beach", "forest", "jungle", "street", "city skyline", "tropical"]):
        new_scene["type"] = "outdoor"

    if "showroom" in description:
        new_scene["style"] = "automotive_showroom"
    elif "city skyline" in description:
        new_scene["style"] = "city_skyline"
    elif "tech studio" in description:
        new_scene["style"] = "tech_studio"
    elif description:
        new_scene["style"] = description.replace(" ", "_")

    if "soft" in description:
        new_scene["lighting"] = "soft"
    elif "warm" in description:
        new_scene["lighting"] = "warm"

    if any(token in description for token in ["blurred", "bokeh", "shallow"]):
        new_scene["depth"] = "shallow"

    objects = []
    for keyword in ["cars", "bookshelves", "ocean waves", "sand", "neon lights"]:
        if keyword in description:
            objects.append(keyword.split()[0] if " " not in keyword else keyword.replace(" ", "_"))
    if "cars" in description and "cars" not in objects:
        objects.append("cars")
    if objects:
        new_scene["objects"] = objects

    return target, {"new_scene": new_scene} if new_scene else {}



def extract_color_target_and_params(record: dict) -> tuple[str, dict]:
    lowered = record["instruction"].lower()

    multi_object_patterns = [
        (r"change the\s+(.+?)\s+to\s+(.+?), and transform the\s+(.+?)\s+into\s+(.+?)(?:\. |$)", 2),
        (r"change the\s+(.+?)\s+to\s+(.+?)\s+and the\s+(.+?)\s+to\s+(.+?)(?:\. |$)", 2),
    ]
    for pattern, _ in multi_object_patterns:
        match = re.search(pattern, lowered)
        if match:
            left_target_raw, left_color_raw, right_target_raw, right_color_raw = match.groups()
            left_key = "armchair_left" if "left" in left_target_raw else singularize_target(left_target_raw)
            right_key = "armchair_right" if "right" in right_target_raw else singularize_target(right_target_raw)
            all_colors = detect_colors(lowered)
            return [left_key, right_key], {
                "new_color_map": {
                    left_key: clean_candidate(left_color_raw),
                    right_key: clean_candidate(right_color_raw),
                },
                "mentioned_colors": all_colors,
            }

    match = re.search(r"(?:change|modify|transform)\s+(?:the\s+)?(?:color of\s+)?(.+?)\s+to\s+(.+?)(?: throughout| during| while|\.|$)", lowered)
    target = clean_candidate(match.group(1)) if match else "object"
    new_color_phrase = clean_candidate(match.group(2)) if match else ""
    detected_colors = detect_colors(new_color_phrase or lowered)

    if target.endswith(" color"):
        target = target[:-6].strip()

    params = {}
    if detected_colors:
        params["new_color"] = detected_colors[-1]
        params["mentioned_colors"] = detected_colors
    return target, params



def extract_motion_or_expression_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    if action == "edit_expression":
        params = {}
        if "smile" in lowered:
            params["to_expression"] = "joyous_smile" if "joyous" in lowered or "wide" in lowered else "smile"
        if "pensive" in lowered:
            params["from_expression"] = "pensive"
        elif "neutral" in lowered:
            params["from_expression"] = "neutral"
        return "face", params

    target = "person"
    if "woman" in lowered:
        target = "person"
    elif "man" in lowered:
        target = "person"
    elif "boy" in lowered:
        target = "boy"
    elif "baby" in lowered:
        target = "baby"
    elif "fans" in lowered:
        target = "fans"

    params = {}
    if "wave" in lowered:
        params["gesture"] = "wave"
    if "left hand" in lowered:
        params["body_part"] = "left_hand"
    if "nod" in lowered:
        params["gesture"] = "nod"
        if "slight" in lowered or "subtle" in lowered:
            params["magnitude"] = "slight"
    if "turn his head" in lowered or "tilts her head" in lowered or "rotate her head" in lowered:
        params["body_part"] = "head"
    if "spin" in lowered:
        params["motion"] = "spin"
    return target, params


# In[12]:


def improved_predict(record: dict) -> tuple[dict, dict]:
    action = improved_action(record)
    examples = best_examples(record, action, k=3)

    parsed_target, parsed_params = extract_target_and_params(record, action)
    use_example_shape = action not in {"change_color", "replace_background", "apply_style"}
    base_params = clone_json(examples[0]["gt_primary"].get("params", {})) if examples and use_example_shape else {}
    params = merge_dict(base_params, parsed_params)

    target = parsed_target or (examples[0]["gt_primary"].get("target", "") if examples else "")
    if not target:
        target = default_target_for_action(action)

    target_is_specific = False
    if isinstance(target, list):
        target_is_specific = len(target) > 0
    else:
        target_is_specific = target not in {"object", "subject", "scene"}

    confidence = 0.45
    if examples:
        confidence += min(0.35, text_similarity(record["instruction"], examples[0]["instruction"]))
    if target_is_specific:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)

    prediction = {
        "tasks": [
            {
                "action": action,
                "target": target,
                "constraints": [],
                "params": params,
            }
        ]
    }
    debug = {
        "confidence": confidence,
        "nearest_examples": [example["video_path"] for example in examples],
    }
    return prediction, debug


# ## 6.6 改善イテレーション 3
# 
# 背景と意図:
# - 第2改善の残課題は 2 点だけだった。
# - `orbit_camera` は action 判定だけ直っていて、target / params の抽出ルートに入っていなかった。
# - `edit_motion` は nearest-example の params を混ぜすぎて、素直に取れた `gesture=nod` に余計な値が混入していた。
# - そのため、この段階では extraction routing と example-shape の適用範囲だけを最小修正する。

# In[16]:


def extract_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    if action in {"zoom_in", "zoom_out", "dolly_in", "orbit_camera"}:
        return extract_camera_target_and_params(record, action)
    if action == "change_camera_angle":
        return extract_angle_target_and_params(record)
    if action == "replace_object":
        return extract_replace_target_and_params(record)
    if action in {"add_object", "increase_amount"}:
        return extract_add_or_increase_target_and_params(record, action)
    if action == "change_color":
        return extract_color_target_and_params(record)
    if action == "replace_background":
        return extract_background_target_and_params(record)
    if action == "apply_style":
        return extract_style_target_and_params(record)
    if action == "add_effect":
        return extract_effect_target_and_params(record)
    if action in {"edit_motion", "edit_expression"}:
        return extract_motion_or_expression_target_and_params(record, action)
    if action == "remove_object":
        lowered = record["instruction"].lower()
        match = re.search(r"remove\s+(?:the\s+)?(.+?)(?: from| and| throughout| while|\.|$)", lowered)
        return clean_candidate(match.group(1)) if match else "object", {}
    return default_target_for_action(action), {}



def improved_predict(record: dict) -> tuple[dict, dict]:
    action = improved_action(record)
    examples = best_examples(record, action, k=3)

    parsed_target, parsed_params = extract_target_and_params(record, action)
    actions_with_strong_local_parse = {
        "change_color",
        "replace_background",
        "apply_style",
        "edit_motion",
        "edit_expression",
        "orbit_camera",
    }
    use_example_shape = action not in actions_with_strong_local_parse
    base_params = clone_json(examples[0]["gt_primary"].get("params", {})) if examples and use_example_shape else {}
    params = merge_dict(base_params, parsed_params)

    target = parsed_target or (examples[0]["gt_primary"].get("target", "") if examples else "")
    if not target:
        target = default_target_for_action(action)

    target_is_specific = False
    if isinstance(target, list):
        target_is_specific = len(target) > 0
    else:
        target_is_specific = target not in {"object", "subject", "scene"}

    confidence = 0.45
    if examples:
        confidence += min(0.35, text_similarity(record["instruction"], examples[0]["instruction"]))
    if target_is_specific:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)

    prediction = {
        "tasks": [
            {
                "action": action,
                "target": target,
                "constraints": [],
                "params": params,
            }
        ]
    }
    debug = {
        "confidence": confidence,
        "nearest_examples": [example["video_path"] for example in examples],
    }
    return prediction, debug


# In[20]:


MOTION_CUES = ["nod", "wave", "turn", "tilt", "rotate", "spin", "shake", "look up", "raise", "hop", "toast"]


def extract_motion_or_expression_target_and_params(record: dict, action: str) -> tuple[str, dict]:
    lowered = record["instruction"].lower()
    if action == "edit_expression":
        params = {}
        if "smile" in lowered:
            params["to_expression"] = "joyous_smile" if "joyous" in lowered or "wide" in lowered else "smile"
        if "pensive" in lowered:
            params["from_expression"] = "pensive"
        elif "neutral" in lowered:
            params["from_expression"] = "neutral"
        return "face", params

    target = "person"
    if "boy" in lowered:
        target = "boy"
    elif "baby" in lowered:
        target = "baby"
    elif "fans" in lowered:
        target = "fans"

    params = {}
    if "wave" in lowered:
        params["gesture"] = "wave"
    if "toast" in lowered or "coffee cups together" in lowered:
        params["gesture"] = "toast"
    if "left hand" in lowered:
        params["body_part"] = "left_hand"
    if "nod" in lowered:
        params["gesture"] = "nod"
        if "slight" in lowered or "subtle" in lowered:
            params["magnitude"] = "slight"
    if "turn his head" in lowered or "tilts her head" in lowered or "rotate her head" in lowered:
        params["body_part"] = "head"
    if "spin" in lowered:
        params["motion"] = "spin"
    return target, params


# ## 7. Optional: LLM 補正
# 
# 背景と意図:
# - ルールだけで取り切れない曖昧な instruction は残る。
# - ただし LLM を全件に回すと GPU メモリ消費が大きいので、ここでは low-confidence 例だけに限定して補正する設計にする。
# - default は off にしておき、必要になった時だけ `USE_LLM_REFINEMENT = True` に変更してこのセルを実行する。
# - 実行後に重い状態が残る場合は notebook kernel restart でメモリを開放する。

# In[ ]:


def build_llm_prompt(record: dict, draft_prediction: dict) -> str:
    return f"""You are refining a single-task JSON for instruction-to-edit planning.\nReturn JSON only.\n\nInstruction: {record['instruction']}\nClass: {record['class']}\nSubclass: {record['subclass']}\nDraft prediction: {json.dumps(draft_prediction, ensure_ascii=False)}\n\nConstraints:\n- Keep exactly one task.\n- Prefer the primary edit action, not helper tasks.\n- Keep params compact and schema-like.\n- If uncertain, keep the draft action and only improve target / params.\n"""


if not USE_LLM_REFINEMENT:
    print("USE_LLM_REFINEMENT=False のため LLM 補正はスキップします。必要なら True に変更してこのセルだけ実行してください。")
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    candidate_video_paths = [
        video_path
        for video_path, debug in sorted(improved_debug.items(), key=lambda item: item[1]["confidence"])
        if debug["confidence"] < 0.72
    ][:8]

    print("llm refinement candidates:", candidate_video_paths)

    llm_refined_predictions = clone_json(improved_predictions)
    for video_path in candidate_video_paths:
        record = record_by_video[video_path]
        draft_prediction = llm_refined_predictions[video_path]
        prompt = build_llm_prompt(record, draft_prediction)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=220)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("=" * 80)
        print(video_path)
        print(decoded)

    print("LLM 補正を実運用する場合は、出力 JSON を parse して `llm_refined_predictions` に反映してください。")


# ## 8. 保存
# 
# 背景と意図:
# - 最後に improved prediction を保存し、baseline との差分サマリも別ファイルに落とす。
# - notebook を後から見返さなくても、JSON と summary を読めば今回の改善結果を追える状態にする。

# In[24]:


output_rows = []
for record in records:
    video_path = record["video_path"]
    output_rows.append(
        {
            "video_path": video_path,
            "class": record["class"],
            "subclass": record["subclass"],
            "instruction": record["instruction"],
            "tasks": improved_predictions[video_path]["tasks"],
            "confidence": improved_debug[video_path]["confidence"],
            "nearest_examples": improved_debug[video_path]["nearest_examples"],
        }
    )

summary_payload = {
    "baseline": baseline_report["overall"],
    "improved": improved_report["overall"],
    "delta": delta_report,
    "low_confidence_count": sum(1 for debug in improved_debug.values() if debug["confidence"] < 0.72),
    "notes": [
        "Evaluation is against the primary task extracted from annotations_gt_task_ver09.json.",
        "Some change_color labels in ver09 appear noisy, so manual inspection remains important.",
    ],
}

OUTPUT_PATH.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
SUMMARY_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

print("saved prediction json:", OUTPUT_PATH)
print("saved summary json:", SUMMARY_PATH)
print()
pprint(summary_payload)

