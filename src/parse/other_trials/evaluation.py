from __future__ import annotations

import re


def normalize_text(value) -> str:
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
        items[prefix] = normalize_text(value)
    return items


def score_action(pred: dict, gt: dict) -> float:
    return float(safe_get(pred, ["tasks", 0, "action"], "") == safe_get(gt, ["tasks", 0, "action"], ""))


def score_target(pred: dict, gt: dict) -> float:
    pred_target = safe_get(pred, ["tasks", 0, "target"], "")
    gt_target = safe_get(gt, ["tasks", 0, "target"], "")

    if isinstance(pred_target, list) and isinstance(gt_target, list):
        pred_join = " ".join(normalize_text(x) for x in pred_target)
        gt_join = " ".join(normalize_text(x) for x in gt_target)
        return float(pred_join in gt_join or gt_join in pred_join)

    pred_text = normalize_text(pred_target)
    gt_text = normalize_text(gt_target)
    if not pred_text or not gt_text:
        return 0.0
    return float(pred_text in gt_text or gt_text in pred_text)


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
    return 0.5 * score_action(pred, gt) + 0.2 * score_target(pred, gt) + 0.3 * score_params(pred, gt)


def evaluate_prediction_map(predictions: dict[str, dict], rows: list[dict]) -> dict:
    scored_rows = []
    for row in rows:
        prediction_key = row.get("prediction_key", row["video_path"])
        gt = {"tasks": [row["gt_primary"]]}
        pred = predictions[prediction_key]
        scored_rows.append(
            {
                "prediction_key": prediction_key,
                "video_path": row["video_path"],
                "variant": row.get("variant", "base"),
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
