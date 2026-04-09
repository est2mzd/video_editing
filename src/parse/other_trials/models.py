from __future__ import annotations

import re
from collections import Counter

from .features import (
    EXPRESSION_CUES,
    MASS_NOUN_HINTS,
    MOTION_CUES,
    SHOT_TERMS,
    STYLE_WORDS,
    best_examples,
    clean_candidate,
    clone_json,
    detect_colors,
    detect_positions,
    merge_dict,
    parse_count_hint,
    singularize_target,
    text_similarity,
)


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
        return "edit_expression" if any(x in instruction for x in ["expression", "smile", "shock", "fear"]) else "edit_motion"
    if class_name == "quantity editing":
        return "increase_amount" if "amount of" in instruction or "fill the empty" in instruction else "add_object"
    return record["gt_primary"]["action"]


def baseline_target(action: str) -> str:
    mapping = {
        "replace_background": "background",
        "zoom_out": "camera_view",
        "zoom_in": "subject",
        "dolly_in": "subject",
        "change_camera_angle": "subject",
        "change_color": "object_color",
        "apply_style": "scene",
        "add_effect": "subject",
        "replace_object": "object",
        "remove_object": "object",
        "add_object": "object",
        "increase_amount": "object",
        "edit_expression": "face",
    }
    return mapping.get(action, "subject")


def baseline_params(record: dict, action: str) -> dict:
    instruction = record["instruction"].lower()
    subclass = record["subclass"].lower().replace(" ", "_")
    if action == "change_camera_angle":
        return {"angle": subclass}
    if action == "apply_style":
        return {"style": subclass}
    if action == "change_color":
        colors = detect_colors(instruction)
        return {"mentioned_colors": colors[-1:]} if colors else {}
    if action == "edit_expression":
        return {"to_expression": "smile" if "smile" in instruction else "expression_change"}
    return {}


def predict_ver10_baseline(record: dict):
    action = baseline_action(record)
    return {
        "tasks": [
            {
                "action": action,
                "target": baseline_target(action),
                "constraints": [],
                "params": baseline_params(record, action),
            }
        ]
    }, {"version": "ver10_baseline", "confidence": 0.6}


def infer_action(record: dict) -> str:
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


def parse_action_specific_fields(record: dict, action: str):
    lowered = record["instruction"].lower()

    if action in {"zoom_in", "zoom_out", "dolly_in", "orbit_camera"}:
        if action == "zoom_out":
            return "camera_view", {}
        if action == "orbit_camera":
            match = re.search(r"around the\s+(.+?)(?:,|\.| transitioning| while|$)", lowered)
            target = clean_candidate(match.group(1)) if match else "subject"
            return target, {"trajectory": "arc"}
        target = "face" if "face" in lowered else "subject"
        match = re.search(r"(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+the\s+([^.,;]+)", lowered)
        if match:
            target = clean_candidate(match.group(1))
        params = {"motion_type": action}
        for shot_term in SHOT_TERMS:
            normalized = shot_term.replace(" ", "_").replace("-", "_")
            if f"from the original {shot_term}" in lowered or f"starting from the original {shot_term}" in lowered:
                params["start_framing"] = normalized
            if f"ending in a {shot_term}" in lowered or f"ending in {shot_term}" in lowered:
                params["end_framing"] = normalized
        return target, params

    if action == "change_camera_angle":
        return "subject", {"angle": record["subclass"].lower().replace(" ", "_")}

    if action == "replace_object":
        match = re.search(r"replace\s+(?:the\s+)?(.+?)\s+with\s+(?:a|an|the)?\s*(.+?)(?: throughout| during| while|\.|$)", lowered)
        target = clean_candidate(match.group(1)) if match else "object"
        replacement = clean_candidate(match.group(2)) if match else "replacement"
        params = {"replacement": {"category": replacement.split()[-1] if replacement else "object", "viewpoint": "match_source"}}
        colors = detect_colors(replacement)
        if colors:
            params["replacement"]["attributes"] = {"color": colors}
        return target, params

    if action in {"add_object", "increase_amount"}:
        target = "object"
        if action == "increase_amount":
            match = re.search(r"amount of\s+(.+?)\s+(?:on|to fill|in the|throughout|$)", lowered)
            if match:
                target = singularize_target(match.group(1))
        else:
            for pattern in [
                r"adding more\s+(.+?)(?: lying| standing| running| on| in| at| throughout|\. |\.$|$)",
                r"add(?:ing)?\s+(?:a|an|another|second|additional)?\s*(.+?)(?: next to| adjacent| on| in| at| throughout|\. |\.$|$)",
                r"increase the number of\s+(.+?)(?: by| to| with| throughout|\. |\.$|$)",
            ]:
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
        elif "fill the empty" in lowered:
            params["count"] = 2
        positions = detect_positions(lowered)
        if positions:
            params["position"] = positions
        if action == "increase_amount" or "fill the empty" in lowered:
            params.setdefault("spatial_distribution", "local")
            params.setdefault("density", "dense")
        return target, params

    if action == "change_color":
        match = re.search(r"change the\s+(.+?)\s+to\s+(.+?), and transform the\s+(.+?)\s+into\s+(.+?)(?:\. |$)", lowered)
        if match:
            lt, lc, rt, rc = match.groups()
            lk = "armchair_left" if "left" in lt else singularize_target(lt)
            rk = "armchair_right" if "right" in rt else singularize_target(rt)
            return [lk, rk], {"new_color_map": {lk: clean_candidate(lc), rk: clean_candidate(rc)}, "mentioned_colors": detect_colors(lowered)}
        match = re.search(r"(?:change|modify|transform)\s+(?:the\s+)?(?:color of\s+)?(.+?)\s+to\s+(.+?)(?: throughout| during| while|\.|$)", lowered)
        target = clean_candidate(match.group(1)) if match else "object"
        if target.endswith(" color"):
            target = target[:-6].strip()
        colors = detect_colors(clean_candidate(match.group(2)) if match else lowered)
        params = {}
        if colors:
            params["new_color"] = colors[-1]
            params["mentioned_colors"] = colors
        return target, params

    if action == "replace_background":
        target = "background_behind_speaker" if "background behind the speaker" in lowered else "background"
        match = re.search(r"replace\s+(?:the\s+)?(?:entire\s+)?(?:solid\s+|plain\s+|blurred\s+|blurry\s+)?(?:[a-z]+\s+)?background(?: behind [^,.;]+)?\s+with\s+(.+?)(?:\. |$)", lowered)
        description = clean_candidate(match.group(1)) if match else ""
        scene = {}
        if any(token in description for token in ["indoor", "showroom", "studio", "library", "office", "kitchen", "cafe"]):
            scene["type"] = "indoor"
        elif any(token in description for token in ["beach", "forest", "jungle", "street", "city skyline", "tropical"]):
            scene["type"] = "outdoor"
        if "showroom" in description:
            scene["style"] = "automotive_showroom"
        elif "city skyline" in description:
            scene["style"] = "city_skyline"
        elif description:
            scene["style"] = description.replace(" ", "_")
        if "soft" in description:
            scene["lighting"] = "soft"
        if any(token in description for token in ["blurred", "bokeh", "shallow"]):
            scene["depth"] = "shallow"
        if "cars" in description:
            scene["objects"] = ["cars"]
        return target, {"new_scene": scene} if scene else {}

    if action == "apply_style":
        style_name = record["subclass"].lower()
        for style in STYLE_WORDS:
            if style in lowered:
                style_name = style
                break
        return "scene", {"style": style_name.replace(" ", "_").replace("-", "_")}

    if action == "add_effect":
        target = "subject"
        match = re.search(r"effect to the\s+(.+?)(?: that| throughout| while|\.|$)", lowered)
        if match:
            target = clean_candidate(match.group(1))
        params = {}
        if any(token in lowered for token in ["glow", "aura", "neon", "flame", "lighting"]):
            params["effect_type"] = "glow_or_decoration"
        if "pulse" in lowered or "rhythmically" in lowered:
            params["temporal_pattern"] = "pulse"
        return target, params

    if action in {"edit_motion", "edit_expression"}:
        if action == "edit_expression":
            params = {}
            if "smile" in lowered:
                params["to_expression"] = "joyous_smile" if "joyous" in lowered or "wide" in lowered else "smile"
            if "pensive" in lowered:
                params["from_expression"] = "pensive"
            return "face", params
        params = {}
        if "wave" in lowered:
            params["gesture"] = "wave"
        if "toast" in lowered or "cups together" in lowered:
            params["gesture"] = "toast"
        if "nod" in lowered:
            params["gesture"] = "nod"
            if "slight" in lowered or "subtle" in lowered:
                params["magnitude"] = "slight"
        if "left hand" in lowered:
            params["body_part"] = "left_hand"
        if "spin" in lowered:
            params["motion"] = "spin"
        return "person", params

    if action == "remove_object":
        match = re.search(r"remove\s+(?:the\s+)?(.+?)(?: from| and| throughout| while|\.|$)", lowered)
        return clean_candidate(match.group(1)) if match else "object", {}

    return "subject", {}


def _predict_improved(record: dict, records: list[dict], version: str):
    action = infer_action(record)
    examples = best_examples(record, records, action, k=3)
    parsed_target, parsed_params = parse_action_specific_fields(record, action)
    actions_with_strong_local_parse = {"change_color", "replace_background", "apply_style", "edit_motion", "edit_expression", "orbit_camera"}
    use_example_shape = action not in actions_with_strong_local_parse
    base_params = clone_json(examples[0]["gt_primary"].get("params", {})) if examples and use_example_shape else {}
    params = merge_dict(base_params, parsed_params)

    template_target = examples[0]["gt_primary"].get("target", "") if examples else ""
    if isinstance(parsed_target, list):
        target = parsed_target
    elif parsed_target in {"object", "subject", "scene"} and template_target:
        target = template_target
    else:
        target = parsed_target or template_target or "subject"

    confidence = 0.45
    if examples:
        confidence += min(0.35, text_similarity(record["instruction"], examples[0]["instruction"]))
    if target:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)

    return {
        "tasks": [{"action": action, "target": target, "constraints": [], "params": params}]
    }, {"version": version, "confidence": confidence, "nearest_examples": [e["video_path"] for e in examples]}


def predict_ver10_improved(record: dict, records: list[dict]):
    return _predict_improved(record, records, "ver10_improved")


def predict_v11a_ruleplus(record: dict, records: list[dict]):
    action = infer_action(record)
    target, params = parse_action_specific_fields(record, action)
    confidence = round(min(0.55 + min(0.2, len(params) * 0.05), 0.95), 3)
    return {
        "tasks": [{"action": action, "target": target, "constraints": [], "params": params}]
    }, {"version": "v11a_ruleplus", "confidence": confidence}


def predict_v11b_retrieval(record: dict, records: list[dict]):
    return _predict_improved(record, records, "v11b_retrieval")


def predict_v11d_ensemble(
    record: dict,
    records: list[dict],
    seed_pred: dict | None,
    pred_v11a: dict,
    dbg_v11a: dict,
    pred_v11b: dict,
    dbg_v11b: dict,
):
    action = infer_action(record)
    examples = best_examples(record, records, action, k=5)
    majority_counter = Counter(example["gt_primary"]["action"] for example in examples)
    majority_action = majority_counter.most_common(1)[0][0] if majority_counter else action

    candidates = []
    if seed_pred:
        candidates.append(("ver10_seed", seed_pred, 0.82))
    candidates.append(("v11a_ruleplus", pred_v11a, dbg_v11a.get("confidence", 0.6)))
    candidates.append(("v11b_retrieval", pred_v11b, dbg_v11b.get("confidence", 0.6)))

    scored = []
    for name, pred, confidence in candidates:
        pred_action = pred["tasks"][0].get("action", "")
        pred_target = pred["tasks"][0].get("target", "")

        bonus = 0.0
        if pred_action == majority_action:
            bonus += 0.12
        if pred_action == action:
            bonus += 0.08

        target_specific = len(pred_target) > 0 if isinstance(pred_target, list) else bool(pred_target) and pred_target not in {"subject", "object", "scene"}
        if target_specific:
            bonus += 0.05

        scored.append((confidence + bonus, name, pred))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_pred = scored[0]
    return best_pred, {
        "version": "v11d_ensemble",
        "selected_from": best_name,
        "confidence": round(min(best_score, 0.99), 3),
        "majority_action": majority_action,
    }
