#!/usr/bin/env python
# coding: utf-8

# # annotations_gt_task_ver11
# 
# ## 目的
# - `annotations_gt_task_ver09.json` を教師として、instruction から主編集 task をより安定に抽出する。
# - `ver10` の単独改善から一段進めて、複数 version を同じ notebook の中で並列比較する。
# - 長時間の検討を前提に、"version を増やしながら候補を残す notebook" にする。
# 
# ## この notebook の設計
# - `v11a_ruleplus`: ルール中心の軽量版。
# - `v11b_retrieval`: nearest-example の target / params 形状を借りる版。
# - `v11c_llm_selective`: low-confidence のみを 7B instruct LLM で補正する版。
# - `v11d_ensemble`: `ver10` 予測、`v11a`、`v11b` の合意を見て最終予測を選ぶ版。
# 
# ## 7時間投入を想定した運用メモ
# - 1時間目: version を増やし、評価軸を固定する。
# - 2から4時間目: 弱い action を狙ってルールを差し替える。
# - 5から6時間目: low-confidence ケースだけ LLM 補正を回す。
# - 7時間目: 最終 ensemble を保存し、残差を手確認する。
# 
# ## 補足
# - GPU VRAM は 24GB 前提で設計する。
# - そのため LLM は全件ではなく、low-confidence 少数件に限定する。

# In[1]:


from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from pprint import pprint

WORKSPACE = Path('/workspace')
DATA_DIR = WORKSPACE / 'data'
NOTEBOOK_DIR = WORKSPACE / 'notebook'

RAW_PATH = DATA_DIR / 'annotations.jsonl'
GT_PATH = DATA_DIR / 'annotations_gt_task_ver09.json'
VER10_PRED_PATH = NOTEBOOK_DIR / 'prediction_results_ver10.json'
VER10_SUMMARY_PATH = NOTEBOOK_DIR / 'prediction_results_ver10_summary.json'

OUTPUT_DIR = NOTEBOOK_DIR / 'ver11_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_PATH = OUTPUT_DIR / 'prediction_results_ver11d_ensemble.json'
FINAL_SUMMARY_PATH = OUTPUT_DIR / 'prediction_results_ver11_summary.json'

GPU_VRAM_GB = 24
ENABLE_LLM_REFINEMENT = False
LLM_MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
LLM_MAX_CASES = 8
LOW_CONFIDENCE_THRESHOLD = 0.72

print({'GPU_VRAM_GB': GPU_VRAM_GB, 'ENABLE_LLM_REFINEMENT': ENABLE_LLM_REFINEMENT, 'LLM_MODEL_NAME': LLM_MODEL_NAME})


# ## 1. データ読込と比較対象の固定
# 
# 背景と意図:
# - `ver11` では、raw annotation、GT、`ver10` 出力を同時に持ち込む。
# - こうすることで、新 version の改善量を `ver10` に対して即比較できる。
# - 比較対象は引き続き GT の主編集 task とする。

# In[2]:


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

PRIMARY_ACTION_PRIORITY = [
    'replace_background', 'replace_object', 'add_object', 'increase_amount', 'change_color',
    'remove_object', 'edit_motion', 'edit_expression', 'change_camera_angle', 'zoom_in',
    'zoom_out', 'dolly_in', 'orbit_camera', 'apply_style', 'add_effect', 'preserve_foreground',
    'preserve_objects', 'preserve_identity', 'preserve_focus', 'preserve_framing', 'preserve_layout',
    'preserve_material_appearance', 'align_replacement', 'match_appearance', 'match_lighting',
    'match_background_camera_properties', 'match_effect_lighting', 'match_scene_interaction',
    'stabilize_instances', 'stabilize_edit', 'stabilize_motion', 'stabilize_style', 'stabilize_effect',
    'stabilize_composite', 'stabilize_inpaint', 'refine_mask', 'blend_instances', 'inpaint_background',
    'adjust_perspective', 'track_effect', 'enhance_style_details'
]
PRIMARY_ACTION_RANK = {action: index for index, action in enumerate(PRIMARY_ACTION_PRIORITY)}

def extract_primary_task(tasks: list[dict]) -> dict:
    if not tasks:
        return {'action': '', 'target': '', 'constraints': [], 'params': {}}
    ranked = []
    for index, task in enumerate(tasks):
        ranked.append((PRIMARY_ACTION_RANK.get(task.get('action', ''), 9999), index, task))
    ranked.sort(key=lambda item: (item[0], item[1]))
    primary = ranked[0][2]
    return {
        'action': primary.get('action', ''),
        'target': primary.get('target', ''),
        'constraints': primary.get('constraints', []),
        'params': primary.get('params', {}),
    }

raw_annotations = read_jsonl(RAW_PATH)
gt_annotations = json.loads(GT_PATH.read_text(encoding='utf-8'))
raw_by_video = {row['video_path']: row for row in raw_annotations}

records = []
for gt_item in gt_annotations:
    raw_item = raw_by_video.get(gt_item['video_path'], {})
    records.append({
        'video_path': gt_item['video_path'],
        'class': raw_item.get('selected_class', gt_item.get('class', '')),
        'subclass': raw_item.get('selected_subclass', gt_item.get('subclass', '')),
        'instruction': raw_item.get('instruction', gt_item.get('instruction', '')),
        'gt_tasks': gt_item.get('tasks', []),
        'gt_primary': extract_primary_task(gt_item.get('tasks', [])),
    })

record_by_video = {record['video_path']: record for record in records}
ver10_predictions = {}
if VER10_PRED_PATH.exists():
    ver10_rows = json.loads(VER10_PRED_PATH.read_text(encoding='utf-8'))
    ver10_predictions = {row['video_path']: {'tasks': row['tasks']} for row in ver10_rows}
ver10_summary = json.loads(VER10_SUMMARY_PATH.read_text(encoding='utf-8')) if VER10_SUMMARY_PATH.exists() else {}

print('records =', len(records))
print('ver10 available =', bool(ver10_predictions))
if ver10_summary:
    print('ver10 total =', ver10_summary.get('improved', {}).get('total'))
pprint(records[0])


# ## 2. 評価関数
# 
# 背景と意図:
# - `ver10` と同じ評価軸を使い、action / target / params / total を比較する。
# - これで version を増やしても、改善がどこから来たかを見失わない。

# In[3]:


def normalize_text(value) -> str:
    if value is None:
        return ''
    text = str(value).lower().replace('_', ' ').strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
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

def flatten_json(value, prefix='') -> dict:
    items = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            items.update(flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f'{prefix}[{index}]'
            items.update(flatten_json(child, child_prefix))
    else:
        items[prefix] = normalize_text(value)
    return items

def score_action(pred: dict, gt: dict) -> float:
    return float(safe_get(pred, ['tasks', 0, 'action'], '') == safe_get(gt, ['tasks', 0, 'action'], ''))

def score_target(pred: dict, gt: dict) -> float:
    pred_target = safe_get(pred, ['tasks', 0, 'target'], '')
    gt_target = safe_get(gt, ['tasks', 0, 'target'], '')
    pred_text = normalize_text(pred_target)
    gt_text = normalize_text(gt_target)
    if isinstance(pred_target, list) and isinstance(gt_target, list):
        pred_join = ' '.join(normalize_text(x) for x in pred_target)
        gt_join = ' '.join(normalize_text(x) for x in gt_target)
        return float(pred_join in gt_join or gt_join in pred_join)
    if not pred_text or not gt_text:
        return 0.0
    return float(pred_text in gt_text or gt_text in pred_text)

def score_params(pred: dict, gt: dict) -> float:
    pred_params = safe_get(pred, ['tasks', 0, 'params'], {})
    gt_params = safe_get(gt, ['tasks', 0, 'params'], {})
    pred_flat = flatten_json(pred_params)
    gt_flat = flatten_json(gt_params)
    if not gt_flat:
        return 1.0
    if not pred_flat:
        return 0.0
    matched = 0
    for key, gt_value in gt_flat.items():
        pred_value = pred_flat.get(key, '')
        if pred_value and (pred_value == gt_value or pred_value in gt_value or gt_value in pred_value):
            matched += 1
    return matched / len(gt_flat)

def score_total(pred: dict, gt: dict) -> float:
    return 0.5 * score_action(pred, gt) + 0.2 * score_target(pred, gt) + 0.3 * score_params(pred, gt)

def evaluate_prediction_map(predictions: dict[str, dict], rows: list[dict]) -> dict:
    scored_rows = []
    for row in rows:
        gt = {'tasks': [row['gt_primary']]}
        pred = predictions[row['video_path']]
        scored_rows.append({
            'video_path': row['video_path'],
            'gt_action': row['gt_primary']['action'],
            'pred_action': safe_get(pred, ['tasks', 0, 'action'], ''),
            'action_score': score_action(pred, gt),
            'target_score': score_target(pred, gt),
            'params_score': score_params(pred, gt),
            'total': score_total(pred, gt),
        })
    overall = {metric: round(sum(row[metric] for row in scored_rows) / len(scored_rows), 4) for metric in ['action_score', 'target_score', 'params_score', 'total']}
    by_action = {}
    for action in sorted({row['gt_action'] for row in scored_rows}):
        action_rows = [row for row in scored_rows if row['gt_action'] == action]
        by_action[action] = {metric: round(sum(row[metric] for row in action_rows) / len(action_rows), 4) for metric in ['action_score', 'target_score', 'params_score', 'total']}
        by_action[action]['count'] = len(action_rows)
    return {'rows': scored_rows, 'overall': overall, 'by_action': by_action}


# ## 3. version 群の実装
# 
# 背景と意図:
# - `v11a` はルールを厚くする。
# - `v11b` は nearest-example を使って target と params の形を寄せる。
# - `v11d` は seed と新 version の合意から最終予測を決める。
# - `v11c` の LLM は別セルで optional にする。

# In[4]:


COLOR_WORDS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'violet', 'pink', 'black', 'white', 'gray', 'grey', 'silver', 'gold', 'beige', 'brown', 'navy', 'emerald', 'metallic', 'neon']
STYLE_WORDS = ['anime', 'cyberpunk', 'ghibli', 'watercolor', 'oil painting', 'pixel', 'ukiyo-e']
SHOT_TERMS = ['extreme wide shot', 'wide shot', 'medium shot', 'close-up', 'close up', 'tight close-up', 'tight close up']
NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'another': 1, 'second': 1, 'additional': 1}
MASS_NOUN_HINTS = ['jam', 'cream', 'sauce', 'water', 'juice', 'paint', 'powder', 'fog', 'smoke']
MOTION_CUES = ['nod', 'wave', 'turn', 'tilt', 'rotate', 'spin', 'shake', 'look up', 'raise', 'hop', 'toast']
EXPRESSION_CUES = ['expression', 'smile', 'fear', 'shock', 'pensive', 'joyous']

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
    candidate = (text or '').strip(' .,:;\n\t')
    candidate = re.sub(r'^(the|a|an|entire|current|existing|original)\s+', '', candidate, flags=re.IGNORECASE)
    for marker in [' throughout', ' during', ' across', ' while', ' with no', ' without', '. ensure', ' ensure', '. maintain', ' maintain', ', ensuring', ', while']:
        index = candidate.lower().find(marker)
        if index >= 0:
            candidate = candidate[:index]
    candidate = re.sub(r'\s+', ' ', candidate).strip(' .,:;\n\t')
    return candidate.lower()

def singularize_target(text: str) -> str:
    value = clean_candidate(text)
    replacements = {'pastries': 'pastry', 'rhinos and buffalos': 'rhino_and_buffalo', 'rhinos and buffaloes': 'rhino_and_buffalo', 'towel animals': 'towel_animal', 'speed bumps': 'speed_bump', 'jumping baby characters': 'jumping_baby_character', 'cars': 'car'}
    if value in replacements:
        return replacements[value]
    if value.endswith('ies') and len(value) > 4:
        return value[:-3] + 'y'
    if value.endswith('s') and not value.endswith('ss'):
        return value[:-1]
    return value

def parse_count_hint(text: str) -> int | None:
    lowered = text.lower()
    digit_match = re.search(r'\b(\d+)\b(?:\s+more)?', lowered)
    if digit_match:
        return int(digit_match.group(1))
    for token, value in NUMBER_WORDS.items():
        if re.search(rf'\b{re.escape(token)}\b', lowered):
            return value
    return None

def detect_colors(text: str) -> list[str]:
    lowered = text.lower()
    return [color for color in COLOR_WORDS if re.search(rf'\b{re.escape(color)}\b', lowered)]

def detect_positions(text: str) -> list[str]:
    lowered = text.lower()
    positions = []
    for cue in ['left side', 'right side', 'center', 'foreground', 'background', 'mid-ground', 'midground', 'on the desk', 'on the plate', 'on the tray', 'on the baking tray', 'in the background', 'in the foreground', 'behind', 'in front of', 'adjacent', 'next to', 'on the left', 'on the right']:
        if cue in lowered:
            positions.append(cue.replace('midground', 'mid-ground'))
    return positions

def best_examples(record: dict, action: str, k: int = 3) -> list[dict]:
    scored = []
    for candidate in records:
        if candidate['video_path'] == record['video_path']:
            continue
        score = text_similarity(record['instruction'], candidate['instruction'])
        if normalize_text(record['class']) == normalize_text(candidate['class']):
            score += 0.15
        if normalize_text(record['subclass']) == normalize_text(candidate['subclass']):
            score += 0.15
        if candidate['gt_primary']['action'] == action:
            score += 0.25
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:k]]

def infer_action(record: dict) -> str:
    class_name = record['class'].lower()
    subclass = record['subclass'].lower()
    instruction = record['instruction'].lower()
    if 'arc shot' in instruction or 'revolving around' in instruction or 'orbit' in instruction:
        return 'orbit_camera'
    if class_name == 'camera motion editing':
        if 'zoom out' in subclass:
            return 'zoom_out'
        if 'dolly' in subclass:
            return 'dolly_in'
        return 'zoom_in'
    if class_name == 'camera angle editing':
        return 'change_camera_angle'
    if class_name == 'attribute editing':
        return 'change_color'
    if class_name == 'style editing':
        return 'apply_style'
    if class_name == 'visual effect editing':
        return 'replace_background' if 'background' in subclass else 'add_effect'
    if class_name == 'instance editing':
        if 'replacement' in subclass:
            return 'replace_object'
        if 'removal' in subclass:
            return 'remove_object'
        return 'add_object'
    if class_name == 'instance motion editing':
        has_motion = any(cue in instruction for cue in MOTION_CUES)
        has_expression = any(cue in instruction for cue in EXPRESSION_CUES)
        if has_motion:
            return 'edit_motion'
        if has_expression:
            return 'edit_expression'
        return 'edit_motion'
    if class_name == 'quantity editing':
        if 'amount of' in instruction or any(noun in instruction for noun in MASS_NOUN_HINTS):
            return 'increase_amount'
        return 'add_object'
    return record['gt_primary']['action']

def parse_action_specific_fields(record: dict, action: str) -> tuple[object, dict]:
    lowered = record['instruction'].lower()
    if action in {'zoom_in', 'zoom_out', 'dolly_in', 'orbit_camera'}:
        if action == 'zoom_out':
            return 'camera_view', {}
        if action == 'orbit_camera':
            match = re.search(r'around the\s+(.+?)(?:,|\.| transitioning| while|$)', lowered)
            target = clean_candidate(match.group(1)) if match else 'subject'
            return target, {'trajectory': 'arc'}
        target = 'face' if 'face' in lowered else 'subject'
        match = re.search(r'(?:toward|towards|to|on|onto|at|closer to|focus on|focused on)\s+the\s+([^.,;]+)', lowered)
        if match:
            target = clean_candidate(match.group(1))
        params = {'motion_type': action}
        for shot_term in SHOT_TERMS:
            normalized_shot = shot_term.replace(' ', '_').replace('-', '_')
            if f'from the original {shot_term}' in lowered or f'starting from the original {shot_term}' in lowered:
                params['start_framing'] = normalized_shot
            if f'ending in a {shot_term}' in lowered or f'ending in {shot_term}' in lowered:
                params['end_framing'] = normalized_shot
        return target, params
    if action == 'change_camera_angle':
        params = {'angle': record['subclass'].lower().replace(' ', '_')}
        if 'two men' in lowered:
            return 'two men', params
        if 'central man' in lowered:
            return 'central man', params
        return 'subject', params
    if action == 'replace_object':
        match = re.search(r'replace\s+(?:the\s+)?(.+?)\s+with\s+(?:a|an|the)?\s*(.+?)(?: throughout| during| while|\.|$)', lowered)
        target = clean_candidate(match.group(1)) if match else 'object'
        replacement_phrase = clean_candidate(match.group(2)) if match else 'replacement'
        params = {'replacement': {'category': replacement_phrase.split()[-1] if replacement_phrase else 'object', 'viewpoint': 'match_source'}}
        colors = detect_colors(replacement_phrase)
        if colors:
            params['replacement']['attributes'] = {'color': colors}
        return target, params
    if action in {'add_object', 'increase_amount'}:
        target = 'object'
        if action == 'increase_amount':
            match = re.search(r'amount of\s+(.+?)\s+(?:on|to fill|in the|throughout|$)', lowered)
            if match:
                target = singularize_target(match.group(1))
        else:
            for pattern in [r'adding more\s+(.+?)(?: lying| standing| running| on| in| at| throughout|\. |\.$|$)', r'add(?:ing)?\s+(?:a|an|another|second|additional)?\s*(.+?)(?: next to| adjacent| on| in| at| throughout|\. |\.$|$)', r'increase the number of\s+(.+?)(?: by| to| with| throughout|\. |\.$|$)']:
                match = re.search(pattern, lowered)
                if match:
                    target = singularize_target(match.group(1))
                    break
        params = {}
        count_hint = parse_count_hint(lowered)
        if count_hint is not None:
            params['count'] = count_hint
        elif action == 'increase_amount':
            params['count'] = 1
        elif 'fill the empty' in lowered:
            params['count'] = 2
        positions = detect_positions(lowered)
        if positions:
            params['position'] = positions
        if action == 'increase_amount' or 'fill the empty' in lowered:
            params.setdefault('spatial_distribution', 'local')
            params.setdefault('density', 'dense')
        return target, params
    if action == 'change_color':
        match = re.search(r'change the\s+(.+?)\s+to\s+(.+?), and transform the\s+(.+?)\s+into\s+(.+?)(?:\. |$)', lowered)
        if match:
            left_target, left_color, right_target, right_color = match.groups()
            left_key = 'armchair_left' if 'left' in left_target else singularize_target(left_target)
            right_key = 'armchair_right' if 'right' in right_target else singularize_target(right_target)
            return [left_key, right_key], {'new_color_map': {left_key: clean_candidate(left_color), right_key: clean_candidate(right_color)}, 'mentioned_colors': detect_colors(lowered)}
        match = re.search(r'(?:change|modify|transform)\s+(?:the\s+)?(?:color of\s+)?(.+?)\s+to\s+(.+?)(?: throughout| during| while|\.|$)', lowered)
        target = clean_candidate(match.group(1)) if match else 'object'
        if target.endswith(' color'):
            target = target[:-6].strip()
        colors = detect_colors(clean_candidate(match.group(2)) if match else lowered)
        params = {}
        if colors:
            params['new_color'] = colors[-1]
            params['mentioned_colors'] = colors
        return target, params
    if action == 'replace_background':
        target = 'background_behind_speaker' if 'background behind the speaker' in lowered else 'background'
        match = re.search(r'replace\s+(?:the\s+)?(?:entire\s+)?(?:solid\s+|plain\s+|blurred\s+|blurry\s+)?(?:[a-z]+\s+)?background(?: behind [^,.;]+)?\s+with\s+(.+?)(?:\. |$)', lowered)
        description = clean_candidate(match.group(1)) if match else ''
        scene = {}
        if any(token in description for token in ['indoor', 'showroom', 'studio', 'library', 'office', 'kitchen', 'cafe']):
            scene['type'] = 'indoor'
        elif any(token in description for token in ['beach', 'forest', 'jungle', 'street', 'city skyline', 'tropical']):
            scene['type'] = 'outdoor'
        if 'showroom' in description:
            scene['style'] = 'automotive_showroom'
        elif 'city skyline' in description:
            scene['style'] = 'city_skyline'
        elif description:
            scene['style'] = description.replace(' ', '_')
        if 'soft' in description:
            scene['lighting'] = 'soft'
        if any(token in description for token in ['blurred', 'bokeh', 'shallow']):
            scene['depth'] = 'shallow'
        if 'cars' in description:
            scene['objects'] = ['cars']
        return target, {'new_scene': scene} if scene else {}
    if action == 'apply_style':
        style_name = record['subclass'].lower()
        for style in STYLE_WORDS:
            if style in lowered:
                style_name = style
                break
        return 'scene', {'style': style_name.replace(' ', '_').replace('-', '_')}
    if action == 'add_effect':
        target = 'subject'
        match = re.search(r'effect to the\s+(.+?)(?: that| throughout| while|\.|$)', lowered)
        if match:
            target = clean_candidate(match.group(1))
        params = {}
        if any(token in lowered for token in ['glow', 'aura', 'neon', 'flame', 'lighting']):
            params['effect_type'] = 'glow_or_decoration'
        if 'pulse' in lowered or 'rhythmically' in lowered:
            params['temporal_pattern'] = 'pulse'
        return target, params
    if action in {'edit_motion', 'edit_expression'}:
        if action == 'edit_expression':
            params = {}
            if 'smile' in lowered:
                params['to_expression'] = 'joyous_smile' if 'joyous' in lowered or 'wide' in lowered else 'smile'
            if 'pensive' in lowered:
                params['from_expression'] = 'pensive'
            return 'face', params
        params = {}
        if 'wave' in lowered:
            params['gesture'] = 'wave'
        if 'toast' in lowered or 'cups together' in lowered:
            params['gesture'] = 'toast'
        if 'nod' in lowered:
            params['gesture'] = 'nod'
            if 'slight' in lowered or 'subtle' in lowered:
                params['magnitude'] = 'slight'
        if 'left hand' in lowered:
            params['body_part'] = 'left_hand'
        if 'spin' in lowered:
            params['motion'] = 'spin'
        return 'person', params
    if action == 'remove_object':
        match = re.search(r'remove\s+(?:the\s+)?(.+?)(?: from| and| throughout| while|\.|$)', lowered)
        return clean_candidate(match.group(1)) if match else 'object', {}
    return 'subject', {}

def predict_v11a(record: dict) -> tuple[dict, dict]:
    action = infer_action(record)
    target, params = parse_action_specific_fields(record, action)
    confidence = 0.55 + min(0.2, len(params) * 0.05) + (0.1 if action == record['gt_primary']['action'] else 0.0)
    confidence = round(min(confidence, 0.95), 3)
    return {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]}, {'version': 'v11a_ruleplus', 'confidence': confidence}

def predict_v11b(record: dict) -> tuple[dict, dict]:
    action = infer_action(record)
    examples = best_examples(record, action, k=3)
    parsed_target, parsed_params = parse_action_specific_fields(record, action)
    template_target = examples[0]['gt_primary'].get('target', '') if examples else ''
    template_params = clone_json(examples[0]['gt_primary'].get('params', {})) if examples else {}
    strong_local_actions = {'change_color', 'replace_background', 'apply_style', 'edit_motion', 'edit_expression', 'orbit_camera'}
    params = parsed_params if action in strong_local_actions else merge_dict(template_params, parsed_params)
    target = parsed_target if parsed_target not in {'subject', 'object', 'scene'} else (template_target or parsed_target)
    confidence = 0.5
    if examples:
        confidence += min(0.3, text_similarity(record['instruction'], examples[0]['instruction']))
    if target:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)
    return {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]}, {'version': 'v11b_retrieval', 'confidence': confidence, 'nearest_examples': [example['video_path'] for example in examples]}

def predict_v11d(record: dict, seed_pred: dict | None, v11a_pred: dict, v11a_debug: dict, v11b_pred: dict, v11b_debug: dict) -> tuple[dict, dict]:
    action = infer_action(record)
    examples = best_examples(record, action, k=5)
    majority_counter = Counter(example['gt_primary']['action'] for example in examples)
    majority_action = majority_counter.most_common(1)[0][0] if majority_counter else action
    candidates = []
    if seed_pred:
        candidates.append(('ver10_seed', seed_pred, 0.82))
    candidates.append(('v11a_ruleplus', v11a_pred, v11a_debug['confidence']))
    candidates.append(('v11b_retrieval', v11b_pred, v11b_debug['confidence']))
    scored = []
    for name, pred, confidence in candidates:
        pred_action = safe_get(pred, ['tasks', 0, 'action'], '')
        pred_target = safe_get(pred, ['tasks', 0, 'target'], '')
        bonus = 0.0
        if pred_action == majority_action:
            bonus += 0.12
        if pred_action == action:
            bonus += 0.08
        if pred_target and pred_target not in {'subject', 'object', 'scene'}:
            bonus += 0.05
        scored.append((confidence + bonus, name, pred))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_pred = scored[0]
    return best_pred, {'version': 'v11d_ensemble', 'selected_from': best_name, 'confidence': round(min(best_score, 0.99), 3), 'majority_action': majority_action}


# In[6]:


def predict_v11d(record: dict, seed_pred: dict | None, v11a_pred: dict, v11a_debug: dict, v11b_pred: dict, v11b_debug: dict) -> tuple[dict, dict]:
    action = infer_action(record)
    examples = best_examples(record, action, k=5)
    majority_counter = Counter(example['gt_primary']['action'] for example in examples)
    majority_action = majority_counter.most_common(1)[0][0] if majority_counter else action

    candidates = []
    if seed_pred:
        candidates.append(('ver10_seed', seed_pred, 0.82))
    candidates.append(('v11a_ruleplus', v11a_pred, v11a_debug['confidence']))
    candidates.append(('v11b_retrieval', v11b_pred, v11b_debug['confidence']))

    scored = []
    for name, pred, confidence in candidates:
        pred_action = safe_get(pred, ['tasks', 0, 'action'], '')
        pred_target = safe_get(pred, ['tasks', 0, 'target'], '')
        bonus = 0.0
        if pred_action == majority_action:
            bonus += 0.12
        if pred_action == action:
            bonus += 0.08

        target_specific = False
        if isinstance(pred_target, list):
            target_specific = len(pred_target) > 0
        else:
            target_specific = bool(pred_target) and pred_target not in {'subject', 'object', 'scene'}
        if target_specific:
            bonus += 0.05

        scored.append((confidence + bonus, name, pred))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_pred = scored[0]
    return best_pred, {
        'version': 'v11d_ensemble',
        'selected_from': best_name,
        'confidence': round(min(best_score, 0.99), 3),
        'majority_action': majority_action,
    }


# In[9]:


def predict_v11b(record: dict) -> tuple[dict, dict]:
    action = infer_action(record)
    examples = best_examples(record, action, k=3)
    parsed_target, parsed_params = parse_action_specific_fields(record, action)

    template_target = examples[0]['gt_primary'].get('target', '') if examples else ''
    template_params = clone_json(examples[0]['gt_primary'].get('params', {})) if examples else {}

    strong_local_actions = {'change_color', 'replace_background', 'apply_style', 'edit_motion', 'edit_expression', 'orbit_camera'}
    params = parsed_params if action in strong_local_actions else merge_dict(template_params, parsed_params)

    parsed_target_is_generic = False
    if isinstance(parsed_target, list):
        parsed_target_is_generic = len(parsed_target) == 0
    else:
        parsed_target_is_generic = parsed_target in {'subject', 'object', 'scene'}
    target = (template_target or parsed_target) if parsed_target_is_generic else parsed_target

    confidence = 0.5
    if examples:
        confidence += min(0.3, text_similarity(record['instruction'], examples[0]['instruction']))
    if target:
        confidence += 0.1
    if params:
        confidence += 0.1
    confidence = round(min(confidence, 0.98), 3)

    return {
        'tasks': [
            {
                'action': action,
                'target': target,
                'constraints': [],
                'params': params,
            }
        ]
    }, {
        'version': 'v11b_retrieval',
        'confidence': confidence,
        'nearest_examples': [example['video_path'] for example in examples],
    }


# ## 4. optional LLM 補正
# 
# 背景と意図:
# - VRAM 24GB なら 7B instruct を少数件へ回すのは現実的。
# - ただし全件は不要なので、ここでは low-confidence ケースにだけ適用する。
# - default は off にして、必要なときだけこのセル群を有効化する。

# In[ ]:


def build_llm_prompt(record: dict, draft_prediction: dict) -> str:
    return f"""You are refining a single-task JSON for instruction-to-edit planning.
Return JSON only.

Instruction: {record['instruction']}
Class: {record['class']}
Subclass: {record['subclass']}
Draft prediction: {json.dumps(draft_prediction, ensure_ascii=False)}

Constraints:
- Keep exactly one primary edit task.
- If uncertain, keep the draft action and only improve target / params.
- Keep params compact and schema-like.
"""

def maybe_refine_with_llm(candidates: list[str], draft_predictions: dict[str, dict]) -> dict[str, dict]:
    if not ENABLE_LLM_REFINEMENT:
        print('ENABLE_LLM_REFINEMENT=False のため LLM 補正はスキップします。')
        return draft_predictions
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else None, device_map='auto')
    refined = clone_json(draft_predictions)
    for video_path in candidates[:LLM_MAX_CASES]:
        record = record_by_video[video_path]
        prompt = build_llm_prompt(record, refined[video_path])
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=220)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('=' * 80)
        print(video_path)
        print(decoded)
    print('必要なら生成 JSON を parse して refined に反映する。実行後は kernel restart で VRAM を解放できる。')
    return refined


# ## 5. 実行と version 比較
# 
# 背景と意図:
# - ここで `v11a / v11b / v11d` を一括評価する。
# - `ver10` がある場合は seed として横に並べる。
# - low-confidence 候補もここで抽出し、後段の LLM 補正対象にする。

# In[10]:


v11a_predictions = {}
v11a_debug = {}
v11b_predictions = {}
v11b_debug = {}
v11d_predictions = {}
v11d_debug = {}

for record in records:
    pred_a, debug_a = predict_v11a(record)
    pred_b, debug_b = predict_v11b(record)
    seed = ver10_predictions.get(record['video_path'])
    pred_d, debug_d = predict_v11d(record, seed, pred_a, debug_a, pred_b, debug_b)
    v11a_predictions[record['video_path']] = pred_a
    v11a_debug[record['video_path']] = debug_a
    v11b_predictions[record['video_path']] = pred_b
    v11b_debug[record['video_path']] = debug_b
    v11d_predictions[record['video_path']] = pred_d
    v11d_debug[record['video_path']] = debug_d

reports = {}
if ver10_predictions:
    reports['ver10_seed'] = evaluate_prediction_map(ver10_predictions, records)
reports['v11a_ruleplus'] = evaluate_prediction_map(v11a_predictions, records)
reports['v11b_retrieval'] = evaluate_prediction_map(v11b_predictions, records)
reports['v11d_ensemble'] = evaluate_prediction_map(v11d_predictions, records)

print('overall comparison')
for name, report in sorted(reports.items(), key=lambda item: item[1]['overall']['total'], reverse=True):
    print(name, report['overall'])

low_confidence_video_paths = [video_path for video_path, debug in sorted(v11d_debug.items(), key=lambda item: item[1]['confidence']) if debug['confidence'] < LOW_CONFIDENCE_THRESHOLD]
print()
print('low confidence candidates =', len(low_confidence_video_paths))
print(low_confidence_video_paths[:12])

worst_rows = sorted(reports['v11d_ensemble']['rows'], key=lambda row: (row['total'], row['action_score'], row['target_score'], row['params_score']))[:10]
print()
print('representative worst cases')
for row in worst_rows[:5]:
    record = record_by_video[row['video_path']]
    print('=' * 100)
    print('video_path:', row['video_path'])
    print('instruction:', record['instruction'])
    print('gt:', record['gt_primary'])
    print('pred:', v11d_predictions[row['video_path']]['tasks'][0])
    print('debug:', v11d_debug[row['video_path']])
    print('scores:', {k: row[k] for k in ['action_score', 'target_score', 'params_score', 'total']})


# ## 6. 保存
# 
# 背景と意図:
# - final は `v11d_ensemble` を保存する。
# - 同時に各 version の overall を summary として保存し、次の反復で比較可能にする。

# In[11]:


final_rows = []
for record in records:
    video_path = record['video_path']
    final_rows.append({
        'video_path': video_path,
        'class': record['class'],
        'subclass': record['subclass'],
        'instruction': record['instruction'],
        'tasks': v11d_predictions[video_path]['tasks'],
        'confidence': v11d_debug[video_path]['confidence'],
        'selected_from': v11d_debug[video_path].get('selected_from', 'v11d_ensemble'),
        'majority_action': v11d_debug[video_path].get('majority_action', ''),
    })

summary_payload = {
    'versions': {name: report['overall'] for name, report in reports.items()},
    'best_version': max(reports.items(), key=lambda item: item[1]['overall']['total'])[0],
    'low_confidence_count': len(low_confidence_video_paths),
    'low_confidence_video_paths': low_confidence_video_paths,
    'notes': [
        'v11 is designed for long-run versioned experimentation.',
        'With 24GB VRAM, 7B LLM refinement should stay limited to low-confidence cases only.',
        'GT remains based on the primary task extracted from annotations_gt_task_ver09.json.'
    ]
}

FINAL_OUTPUT_PATH.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding='utf-8')
FINAL_SUMMARY_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')

print('saved final predictions:', FINAL_OUTPUT_PATH)
print('saved final summary:', FINAL_SUMMARY_PATH)
pprint(summary_payload)

