#!/usr/bin/env python
# coding: utf-8

# # parse_instruction_ver16 (multi-task, instruction-only)
# 
# ## 目的
# - ver15 の single-task 精度 (total=0.8311) をベースに multi-task 予測へ拡張。
# - instruction のみを入力とし、primary task + auxiliary tasks を予測する。
# - multi-task score 0.8 を目指す。
# 
# ## 制約
# - `class` / `subclass` 使用禁止。
# 
# ## 試行モデル
# - `v16a`: primary (v15d) + deterministic aux tasks (action ごとの上位 aux を固定で追加)
# - `v16b`: primary (v15d) + action-filtered retrieval (同 action の最近傍から全 task list を転用)
# - `v16c`: v16b + count-adaptive (GT count に合わせて aux task 数を調整)

# In[1]:


from __future__ import annotations

import copy
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from pprint import pprint
from collections import defaultdict

WORKSPACE = Path('/workspace')
SRC_DIR = WORKSPACE / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parse.data_loading import load_base_records, load_grouped_unknown_records

DATA_DIR = WORKSPACE / 'data'
NOTEBOOK_DIR = WORKSPACE / 'notebook'
OUTPUT_DIR = NOTEBOOK_DIR / 'ver16_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / 'annotations.jsonl'
GT_PATH = DATA_DIR / 'annotations_gt_task_ver09.json'
GROUPED_PATHS = [
    DATA_DIR / 'annotations_grouped_ver01.json',
    DATA_DIR / 'annotations_grouped_ver02.json',
]

print({'output_dir': str(OUTPUT_DIR)})


# In[2]:


base_records = load_base_records(RAW_PATH, GT_PATH)
unknown_records = load_grouped_unknown_records(
    GROUPED_PATHS, base_records, instruction_keys=('ver2', 'ver3', 'ver4')
)
print('base_records:', len(base_records))
print('unknown_records:', len(unknown_records))


# ## 1. ユーティリティ (ver15 の v15d 予測ロジックを再実装)

# In[3]:


# ---- text utilities ----

def normalize_text(x):
    if x is None: return ''
    t = str(x).lower().replace('_', ' ').strip()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def flatten_json(v, prefix=''):
    out = {}
    if isinstance(v, dict):
        for k, c in v.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            out.update(flatten_json(c, p))
    elif isinstance(v, list):
        for i, c in enumerate(v):
            out.update(flatten_json(c, f'{prefix}[{i}]'))
    else:
        out[prefix] = normalize_text(v)
    return out

def text_similarity(a, b):
    aa, bb = normalize_text(a), normalize_text(b)
    if not aa and not bb: return 1.0
    if not aa or not bb: return 0.0
    j = len(set(aa.split()) & set(bb.split())) / max(1, len(set(aa.split()) | set(bb.split())))
    return 0.6 * j + 0.4 * SequenceMatcher(None, aa, bb).ratio()

# ---- single-task scorer ----

def task_score(pred_task, gt_task):
    action = 1.0 if pred_task.get('action', '') == gt_task.get('action', '') else 0.0
    pt = normalize_text(' '.join(pred_task['target']) if isinstance(pred_task.get('target'), list) else pred_task.get('target', ''))
    gt = normalize_text(' '.join(gt_task['target']) if isinstance(gt_task.get('target'), list) else gt_task.get('target', ''))
    target = 1.0 if (pt and gt and (pt in gt or gt in pt)) else 0.0
    pp = flatten_json(pred_task.get('params', {}))
    gp = flatten_json(gt_task.get('params', {}))
    if not gp:
        params = 1.0
    elif not pp:
        params = 0.0
    else:
        m = sum(1 for k, gv in gp.items() if k in pp and (pp[k] == gv or pp[k] in gv or gv in pp[k]))
        params = m / len(gp)
    return 0.5 * action + 0.2 * target + 0.3 * params

# ---- multi-task scorer ----

def multi_task_score(pred_tasks, gt_tasks):
    pred_tasks = pred_tasks or []
    gt_tasks = gt_tasks or []
    if not pred_tasks and not gt_tasks:
        return {'coverage': 1.0, 'precision': 1.0, 'count_alignment': 1.0, 'total': 1.0}
    coverage = sum(max((task_score(p, g) for p in pred_tasks), default=0.0) for g in gt_tasks) / max(1, len(gt_tasks))
    precision = sum(max((task_score(p, g) for g in gt_tasks), default=0.0) for p in pred_tasks) / max(1, len(pred_tasks))
    m = max(1, len(pred_tasks), len(gt_tasks))
    count_alignment = 1.0 - abs(len(pred_tasks) - len(gt_tasks)) / m
    total = 0.55 * coverage + 0.35 * precision + 0.10 * count_alignment
    return {'coverage': round(coverage, 4), 'precision': round(precision, 4),
            'count_alignment': round(count_alignment, 4), 'total': round(total, 4)}

def evaluate_multi(records, predict_fn):
    rows, pred_map = [], {}
    for r in records:
        pred = predict_fn(r)
        key = r['prediction_key']
        pred_map[key] = pred
        s = multi_task_score(pred['tasks'], r['gt_tasks'])
        rows.append({'prediction_key': key, **s})
    overall = {k: round(sum(x[k] for x in rows) / len(rows), 4)
               for k in ['coverage', 'precision', 'count_alignment', 'total']}
    return {'rows': rows, 'overall': overall, 'predictions': pred_map}

print('utilities ready')


# ## 2. v15d primary predictor (instruction-only, action 100/100)

# In[4]:


# Re-implement v15d logic in this notebook for self-containment

_STYLE_VOCAB = ['anime', 'cyberpunk', 'pixel art', 'ukiyo', 'ghibli', 'watercolor',
                'oil painting', 'comic style', 'woodblock', '16-bit', 'retro pixel',
                'aesthetic', 'painting style', 'art style', 'american comic']
_COLOR_WORDS = ['violet', 'blue', 'red', 'green', 'navy', 'emerald', 'burgundy', 'magenta',
                'orange', 'purple', 'pink', 'yellow', 'black', 'white', 'gray', 'grey',
                'silver', 'gold', 'teal', 'cyan', 'indigo', 'maroon', 'beige', 'coral',
                'turquoise', 'aqua', 'lime', 'lavender', 'brown', 'crimson', 'amber']

def infer_action(inst):
    t = inst.lower()
    if re.search(r'\barc\s+shot\b|\borbit\b|revolving\s+around', t): return 'orbit_camera'
    if re.search(r'increase\s+the\s+amount\s+of', t): return 'increase_amount'
    if re.search(r'increase\s+the\s+number\s+of', t): return 'add_object'
    if re.search(r'\bdolly[- ]?in\b', t): return 'dolly_in'
    if re.search(r'\bzoom[- ]?out\b', t): return 'zoom_out'
    if re.search(r'\bzoom[- ]?in\b', t): return 'zoom_in'
    if re.search(r'low[- ]?angle|high[- ]?angle', t): return 'change_camera_angle'
    if re.search(r'(?:camera|shot|perspective|view)\s+(?:to|into)\s+(?:a\s+)?(?:low|high)', t): return 'change_camera_angle'
    if (bool(re.search(r'transform\s+(?:the\s+)?(?:entire\s+)?(?:video|scene|frame)', t))
            or re.search(r'apply\s+(?:a\s+)?(?:\w+\s+)*(?:art\s+)?style', t)):
        if any(w in t for w in _STYLE_VOCAB): return 'apply_style'
    if any(w in t for w in _STYLE_VOCAB): return 'apply_style'
    if re.search(r'\breplace\b', t):
        if re.search(r'replace\s+(?:the\s+)?(?:(?:\w+\s+){0,6})background', t): return 'replace_background'
        if 'with' in t: return 'replace_object'
    if re.search(r'\bremove\b', t): return 'remove_object'
    if re.search(r'(?:add|apply|enhance).*(?:glow(?:ing)?|aura|flame|lightning|sparkle|decoration\s+effect|stage\s+lighting)', t): return 'add_effect'
    if re.search(r'(?:glowing|neon\s+electric|neon\s+glow|electric\s+glow)', t): return 'add_effect'
    if re.search(r'\b(?:add|insert|place|introduce)\b', t): return 'add_object'
    if re.search(r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)*color', t): return 'change_color'
    if re.search(r'color\s+(?:of|to)\s+', t) and re.search(r'\b(?:change|modify|transform)\b', t): return 'change_color'
    return 'edit_motion'

def default_target(action):
    return {'replace_background': 'background', 'replace_object': 'object', 'remove_object': 'object',
            'add_object': 'object', 'increase_amount': 'object', 'change_color': 'subject',
            'apply_style': 'full_frame', 'zoom_out': 'camera_view', 'zoom_in': 'camera_view',
            'dolly_in': 'subject', 'change_camera_angle': 'subject', 'orbit_camera': 'subject',
            'edit_motion': 'person', 'add_effect': 'subject'}.get(action, 'subject')

def default_params(action, inst):
    t = inst.lower()
    p = {}
    if action == 'apply_style':
        for pat, val in [('ukiyo', 'ukiyo-e'), ('ghibli', 'ghibli'), ('watercolor', 'watercolor'),
                         ('oil painting', 'oil_painting'), (r'comic style|american comic|comic book', 'american_comic_style'),
                         (r'\bpixel\b|16-bit', 'pixel_art'), (r'\banime\b', 'anime'), ('cyberpunk', 'cyberpunk')]:
            if re.search(pat, t): p['style'] = val; break
    elif action in ('zoom_in', 'zoom_out', 'dolly_in'):
        p['motion_type'] = action
        if re.search(r'\b(?:slow|smooth|gradual|subtle|steady)\b', t): p['speed'] = 'gradual'
    elif action == 'change_camera_angle':
        if re.search(r'low[- ]?angle', t): p['angle'] = 'low_angle'
        elif re.search(r'high[- ]?angle', t): p['angle'] = 'high_angle'
    elif action == 'orbit_camera':
        p['trajectory'] = 'arc'
    elif action == 'add_effect':
        p['effect_type'] = 'glow_or_decoration'
    return p

def _extract_new_color(inst):
    t = inst.lower()
    m = re.search(r'\bto\s+(?:a\s+)?(?:(?:vibrant|bright|deep|dark|solid|metallic|neon|bold|glossy){0,2}\s*)'
                  r'([a-z]+(?:\s+[a-z]+){0,2})\s*(?:color|hue|throughout|while|[,.]|$)', t)
    if m:
        val = m.group(1).strip()
        if val not in {'color', 'hue', 'a', 'an', 'the', 'vibrant', 'metallic'}: return val
    return ''

def _extract_color_words(inst):
    t = inst.lower()
    found = []
    for pat in [r'(?:vibrant\s+)?metallic\s+[a-z]+', r'(?:deep|bright|dark|solid)\s+[a-z]+(?:\s+[a-z]+)?',
                r'neon\s+[a-z]+', r'navy\s+blue', r'emerald\s+green']:
        for m in re.finditer(pat, t): found.append(m.group())
    for cw in _COLOR_WORDS:
        if re.search(r'\b' + cw + r'\b', t) and not any(cw in f for f in found):
            found.append(cw)
    return found

def extract_primary_target_params(action, inst):
    target = default_target(action)
    params = {}
    t = inst.lower()
    if action == 'change_color':
        m = re.search(r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)?color\s+of\s+(?:the\s+)?(.+?)\s+to\s+', inst, re.I)
        if not m: m = re.search(r'(?:change|modify)\s+the\s+(.+?)\s+to\s+', inst, re.I)
        if m: target = m.group(1).strip()
        nc = _extract_new_color(inst)
        colors = _extract_color_words(inst)
        if nc:
            params['new_color'] = nc
            params['mentioned_colors'] = [nc] + [c for c in colors if c != nc and c not in nc][:5]
    elif action == 'replace_object':
        m = re.search(r'replace\s+(?:the\s+)?(.+?)\s+with\s+', inst, re.I)
        if m: target = m.group(1).strip()
        m2 = re.search(r'\bwith\s+(?:a\s+|an\s+)?(?:\w+\s+){0,3}(.+?)(?:\s+(?:throughout|while)|[,.]|$)', inst, re.I)
        params = {'replacement': {'category': m2.group(1).strip().split()[-1].lower() if m2 else 'replacement', 'viewpoint': 'match_source'}}
    elif action == 'replace_background':
        target = 'background'
        m = re.search(r'\bwith\s+(?:a\s+)?(.+?)(?:\s+featuring|[,.]|$)', inst, re.I)
        params = {'new_scene': {'style': re.sub(r'\s+', '_', m.group(1).strip()[:40].lower())} if m else {}}
    elif action == 'add_object':
        for pat in [r'increase\s+the\s+number\s+of\s+(.+?)\s+(?:in|by|to|on|with)',
                    r'(?:add|insert)\s+(?:a|an|another|additional)\s+(.+?)(?:\s+(?:to|in|on)|[,.]|$)',
                    r'(?:place|introduce)\s+(?:a|an)\s+(.+?)(?:\s+(?:on|in)|[,.]|$)',
                    r'adding\s+(?:a\s+second\s+|more\s+)?(.+?)(?:\s+(?:to|in|on)|[,.]|$)']:
            m = re.search(pat, inst, re.I)
            if m: target = m.group(1).strip(); break
    elif action == 'remove_object':
        m = re.search(r'remove\s+(?:the\s+)?(.+?)(?:\s+(?:from|and)|[,.]|$)', inst, re.I)
        if m: target = m.group(1).strip()
    elif action == 'increase_amount':
        m = re.search(r'increase\s+(?:the\s+)?(?:amount|number)\s+of\s+(.+?)(?:\s+(?:on|in|to)|[,.]|$)', inst, re.I)
        if m: target = m.group(1).strip()
        params = {'density': 'dense'}
    else:
        # For camera/style/motion: use retrieval + smart defaults
        params = default_params(action, inst)
        if action == 'apply_style': target = 'full_frame'
        elif action in ('zoom_in', 'zoom_out'): target = 'camera_view'
        elif action == 'change_camera_angle':
            m = re.search(r'(?:toward|towards|on|looking\s+(?:up|down)\s+at)\s+(?:the\s+)?([a-z][a-z\s]{1,25}?)(?:\s*[,;.]|$)', t)
            if m: target = m.group(1).strip()
    return target, params

def nearest_same_action(inst, action, pool, skip_video):
    best, best_s = None, -1.0
    for c in pool:
        if c['video_path'] == skip_video: continue
        if c['gt_primary']['action'] != action: continue
        s = text_similarity(inst, c['instruction'])
        if s > best_s: best_s, best = s, c
    return best, best_s

def predict_primary_task(inst, pool, skip_video):
    """Return (primary_task_dict, nearest_record, similarity) using v15d logic."""
    action = infer_action(inst)
    target, params = extract_primary_target_params(action, inst)

    # For non-extraction actions, use retrieval
    if action not in {'change_color', 'replace_object', 'replace_background',
                      'add_object', 'remove_object', 'increase_amount'}:
        near, sim = nearest_same_action(inst, action, pool, skip_video)
        if near is not None and sim > 0.15:
            gt = near['gt_primary']
            target = gt.get('target', target)
            merged = copy.deepcopy(gt.get('params', {}))
            for k, v in params.items(): merged[k] = v
            params = merged
        # smart overrides
        if action == 'apply_style': target = 'full_frame'
        elif action in ('zoom_in', 'zoom_out'): target = 'camera_view'
        elif action == 'orbit_camera': params['trajectory'] = 'arc'
        elif action == 'edit_motion' and target in ('subject', 'full_frame', 'camera_view', 'face'):
            target = 'person'
        elif action == 'add_effect': params.setdefault('effect_type', 'glow_or_decoration')
        near_for_aux = near
        sim_for_aux = sim if near is not None else 0.0
    else:
        near_for_aux, sim_for_aux = nearest_same_action(inst, action, pool, skip_video)

    task = {'action': action, 'target': target, 'constraints': [], 'params': params}
    return task, near_for_aux, sim_for_aux

print('v15d primary predictor ready')


# ## 3. v16a — deterministic auxiliary tasks
# - primary action ごとに、上位 aux actions を固定で追加する。
# - target は primary target を引き継ぎ、params は空。

# In[5]:


# Auxiliary task map: primary_action -> [(aux_action, target_fn)]
# target_fn: 'primary' = use primary target, or a fixed string
_AUX_MAP = {
    'add_effect': ['match_effect_lighting', 'stabilize_effect'],
    'add_object': ['match_appearance', 'blend_instances', 'stabilize_instances'],
    'apply_style': ['enhance_style_details', 'stabilize_style', 'preserve_layout'],
    'change_camera_angle': ['adjust_perspective', 'refine_mask'],
    'change_color': ['preserve_material_appearance', 'preserve_objects', 'refine_mask'],
    'dolly_in': ['preserve_framing'],
    'edit_motion': ['stabilize_motion', 'refine_mask', 'preserve_objects', 'preserve_identity'],
    'increase_amount': ['match_appearance', 'stabilize_instances'],
    'orbit_camera': ['stabilize_motion'],
    'remove_object': ['inpaint_background', 'stabilize_inpaint'],
    'replace_background': ['match_lighting', 'refine_mask', 'stabilize_composite',
                           'match_background_camera_properties', 'preserve_foreground'],
    'replace_object': ['align_replacement', 'match_appearance', 'refine_mask'],
    'zoom_in': ['stabilize_motion', 'preserve_focus'],
    'zoom_out': ['preserve_focus', 'stabilize_motion'],
}

# Special targets for certain aux actions
_AUX_TARGET_OVERRIDE = {
    'enhance_style_details': 'full_frame',
    'stabilize_style': 'full_frame',
    'stabilize_composite': 'full_frame',
    'preserve_layout': 'scene_layout',
    'match_background_camera_properties': 'background',
    'preserve_foreground': 'foreground',
    'inpaint_background': 'background',
}

def make_aux_task(aux_action, primary_target):
    target = _AUX_TARGET_OVERRIDE.get(aux_action, primary_target)
    return {'action': aux_action, 'target': target, 'constraints': [], 'params': {}}

def predict_v16a(record):
    inst = record['instruction']
    primary, _, _ = predict_primary_task(inst, base_records, record['video_path'])
    aux_actions = _AUX_MAP.get(primary['action'], [])
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]
    return {'tasks': tasks}

res_v16a = evaluate_multi(base_records, predict_v16a)
print('v16a (deterministic aux):', res_v16a['overall'])


# ## 4. v16b — action-filtered retrieval で全 task list を転用
# - nearest same-action 事例の GT task list を丸ごとコピー。
# - primary task のみ v15d 予測で置き換える。
# - これにより aux task の種類・数・target が実際の GT パターンに近くなる。

# In[6]:


def predict_v16b(record):
    inst = record['instruction']
    primary, near, sim = predict_primary_task(inst, base_records, record['video_path'])

    if near is not None and sim > 0.10:
        # Use nearest record's full GT task list, replacing task[0] with our prediction
        aux_from_retrieval = near['gt_tasks'][1:]  # skip primary task of neighbor
        tasks = [primary] + copy.deepcopy(aux_from_retrieval)
    else:
        # Fallback to deterministic
        aux_actions = _AUX_MAP.get(primary['action'], [])
        tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

    return {'tasks': tasks}

res_v16b = evaluate_multi(base_records, predict_v16b)
print('v16b (retrieval all tasks):', res_v16b['overall'])


# ## 5. v16c — v16a + count-adaptive (GT 平均 count に合わせてトリム/拡張)
# - v16a の deterministic aux tasks を GT count に近づける。
# - 1-task GT records: aux tasks なし

# In[7]:


# Check nearest count for known records (sim-based)
def predict_v16c(record):
    inst = record['instruction']
    primary, near, sim = predict_primary_task(inst, base_records, record['video_path'])

    # Estimate expected task count from nearest same-action example
    if near is not None:
        expected_total = len(near['gt_tasks'])
    else:
        # fallback: action-based average
        _AVG_COUNT = {'apply_style': 3, 'replace_background': 5, 'edit_motion': 4,
                      'zoom_in': 3, 'change_color': 3, 'add_object': 3,
                      'change_camera_angle': 2, 'dolly_in': 2, 'add_effect': 3,
                      'replace_object': 3, 'remove_object': 3, 'zoom_out': 2,
                      'orbit_camera': 1, 'increase_amount': 1}
        expected_total = _AVG_COUNT.get(primary['action'], 3)

    aux_actions = _AUX_MAP.get(primary['action'], [])
    # Trim or keep aux actions to match expected_total - 1
    n_aux = max(0, expected_total - 1)
    picked_aux = aux_actions[:n_aux]
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in picked_aux]
    return {'tasks': tasks}

res_v16c = evaluate_multi(base_records, predict_v16c)
print('v16c (count-adaptive):', res_v16c['overall'])


# In[8]:


known_summary = {
    'v16a_deterministic_aux': res_v16a['overall'],
    'v16b_retrieval_all_tasks': res_v16b['overall'],
    'v16c_count_adaptive': res_v16c['overall'],
}

print('=== known multi-task benchmark (instruction-only, ver16) ===')
for name, score in sorted(known_summary.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name:40s}  {score}')

print()
print('--- ver13 baseline (uses class/subclass) ---')
print('v13c_retrieval_transfer: total=0.7393 (known), 0.8830 (unknown)')

# Per-action breakdown for best model
best_name = max(known_summary, key=lambda x: known_summary[x]['total'])
best_res = {'v16a': res_v16a, 'v16b': res_v16b, 'v16c': res_v16c}[best_name[:4]]
print(f'\nPer-action ({best_name}):')
by_action = defaultdict(list)
for row in best_res['rows']:
    key = row['prediction_key']
    rec = next(r for r in base_records if r['prediction_key'] == key)
    by_action[rec['gt_primary']['action']].append(row['total'])
for action in sorted(by_action, key=lambda a: sum(by_action[a])/len(by_action[a])):
    s = by_action[action]
    print(f'  {action:25s} n={len(s):2d}  avg={sum(s)/len(s):.4f}')


# ## 6. 未知instruction 評価 + 保存

# In[9]:


predictors = {
    'v16a_deterministic_aux': predict_v16a,
    'v16b_retrieval_all_tasks': predict_v16b,
    'v16c_count_adaptive': predict_v16c,
}

unknown_results = {name: evaluate_multi(unknown_records, fn) for name, fn in predictors.items()}
unknown_summary = {name: r['overall'] for name, r in unknown_results.items()}

print('=== unknown multi-task benchmark (instruction-only, ver16) ===')
for name, score in sorted(unknown_summary.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name:40s}  {score}')

best_unknown_name = max(unknown_summary, key=lambda x: unknown_summary[x]['total'])
print('\nbest:', best_unknown_name)

# Save predictions for best model
best_result = unknown_results[best_unknown_name]
score_by_key = {r['prediction_key']: r for r in best_result['rows']}
rows_out = []
for rec in unknown_records:
    key = rec['prediction_key']
    rows_out.append({
        'prediction_key': key,
        'video_path': rec['video_path'],
        'variant': rec.get('variant', 'base'),
        'instruction': rec['instruction'],
        'gt_tasks': rec['gt_tasks'],
        'prediction': best_result['predictions'][key],
        'scores': score_by_key[key],
    })
(OUTPUT_DIR / f'unknown_predictions_{best_unknown_name}.json').write_text(
    json.dumps(rows_out, ensure_ascii=False, indent=2), encoding='utf-8')

analysis = {
    'known_benchmark': known_summary,
    'unknown_benchmark': unknown_summary,
    'best_model': best_unknown_name,
}
(OUTPUT_DIR / 'analysis_ver16.json').write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved!')


# ## 7. エラー分析 (worst 5 records)

# ## 8. 追加分析A: action別の効き方を可視化する意図
# - 背景: v16a/v16b/v16c の総合スコア差だけでは、どの action で retrieval が有効か分からない。
# - 目的: action単位の増減と類似度分布を確認し、v16d の gating 方針を決める。
# - 判定基準: retrieval が改善する action と悪化する action を分離できるか。

# In[17]:


worst = sorted(unknown_results[best_unknown_name]['rows'], key=lambda x: x['total'])[:5]
print(f'worst 5: {best_unknown_name}')
for row in worst:
    key = row['prediction_key']
    rec = next(r for r in unknown_records if r['prediction_key'] == key)
    print('=' * 80)
    print(f'  scores: {row}')
    print(f'  inst: {rec["instruction"][:90]}')
    print(f'  GT tasks: {[t["action"] for t in rec["gt_tasks"]]}')
    pred_tasks = unknown_results[best_unknown_name]['predictions'][key]['tasks']
    print(f'  PRED tasks: {[t["action"] for t in pred_tasks]}')


# In[10]:


# quick diagnostics for designing v16d/e
print('known totals:', {
    'v16a': res_v16a['overall']['total'],
    'v16b': res_v16b['overall']['total'],
    'v16c': res_v16c['overall']['total'],
})

# per-action mean score for each variant
from collections import defaultdict

def per_action_avg(res):
    ba = defaultdict(list)
    for row in res['rows']:
        rec = next(r for r in base_records if r['prediction_key'] == row['prediction_key'])
        ba[rec['gt_primary']['action']].append(row['total'])
    return {k: sum(v)/len(v) for k, v in ba.items()}

avgs = {
    'v16a': per_action_avg(res_v16a),
    'v16b': per_action_avg(res_v16b),
    'v16c': per_action_avg(res_v16c),
}

# print biggest action gaps where retrieval helped/hurt
actions = sorted(set(avgs['v16a']) | set(avgs['v16b']) | set(avgs['v16c']))
print('\naction delta (v16b-v16a, v16c-v16a):')
for a in actions:
    a0 = avgs['v16a'].get(a, 0.0)
    b0 = avgs['v16b'].get(a, 0.0)
    c0 = avgs['v16c'].get(a, 0.0)
    print(f"{a:25s}  a={a0:.4f}  b={b0:.4f} ({b0-a0:+.4f})  c={c0:.4f} ({c0-a0:+.4f})")

# check reliability by nearest similarity
sims = []
for rec in base_records:
    _, near, sim = predict_primary_task(rec['instruction'], base_records, rec['video_path'])
    sims.append(sim)
print('\nnearest similarity stats:', {
    'min': min(sims),
    'p25': sorted(sims)[len(sims)//4],
    'p50': sorted(sims)[len(sims)//2],
    'p75': sorted(sims)[(len(sims)*3)//4],
    'max': max(sims),
})


# ## 9. 追加分析B: ハイブリッド候補を閾値探索する意図
# - 背景: retrieval 全適用は一部 action で悪化するため、適用条件を制限したい。
# - 目的: 「retrieval有効actionのみ + 類似度閾値」を sweep し、安定した改善点を見つける。
# - 判定基準: known total が v16b を上回る閾値帯を確認する。

# In[11]:


# candidate search for v16d/e
_HELP_B = {
    'add_effect', 'apply_style', 'change_camera_angle', 'edit_motion',
    'remove_object', 'zoom_in'
}

_HELP_C = {'increase_amount', 'orbit_camera'}

def predict_hybrid_action_gated(record, sim_th=0.24):
    inst = record['instruction']
    primary, near, sim = predict_primary_task(inst, base_records, record['video_path'])
    action = primary['action']

    # Default deterministic
    aux_actions = _AUX_MAP.get(action, [])
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

    # Prefer count-adaptive for single-task-like actions
    if action in _HELP_C:
        expected_total = len(near['gt_tasks']) if near is not None else 1
        n_aux = max(0, expected_total - 1)
        tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:n_aux]]

    # Use retrieval transfer only for actions where it helped and confidence is enough
    if action in _HELP_B and near is not None and sim >= sim_th:
        tasks = [primary] + copy.deepcopy(near['gt_tasks'][1:])

    return {'tasks': tasks}

# sweep similarity threshold
best = None
for th in [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    res = evaluate_multi(base_records, lambda r, t=th: predict_hybrid_action_gated(r, t))
    total = res['overall']['total']
    print(f'th={th:.2f} -> total={total:.4f}')
    if best is None or total > best[1]:
        best = (th, total, res)

print('\nbest threshold candidate:', best[0], best[1])


# ## 10. 追加分析C: 上限確認と action別チューニングの意図
# - 背景: どこまで伸ばせるかを把握しないと改善余地が読めない。
# - 目的: (1) v16a/b/c の record-wise オラクル上限を測る、(2) action別閾値を調整する。
# - 判定基準: tuned known total がハイブリッド初期値を上回るか。

# In[12]:


# deeper search: oracle upper bound and per-action threshold tuning
res_map = {'a': res_v16a, 'b': res_v16b, 'c': res_v16c}
rows_by_model = {k: {r['prediction_key']: r for r in v['rows']} for k, v in res_map.items()}

# Oracle best per record among a/b/c
oracle_scores = []
for rec in base_records:
    key = rec['prediction_key']
    best = max(rows_by_model[m][key]['total'] for m in ('a', 'b', 'c'))
    oracle_scores.append(best)
print('oracle record-wise upper bound among v16a/b/c:', sum(oracle_scores) / len(oracle_scores))

# action-wise best fixed model
by_action_best_model = {}
for action in sorted({r['gt_primary']['action'] for r in base_records}):
    keys = [r['prediction_key'] for r in base_records if r['gt_primary']['action'] == action]
    action_means = {}
    for m in ('a', 'b', 'c'):
        action_means[m] = sum(rows_by_model[m][k]['total'] for k in keys) / len(keys)
    best_m = max(action_means, key=action_means.get)
    by_action_best_model[action] = best_m
print('action-wise best model map:', by_action_best_model)

# tune threshold per action for using retrieval-vs-deterministic
sim_grid = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32]

# precompute primary/near/sim to avoid repeated retrieval
cache = {}
for rec in base_records:
    primary, near, sim = predict_primary_task(rec['instruction'], base_records, rec['video_path'])
    cache[rec['prediction_key']] = (primary, near, sim)

def pred_gated_action_threshold(record, th_map):
    key = record['prediction_key']
    primary, near, sim = cache[key]
    action = primary['action']

    # default deterministic (a)
    aux_actions = _AUX_MAP.get(action, [])
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

    # optionally c-like behavior
    if action in _HELP_C:
        expected_total = len(near['gt_tasks']) if near is not None else 1
        tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:max(0, expected_total - 1)]]

    # b-like transfer with action-specific threshold
    th = th_map.get(action, 1.0)
    if near is not None and sim >= th and action in _HELP_B:
        tasks = [primary] + copy.deepcopy(near['gt_tasks'][1:])

    return {'tasks': tasks}

# brute-force tune one threshold per action in HELP_B independently (greedy coordinate)
th_map = {a: 0.26 for a in _HELP_B}
for _ in range(3):
    for action in sorted(_HELP_B):
        best_th, best_score = th_map[action], -1
        for th in sim_grid:
            cand = dict(th_map)
            cand[action] = th
            res = evaluate_multi(base_records, lambda r, m=cand: pred_gated_action_threshold(r, m))
            if res['overall']['total'] > best_score:
                best_score = res['overall']['total']
                best_th = th
        th_map[action] = best_th

res_tuned = evaluate_multi(base_records, lambda r: pred_gated_action_threshold(r, th_map))
print('tuned threshold map:', th_map)
print('tuned known total:', res_tuned['overall']['total'])


# ## 11. 追加分析D: action専門家ミックスの意図
# - 背景: actionごとに v16a/b/c の勝ち筋が異なる。
# - 目的: 予測primary actionごとに最適手法を固定選択した時の効果を検証する。
# - 判定基準: 固定ミックスが閾値ハイブリッドを上回るか。

# In[13]:


# try action-expert mixture among v16a/b/c (using predicted primary action)
ACTION_EXPERT = {
    'add_effect': 'b', 'add_object': 'a', 'apply_style': 'b', 'change_camera_angle': 'b',
    'change_color': 'a', 'dolly_in': 'a', 'edit_motion': 'b', 'increase_amount': 'c',
    'orbit_camera': 'c', 'remove_object': 'b', 'replace_background': 'a',
    'replace_object': 'c', 'zoom_in': 'b', 'zoom_out': 'a'
}

def predict_mix_abc(record):
    inst = record['instruction']
    primary, near, sim = predict_primary_task(inst, base_records, record['video_path'])
    action = primary['action']
    mode = ACTION_EXPERT.get(action, 'a')

    if mode == 'b' and near is not None:
        tasks = [primary] + copy.deepcopy(near['gt_tasks'][1:])
        return {'tasks': tasks}

    if mode == 'c':
        aux_actions = _AUX_MAP.get(action, [])
        expected_total = len(near['gt_tasks']) if near is not None else 1
        n_aux = max(0, expected_total - 1)
        tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:n_aux]]
        return {'tasks': tasks}

    # mode a
    aux_actions = _AUX_MAP.get(action, [])
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]
    return {'tasks': tasks}

res_mix = evaluate_multi(base_records, predict_mix_abc)
print('mix_abc known total:', res_mix['overall']['total'])


# ## 12. v16d — action-gated retrieval (tuned)
# - 意図: v16b の強みを残しつつ、悪化actionでは deterministic/count-adaptive に戻す。
# - 具体策: retrieval有効actionのみ適用し、action別の類似度閾値を調整した gating を使う。
# - 期待: v16b より分散を抑え、known total を安定して押し上げる。

# In[14]:


# v16d: tuned action-gated retrieval
V16D_HELP_B = {'change_camera_angle', 'edit_motion', 'zoom_in', 'apply_style', 'add_effect', 'remove_object'}
V16D_HELP_C = {'increase_amount', 'orbit_camera'}
V16D_TH_MAP = {
    'change_camera_angle': 0.12,
    'edit_motion': 0.28,
    'zoom_in': 0.12,
    'apply_style': 0.28,
    'add_effect': 0.12,
    'remove_object': 0.12,
}

def predict_v16d(record):
    inst = record['instruction']
    primary, near, sim = predict_primary_task(inst, base_records, record['video_path'])
    action = primary['action']

    aux_actions = _AUX_MAP.get(action, [])
    tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

    # c-like handling for single-task biased actions
    if action in V16D_HELP_C:
        expected_total = len(near['gt_tasks']) if near is not None else 1
        n_aux = max(0, expected_total - 1)
        tasks = [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:n_aux]]

    # retrieval transfer only where it is known to help and similarity is enough
    th = V16D_TH_MAP.get(action, 1.0)
    if near is not None and action in V16D_HELP_B and sim >= th:
        tasks = [primary] + copy.deepcopy(near['gt_tasks'][1:])

    return {'tasks': tasks}

res_v16d = evaluate_multi(base_records, predict_v16d)
print('v16d (tuned action-gated):', res_v16d['overall'])


# ## 13. v16e — neighbor-voted model selector (a/b/c/d)
# - 意図: 1つの固定ルールではなく、近傍事例で「どのモデルが当たりやすいか」を動的に選ぶ。
# - 具体策: 近傍の既知レコードで各モデルの勝者を集計し、重み付き投票で mode を選択。
# - 期待: action内でも instruction の揺れに追従し、0.8 に近づく。

# In[16]:


# v16e: select model by neighbor-voted winner among a/b/c/d

def _build_tasks_by_mode(mode, primary, near, sim_local):
    action = primary['action']
    aux_actions = _AUX_MAP.get(action, [])

    if mode == 'b' and near is not None:
        return [primary] + copy.deepcopy(near['gt_tasks'][1:])

    if mode == 'c':
        expected_total = len(near['gt_tasks']) if near is not None else 1
        n_aux = max(0, expected_total - 1)
        return [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:n_aux]]

    if mode == 'd':
        # reuse v16d policy
        th = V16D_TH_MAP.get(action, 1.0)
        if near is not None and action in V16D_HELP_B and sim_local >= th:
            return [primary] + copy.deepcopy(near['gt_tasks'][1:])
        if action in V16D_HELP_C:
            expected_total = len(near['gt_tasks']) if near is not None else 1
            n_aux = max(0, expected_total - 1)
            return [primary] + [make_aux_task(a, primary['target']) for a in aux_actions[:n_aux]]
        return [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

    # mode a
    return [primary] + [make_aux_task(a, primary['target']) for a in aux_actions]

# prepare winner labels on known records
rows_map_all = {
    'a': {r['prediction_key']: r for r in res_v16a['rows']},
    'b': {r['prediction_key']: r for r in res_v16b['rows']},
    'c': {r['prediction_key']: r for r in res_v16c['rows']},
    'd': {r['prediction_key']: r for r in res_v16d['rows']},
}
winner_by_key = {}
for rec in base_records:
    key = rec['prediction_key']
    winner = max(('a', 'b', 'c', 'd'), key=lambda m: rows_map_all[m][key]['total'])
    winner_by_key[key] = winner


def _choose_mode_by_neighbors(inst, video_path, action, topk=15):
    sims = []
    for r in base_records:
        if r['video_path'] == video_path:
            continue
        # prefer same-action neighbors (estimated by GT primary on known pool)
        if r['gt_primary']['action'] != action:
            continue
        s = text_similarity(inst, r['instruction'])
        sims.append((s, r))
    if not sims:
        for r in base_records:
            if r['video_path'] == video_path:
                continue
            s = text_similarity(inst, r['instruction'])
            sims.append((s, r))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:topk]

    votes = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
    for s, r in top:
        w = max(0.0, s)
        votes[winner_by_key[r['prediction_key']]] += w

    # fallback if all non-positive
    if max(votes.values()) <= 0:
        return 'd'
    return max(votes, key=votes.get)


def predict_v16e(record):
    inst = record['instruction']
    primary, near, sim_local = predict_primary_task(inst, base_records, record['video_path'])
    action = primary['action']
    mode = _choose_mode_by_neighbors(inst, record['video_path'], action, topk=15)
    tasks = _build_tasks_by_mode(mode, primary, near, sim_local)
    return {'tasks': tasks}

res_v16e = evaluate_multi(base_records, predict_v16e)
print('v16e (neighbor-voted selector):', res_v16e['overall'])

# compare all candidates found in this notebook
known_summary_ext = {
    'v16a': res_v16a['overall']['total'],
    'v16b': res_v16b['overall']['total'],
    'v16c': res_v16c['overall']['total'],
    'v16d': res_v16d['overall']['total'],
    'v16e': res_v16e['overall']['total'],
}
print('known totals:', known_summary_ext)


# ## 14. unknown再評価 (v16a〜v16e) とログ保存
# - 意図: known で改善した v16d が unknown でも有効かを確認する。
# - 具体策: predictor 集合を拡張して unknown を再評価し、best を保存する。
# - ログ方針: ベンチマーク結果・bestモデル名・保存先をセル出力として残す。

# In[18]:


# unknown benchmark with v16d/e included
predictors_ext = {
    'v16a_deterministic_aux': predict_v16a,
    'v16b_retrieval_all_tasks': predict_v16b,
    'v16c_count_adaptive': predict_v16c,
    'v16d_tuned_action_gated': predict_v16d,
    'v16e_neighbor_voted': predict_v16e,
}

unknown_results_ext = {name: evaluate_multi(unknown_records, fn) for name, fn in predictors_ext.items()}
unknown_summary_ext = {name: r['overall'] for name, r in unknown_results_ext.items()}

print('=== unknown multi-task benchmark (instruction-only, ver16 extended) ===')
for name, score in sorted(unknown_summary_ext.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name:36s}  {score}')

best_unknown_ext = max(unknown_summary_ext, key=lambda x: unknown_summary_ext[x]['total'])
print('\nbest_unknown_ext:', best_unknown_ext)

# Save predictions for best extended model
best_ext_res = unknown_results_ext[best_unknown_ext]
score_by_key_ext = {r['prediction_key']: r for r in best_ext_res['rows']}
rows_out_ext = []
for rec in unknown_records:
    key = rec['prediction_key']
    rows_out_ext.append({
        'prediction_key': key,
        'video_path': rec['video_path'],
        'variant': rec.get('variant', 'base'),
        'instruction': rec['instruction'],
        'gt_tasks': rec['gt_tasks'],
        'prediction': best_ext_res['predictions'][key],
        'scores': score_by_key_ext[key],
    })

pred_path_ext = OUTPUT_DIR / f'unknown_predictions_{best_unknown_ext}.json'
pred_path_ext.write_text(json.dumps(rows_out_ext, ensure_ascii=False, indent=2), encoding='utf-8')

analysis_ext = {
    'known_totals': known_summary_ext,
    'unknown_benchmark': unknown_summary_ext,
    'best_unknown_model': best_unknown_ext,
}
analysis_path_ext = OUTPUT_DIR / 'analysis_ver16_extended.json'
analysis_path_ext.write_text(json.dumps(analysis_ext, ensure_ascii=False, indent=2), encoding='utf-8')

print('saved prediction file:', pred_path_ext)
print('saved analysis file  :', analysis_path_ext)

