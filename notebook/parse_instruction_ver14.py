#!/usr/bin/env python
# coding: utf-8

# # parse_instruction_ver14 (instruction only)
# 
# ## 目的
# - コンペ制約に合わせて、入力を `instruction` のみに限定した予測系へ修正する。
# - 旧ver10/ver11相当の流れを `*_only_inst_input` として再実装し、比較する。
# 
# ## 制約
# - 予測ロジック内で `class` / `subclass` を使わない。
# - 使用可能入力は instruction 文字列のみ。
# 
# ## 比較モデル
# - `ver10_only_inst_input`
# - `ver11a_ruleplus_only_inst_input`
# - `ver11b_retrieval_only_inst_input`
# - `ver11d_ensemble_only_inst_input`

# In[1]:


from __future__ import annotations

import copy
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from pprint import pprint

WORKSPACE = Path('/workspace')
SRC_DIR = WORKSPACE / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parse.data_loading import load_base_records, load_grouped_unknown_records

DATA_DIR = WORKSPACE / 'data'
NOTEBOOK_DIR = WORKSPACE / 'notebook'
OUTPUT_DIR = NOTEBOOK_DIR / 'ver14_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / 'annotations.jsonl'
GT_PATH = DATA_DIR / 'annotations_gt_task_ver09.json'
GROUPED_PATHS = [DATA_DIR / 'annotations_grouped_ver01.json', DATA_DIR / 'annotations_grouped_ver02.json']

print({'output_dir': str(OUTPUT_DIR)})


# ## 1. データ読み込み
# - 既知データで試行し、未知instructionへ適用する。

# In[2]:


base_records = load_base_records(RAW_PATH, GT_PATH)
unknown_records = load_grouped_unknown_records(GROUPED_PATHS, base_records, instruction_keys=('ver2', 'ver3', 'ver4'))

print('base_records:', len(base_records))
print('unknown_records:', len(unknown_records))
print('sample instruction:')
print(base_records[0]['instruction'])


# ## 2. ユーティリティ（instruction only）

# In[3]:


def normalize_text(x):
    if x is None:
        return ''
    t = str(x).lower().replace('_', ' ').strip()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def flatten_json(v, prefix=''):
    out = {}
    if isinstance(v, dict):
        for k, c in v.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            out.update(flatten_json(c, p))
    elif isinstance(v, list):
        for i, c in enumerate(v):
            p = f'{prefix}[{i}]'
            out.update(flatten_json(c, p))
    else:
        out[prefix] = normalize_text(v)
    return out

def score_primary(pred, gt_primary):
    task = pred.get('tasks', [{}])[0]

    action = 1.0 if task.get('action', '') == gt_primary.get('action', '') else 0.0

    pt = task.get('target', '')
    gt = gt_primary.get('target', '')
    if isinstance(pt, list):
        pt = ' '.join(normalize_text(x) for x in pt)
    else:
        pt = normalize_text(pt)
    if isinstance(gt, list):
        gt = ' '.join(normalize_text(x) for x in gt)
    else:
        gt = normalize_text(gt)

    target = 1.0 if (pt and gt and (pt in gt or gt in pt)) else 0.0

    pp = flatten_json(task.get('params', {}))
    gp = flatten_json(gt_primary.get('params', {}))
    if not gp:
        params = 1.0
    elif not pp:
        params = 0.0
    else:
        m = 0
        for k, gv in gp.items():
            pv = pp.get(k, '')
            if pv and (pv == gv or pv in gv or gv in pv):
                m += 1
        params = m / len(gp)

    total = 0.5 * action + 0.2 * target + 0.3 * params
    return {
        'action_score': round(action, 4),
        'target_score': round(target, 4),
        'params_score': round(params, 4),
        'total': round(total, 4),
    }

def text_similarity(a, b):
    aa = normalize_text(a)
    bb = normalize_text(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    tok_a = set(aa.split())
    tok_b = set(bb.split())
    j = len(tok_a & tok_b) / max(1, len(tok_a | tok_b))
    r = SequenceMatcher(None, aa, bb).ratio()
    return 0.6 * j + 0.4 * r

def evaluate_records(records, predict_fn):
    rows = []
    pred_map = {}
    dbg_map = {}
    for r in records:
        pred, dbg = predict_fn(r)
        key = r['prediction_key']
        pred_map[key] = pred
        dbg_map[key] = dbg
        s = score_primary(pred, r['gt_primary'])
        rows.append({'prediction_key': key, 'video_path': r['video_path'], **s})

    overall = {k: round(sum(x[k] for x in rows) / len(rows), 4) for k in ['action_score', 'target_score', 'params_score', 'total']}
    return {'rows': rows, 'overall': overall, 'predictions': pred_map, 'debug': dbg_map}

print('utility ready')


# ## 3. ver10_only_inst_input（class/subclass 非使用）

# In[4]:


def infer_action_from_instruction(inst):
    t = inst.lower()

    if 'replace the' in t and 'background' in t:
        return 'replace_background'
    if ('replace' in t and 'with' in t) and ('background' not in t):
        return 'replace_object'
    if 'remove' in t:
        return 'remove_object'
    if 'add' in t or 'insert' in t:
        return 'add_object'
    if 'increase the amount of' in t or 'increase amount' in t:
        return 'increase_amount'
    if 'color' in t and any(k in t for k in ['change', 'transform', 'modify']):
        return 'change_color'
    if 'style' in t or any(k in t for k in ['anime', 'cyberpunk', 'pixel', 'ukiyo', 'ghibli', 'watercolor']):
        return 'apply_style'
    if 'zoom out' in t:
        return 'zoom_out'
    if 'dolly in' in t:
        return 'dolly_in'
    if 'zoom in' in t:
        return 'zoom_in'
    if 'low angle' in t or 'high angle' in t:
        return 'change_camera_angle'
    if any(k in t for k in ['smile', 'expression', 'pensive', 'joyous']):
        return 'edit_expression'
    if any(k in t for k in ['wave', 'nod', 'raise', 'spin', 'rotate', 'turn']):
        return 'edit_motion'
    if any(k in t for k in ['glow', 'effect', 'neon', 'lightning']):
        return 'add_effect'

    return 'edit_motion'

def default_target_for_action(action):
    m = {
        'replace_background': 'background',
        'replace_object': 'object',
        'remove_object': 'object',
        'add_object': 'object',
        'increase_amount': 'object',
        'change_color': 'object_color',
        'apply_style': 'full_frame',
        'zoom_out': 'camera_view',
        'zoom_in': 'subject',
        'dolly_in': 'subject',
        'change_camera_angle': 'subject',
        'edit_expression': 'face',
        'edit_motion': 'person',
        'add_effect': 'subject',
    }
    return m.get(action, 'subject')

def parse_target_params(inst, action):
    t = inst.lower()
    target = default_target_for_action(action)
    params = {}

    if action == 'replace_background':
        m = re.search(r'replace\s+(?:the\s+)?(?:entire\s+)?(?:solid\s+|plain\s+|blurred\s+|blurry\s+)?(?:[a-z]+\s+)?background(?: behind [^,.;]+)?\s+with\s+(.+?)(?:\.|$)', t)
        if m:
            params['new_scene'] = {'style': m.group(1).strip().replace(' ', '_')}

    if action == 'replace_object':
        m = re.search(r'replace\s+(?:the\s+)?(.+?)\s+with\s+(?:a|an|the)?\s*(.+?)(?: throughout| during| while|\.|$)', t)
        if m:
            target = m.group(1).strip()
            params['replacement'] = {'category': m.group(2).strip().split()[-1], 'viewpoint': 'match_source'}

    if action in {'zoom_in', 'zoom_out', 'dolly_in'}:
        if action == 'zoom_out':
            params['motion_type'] = 'zoom_out'
            target = 'camera_view'
        else:
            params['motion_type'] = action

    if action == 'change_color':
        m = re.search(r'(?:change|modify|transform)\s+(?:the\s+)?(?:color of\s+)?(.+?)\s+to\s+(.+?)(?: throughout| during| while|\.|$)', t)
        if m:
            target = m.group(1).strip()
            params['new_color'] = m.group(2).strip().split()[0]

    if action == 'edit_motion':
        if 'wave' in t:
            params['gesture'] = 'wave'
        elif 'nod' in t:
            params['gesture'] = 'nod'
        elif 'spin' in t:
            params['gesture'] = 'spin'

    if action == 'edit_expression':
        if 'smile' in t:
            params['to_expression'] = 'smile'

    return target, params

def predict_ver10_only_inst_input(record):
    inst = record['instruction']
    action = infer_action_from_instruction(inst)
    target, params = parse_target_params(inst, action)
    pred = {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]}
    dbg = {'version': 'ver10_only_inst_input'}
    return pred, dbg

res_v10_only = evaluate_records(base_records, predict_ver10_only_inst_input)
print('ver10_only_inst_input:', res_v10_only['overall'])


# ## 4. ver11a_ruleplus_only_inst_input
# - preserve/stabilize などの保護系 action を instruction から追加。

# In[5]:


def add_ruleplus_tasks(inst, tasks):
    t = inst.lower()
    out = copy.deepcopy(tasks)

    if any(k in t for k in ['preserve identity', 'keep identity', 'identity']) and any(k in t for k in ['replace', 'change', 'edit']):
        out.append({'action': 'preserve_identity', 'target': 'subject_identity', 'constraints': [], 'params': {}})

    if any(k in t for k in ['no flicker', 'without flicker', 'flickering', 'temporally consistent', 'stable']):
        out.append({'action': 'stabilize_edit', 'target': 'full_frame', 'constraints': [], 'params': {}})

    return out

def predict_ver11a_ruleplus_only_inst_input(record):
    p0, d0 = predict_ver10_only_inst_input(record)
    tasks = add_ruleplus_tasks(record['instruction'], p0['tasks'])
    return {'tasks': tasks}, {'version': 'ver11a_ruleplus_only_inst_input', 'base': d0['version']}

res_v11a_only = evaluate_records(base_records, predict_ver11a_ruleplus_only_inst_input)
print('ver11a_ruleplus_only_inst_input:', res_v11a_only['overall'])


# ## 5. ver11b_retrieval_only_inst_input
# - instruction 類似度のみで近傍例を探索し、target/params 形状を補助。

# In[6]:


def nearest_by_instruction(inst, pool, skip_video):
    best = None
    best_s = -1.0
    for c in pool:
        if c['video_path'] == skip_video:
            continue
        s = text_similarity(inst, c['instruction'])
        if s > best_s:
            best_s = s
            best = c
    return best, best_s

def predict_ver11b_retrieval_only_inst_input(record):
    p0, _ = predict_ver10_only_inst_input(record)
    task = copy.deepcopy(p0['tasks'][0])

    near, sim = nearest_by_instruction(record['instruction'], base_records, record['video_path'])
    if near is not None:
        gt_primary = near['gt_primary']
        if task['target'] in {'subject', 'object', 'object_color', 'person'}:
            task['target'] = gt_primary.get('target', task['target'])
        if not task['params']:
            task['params'] = copy.deepcopy(gt_primary.get('params', {}))

    return {'tasks': [task]}, {
        'version': 'ver11b_retrieval_only_inst_input',
        'nearest_similarity': round(sim, 4) if near is not None else 0.0,
        'nearest_video': near['video_path'] if near is not None else None,
    }

res_v11b_only = evaluate_records(base_records, predict_ver11b_retrieval_only_inst_input)
print('ver11b_retrieval_only_inst_input:', res_v11b_only['overall'])


# ## 6. ver11d_ensemble_only_inst_input
# - instruction only の `v11a` と `v11b` から1つ選択。

# In[7]:


def predict_ver11d_ensemble_only_inst_input(record):
    p1, d1 = predict_ver11a_ruleplus_only_inst_input(record)
    p2, d2 = predict_ver11b_retrieval_only_inst_input(record)

    s2 = d2.get('nearest_similarity', 0.0)
    t = record['instruction'].lower()
    preserve_dense = any(k in t for k in ['preserve', 'maintain', 'keep', 'stable', 'flicker'])

    use_retrieval = (s2 >= 0.72) and (not preserve_dense)
    chosen = p2 if use_retrieval else p1
    src = 'ver11b_retrieval_only_inst_input' if use_retrieval else 'ver11a_ruleplus_only_inst_input'

    return chosen, {'version': 'ver11d_ensemble_only_inst_input', 'selected_from': src, 'nearest_similarity': s2}

res_v11d_only = evaluate_records(base_records, predict_ver11d_ensemble_only_inst_input)
print('ver11d_ensemble_only_inst_input:', res_v11d_only['overall'])

known_summary = {
    'ver10_only_inst_input': res_v10_only['overall'],
    'ver11a_ruleplus_only_inst_input': res_v11a_only['overall'],
    'ver11b_retrieval_only_inst_input': res_v11b_only['overall'],
    'ver11d_ensemble_only_inst_input': res_v11d_only['overall'],
}
print('\nknown benchmark (instruction-only)')
for k, v in sorted(known_summary.items(), key=lambda kv: kv[1]['total'], reverse=True):
    print(k, v)


# ## 7. 未知instruction評価 + 保存

# In[8]:


predictors = {
    'ver10_only_inst_input': predict_ver10_only_inst_input,
    'ver11a_ruleplus_only_inst_input': predict_ver11a_ruleplus_only_inst_input,
    'ver11b_retrieval_only_inst_input': predict_ver11b_retrieval_only_inst_input,
    'ver11d_ensemble_only_inst_input': predict_ver11d_ensemble_only_inst_input,
}

unknown_results = {name: evaluate_records(unknown_records, fn) for name, fn in predictors.items()}
unknown_summary = {name: result['overall'] for name, result in unknown_results.items()}

print('unknown benchmark (instruction-only)')
for k, v in sorted(unknown_summary.items(), key=lambda kv: kv[1]['total'], reverse=True):
    print(k, v)

best_name = max(unknown_summary, key=lambda x: unknown_summary[x]['total'])
print('best:', best_name)

for name, result in unknown_results.items():
    rows = []
    score_by_key = {r['prediction_key']: r for r in result['rows']}
    for rec in unknown_records:
        key = rec['prediction_key']
        rows.append({
            'prediction_key': key,
            'video_path': rec['video_path'],
            'variant': rec.get('variant', 'base'),
            'variant_source': rec.get('variant_source', 'base'),
            'instruction': rec['instruction'],
            'gt_primary': rec['gt_primary'],
            'prediction': result['predictions'][key],
            'debug': result['debug'][key],
            'scores': {
                'action_score': score_by_key[key]['action_score'],
                'target_score': score_by_key[key]['target_score'],
                'params_score': score_by_key[key]['params_score'],
                'total': score_by_key[key]['total'],
            },
        })
    path = OUTPUT_DIR / f'unknown_predictions_{name}.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

analysis = {
    'constraint': 'instruction only input (class/subclass unused)',
    'known_benchmark': known_summary,
    'unknown_benchmark': unknown_summary,
    'best_model': best_name,
    'record_count': len(unknown_records),
}
analysis_path = OUTPUT_DIR / 'instruction_only_analysis_ver14.json'
analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved:', analysis_path)


# ## 8. エラー分析（best model）

# In[9]:


best_rows = sorted(unknown_results[best_name]['rows'], key=lambda x: x['total'])
print('worst 5:', best_name)
for row in best_rows[:5]:
    key = row['prediction_key']
    rec = next(r for r in unknown_records if r['prediction_key'] == key)
    pred = unknown_results[best_name]['predictions'][key]['tasks'][0]
    print('=' * 100)
    print('prediction_key:', key)
    print('instruction:', rec['instruction'])
    print('gt_primary:', rec['gt_primary'])
    print('pred:', pred)
    print('scores:', {k: row[k] for k in ['action_score', 'target_score', 'params_score', 'total']})

