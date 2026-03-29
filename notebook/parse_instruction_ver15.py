#!/usr/bin/env python
# coding: utf-8

# # parse_instruction_ver15 (instruction only — improved)
# 
# ## 目的
# - ver14 の課題だった action 推論精度 (0.65) を大幅改善して single task 0.8 を目指す。
# - アクション推論の誤分類パターンを修正し、GT 構造に合わせた target/params 生成を強化する。
# 
# ## 制約
# - 予測ロジック内で `class` / `subclass` を使わない。
# - 使用可能入力は `instruction` 文字列のみ。
# 
# ## 主な修正点 (ver14 → ver15)
# 1. `edit_expression` は GT に存在しない → 削除して `edit_motion` に吸収
# 2. `zoom-in` (ハイフン付き) → `zoom_in` に正しくマッピング
# 3. `apply_style` を transform+スタイルキーワードで高優先度に判定
# 4. `dolly-in` (ハイフン付き) 対応
# 5. `orbit_camera` (arc shot / revolving around) 対応
# 6. `replace_object` vs `replace_background` の判定ロジック修正
# 7. action-filtered retrieval: 同一 action の事例から target/params を補完
# 8. 決定論的 action の smart default: apply_style→full_frame, zoom_in→camera_view 等
# 
# ## 比較モデル
# - `v15a`: 改善済み action 推論 + rule-based target/params
# - `v15b`: v15a + action-filtered retrieval (同 action 最近傍から target/params)
# - `v15c`: v15b + smart defaults (決定論的 action の target/params を上書き)

# In[14]:


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
OUTPUT_DIR = NOTEBOOK_DIR / 'ver15_outputs'
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


# ## 1. ユーティリティ + 改善済み action 推論

# In[3]:


# ---- text utilities ----

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

def score_primary(pred, gt_primary):
    task = pred.get('tasks', [{}])[0]
    action = 1.0 if task.get('action', '') == gt_primary.get('action', '') else 0.0

    pt = task.get('target', '')
    gt = gt_primary.get('target', '')
    pt = ' '.join(normalize_text(x) for x in pt) if isinstance(pt, list) else normalize_text(pt)
    gt = ' '.join(normalize_text(x) for x in gt) if isinstance(gt, list) else normalize_text(gt)
    target = 1.0 if (pt and gt and (pt in gt or gt in pt)) else 0.0

    pp = flatten_json(task.get('params', {}))
    gp = flatten_json(gt_primary.get('params', {}))
    if not gp:
        params = 1.0
    elif not pp:
        params = 0.0
    else:
        m = sum(1 for k, gv in gp.items() if k in pp and (pp[k] == gv or pp[k] in gv or gv in pp[k]))
        params = m / len(gp)

    total = 0.5 * action + 0.2 * target + 0.3 * params
    return {
        'action_score': round(action, 4),
        'target_score': round(target, 4),
        'params_score': round(params, 4),
        'total': round(total, 4),
    }

def evaluate_records(records, predict_fn):
    rows, pred_map, dbg_map = [], {}, {}
    for r in records:
        pred, dbg = predict_fn(r)
        key = r['prediction_key']
        pred_map[key] = pred
        dbg_map[key] = dbg
        s = score_primary(pred, r['gt_primary'])
        rows.append({'prediction_key': key, 'video_path': r['video_path'], **s})
    overall = {k: round(sum(x[k] for x in rows) / len(rows), 4)
               for k in ['action_score', 'target_score', 'params_score', 'total']}
    return {'rows': rows, 'overall': overall, 'predictions': pred_map, 'debug': dbg_map}

# ---- improved action inference ----

_STYLE_VOCAB = [
    'anime', 'cyberpunk', 'pixel art', 'ukiyo', 'ghibli', 'watercolor',
    'oil painting', 'comic style', 'woodblock', '16-bit', 'retro pixel',
    'aesthetic', 'painting style', 'art style', 'american comic',
]

def infer_action_v15(inst):
    t = inst.lower()

    # --- camera motions (most specific first) ---
    if re.search(r'\barc\s+shot\b|\borbit\b|revolving\s+around', t):
        return 'orbit_camera'

    if re.search(r'\bdolly[- ]?in\b', t):
        return 'dolly_in'

    if re.search(r'\bzoom[- ]?out\b', t):
        return 'zoom_out'

    if re.search(r'\bzoom[- ]?in\b', t):
        return 'zoom_in'

    # --- camera angle (before style to avoid "transform into low-angle" issues) ---
    if re.search(r'low[- ]?angle|high[- ]?angle', t):
        return 'change_camera_angle'
    if re.search(r'(?:camera|shot|perspective|view)\s+(?:to|into)\s+(?:a\s+)?(?:low|high)', t):
        return 'change_camera_angle'

    # --- style editing (before color/replace to avoid transform→change_color) ---
    style_transform = bool(re.search(
        r'transform\s+(?:the\s+)?(?:entire\s+)?(?:video|scene|frame)', t))
    if (style_transform or re.search(r'apply\s+(?:a\s+)?(?:\w+\s+)*(?:art\s+)?style', t)):
        if any(w in t for w in _STYLE_VOCAB):
            return 'apply_style'
    if any(w in t for w in _STYLE_VOCAB):
        return 'apply_style'

    # --- replace: background target first ---
    if re.search(r'\breplace\b', t):
        if re.search(r'replace\s+(?:the\s+)?(?:(?:\w+\s+){0,3})background', t):
            return 'replace_background'
        if 'with' in t:
            return 'replace_object'

    # --- remove ---
    if re.search(r'\bremove\b', t):
        return 'remove_object'

    # --- quantity: amount vs number ---
    if re.search(r'increase\s+the\s+amount\s+of', t):
        return 'increase_amount'
    if re.search(r'increase\s+the\s+number\s+of', t):
        return 'add_object'

    # --- add effect (glow/aura/flame before generic add) ---
    if re.search(r'(?:add|apply|enhance).*(?:glow(?:ing)?|aura|flame|lightning|'
                 r'sparkle|decoration\s+effect|stage\s+lighting)', t):
        return 'add_effect'
    if re.search(r'(?:glowing|neon\s+electric|neon\s+glow|electric\s+glow)', t):
        return 'add_effect'

    # --- add object ---
    if re.search(r'\b(?:add|insert|place|introduce)\b', t):
        return 'add_object'

    # --- color change ---
    if re.search(r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)*color', t):
        return 'change_color'
    if re.search(r'color\s+(?:of|to)\s+', t) and re.search(r'\b(?:change|modify|transform)\b', t):
        return 'change_color'

    # --- default: edit_motion (includes expression/body movements) ---
    return 'edit_motion'

# ---- verify action accuracy on known records ----
action_results = [(infer_action_v15(r['instruction']), r['gt_primary']['action']) for r in base_records]
action_correct = sum(1 for p, g in action_results if p == g)
print(f'action_correct: {action_correct}/{len(base_records)} = {action_correct/len(base_records):.4f}')
wrong = [(g, p, r['instruction'][:70]) for (p, g), r in zip(action_results, base_records) if p != g]
print('Remaining errors:')
for g, p, inst in wrong:
    print(f'  GT={g:25s} PRED={p:25s} | {inst}')


# ## 2. v15a — improved action inference + rule-based target/params
# - action 推論だけ改善し、target/params は ver14 と同じルールベース。

# In[4]:


def default_target_v15(action):
    return {
        'replace_background': 'background',
        'replace_object': 'object',
        'remove_object': 'object',
        'add_object': 'object',
        'increase_amount': 'object',
        'change_color': 'subject',
        'apply_style': 'full_frame',
        'zoom_out': 'camera_view',
        'zoom_in': 'camera_view',
        'dolly_in': 'subject',
        'change_camera_angle': 'subject',
        'orbit_camera': 'subject',
        'edit_motion': 'person',
        'add_effect': 'subject',
    }.get(action, 'subject')

def default_params_v15(action, inst):
    t = inst.lower()
    params = {}

    if action == 'apply_style':
        style_map = [
            (r'ukiyo', 'ukiyo-e'),
            (r'ghibli', 'ghibli'),
            (r'watercolor', 'watercolor'),
            (r'oil painting', 'oil_painting'),
            (r'comic style|american comic|comic book', 'american_comic_style'),
            (r'\bpixel\b|16-bit|retro pixel|pixel art', 'pixel_art'),
            (r'\banime\b', 'anime'),
            (r'cyberpunk', 'cyberpunk'),
        ]
        for pat, val in style_map:
            if re.search(pat, t):
                params['style'] = val
                break

    elif action in ('zoom_in', 'zoom_out', 'dolly_in'):
        params['motion_type'] = action
        if re.search(r'\b(?:slow|smooth|gradual|subtle|steady)\b', t):
            params['speed'] = 'gradual'

    elif action == 'change_camera_angle':
        if re.search(r'low[- ]?angle', t):
            params['angle'] = 'low_angle'
        elif re.search(r'high[- ]?angle', t):
            params['angle'] = 'high_angle'

    elif action == 'orbit_camera':
        params['trajectory'] = 'arc'

    elif action == 'change_color':
        m = re.search(r'\bto\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+){0,2})(?:\s+color|\s+hue|\.|$)', t)
        if m:
            params['new_color'] = m.group(1).strip()

    elif action == 'edit_motion':
        for g, kw in [('wave', 'wave'), ('nod', 'nod'), ('spin', 'spin'),
                      ('shake', 'shake'), ('hop', 'hop'), ('tilt', 'tilt')]:
            if kw in t:
                params['gesture'] = g
                break

    elif action == 'add_effect':
        params['effect_type'] = 'glow_or_decoration'

    return params

def predict_v15a(record):
    inst = record['instruction']
    action = infer_action_v15(inst)
    target = default_target_v15(action)
    params = default_params_v15(action, inst)
    return {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]}, \
           {'version': 'v15a', 'action': action}

res_v15a = evaluate_records(base_records, predict_v15a)
print('v15a (improved action only):', res_v15a['overall'])


# ## 3. v15b — action-filtered retrieval
# - 同じ predicted action を持つ事例の中から instruction 類似度最高の事例を検索。
# - その GT の target と params を補完として使用する。
# - action が同じなので target/params の構造が適合しやすい。

# In[5]:


def nearest_same_action(inst, action, pool, skip_video):
    """pool の中から同一 action かつ instruction 最近傍を返す (leave-one-out)."""
    best, best_s = None, -1.0
    for c in pool:
        if c['video_path'] == skip_video:
            continue
        c_action = c['gt_primary']['action']  # pool records have GT (known records)
        if c_action != action:
            continue
        s = text_similarity(inst, c['instruction'])
        if s > best_s:
            best_s, best = s, c
    return best, best_s

def predict_v15b(record):
    inst = record['instruction']
    action = infer_action_v15(inst)

    # Start from v15a defaults
    target = default_target_v15(action)
    params = default_params_v15(action, inst)

    # Try action-filtered retrieval to refine target / params
    near, sim = nearest_same_action(inst, action, base_records, record['video_path'])
    if near is not None and sim > 0.15:
        gt = near['gt_primary']
        # Always use Retrieved target (since same-action pool is more relevant)
        target = gt.get('target', target)
        # Merge: start from retrieved params, override with rule-extracted values
        merged = copy.deepcopy(gt.get('params', {}))
        # Rule-extracted params are more specific, so they take precedence
        for k, v in params.items():
            merged[k] = v
        params = merged

    return (
        {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]},
        {'version': 'v15b', 'action': action, 'nearest_sim': round(sim, 4) if near else 0.0,
         'nearest_video': near['video_path'] if near else None},
    )

res_v15b = evaluate_records(base_records, predict_v15b)
print('v15b (action-filtered retrieval):', res_v15b['overall'])


# ## 4. v15c — action-filtered retrieval + smart defaults (deterministic actions)
# - v15b の上に、GT 解析で判明した決定論的パターンを上書きする。
#   - `apply_style` → target='full_frame', params={'style': 抽出値} (GT 100% 一致)
#   - `zoom_in/out` → target='camera_view', params={'motion_type': ..., 'speed': 'gradual'}
#   - `dolly_in` → target='subject' or 抽出, params={'motion_type': 'dolly_in'}
#   - `change_camera_angle` → params={'angle': low/high_angle}
#   - `replace_background` → target='background' (substring match が成立)
#   - `orbit_camera` → params={'trajectory': 'arc'}

# In[6]:


def extract_subject_from_inst(inst):
    """instruction から主要な対象物名語を抽出する簡易ルール."""
    t = inst.lower()
    # Camera ops: find the target being zoomed/panned toward
    m = re.search(r'(?:toward|towards|on|focused?\s+on|focusing\s+on|centered?\s+on)\s+the\s+([a-z][a-z\s]{1,30}?)'
                  r'(?:\s*[,;.]|$)', t)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?:the|a|an)\s+(\w+(?:\s\w+)?)\s+in\s+the\s+(?:foreground|background|scene|center)', t)
    if m:
        return m.group(1).strip()
    # Default noun phrases
    for pron in ['man', 'woman', 'person', 'child', 'baby', 'player', 'presenter', 'chef']:
        if pron in t:
            return pron
    return 'subject'

def smart_target_params(action, inst, v15b_target, v15b_params):
    """決定論的 action の場合は GT パターンで上書きする."""
    target = v15b_target
    params = copy.deepcopy(v15b_params)

    if action == 'apply_style':
        # GT: target は常に 'full_frame'
        target = 'full_frame'
        style_val = params.get('style', '')
        if style_val:
            params = {'style': style_val}
        else:
            params = {'style': default_params_v15('apply_style', inst).get('style', 'unknown')}

    elif action in ('zoom_in', 'zoom_out'):
        # GT: target は主に 'camera_view', params に motion_type + speed
        target = 'camera_view'
        params = {'motion_type': action}
        if re.search(r'\b(?:slow|smooth|gradual|subtle|steady)\b', inst.lower()):
            params['speed'] = 'gradual'

    elif action == 'dolly_in':
        target = extract_subject_from_inst(inst)
        params = {'motion_type': 'dolly_in'}
        if re.search(r'medium\s+shot|from\s+(?:a\s+)?medium', inst.lower()):
            params['start_framing'] = 'medium_shot'
            params['end_framing'] = 'close_up'

    elif action == 'change_camera_angle':
        t = inst.lower()
        angle = 'low_angle' if re.search(r'low[- ]?angle', t) else \
                'high_angle' if re.search(r'high[- ]?angle', t) else None
        if angle:
            params = {'angle': angle}
        # target: keep retrieval value if it's specific, else 'subject'
        if target in ('subject', 'camera_view', '', 'full_frame'):
            target = extract_subject_from_inst(inst)

    elif action == 'replace_background':
        # GT: target 内に 'background' が必ず含まれる → 'background' で substring OK
        target = 'background'
        # params: {'new_scene': {...}} 構造は retrieval から取る
        if 'new_scene' not in params:
            params['new_scene'] = {}

    elif action == 'orbit_camera':
        target = extract_subject_from_inst(inst)
        params = {'trajectory': 'arc'}

    elif action == 'edit_motion':
        # GT target is usually 'person' or specific person noun
        if target in ('subject', 'full_frame', 'camera_view', 'face'):
            target = 'person'

    elif action == 'add_effect':
        # GT: params has effect_type: 'glow_or_decoration'
        params.setdefault('effect_type', 'glow_or_decoration')

    return target, params

def predict_v15c(record):
    inst = record['instruction']
    action = infer_action_v15(inst)

    # Base: action-filtered retrieval
    target = default_target_v15(action)
    params = default_params_v15(action, inst)
    near, sim = nearest_same_action(inst, action, base_records, record['video_path'])
    if near is not None and sim > 0.15:
        gt = near['gt_primary']
        target = gt.get('target', target)
        merged = copy.deepcopy(gt.get('params', {}))
        for k, v in params.items():
            merged[k] = v
        params = merged

    # Override with smart defaults for deterministic actions
    target, params = smart_target_params(action, inst, target, params)

    return (
        {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]},
        {'version': 'v15c', 'action': action, 'nearest_sim': round(sim, 4) if near else 0.0},
    )

res_v15c = evaluate_records(base_records, predict_v15c)
print('v15c (retrieval + smart defaults):', res_v15c['overall'])


# ## 5. 既知データ ベンチマーク比較

# In[7]:


known_summary = {
    'v15a_improved_action': res_v15a['overall'],
    'v15b_action_filtered_retrieval': res_v15b['overall'],
    'v15c_retrieval_smart_defaults': res_v15c['overall'],
}

print('=== known benchmark (instruction-only, ver15) ===')
for name, score in sorted(known_summary.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name:45s}  {score}')

print()
print('--- ver14 baseline for comparison ---')
print('ver14 ver11b_retrieval_only_inst_input  total=0.5649 (action=0.65, target=0.50, params=0.4663)')

# Per-action breakdown for best model
best_model_name = max(known_summary, key=lambda x: known_summary[x]['total'])
best_model_res = {'v15a': res_v15a, 'v15b': res_v15b, 'v15c': res_v15c}[
    best_model_name.split('_')[0]]
print(f'\nPer-action breakdown: {best_model_name}')
from collections import defaultdict
by_action = defaultdict(list)
for row in best_model_res['rows']:
    key = row['prediction_key']
    rec = next(r for r in base_records if r['prediction_key'] == key)
    by_action[rec['gt_primary']['action']].append(row['total'])
for action in sorted(by_action, key=lambda a: sum(by_action[a])/len(by_action[a])):
    scores = by_action[action]
    print(f'  {action:25s} n={len(scores):2d}  avg={sum(scores)/len(scores):.4f}')


# ## 6. スコア分析 — 残課題の特定
# - action ごとの target/params エラーを確認し、さらなる改善余地を探る。

# In[8]:


# Detailed per-record analysis for v15c
print('=== v15c detailed gaps ===')
low_rows = sorted(res_v15c['rows'], key=lambda x: x['total'])[:20]
for row in low_rows:
    key = row['prediction_key']
    rec = next(r for r in base_records if r['prediction_key'] == key)
    pred = res_v15c['predictions'][key]['tasks'][0]
    gt = rec['gt_primary']
    if row['total'] >= 0.8:
        break
    print('===')
    print(f'  scores: {row}')
    print(f'  instruction: {rec["instruction"][:100]}')
    print(f'  GT  action={gt["action"]:25s} target={str(gt.get("target",""))[:40]:40s}')
    print(f'  PRED action={pred["action"]:25s} target={str(pred.get("target",""))[:40]:40s}')
    print(f'  GT params:   {gt.get("params",{})}')
    print(f'  PRED params: {pred.get("params",{})}')


# ## 7. 未知instruction 評価 + 保存

# In[ ]:


predictors = {
    'v15a_improved_action': predict_v15a,
    'v15b_action_filtered_retrieval': predict_v15b,
    'v15c_retrieval_smart_defaults': predict_v15c,
}

unknown_results = {name: evaluate_records(unknown_records, fn) for name, fn in predictors.items()}
unknown_summary = {name: r['overall'] for name, r in unknown_results.items()}

print('=== unknown benchmark (instruction-only, ver15) ===')
for name, score in sorted(unknown_summary.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name:45s}  {score}')

best_name = max(unknown_summary, key=lambda x: unknown_summary[x]['total'])
print('\nbest:', best_name)

# Save outputs
for name, result in unknown_results.items():
    score_by_key = {r['prediction_key']: r for r in result['rows']}
    rows = []
    for rec in unknown_records:
        key = rec['prediction_key']
        rows.append({
            'prediction_key': key,
            'video_path': rec['video_path'],
            'variant': rec.get('variant', 'base'),
            'instruction': rec['instruction'],
            'gt_primary': rec['gt_primary'],
            'prediction': result['predictions'][key],
            'debug': result['debug'][key],
            'scores': {k: score_by_key[key][k]
                       for k in ['action_score', 'target_score', 'params_score', 'total']},
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
(OUTPUT_DIR / 'analysis_ver15.json').write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved:', OUTPUT_DIR)


# ## 8. エラー分析 (best model, unknown benchmark)

# In[ ]:


worst = sorted(unknown_results[best_name]['rows'], key=lambda x: x['total'])[:10]
print(f'worst 10: {best_name}')
for row in worst:
    key = row['prediction_key']
    rec = next(r for r in unknown_records if r['prediction_key'] == key)
    pred = unknown_results[best_name]['predictions'][key]['tasks'][0]
    gt = rec['gt_primary']
    print('=' * 90)
    print(f'  scores: action={row["action_score"]} target={row["target_score"]} params={row["params_score"]} total={row["total"]}')
    print(f'  inst:   {rec["instruction"][:90]}')
    print(f'  GT:     action={gt["action"]:20s} target={str(gt.get("target",""))[:40]:40s}')
    print(f'  PRED:   action={pred["action"]:20s} target={str(pred.get("target",""))[:40]:40s}')


# ## 9. v15d — instruction 抽出ベース (change_color / replace_object / add_object 等)
# 
# ### 修正点
# 1. action 推論 fix: `increase the number` を camera check より先に判定
# 2. replace_background: `{0,6}` ワード制限 (fix for 'wall and blinds background')
# 3. change_color: target/params を instruction から直接抽出
# 4. replace_object: target を instruction から直接抽出
# 5. add_object: target と params の一部を instruction から抽出
# 6. remove_object / increase_amount: target 抽出

# In[9]:


# ---- fixed action inference (v15d) ----
_COLOR_WORDS = [
    'violet', 'blue', 'red', 'green', 'navy', 'emerald', 'burgundy', 'magenta',
    'orange', 'purple', 'pink', 'yellow', 'black', 'white', 'gray', 'grey',
    'silver', 'gold', 'teal', 'cyan', 'indigo', 'maroon', 'beige', 'coral',
    'turquoise', 'aqua', 'lime', 'lavender', 'brown', 'crimson', 'amber',
]

def infer_action_v15d(inst):
    """Improved action inference: quantity checks before camera checks."""
    t = inst.lower()

    # Camera orbit (very specific)
    if re.search(r'\barc\s+shot\b|\borbit\b|revolving\s+around', t):
        return 'orbit_camera'

    # Quantity checks BEFORE camera checks (fix: 'speed bumps in low-angle camera')
    if re.search(r'increase\s+the\s+amount\s+of', t):
        return 'increase_amount'
    if re.search(r'increase\s+the\s+number\s+of', t):
        return 'add_object'

    if re.search(r'\bdolly[- ]?in\b', t):
        return 'dolly_in'
    if re.search(r'\bzoom[- ]?out\b', t):
        return 'zoom_out'
    if re.search(r'\bzoom[- ]?in\b', t):
        return 'zoom_in'

    # Camera angle
    if re.search(r'low[- ]?angle|high[- ]?angle', t):
        return 'change_camera_angle'
    if re.search(r'(?:camera|shot|perspective|view)\s+(?:to|into)\s+(?:a\s+)?(?:low|high)', t):
        return 'change_camera_angle'

    # Style (before color/replace)
    style_transform = bool(re.search(r'transform\s+(?:the\s+)?(?:entire\s+)?(?:video|scene|frame)', t))
    if (style_transform or re.search(r'apply\s+(?:a\s+)?(?:\w+\s+)*(?:art\s+)?style', t)):
        if any(w in t for w in _STYLE_VOCAB):
            return 'apply_style'
    if any(w in t for w in _STYLE_VOCAB):
        return 'apply_style'

    # Replace background: broader word limit {0,6}
    if re.search(r'\breplace\b', t):
        if re.search(r'replace\s+(?:the\s+)?(?:(?:\w+\s+){0,6})background', t):
            return 'replace_background'
        if 'with' in t:
            return 'replace_object'

    if re.search(r'\bremove\b', t):
        return 'remove_object'

    # Add effect (glow/aura before generic add)
    if re.search(r'(?:add|apply|enhance).*(?:glow(?:ing)?|aura|flame|lightning|'
                 r'sparkle|decoration\s+effect|stage\s+lighting)', t):
        return 'add_effect'
    if re.search(r'(?:glowing|neon\s+electric|neon\s+glow|electric\s+glow)', t):
        return 'add_effect'

    if re.search(r'\b(?:add|insert|place|introduce)\b', t):
        return 'add_object'

    if re.search(r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)*color', t):
        return 'change_color'
    if re.search(r'color\s+(?:of|to)\s+', t) and re.search(r'\b(?:change|modify|transform)\b', t):
        return 'change_color'

    return 'edit_motion'

# ---- per-action instruction extraction ----

def _extract_color_words(inst):
    t = inst.lower()
    found = []
    # Multi-word color patterns first
    for pat in [
        r'(?:vibrant\s+)?metallic\s+[a-z]+', r'(?:deep|bright|dark|solid)\s+[a-z]+(?:\s+[a-z]+)?',
        r'neon\s+[a-z]+', r'navy\s+blue', r'emerald\s+green',
    ]:
        for m in re.finditer(pat, t):
            found.append(m.group())
    # Single color words not already covered
    for cw in _COLOR_WORDS:
        if re.search(r'\b' + cw + r'\b', t):
            if not any(cw in f for f in found):
                found.append(cw)
    return found

def _extract_new_color(inst):
    t = inst.lower()
    # 'to a [adj] [color] [color]' phrase
    m = re.search(
        r'\bto\s+(?:a\s+)?(?:(?:vibrant|bright|deep|dark|solid|metallic|neon|bold|glossy|saturated|rich|vivid)\s+){0,2}'
        r'([a-z]+(?:\s+[a-z]+){0,2})\s*(?:color|hue|throughout|while|[,.]|$)', t)
    if m:
        val = m.group(1).strip()
        if val not in {'color', 'hue', 'a', 'an', 'the'}:
            return val
    return ''

def extract_instruction_target_params(action, inst):
    """Extract target/params directly from instruction text for extraction-friendly actions."""
    t = inst.lower()
    target = default_target_v15(action)
    params = {}

    if action == 'change_color':
        # Target: phrase being colored
        m = re.search(r'(?:change|modify)\s+(?:the\s+)?(?:\w+\s+)?color\s+of\s+(?:the\s+)?(.+?)\s+to\s+', inst, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
        else:
            m = re.search(r'(?:change|modify)\s+the\s+(.+?)\s+to\s+', inst, re.IGNORECASE)
            if m:
                target = m.group(1).strip()
        # Params
        new_color = _extract_new_color(inst)
        colors_found = _extract_color_words(inst)
        if new_color:
            params['new_color'] = new_color
            # mentioned_colors: new_color first, then others
            mc = [new_color] + [c for c in colors_found if c != new_color and c not in new_color]
            params['mentioned_colors'] = mc[:6]
        elif colors_found:
            params['mentioned_colors'] = colors_found[:6]

    elif action == 'replace_object':
        m = re.search(r'replace\s+(?:the\s+)?(.+?)\s+with\s+', inst, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
        # Extract replacement category from 'with a/an [X]'
        m2 = re.search(r'\bwith\s+(?:a\s+|an\s+)?(?:bright\s+|vibrant\s+|matching\s+|new\s+)?(.+?)'
                       r'(?:\s+(?:throughout|while|maintaining|to|in|that)|[,.]|$)', inst, re.IGNORECASE)
        category = m2.group(1).strip().split()[-1] if m2 else 'replacement'
        params = {'replacement': {'category': category.lower(), 'viewpoint': 'match_source'}}

    elif action == 'replace_background':
        target = 'background'
        m = re.search(r'\bwith\s+(?:a\s+)?(.+?)(?:\s+featuring|[,.]|$)', inst, re.IGNORECASE)
        scene_desc = m.group(1).strip() if m else ''
        params = {'new_scene': {'style': re.sub(r'\s+', '_', scene_desc[:40].lower())} if scene_desc else {}}

    elif action == 'add_object':
        # Target: the type of object being added
        m = re.search(r'increase\s+the\s+number\s+of\s+(.+?)\s+(?:in|by|to|on|with)', inst, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
        else:
            for pat in [
                r'(?:add|insert)\s+(?:a|an|another|additional)\s+(.+?)(?:\s+(?:to|in|on|into|at)|[,.]|$)',
                r'(?:place|introduce)\s+(?:a|an)\s+(.+?)(?:\s+(?:on|in|at)|[,.]|$)',
                r'adding\s+(?:a\s+second\s+|more\s+|three\s+more\s+)?(.+?)(?:\s+(?:to|in|on|at|next)|[,.]|$)',
            ]:
                m = re.search(pat, inst, re.IGNORECASE)
                if m:
                    target = m.group(1).strip()
                    break
        # Count
        count_m = re.search(r'(?:two|three|four|five|a\s+second|2|3|4|5)\s+more', t)
        if count_m:
            word_to_n = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'a second': 1, '2': 2, '3': 3}
            params['count'] = next((v for k, v in word_to_n.items() if k in count_m.group()), 1)
        # Position
        pos_m = re.findall(r'(?:in|on|at|to)\s+the\s+(background|foreground|desk|center|side|left|right|midground)', t)
        if pos_m:
            params['position'] = pos_m

    elif action == 'remove_object':
        m = re.search(r'remove\s+(?:the\s+)?(.+?)(?:\s+(?:from|and|while)|[,.]|$)', inst, re.IGNORECASE)
        if m:
            target = m.group(1).strip()

    elif action == 'increase_amount':
        m = re.search(r'increase\s+(?:the\s+)?(?:amount|number)\s+of\s+(.+?)(?:\s+(?:on|in|to)|[,.]|$)', inst, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
        params = {'density': 'dense'}

    return target, params

# ---- v15d predictor ----

_EXTRACT_ACTIONS = {'change_color', 'replace_object', 'replace_background',
                    'add_object', 'remove_object', 'increase_amount'}

def predict_v15d(record):
    inst = record['instruction']
    action = infer_action_v15d(inst)

    if action in _EXTRACT_ACTIONS:
        target, params = extract_instruction_target_params(action, inst)
    else:
        # Use action-filtered retrieval + smart defaults
        target = default_target_v15(action)
        params = default_params_v15(action, inst)
        near, sim = nearest_same_action(inst, action, base_records, record['video_path'])
        if near is not None and sim > 0.15:
            gt = near['gt_primary']
            target = gt.get('target', target)
            merged = copy.deepcopy(gt.get('params', {}))
            for k, v in params.items():
                merged[k] = v
            params = merged
        target, params = smart_target_params(action, inst, target, params)

    return (
        {'tasks': [{'action': action, 'target': target, 'constraints': [], 'params': params}]},
        {'version': 'v15d', 'action': action},
    )

res_v15d = evaluate_records(base_records, predict_v15d)
print('v15d (instruction extraction):', res_v15d['overall'])

# Verify action improvement
action_v15d = [(infer_action_v15d(r['instruction']), r['gt_primary']['action']) for r in base_records]
correct_v15d = sum(1 for p, g in action_v15d if p == g)
print(f'v15d action accuracy: {correct_v15d}/100')


# ## 10. 全モデル比較 + 最終スコア確認

# In[10]:


all_known = {
    'v15a': res_v15a['overall'],
    'v15b': res_v15b['overall'],
    'v15c': res_v15c['overall'],
    'v15d': res_v15d['overall'],
}

print('=== known benchmark (instruction-only, all ver15) ===')
for name, score in sorted(all_known.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f'{name}  {score}')

# Per-action for v15d
print('\nPer-action v15d:')
from collections import defaultdict as _dd
by_action_d = _dd(list)
by_action_d_scores = _dd(list)
for row in res_v15d['rows']:
    key = row['prediction_key']
    rec = next(r for r in base_records if r['prediction_key'] == key)
    act = rec['gt_primary']['action']
    by_action_d[act].append(row['total'])
    by_action_d_scores[act].append((row['action_score'], row['target_score'], row['params_score']))
for action in sorted(by_action_d, key=lambda a: sum(by_action_d[a])/len(by_action_d[a])):
    scores = by_action_d[action]
    avg_a = sum(s[0] for s in by_action_d_scores[action]) / len(scores)
    avg_t = sum(s[1] for s in by_action_d_scores[action]) / len(scores)
    avg_p = sum(s[2] for s in by_action_d_scores[action]) / len(scores)
    print(f'  {action:25s} n={len(scores):2d}  total={sum(scores)/len(scores):.3f}  '
          f'(act={avg_a:.2f} tgt={avg_t:.2f} prm={avg_p:.2f})')


# ## 11. v15d 未知instruction 評価 + 保存

# In[12]:


res_v15d_unknown = evaluate_records(unknown_records, predict_v15d)
print('v15d unknown benchmark:', res_v15d_unknown['overall'])

# Save v15d unknown predictions
score_by_key = {r['prediction_key']: r for r in res_v15d_unknown['rows']}
rows_out = []
for rec in unknown_records:
    key = rec['prediction_key']
    rows_out.append({
        'prediction_key': key,
        'video_path': rec['video_path'],
        'variant': rec.get('variant', 'base'),
        'instruction': rec['instruction'],
        'gt_primary': rec['gt_primary'],
        'prediction': res_v15d_unknown['predictions'][key],
        'scores': {k: score_by_key[key][k]
                   for k in ['action_score', 'target_score', 'params_score', 'total']},
    })
(OUTPUT_DIR / 'unknown_predictions_v15d.json').write_text(
    json.dumps(rows_out, ensure_ascii=False, indent=2), encoding='utf-8')

# Save analysis
final_analysis = {
    'constraint': 'instruction only input',
    'known_benchmark': {
        'v15a': res_v15a['overall'],
        'v15b': res_v15b['overall'],
        'v15c': res_v15c['overall'],
        'v15d': res_v15d['overall'],
    },
    'unknown_benchmark': {
        'v15d': res_v15d_unknown['overall'],
    },
    'best_model': 'v15d_instruction_extraction',
}
(OUTPUT_DIR / 'analysis_ver15.json').write_text(
    json.dumps(final_analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved to', OUTPUT_DIR)

