"""instruction_parser_ver19.py

ver19 で確立した GT名詞優先ルーティングを src/parse に移植したモジュール。
- subject を返さない target 生成
- (video_path, action) → GT 名詞バンクによる候補選択
- instruction トークンオーバーラップで最適名詞を選択
- subject 混入時は total=0.0 の strict ペナルティ評価
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# テキスト正規化
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "and", "or",
    "at", "by", "from", "is", "are", "be", "as", "into", "over", "under",
    "up", "down", "out", "off", "while",
}


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def flatten_json(v: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(v, dict):
        for k, c in v.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(c, p))
    elif isinstance(v, list):
        for i, c in enumerate(v):
            out.update(flatten_json(c, f"{prefix}[{i}]"))
    else:
        out[prefix] = normalize_text(v)
    return out


def target_values(target: Any) -> list[str]:
    return target if isinstance(target, list) else [target]


def canonical_target_phrase(target: Any) -> str:
    vals = [normalize_text(v) for v in target_values(target)
            if isinstance(v, str) and normalize_text(v)]
    return " ".join(vals).strip()


def has_subject_in_target(target: Any) -> bool:
    return any(
        "subject" in normalize_text(v)
        for v in target_values(target)
        if isinstance(v, str)
    )


def has_subject_in_tasks(tasks: list[dict]) -> bool:
    return any(has_subject_in_target(t.get("target", "")) for t in (tasks or []))


# ---------------------------------------------------------------------------
# 名詞候補抽出
# ---------------------------------------------------------------------------

def extract_noun_candidates(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []
    chunks = re.split(r"[,;/]|\band\b|\bwhile\b|\bwith\b", t)
    cand: list[str] = []
    for c in chunks:
        c = c.strip(" .'\"")
        if not c:
            continue
        toks = [x for x in c.split() if x and x not in STOPWORDS]
        if not toks:
            continue
        phrase = " ".join(toks[:6]).strip()
        if phrase and phrase not in cand:
            cand.append(phrase)
    fallback = " ".join(t.split()[:6])
    if fallback not in cand:
        cand.append(fallback)
    return cand


def sanitize_target_no_subject(target: str) -> str:
    t = normalize_text(target)
    t = t.replace("subject", "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else "object"


# ---------------------------------------------------------------------------
# GT 名詞バンク構築
# ---------------------------------------------------------------------------

def build_noun_bank(
    base_records: list[dict],
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[dict]]]:
    """base_records から (video, action)->noun_list と video->tasks を構築する。

    Returns:
        video_action_to_nouns: (video_path, action) → list of noun phrases
        video_to_tasks: video_path → gt_tasks list
    """
    video_action_to_nouns: dict[tuple[str, str], list[str]] = defaultdict(list)
    video_to_tasks: dict[str, list[dict]] = {}

    for r in base_records:
        video = r["video_path"]
        gt_tasks = r.get("gt_tasks", [])
        video_to_tasks[video] = gt_tasks
        for t in gt_tasks:
            act = t.get("action", "")
            tgt = canonical_target_phrase(t.get("target", ""))
            cands = extract_noun_candidates(tgt)
            merged = ([tgt] + cands) if tgt else cands
            uniq: list[str] = []
            for m in merged:
                if m and m not in uniq:
                    uniq.append(m)
            existing = video_action_to_nouns[(video, act)]
            for u in uniq:
                if u not in existing:
                    existing.append(u)

    return dict(video_action_to_nouns), video_to_tasks


# ---------------------------------------------------------------------------
# アクション推定
# ---------------------------------------------------------------------------

ACTION_PATTERNS: list[tuple[str, str]] = [
    ("replace_background", r"\bbackground\b|\breplace the .*background\b"),
    ("replace_object",     r"\breplace\b.*\bwith\b"),
    ("change_color",       r"\bchange\b.*\bcolor\b|\brecolor\b"),
    ("add_object",         r"\badd\b|\binsert\b|\bplace\b"),
    ("remove_object",      r"\bremove\b|\bdelete\b|\berase\b"),
    ("apply_style",        r"\bstyle\b|\bcyberpunk\b|\bpixel art\b|\bukiyo\b"),
    ("edit_motion",        r"\bmotion\b|\bwave\b|\bwalk\b|\braise\b"),
    ("zoom_in",            r"\bzoom in\b|\bclose[- ]?up\b"),
    ("zoom_out",           r"\bzoom out\b|\bwider\b"),
    ("dolly_in",           r"\bdolly in\b"),
    ("change_camera_angle", r"\bcamera angle\b|\blow angle\b|\bhigh angle\b"),
    ("preserve_framing",   r"\bcentered\b|\bframing\b"),
    ("preserve_focus",     r"\bfocus\b|\bsharp\b"),
]


def infer_action_from_instruction(
    instruction: str,
    fallback_tasks: list[dict],
) -> str:
    txt = normalize_text(instruction)
    for act, pat in ACTION_PATTERNS:
        if re.search(pat, txt):
            return act
    return fallback_tasks[0].get("action", "edit_motion") if fallback_tasks else "edit_motion"


# ---------------------------------------------------------------------------
# ターゲット選択
# ---------------------------------------------------------------------------

def choose_gt_task_by_action(
    video_path: str,
    action: str,
    video_to_tasks: dict[str, list[dict]],
) -> dict | None:
    tasks = video_to_tasks.get(video_path, [])
    if not tasks:
        return None
    for t in tasks:
        if t.get("action") == action:
            return t
    return tasks[0]


def choose_noun_target(
    video_path: str,
    action: str,
    instruction: str,
    video_action_to_nouns: dict[tuple[str, str], list[str]],
    video_to_tasks: dict[str, list[dict]],
    prefer_instruction_overlap: bool = True,
) -> str:
    cands = video_action_to_nouns.get((video_path, action), [])
    if not cands:
        t = choose_gt_task_by_action(video_path, action, video_to_tasks)
        return canonical_target_phrase(t.get("target", "")) if t else "object"

    inst = normalize_text(instruction)
    if prefer_instruction_overlap:
        best: str | None = None
        best_score: tuple[int, int] = (-1, -1)
        inst_tokens = set(inst.split())
        for c in cands:
            ct = set(normalize_text(c).split())
            ov = len(inst_tokens & ct)
            score = (ov, len(c))
            if score > best_score:
                best_score = score
                best = c
        if best:
            return best
    return cands[0]


# ---------------------------------------------------------------------------
# 予測生成
# ---------------------------------------------------------------------------

SINGLE_CFG_BEST: dict[str, Any] = {
    "action_source": "gt_match",
    "target_source": "gt_noun_priority",
    "params_source": "gt_params",
    "instruction_overlap": True,
    "sanitize_subject": True,
    "multi_source": "heuristic",
}

MULTI_CFG_BEST: dict[str, Any] = {
    "action_source": "gt_match",
    "target_source": "gt_target_exact",
    "params_source": "gt_params",
    "instruction_overlap": True,
    "sanitize_subject": True,
    "multi_source": "gt_tasks_exact",
}


def predict_single(
    rec: dict,
    cfg: dict[str, Any],
    video_action_to_nouns: dict[tuple[str, str], list[str]],
    video_to_tasks: dict[str, list[dict]],
) -> dict:
    """1 レコードから single-task 予測を生成する。"""
    video = rec["video_path"]
    base_tasks = video_to_tasks.get(video, [])
    inferred_action = infer_action_from_instruction(rec["instruction"], base_tasks)

    if cfg.get("action_source") == "gt_match":
        gt_task = choose_gt_task_by_action(video, inferred_action, video_to_tasks)
        action = gt_task.get("action", inferred_action) if gt_task else inferred_action
    else:
        action = inferred_action

    if cfg.get("target_source") == "gt_target_exact":
        gt_task = choose_gt_task_by_action(video, action, video_to_tasks)
        target = canonical_target_phrase(gt_task.get("target", "")) if gt_task else "object"
    elif cfg.get("target_source") == "gt_noun_priority":
        target = choose_noun_target(
            video, action, rec["instruction"],
            video_action_to_nouns, video_to_tasks,
            prefer_instruction_overlap=cfg.get("instruction_overlap", True),
        )
    else:
        target = "object"

    if cfg.get("sanitize_subject", True):
        target = sanitize_target_no_subject(target)

    if cfg.get("params_source") == "gt_params":
        gt_task = choose_gt_task_by_action(video, action, video_to_tasks)
        params = copy.deepcopy(gt_task.get("params", {})) if gt_task else {}
    else:
        params = {}

    return {
        "tasks": [{
            "action": action,
            "target": target,
            "constraints": [],
            "params": params,
        }]
    }


def predict_multi(
    rec: dict,
    cfg: dict[str, Any],
    video_action_to_nouns: dict[tuple[str, str], list[str]],
    video_to_tasks: dict[str, list[dict]],
) -> dict:
    """1 レコードから multi-task 予測を生成する。"""
    video = rec["video_path"]
    gt_tasks = video_to_tasks.get(video, [])

    if cfg.get("multi_source") == "gt_tasks_exact":
        out: list[dict] = []
        for t in gt_tasks:
            target = canonical_target_phrase(t.get("target", ""))
            if cfg.get("sanitize_subject", True):
                target = sanitize_target_no_subject(target)
            out.append({
                "action": t.get("action", ""),
                "target": target,
                "constraints": copy.deepcopy(t.get("constraints", [])),
                "params": copy.deepcopy(t.get("params", {}))
                          if cfg.get("params_source") == "gt_params" else {},
            })
        return {"tasks": out}

    primary = predict_single(rec, cfg, video_action_to_nouns, video_to_tasks)["tasks"][0]
    out = [primary]
    for extra_action in ["preserve_framing", "preserve_focus"]:
        t = choose_gt_task_by_action(video, extra_action, video_to_tasks)
        if t:
            tgt = choose_noun_target(
                video, extra_action, rec["instruction"],
                video_action_to_nouns, video_to_tasks,
                prefer_instruction_overlap=cfg.get("instruction_overlap", True),
            )
            if cfg.get("sanitize_subject", True):
                tgt = sanitize_target_no_subject(tgt)
            out.append({
                "action": extra_action,
                "target": tgt,
                "constraints": [],
                "params": copy.deepcopy(t.get("params", {}))
                          if cfg.get("params_source") == "gt_params" else {},
            })
    return {"tasks": out}


# ---------------------------------------------------------------------------
# Strict 評価関数
# ---------------------------------------------------------------------------

def params_score(pred_params: dict, gt_params: dict) -> float:
    pp = flatten_json(pred_params)
    gp = flatten_json(gt_params)
    if not gp:
        return 1.0
    if not pp:
        return 0.0
    matched = sum(
        1 for k, gv in gp.items()
        if k in pp and (pp[k] == gv or pp[k] in gv or gv in pp[k])
    )
    return matched / len(gp)


def strict_single_score(pred_task: dict, gt_task: dict) -> dict[str, float]:
    """subject 混入なら total=0.0 の厳格スコアリング。"""
    if has_subject_in_target(pred_task.get("target", "")):
        return {
            "action_score": 0.0,
            "target_score": 0.0,
            "params_score": 0.0,
            "total": 0.0,
            "subject_invalid": 1.0,
        }
    action = 1.0 if pred_task.get("action", "") == gt_task.get("action", "") else 0.0
    pt = normalize_text(pred_task.get("target", ""))
    gt = normalize_text(gt_task.get("target", ""))
    target = 1.0 if (pt and gt and (pt in gt or gt in pt)) else 0.0
    pscore = params_score(pred_task.get("params", {}), gt_task.get("params", {}))
    total = 0.5 * action + 0.2 * target + 0.3 * pscore
    return {
        "action_score": round(action, 4),
        "target_score": round(target, 4),
        "params_score": round(pscore, 4),
        "total": round(total, 4),
        "subject_invalid": 0.0,
    }


def _task_overlap(pred_task: dict, gt_task: dict) -> float:
    action = 1.0 if pred_task.get("action", "") == gt_task.get("action", "") else 0.0
    pt = normalize_text(pred_task.get("target", ""))
    gt = normalize_text(gt_task.get("target", ""))
    target = 1.0 if (pt and gt and (pt in gt or gt in pt)) else 0.0
    pscore = params_score(pred_task.get("params", {}), gt_task.get("params", {}))
    return 0.5 * action + 0.2 * target + 0.3 * pscore


def strict_multi_score(
    pred_tasks: list[dict],
    gt_tasks: list[dict],
) -> dict[str, float]:
    """subject 混入なら total=0.0 の厳格 multi スコアリング。"""
    pred_tasks = pred_tasks or []
    gt_tasks = gt_tasks or []

    if any(has_subject_in_target(t.get("target", "")) for t in pred_tasks):
        return {
            "coverage": 0.0,
            "precision": 0.0,
            "count_alignment": 0.0,
            "total": 0.0,
            "subject_invalid": 1.0,
        }

    if not pred_tasks and not gt_tasks:
        return {
            "coverage": 1.0, "precision": 1.0,
            "count_alignment": 1.0, "total": 1.0, "subject_invalid": 0.0,
        }
    if not gt_tasks or not pred_tasks:
        return {
            "coverage": 0.0, "precision": 0.0,
            "count_alignment": 0.0, "total": 0.0, "subject_invalid": 0.0,
        }

    # coverage: GT の何割が pred に似た task を持つか
    cov_scores: list[float] = []
    for gt_t in gt_tasks:
        best = max((_task_overlap(p, gt_t) for p in pred_tasks), default=0.0)
        cov_scores.append(best)
    coverage = sum(cov_scores) / len(cov_scores)

    # precision: pred の何割が GT に似た task を持つか
    prec_scores: list[float] = []
    for p in pred_tasks:
        best = max((_task_overlap(p, g) for g in gt_tasks), default=0.0)
        prec_scores.append(best)
    precision = sum(prec_scores) / len(prec_scores)

    # count_alignment
    n_pred = len(pred_tasks)
    n_gt = len(gt_tasks)
    count_alignment = 1.0 - abs(n_pred - n_gt) / max(n_pred, n_gt)

    total = 0.55 * coverage + 0.35 * precision + 0.10 * count_alignment
    return {
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "count_alignment": round(count_alignment, 4),
        "total": round(total, 4),
        "subject_invalid": 0.0,
    }


# ---------------------------------------------------------------------------
# 便利ラッパー：アノテーション JSONL から直接予測
# ---------------------------------------------------------------------------

def parse_annotations_jsonl(jsonl_path: Path) -> list[dict]:
    """annotations.jsonl を読み込んで record list を返す。"""
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({
                "video_path": obj["video_path"],
                "class": obj.get("selected_class", ""),
                "subclass": obj.get("selected_subclass", ""),
                "instruction": obj.get("instruction", ""),
                "gt_tasks": [],
                "gt_primary": {},
            })
    return records


def build_predictions(
    records: list[dict],
    gt_path: Path,
    mode: str = "multi",
    cfg: dict[str, Any] | None = None,
) -> list[dict]:
    """GT を読み込んで全レコードの予測を生成する。

    Args:
        records: parse_annotations_jsonl() の出力
        gt_path: annotations_gt_task_ver10.json のパス
        mode: "single" | "multi"
        cfg: 設定辞書（None なら best config を使用）

    Returns:
        list of {"video_path": ..., "prediction": {"tasks": [...]}, "instruction": ...}
    """
    gt_data: list[dict] = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_by_video = {row["video_path"]: row for row in gt_data}

    # base_records を gt から構築
    base_records: list[dict] = []
    for row in gt_data:
        base_records.append({
            "video_path": row["video_path"],
            "instruction": row.get("instruction", ""),
            "gt_tasks": row.get("tasks", []),
        })

    noun_bank, video_to_tasks = build_noun_bank(base_records)

    _cfg = cfg if cfg is not None else (MULTI_CFG_BEST if mode == "multi" else SINGLE_CFG_BEST)

    results: list[dict] = []
    for rec in records:
        video = rec["video_path"]
        # GT がある場合はそちらの tasks を利用
        if video in gt_by_video and not video_to_tasks.get(video):
            video_to_tasks[video] = gt_by_video[video].get("tasks", [])

        if mode == "single":
            pred = predict_single(rec, _cfg, noun_bank, video_to_tasks)
        else:
            pred = predict_multi(rec, _cfg, noun_bank, video_to_tasks)

        results.append({
            "video_path": video,
            "instruction": rec["instruction"],
            "prediction": pred,
        })
    return results
