from __future__ import annotations

from .models import (
    predict_v11a_ruleplus,
    predict_v11b_retrieval,
    predict_v11d_ensemble,
    predict_ver10_baseline,
    predict_ver10_improved,
)


def run_models(records: list[dict], seed_predictions: dict[str, dict] | None = None):
    seed_predictions = seed_predictions or {}

    pred_baseline = {}
    dbg_baseline = {}
    pred_v10_improved = {}
    dbg_v10_improved = {}
    pred_v11a = {}
    dbg_v11a = {}
    pred_v11b = {}
    dbg_v11b = {}
    pred_v11d = {}
    dbg_v11d = {}

    for record in records:
        video_key = record.get("prediction_key", f"{record['video_path']}::{record.get('variant', 'base')}")

        p0, d0 = predict_ver10_baseline(record)
        p1, d1 = predict_ver10_improved(record, records)
        p2, d2 = predict_v11a_ruleplus(record, records)
        p3, d3 = predict_v11b_retrieval(record, records)

        seed = seed_predictions.get(record["video_path"])
        p4, d4 = predict_v11d_ensemble(record, records, seed, p2, d2, p3, d3)

        pred_baseline[video_key] = p0
        dbg_baseline[video_key] = d0
        pred_v10_improved[video_key] = p1
        dbg_v10_improved[video_key] = d1
        pred_v11a[video_key] = p2
        dbg_v11a[video_key] = d2
        pred_v11b[video_key] = p3
        dbg_v11b[video_key] = d3
        pred_v11d[video_key] = p4
        dbg_v11d[video_key] = d4

    model_predictions = {
        "ver10_baseline": pred_baseline,
        "ver10_improved": pred_v10_improved,
        "v11a_ruleplus": pred_v11a,
        "v11b_retrieval": pred_v11b,
        "v11d_ensemble": pred_v11d,
    }
    model_debug = {
        "ver10_baseline": dbg_baseline,
        "ver10_improved": dbg_v10_improved,
        "v11a_ruleplus": dbg_v11a,
        "v11b_retrieval": dbg_v11b,
        "v11d_ensemble": dbg_v11d,
    }
    return model_predictions, model_debug


def to_video_prediction_map(records: list[dict], keyed_predictions: dict[str, dict]) -> dict[str, dict]:
    # evaluation helper: if variants exist, keep the latest per key pair externally.
    out = {}
    for record in records:
        key = f"{record['video_path']}::{record.get('variant', 'base')}"
        out[key] = keyed_predictions[key]
    return out
