"""
model_dev/evaluate_baseline.py — Evaluate a trained baseline model.

Usage:
    python model_dev/evaluate_baseline.py \\
        --dataset /tmp/derivation_chain_dataset.jsonl \\
        --model /tmp/model_dir/baseline_model.json \\
        --out /tmp/eval.json

Emits evaluation.json with:
    row_count, source_count, spearman_pass_degradation,
    mean_same_intent_stability_by_pass, worst_pass_by_source, intent_state_means
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation (no scipy dependency required)
# ─────────────────────────────────────────────────────────────────────────────

def _rankdata(a: List[float]) -> List[float]:
    """Compute ranks with average ties."""
    sorted_idx = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j < len(sorted_idx) - 1 and a[sorted_idx[j]] == a[sorted_idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_row(model: dict, row: dict) -> dict:
    """Predict all targets for one dataset row."""
    feature_names = model["feature_names"]
    weights = model["weights"]
    intercepts = model["intercepts"]
    target_keys = model.get("target_keys", list(weights.keys()))
    feats = row.get("features", {})

    x = {fn: feats.get(fn, 0.0) for fn in feature_names}
    x = {fn: (0.0 if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v))
         for fn, v in x.items()}

    preds = {}
    for target in target_keys:
        w = weights.get(target, {})
        b = intercepts.get(target, 0.5)
        val = b + sum(w.get(fn, 0.0) * x[fn] for fn in feature_names)
        preds[target] = max(0.0, min(1.0, val))
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dataset_path: str, model_path: str, out_path: str) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    if model.get("type") == "passthrough":
        # No real model — emit baseline metrics from label columns
        for row in rows:
            row["_pred"] = row.get("labels", {})
    else:
        for row in rows:
            row["_pred"] = predict_row(model, row)

    sources = {row["source_id"] for row in rows}

    # spearman: predicted degradation vs step_index (should increase over passes)
    pass_indices = [int(row["step_index"]) for row in rows]
    pred_deg = [row["_pred"].get("divergence_target", 0.5) for row in rows]
    spearman_pass_deg = spearman_r(pass_indices, pred_deg)

    # mean same-intent stability by pass
    stability_by_pass: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        stab = row["_pred"].get("temporal_consistency_target")
        if stab is not None and math.isfinite(float(stab)):
            stability_by_pass[int(row["step_index"])].append(float(stab))
    mean_stability_by_pass = {
        str(k): round(float(np.mean(v)), 6)
        for k, v in sorted(stability_by_pass.items())
    }

    # worst pass per source (highest predicted degradation)
    worst_pass_by_source: Dict[str, int] = {}
    for source in sources:
        source_rows = [r for r in rows if r["source_id"] == source]
        if not source_rows:
            continue
        worst = max(source_rows, key=lambda r: r["_pred"].get("divergence_target", 0.0))
        worst_pass_by_source[source] = int(worst["step_index"])

    # prompt state means for each target
    state_groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    target_keys = model.get("target_keys", ["quality_target", "divergence_target"])
    for row in rows:
        ps = row.get("intent_state", "unknown")
        for tk in target_keys:
            v = row["_pred"].get(tk)
            if v is not None and math.isfinite(float(v)):
                state_groups[ps][tk].append(float(v))

    intent_state_means: Dict[str, Dict[str, float]] = {}
    for ps, targets in sorted(state_groups.items()):
        intent_state_means[ps] = {
            tk: round(float(np.mean(vals)), 6)
            for tk, vals in sorted(targets.items())
        }

    result = {
        "row_count": len(rows),
        "source_count": len(sources),
        "spearman_pass_degradation": round(float(spearman_pass_deg), 6)
            if math.isfinite(spearman_pass_deg) else None,
        "mean_same_intent_stability_by_pass": mean_stability_by_pass,
        "worst_pass_by_source": worst_pass_by_source,
        "intent_state_means": intent_state_means,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python model_dev/evaluate_baseline.py",
        description="Evaluate a trained baseline model against a JSONL dataset.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result = evaluate(args.dataset, args.model, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
