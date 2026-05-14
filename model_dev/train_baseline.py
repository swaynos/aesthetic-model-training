"""
model_dev/train_baseline.py — Deterministic ridge regression baseline.

Usage:
    python model_dev/train_baseline.py \\
        --dataset /tmp/derivation_chain_dataset.jsonl \\
        --out /tmp/model_dir \\
        --seed 42

Outputs (in --out directory):
    baseline_model.json   — weights, intercepts, feature_names, metadata
    feature_schema.json   — feature names + types + version

Determinism guarantee:
    Two runs with the same dataset and --seed produce byte-identical JSON.
    Ridge regression uses closed-form solution (no iterative solver randomness).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Targets to train one ridge model per target
TARGET_KEYS = [
    "quality_target",
    "divergence_target",
    "artifact_target",
    "outside_intent_drift_target",
    "inside_intent_quality_target",
    "temporal_consistency_target",
]

# Features extracted from RegionalFeatureVector that are numeric scalars
FEATURE_NAMES = [
    "ssim", "lpips_proxy", "rgb_l1_delta", "edge_retention", "hf_retention",
    "grad_ratio", "banding",
    "inside_ssim", "inside_lpips_proxy", "inside_rgb_l1_delta", "inside_edge_retention",
    "outside_ssim", "outside_lpips_proxy", "outside_rgb_l1_delta", "outside_edge_retention",
    "boundary_seam_intensity", "boundary_edge_continuity",
    "tile_mean", "tile_worst", "tile_variance", "tile_high_risk_count",
    "temporal_ssim", "temporal_lpips_proxy", "temporal_rgb_l1_delta",
]

MODEL_VERSION = "baseline-1.0"
FEATURE_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def extract_X_y(rows: List[dict], feature_names: List[str], target_key: str,
                seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target vector y, replacing NaN with column mean."""
    rng = np.random.default_rng(seed)

    X_raw = []
    y_raw = []
    for row in rows:
        feats = row.get("features", {})
        x = [feats.get(fn) for fn in feature_names]
        # Convert None/nan to nan
        x = [float(v) if v is not None else float("nan") for v in x]
        y_val = row.get("labels", {}).get(target_key)
        if y_val is None:
            continue
        X_raw.append(x)
        y_raw.append(float(y_val))

    X = np.array(X_raw, dtype=np.float64)
    y = np.array(y_raw, dtype=np.float64)

    # Replace NaN with column mean (deterministic: computed before any shuffling)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_means[j]

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression (closed-form, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
    """Closed-form ridge regression: w = (X^T X + alpha I)^{-1} X^T y.

    Returns (weights, intercept).  Adds bias column internally.
    """
    n, p = X.shape
    # Add bias column
    X_b = np.hstack([np.ones((n, 1)), X])
    # Ridge: regularize all except bias
    reg = alpha * np.eye(p + 1)
    reg[0, 0] = 0.0  # don't regularize bias
    A = X_b.T @ X_b + reg
    b = X_b.T @ y
    # Solve via Cholesky (deterministic, stable)
    try:
        L = np.linalg.cholesky(A)
        w_full = np.linalg.solve(L.T, np.linalg.solve(L, b))
    except np.linalg.LinAlgError:
        # Fallback to lstsq if not positive definite
        w_full, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    intercept = float(w_full[0])
    weights = w_full[1:]
    return weights, intercept


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str, out_dir: str, seed: int, alpha: float = 1.0) -> dict:
    rows = load_jsonl(dataset_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    weights_dict: Dict[str, Dict[str, float]] = {}
    intercepts_dict: Dict[str, float] = {}

    for target in TARGET_KEYS:
        X, y = extract_X_y(rows, FEATURE_NAMES, target, seed)
        if len(y) < 2:
            # Not enough data — store zero weights
            weights_dict[target] = {fn: 0.0 for fn in FEATURE_NAMES}
            intercepts_dict[target] = 0.5
            continue
        w, b = ridge_fit(X, y, alpha=alpha)
        weights_dict[target] = {fn: round(float(w[i]), 10) for i, fn in enumerate(FEATURE_NAMES)}
        intercepts_dict[target] = round(float(b), 10)

    # Compute feature schema hash for traceability
    schema_hash = hashlib.sha256(
        json.dumps(FEATURE_NAMES, sort_keys=True).encode()
    ).hexdigest()[:16]

    model = {
        "type": "ridge",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "target_keys": TARGET_KEYS,
        "alpha": alpha,
        "seed": seed,
        "row_count": len(rows),
        "schema_hash": schema_hash,
        "weights": weights_dict,
        "intercepts": intercepts_dict,
    }

    feature_schema = {
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "schema_hash": schema_hash,
    }

    # Write with sorted keys for byte-identical output across runs
    model_path = out_path / "baseline_model.json"
    schema_path = out_path / "feature_schema.json"
    model_path.write_text(json.dumps(model, sort_keys=True, indent=2) + "\n")
    schema_path.write_text(json.dumps(feature_schema, sort_keys=True, indent=2) + "\n")

    return {
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "row_count": len(rows),
        "target_keys": TARGET_KEYS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python model_dev/train_baseline.py",
        description="Train a deterministic ridge regression baseline model.",
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    args = parser.parse_args(argv)

    result = train(args.dataset, args.out, args.seed, args.alpha)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
