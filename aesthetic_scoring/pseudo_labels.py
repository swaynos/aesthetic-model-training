"""
pseudo_labels.py — No-human-label pseudo-target generation.

Rules:
- All outputs are in [0, 1].
- Per-source_id normalization only; never normalize across mixed sources.
- Targets derive from regional/temporal features only.
- divergence_target and quality_target are inverses by design.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .types import PseudoLabelVector, RegionalFeatureVector


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(v):
        return 0.5
    return max(lo, min(hi, v))


def _raw_targets(fv: RegionalFeatureVector) -> Dict[str, float]:
    """Derive raw (un-normalized) pseudo-target signals for one feature vector."""

    ssim_w = _clamp(fv.ssim)
    er_w = _clamp(fv.edge_retention / 1.5)
    hfr_w = _clamp(fv.hf_retention / 1.5)
    quality_raw = 0.5 * ssim_w + 0.25 * er_w + 0.25 * hfr_w

    tile_worst_w = _clamp(fv.tile_worst / 0.5)
    band_w = _clamp(fv.banding * 10.0)
    degradation_raw = _clamp(
        (1.0 - quality_raw) * 0.6 + tile_worst_w * 0.25 + band_w * 0.15
    )

    hr_norm = _clamp(fv.tile_high_risk_count / max(1, 100))
    technical_raw = _clamp(0.6 * _clamp(fv.lpips_proxy) + 0.4 * hr_norm)

    if math.isfinite(fv.outside_ssim):
        outside_ssim_w = _clamp(fv.outside_ssim)
        out_rgb_w = _clamp(fv.outside_rgb_l1_delta / 0.2)
        outside_drift_raw = _clamp(0.5 * (1.0 - outside_ssim_w) + 0.5 * out_rgb_w)
    else:
        outside_drift_raw = 0.0

    if math.isfinite(fv.inside_ssim):
        in_ssim_w = _clamp(fv.inside_ssim)
        in_er_w = _clamp(fv.inside_edge_retention / 1.5)
        inside_quality_raw = 0.6 * in_ssim_w + 0.4 * in_er_w
    else:
        inside_quality_raw = quality_raw

    if math.isfinite(fv.temporal_ssim):
        stability_raw = _clamp(fv.temporal_ssim)
    else:
        stability_raw = 0.5

    return {
        "quality_target": _clamp(quality_raw),
        "divergence_target": _clamp(degradation_raw),
        "artifact_target": _clamp(technical_raw),
        "outside_intent_drift_target": _clamp(outside_drift_raw),
        "inside_intent_quality_target": _clamp(inside_quality_raw),
        "temporal_consistency_target": _clamp(stability_raw),
    }


_TARGET_KEYS = [
    "quality_target",
    "divergence_target",
    "artifact_target",
    "outside_intent_drift_target",
    "inside_intent_quality_target",
    "temporal_consistency_target",
]


def normalize_per_source(raw_rows: List[Dict]) -> List[Dict]:
    """Min-max normalize each target key per source_id."""
    sources: Dict[str, List[int]] = {}
    for i, row in enumerate(raw_rows):
        sid = row["source_id"]
        sources.setdefault(sid, []).append(i)

    result = [dict(r) for r in raw_rows]

    for sid, indices in sources.items():
        for key in _TARGET_KEYS:
            vals = [raw_rows[i][key] for i in indices]
            lo, hi = min(vals), max(vals)
            span = hi - lo
            for i in indices:
                if span < 1e-9:
                    result[i][key] = 0.5
                else:
                    result[i][key] = _clamp((result[i][key] - lo) / span)

    return result


def generate_pseudo_labels(features: RegionalFeatureVector) -> PseudoLabelVector:
    """Generate pseudo-labels for a single feature vector (pre-normalization)."""
    raw = _raw_targets(features)
    return PseudoLabelVector(
        source_id=features.source_id,
        step_index=features.step_index,
        intent_state=features.intent_state,
        quality_target=raw["quality_target"],
        divergence_target=raw["divergence_target"],
        artifact_target=raw["artifact_target"],
        outside_intent_drift_target=raw["outside_intent_drift_target"],
        inside_intent_quality_target=raw["inside_intent_quality_target"],
        temporal_consistency_target=raw["temporal_consistency_target"],
    )
