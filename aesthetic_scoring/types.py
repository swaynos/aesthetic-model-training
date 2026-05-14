"""Typed dataclasses for the aesthetic-model-training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RegionalFeatureVector:
    """Features extracted from a reference→derivative image pair.

    ``step_index`` and ``intent_state`` are caller-defined identifiers.
    Common uses: step_index as a sequence position (edit pass, frame number,
    variant index); intent_state as a group/category label (prompt variant,
    style bucket, A/B group).
    """
    # caller-defined identifiers
    source_id: str
    step_index: int
    intent_state: str

    # global metrics
    ssim: float
    lpips_proxy: float
    rgb_l1_delta: float
    edge_retention: float
    hf_retention: float
    grad_ratio: float
    banding: float

    # inside-intent metrics (NaN when no mask)
    inside_ssim: float
    inside_lpips_proxy: float
    inside_rgb_l1_delta: float
    inside_edge_retention: float

    # outside-intent metrics (NaN when no mask)
    outside_ssim: float
    outside_lpips_proxy: float
    outside_rgb_l1_delta: float
    outside_edge_retention: float

    # boundary metrics
    boundary_seam_intensity: float
    boundary_edge_continuity: float

    # tile metrics
    tile_mean: float
    tile_worst: float
    tile_variance: float
    tile_high_risk_count: int

    # temporal metrics (NaN when no prior reference)
    temporal_ssim: float
    temporal_lpips_proxy: float
    temporal_rgb_l1_delta: float

    # feature metadata
    feature_version: str
    mask_policy: str
    finite: bool


@dataclass
class PseudoLabelVector:
    """Normalized [0, 1] training targets for one sequence row."""
    source_id: str
    step_index: int
    intent_state: str

    quality_target: float
    divergence_target: float
    artifact_target: float
    outside_intent_drift_target: float
    inside_intent_quality_target: float
    temporal_consistency_target: float


@dataclass
class DerivationChainDatasetRow:
    """One row in the derivation-chain JSONL dataset."""
    source_id: str
    reference_path: str
    derivative_path: str
    step_index: int
    intent_state: str
    caption: str

    features: RegionalFeatureVector
    labels: PseudoLabelVector
