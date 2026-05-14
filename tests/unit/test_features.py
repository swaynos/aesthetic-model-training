"""Unit tests for aesthetic_scoring.features."""
import dataclasses
import math
import pytest
from aesthetic_scoring.features import extract_reference_features
from aesthetic_scoring.types import RegionalFeatureVector
from tests.conftest import (
    OSC_HUMAN_SRC, OSC_HUMAN_MASK, OSC_HUMAN_EDITS,
    MORPH_LADDER_SRC, MORPH_LADDER_EDITS,
)

# Convenience str aliases
SRC   = str(OSC_HUMAN_SRC)
MASK  = str(OSC_HUMAN_MASK)
E0    = str(OSC_HUMAN_EDITS[0])   # state A
E1    = str(OSC_HUMAN_EDITS[1])   # state B  (large cross-state drift)
E2    = str(OSC_HUMAN_EDITS[2])   # state A  (near-identical to E0)


def test_returns_regional_feature_vector():
    fv = extract_reference_features(SRC, E0)
    assert isinstance(fv, RegionalFeatureVector)


def test_global_metrics_finite():
    fv = extract_reference_features(SRC, E0)
    for field in ["ssim", "lpips_proxy", "rgb_l1_delta", "edge_retention",
                  "hf_retention", "grad_ratio", "banding"]:
        val = getattr(fv, field)
        assert math.isfinite(val), f"{field}={val} is not finite"


def test_ssim_range():
    fv = extract_reference_features(SRC, E0)
    assert -1.0 <= fv.ssim <= 1.0


def test_rgb_l1_nonnegative():
    fv = extract_reference_features(SRC, E0)
    assert fv.rgb_l1_delta >= 0.0


def test_tile_metrics():
    fv = extract_reference_features(SRC, E0)
    assert fv.tile_mean >= 0.0
    assert fv.tile_worst >= fv.tile_mean
    assert fv.tile_variance >= 0.0
    assert fv.tile_high_risk_count >= 0


def test_no_mask_gives_nan_inside_outside():
    fv = extract_reference_features(SRC, E0, mask_policy="none")
    assert math.isnan(fv.inside_ssim)
    assert math.isnan(fv.outside_ssim)


def test_with_mask_gives_finite_inside_metrics():
    fv = extract_reference_features(
        SRC, E0,
        intent_mask_path=MASK,
        mask_policy="custom_intent",
    )
    assert math.isfinite(fv.inside_ssim)
    assert math.isfinite(fv.inside_rgb_l1_delta)


def test_temporal_nan_when_no_reference():
    fv = extract_reference_features(SRC, E0)
    assert math.isnan(fv.temporal_ssim)
    assert math.isnan(fv.temporal_lpips_proxy)


def test_temporal_finite_when_reference_given():
    fv = extract_reference_features(SRC, E2, prior_reference_path=E0)
    assert math.isfinite(fv.temporal_ssim)


def test_feature_version_set():
    fv = extract_reference_features(SRC, E0)
    assert fv.feature_version != ""


def test_json_serializable():
    import json
    fv = extract_reference_features(SRC, E0)
    d = dataclasses.asdict(fv)
    cleaned = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
               for k, v in d.items()}
    json.dumps(cleaned)


def test_boundary_metrics_finite():
    fv = extract_reference_features(
        SRC, E0,
        intent_mask_path=MASK,
        mask_policy="custom_intent",
    )
    assert math.isfinite(fv.boundary_seam_intensity)
    assert math.isfinite(fv.boundary_edge_continuity)


# ── Directional tests enabled by real oscillating-intent content ──────────────

def test_identical_images_ssim_near_one():
    """Identity must produce SSIM ≈ 1.0."""
    fv = extract_reference_features(SRC, SRC)
    assert fv.ssim > 0.99


def test_same_state_ssim_high():
    """Same-state pair (src vs edit_0) must be very similar. Measured: 0.9943."""
    fv = extract_reference_features(SRC, E0)
    assert fv.ssim > 0.95


def test_cross_state_ssim_low():
    """Cross-state pair (src vs edit_1) must be very different. Measured: 0.4313."""
    fv = extract_reference_features(SRC, E1)
    assert fv.ssim < 0.70


def test_same_state_ssim_higher_than_cross_state():
    """Same-state SSIM must exceed cross-state SSIM (directional)."""
    fv_same  = extract_reference_features(SRC, E0)
    fv_cross = extract_reference_features(SRC, E1)
    assert fv_same.ssim > fv_cross.ssim


def test_temporal_ssim_high_for_same_state_reference():
    """edit_2 vs ref=edit_0: both state A, temporal SSIM must be high. Measured: 0.9761."""
    fv = extract_reference_features(SRC, E2, prior_reference_path=E0)
    assert fv.temporal_ssim > 0.90


def test_temporal_ssim_low_for_cross_state_reference():
    """edit_1 vs ref=edit_0: A vs B cross-state, temporal SSIM must be low. Measured: 0.4190."""
    fv = extract_reference_features(SRC, E1, prior_reference_path=E0)
    assert fv.temporal_ssim < 0.70


def test_inside_and_outside_metrics_differ_with_mask():
    fv = extract_reference_features(
        SRC, E1, intent_mask_path=MASK, mask_policy="custom_intent"
    )
    assert math.isfinite(fv.inside_ssim)
    assert math.isfinite(fv.outside_ssim)
    assert abs(fv.inside_ssim - fv.outside_ssim) > 1e-4


def test_morphology_ladder_shows_progressive_drift():
    src = str(MORPH_LADDER_SRC)
    e0 = str(MORPH_LADDER_EDITS[0])
    e2 = str(MORPH_LADDER_EDITS[2])
    e4 = str(MORPH_LADDER_EDITS[4])

    fv0 = extract_reference_features(src, e0)
    fv2 = extract_reference_features(src, e2)
    fv4 = extract_reference_features(src, e4)

    assert fv0.rgb_l1_delta <= fv2.rgb_l1_delta <= fv4.rgb_l1_delta
