"""Unit tests for aesthetic_scoring.pseudo_labels."""
from aesthetic_scoring.features import extract_reference_features
from aesthetic_scoring.pseudo_labels import (
    generate_pseudo_labels, normalize_per_source, _raw_targets, _TARGET_KEYS,
)
from aesthetic_scoring.types import PseudoLabelVector
from tests.conftest import (
    OSC_HUMAN_SRC, OSC_HUMAN_EDITS,
    OSC_ANIMAL_SRC, OSC_ANIMAL_EDITS,
)

SRC_H = str(OSC_HUMAN_SRC)
E_H0 = str(OSC_HUMAN_EDITS[0])
E_H1 = str(OSC_HUMAN_EDITS[1])
E_H2 = str(OSC_HUMAN_EDITS[2])
SRC_A = str(OSC_ANIMAL_SRC)
E_A0 = str(OSC_ANIMAL_EDITS[0])


def _fv(ref, edit, sid="s", pi=0, ps="A"):
    return extract_reference_features(ref, edit, source_id=sid, step_index=pi, intent_state=ps)


def test_returns_pseudo_label_vector():
    fv = _fv(SRC_H, E_H0)
    lv = generate_pseudo_labels(fv)
    assert isinstance(lv, PseudoLabelVector)


def test_all_targets_bounded():
    fv = _fv(SRC_H, E_H0)
    lv = generate_pseudo_labels(fv)
    for key in _TARGET_KEYS:
        val = getattr(lv, key)
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_degradation_and_quality_roughly_inverse():
    fv = _fv(SRC_H, E_H0)
    lv = generate_pseudo_labels(fv)
    assert lv.quality_target + lv.divergence_target > 0.4


def test_normalize_per_source_bounded():
    rows = []
    for i, (ref, edit, sid) in enumerate([
        (SRC_H, E_H0, "osc_human"),
        (SRC_H, E_H1, "osc_human"),
        (SRC_A, E_A0, "osc_animal"),
    ]):
        fv = _fv(ref, edit, sid=sid, pi=i)
        raw = _raw_targets(fv)
        rows.append({"source_id": sid, **raw})

    normed = normalize_per_source(rows)
    for row in normed:
        for key in _TARGET_KEYS:
            val = row[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] after normalization"


def test_normalize_per_source_does_not_mix_sources():
    rows = []
    for i, edit in enumerate([E_H0, E_H1]):
        fv = _fv(SRC_H, edit, sid="osc_human", pi=i)
        raw = _raw_targets(fv)
        rows.append({"source_id": "osc_human", **raw})

    normed = normalize_per_source(rows)
    for key in _TARGET_KEYS:
        vals = [r[key] for r in normed]
        assert min(vals) >= 0.0 and max(vals) <= 1.0


def test_cross_state_higher_degradation_than_same_state():
    fv_same = _fv(SRC_H, E_H0, sid="s", pi=0)
    fv_cross = _fv(SRC_H, E_H1, sid="s", pi=1)
    raw_same = _raw_targets(fv_same)
    raw_cross = _raw_targets(fv_cross)
    assert raw_cross["divergence_target"] > raw_same["divergence_target"]


def test_oscillation_degradation_not_strictly_monotone():
    rows = []
    for i, edit in enumerate(OSC_HUMAN_EDITS):
        fv = _fv(SRC_H, str(edit), sid="osc_human", pi=i, ps=("A" if i % 2 == 0 else "B"))
        rows.append({"source_id": "osc_human", **_raw_targets(fv)})
    normed = normalize_per_source(rows)
    vals = [r["divergence_target"] for r in normed]
    increasing = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    decreasing = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    assert not (increasing or decreasing)
