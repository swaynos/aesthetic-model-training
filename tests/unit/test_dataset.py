"""Unit tests for aesthetic_scoring.dataset."""
import json
from pathlib import Path
import pytest
from aesthetic_scoring.dataset import (
    build_derivation_chain_dataset,
    build_derivation_chain_dataset_from_rows,
    load_manifest,
)
from tests.conftest import (
    FIXTURES,
    OSC_HUMAN_SRC,
    OSC_HUMAN_EDITS,
    OSC_HUMAN_MASK,
    OSC_ANIMAL_SRC,
    OSC_ANIMAL_EDITS,
    OSC_ANIMAL_MASK,
    MORPH_LADDER_SRC,
    MORPH_LADDER_EDITS,
    MORPH_LADDER_MASK,
)

MANIFEST = str(FIXTURES / "derivation_chain_manifest.csv")


def _fixture_rows():
    rows = []

    for i in range(6):
        rows.append({
            "source_id": "osc_human",
            "reference_path": str(OSC_HUMAN_SRC),
            "derivative_path": str(OSC_HUMAN_EDITS[i]),
            "step_index": str(i),
            "intent_state": "A" if i % 2 == 0 else "B",
            "caption_path": str(FIXTURES / "caption_osc_human.txt"),
            "prior_reference_path": str(OSC_HUMAN_EDITS[i - 2]) if i >= 2 else "",
            "intent_mask_path": str(OSC_HUMAN_MASK),
            "subject_mask_path": "",
            "mask_policy": "custom_intent",
        })

    for i in range(6):
        rows.append({
            "source_id": "osc_animal",
            "reference_path": str(OSC_ANIMAL_SRC),
            "derivative_path": str(OSC_ANIMAL_EDITS[i]),
            "step_index": str(i),
            "intent_state": "A" if i % 2 == 0 else "B",
            "caption_path": str(FIXTURES / "caption_osc_animal.txt"),
            "prior_reference_path": str(OSC_ANIMAL_EDITS[i - 2]) if i >= 2 else "",
            "intent_mask_path": str(OSC_ANIMAL_MASK),
            "subject_mask_path": "",
            "mask_policy": "custom_intent",
        })

    for i in range(5):
        rows.append({
            "source_id": "morph_ladder",
            "reference_path": str(MORPH_LADDER_SRC),
            "derivative_path": str(MORPH_LADDER_EDITS[i]),
            "step_index": str(i),
            "intent_state": "L",
            "caption_path": str(FIXTURES / "caption_morph_ladder_robot.txt"),
            "prior_reference_path": str(MORPH_LADDER_EDITS[i - 1]) if i >= 1 else "",
            "intent_mask_path": str(MORPH_LADDER_MASK),
            "subject_mask_path": "",
            "mask_policy": "custom_intent",
        })

    return rows


def test_load_manifest_returns_rows():
    rows = load_manifest(MANIFEST)
    assert len(rows) == 18


def test_load_manifest_required_columns():
    rows = load_manifest(MANIFEST)
    for row in rows:
        for col in ("source_id", "reference_path", "derivative_path", "step_index",
                    "intent_state", "caption_path"):
            assert col in row


def test_load_manifest_three_sources():
    rows = load_manifest(MANIFEST)
    sources = {r["source_id"] for r in rows}
    assert sources == {"osc_human", "osc_animal", "morph_robot"}


def test_load_manifest_missing_file():
    with pytest.raises(FileNotFoundError):
        load_manifest("/nonexistent/path.csv")


def test_load_manifest_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("source_id,reference_path\ns1,/a\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(str(bad))


def test_build_dataset_writes_jsonl(tmp_path):
    out = str(tmp_path / "out.jsonl")
    summary = build_derivation_chain_dataset(MANIFEST, out)
    assert summary["row_count"] == 18
    assert summary["source_count"] == 3
    assert Path(out).exists()


def test_build_dataset_jsonl_valid_rows(tmp_path):
    out = str(tmp_path / "out.jsonl")
    build_derivation_chain_dataset(MANIFEST, out)
    rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
    assert len(rows) == 18
    for row in rows:
        assert "features" in row
        assert "labels" in row
        assert "source_id" in row


def test_build_dataset_labels_bounded(tmp_path):
    out = str(tmp_path / "out.jsonl")
    build_derivation_chain_dataset(MANIFEST, out)
    rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
    for row in rows:
        for key, val in row["labels"].items():
            if isinstance(val, float):
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_build_dataset_summary_fields(tmp_path):
    out = str(tmp_path / "out.jsonl")
    summary = build_derivation_chain_dataset(MANIFEST, out)
    assert "row_count" in summary
    assert "source_count" in summary
    assert "skipped_count" in summary
    assert "error_count" in summary


def test_build_dataset_temporal_rows_have_finite_temporal_ssim(tmp_path):
    out = str(tmp_path / "out.jsonl")
    build_derivation_chain_dataset(MANIFEST, out)
    rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
    temporal_rows = [
        r for r in rows
        if r.get("features", {}).get("temporal_ssim") is not None
        and r["features"]["temporal_ssim"] == r["features"]["temporal_ssim"]
    ]
    assert len(temporal_rows) >= 6


def test_build_dataset_from_generated_fixture_rows(tmp_path):
    out = str(tmp_path / "rows_out.jsonl")
    summary = build_derivation_chain_dataset_from_rows(_fixture_rows(), out)
    assert summary["row_count"] == 17
    assert summary["source_count"] == 3

    rows = [json.loads(line) for line in Path(out).read_text().splitlines()]
    assert len(rows) == 17
    assert all("labels" in row for row in rows)


def test_dataset_cli(tmp_path):
    out = str(tmp_path / "cli_out.jsonl")
    from aesthetic_scoring.dataset import main
    main(["--manifest", MANIFEST, "--out", out])
    assert Path(out).exists()
