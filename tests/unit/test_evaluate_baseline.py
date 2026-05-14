"""Unit tests for model_dev/evaluate_baseline.py."""
import json
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from model_dev.train_baseline import train
from model_dev.evaluate_baseline import evaluate
from aesthetic_scoring.dataset import build_derivation_chain_dataset
from tests.conftest import FIXTURES

MANIFEST = str(FIXTURES / "derivation_chain_manifest.csv")


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    ds = str(tmp / "ds.jsonl")
    build_derivation_chain_dataset(MANIFEST, ds)
    result = train(ds, str(tmp / "model"), seed=42)
    return ds, result["model_path"]


def test_evaluate_writes_json(trained_model, tmp_path):
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    evaluate(ds, model, out)
    assert Path(out).exists()


def test_evaluate_required_fields(trained_model, tmp_path):
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    result = evaluate(ds, model, out)
    for field in [
        "row_count", "source_count", "spearman_pass_degradation",
        "mean_same_intent_stability_by_pass", "worst_pass_by_source",
        "intent_state_means",
    ]:
        assert field in result, f"Missing field: {field}"


def test_evaluate_row_and_source_count(trained_model, tmp_path):
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    result = evaluate(ds, model, out)
    assert result["row_count"] == 18
    assert result["source_count"] == 3


def test_evaluate_both_intent_states_present(trained_model, tmp_path):
    """Both state A and state B must appear in intent_state_means."""
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    result = evaluate(ds, model, out)
    assert "A" in result["intent_state_means"]
    assert "B" in result["intent_state_means"]


def test_evaluate_worst_pass_by_source_all_sources(trained_model, tmp_path):
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    result = evaluate(ds, model, out)
    assert set(result["worst_pass_by_source"].keys()) == {"osc_human", "osc_animal", "morph_robot"}
    for source, pass_idx in result["worst_pass_by_source"].items():
        assert isinstance(pass_idx, int)


def test_evaluate_stability_by_pass_has_entries(trained_model, tmp_path):
    ds, model = trained_model
    out = str(tmp_path / "eval.json")
    result = evaluate(ds, model, out)
    assert len(result["mean_same_intent_stability_by_pass"]) >= 1
