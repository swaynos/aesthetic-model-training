"""Unit tests for model_dev/train_baseline.py — determinism is the key property."""
import json
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from model_dev.train_baseline import train, ridge_fit
from aesthetic_scoring.dataset import build_derivation_chain_dataset
from tests.conftest import FIXTURES

MANIFEST = str(FIXTURES / "derivation_chain_manifest.csv")


@pytest.fixture(scope="module")
def dataset_jsonl(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ds")
    out = str(tmp / "ds.jsonl")
    build_derivation_chain_dataset(MANIFEST, out)
    return out


def test_train_produces_output_files(dataset_jsonl, tmp_path):
    result = train(dataset_jsonl, str(tmp_path / "model"), seed=42)
    assert Path(result["model_path"]).exists()
    assert Path(result["schema_path"]).exists()


def test_model_json_valid(dataset_jsonl, tmp_path):
    result = train(dataset_jsonl, str(tmp_path / "model"), seed=42)
    model = json.loads(Path(result["model_path"]).read_text())
    assert "weights" in model
    assert "intercepts" in model
    assert "feature_names" in model
    assert model["type"] == "ridge"


def test_feature_schema_json(dataset_jsonl, tmp_path):
    result = train(dataset_jsonl, str(tmp_path / "model"), seed=42)
    schema = json.loads(Path(result["schema_path"]).read_text())
    assert "feature_names" in schema
    assert "feature_version" in schema
    assert "schema_hash" in schema


def test_determinism(dataset_jsonl, tmp_path):
    """Two runs with same seed must produce byte-identical model JSON."""
    out_a = str(tmp_path / "model_a")
    out_b = str(tmp_path / "model_b")
    r_a = train(dataset_jsonl, out_a, seed=42)
    r_b = train(dataset_jsonl, out_b, seed=42)
    text_a = Path(r_a["model_path"]).read_text()
    text_b = Path(r_b["model_path"]).read_text()
    assert text_a == text_b, "Model JSON differs between runs with same seed"


def test_different_seeds_produce_valid_models(dataset_jsonl, tmp_path):
    r_a = train(dataset_jsonl, str(tmp_path / "m42"), seed=42)
    r_b = train(dataset_jsonl, str(tmp_path / "m99"), seed=99)
    m_a = json.loads(Path(r_a["model_path"]).read_text())
    m_b = json.loads(Path(r_b["model_path"]).read_text())
    assert m_a["type"] == "ridge"
    assert m_b["type"] == "ridge"


def test_row_count_matches_dataset(dataset_jsonl, tmp_path):
    result = train(dataset_jsonl, str(tmp_path / "model"), seed=42)
    model = json.loads(Path(result["model_path"]).read_text())
    assert model["row_count"] == 18


def test_ridge_fit_basic():
    import numpy as np
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 0.0, 1.0])
    w, b = ridge_fit(X, y, alpha=0.1)
    assert len(w) == 2
    assert isinstance(b, float)
