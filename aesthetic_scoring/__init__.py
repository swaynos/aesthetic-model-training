"""
aesthetic_scoring (aesthetic-model-training)

Public API:
    extract_reference_features   — deterministic feature extraction
    generate_pseudo_labels       — no-human-label target generation
    normalize_per_source         — per-source-id min-max normalization
    build_derivation_chain_dataset          — build JSONL from CSV manifest
    build_derivation_chain_dataset_from_rows — build JSONL from in-memory rows

Types:
    RegionalFeatureVector, PseudoLabelVector, DerivationChainDatasetRow
"""

from .features import extract_reference_features
from .pseudo_labels import generate_pseudo_labels, normalize_per_source
from .dataset import (
    build_derivation_chain_dataset,
    build_derivation_chain_dataset_from_rows,
    load_manifest,
)
from .types import RegionalFeatureVector, PseudoLabelVector, DerivationChainDatasetRow

__all__ = [
    "extract_reference_features",
    "generate_pseudo_labels",
    "normalize_per_source",
    "build_derivation_chain_dataset",
    "build_derivation_chain_dataset_from_rows",
    "load_manifest",
    "RegionalFeatureVector",
    "PseudoLabelVector",
    "DerivationChainDatasetRow",
]
