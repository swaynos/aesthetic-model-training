"""
dataset.py — Edit-chain dataset builder.

CLI:
    python -m aesthetic_scoring.dataset --manifest <path.csv> --out <path.jsonl>

Manifest CSV required columns:
    source_id, reference_path, derivative_path, step_index, intent_state, caption_path

Optional columns:
    prior_reference_path, intent_mask_path, subject_mask_path, mask_policy
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List

from .features import extract_reference_features
from .pseudo_labels import _TARGET_KEYS, _raw_targets, normalize_per_source

REQUIRED_COLUMNS = {
    "source_id", "reference_path", "derivative_path",
    "step_index", "intent_state", "caption_path",
}


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load and validate a manifest CSV. Returns list of row dicts."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )
        for i, row in enumerate(reader):
            for col in REQUIRED_COLUMNS:
                if not row.get(col, "").strip():
                    raise ValueError(
                        f"Row {i+1}: required column '{col}' is empty"
                    )
            rows.append(dict(row))

    if not rows:
        raise ValueError("Manifest has no data rows.")

    return rows


def build_derivation_chain_dataset_from_rows(rows: List[Dict], output_jsonl: str) -> Dict:
    """Build an derivation-chain JSONL dataset from in-memory manifest rows."""
    if not rows:
        raise ValueError("Manifest row list is empty.")

    processed = []
    skipped = 0
    errors = 0

    for row in rows:
        try:
            caption = Path(row["caption_path"]).read_text(encoding="utf-8").strip()

            fv = extract_reference_features(
                reference_path=row["reference_path"],
                derivative_path=row["derivative_path"],
                source_id=row["source_id"],
                step_index=int(row["step_index"]),
                intent_state=row["intent_state"],
                prior_reference_path=row.get("prior_reference_path") or None,
                intent_mask_path=row.get("intent_mask_path") or None,
                subject_mask_path=row.get("subject_mask_path") or None,
                mask_policy=row.get("mask_policy", "none") or "none",
            )

            raw = _raw_targets(fv)

            processed.append({
                "source_id": row["source_id"],
                "reference_path": row["reference_path"],
                "derivative_path": row["derivative_path"],
                "step_index": int(row["step_index"]),
                "intent_state": row["intent_state"],
                "caption": caption,
                "features": dataclasses.asdict(fv),
                "raw_labels": raw,
            })

        except Exception as exc:
            errors += 1
            print(
                f"WARNING: skipping row {row.get('source_id','?')} "
                f"pass={row.get('step_index','?')}: {exc}",
                file=sys.stderr,
            )
            skipped += 1

    raw_label_dicts = [
        {"source_id": p["source_id"], **p["raw_labels"]}
        for p in processed
    ]
    normed = normalize_per_source(raw_label_dicts)

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources_seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for p, n in zip(processed, normed):
            sources_seen.add(p["source_id"])
            label_dict = {k: n[k] for k in _TARGET_KEYS}
            label_dict["source_id"] = p["source_id"]
            label_dict["step_index"] = p["step_index"]
            label_dict["intent_state"] = p["intent_state"]

            record = {
                "source_id": p["source_id"],
                "reference_path": p["reference_path"],
                "derivative_path": p["derivative_path"],
                "step_index": p["step_index"],
                "intent_state": p["intent_state"],
                "caption": p["caption"],
                "features": p["features"],
                "labels": label_dict,
            }
            f.write(json.dumps(record) + "\n")

    return {
        "row_count": len(processed),
        "source_count": len(sources_seen),
        "skipped_count": skipped,
        "error_count": errors,
    }


def build_derivation_chain_dataset(manifest_path: str, output_jsonl: str) -> Dict:
    """Build an derivation-chain JSONL dataset from a manifest CSV."""
    rows = load_manifest(manifest_path)
    return build_derivation_chain_dataset_from_rows(rows, output_jsonl)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m aesthetic_scoring.dataset",
        description="Build an derivation-chain JSONL dataset from a manifest CSV.",
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--out", required=True, help="Path for output JSONL")
    args = parser.parse_args(argv)

    summary = build_derivation_chain_dataset(args.manifest, args.out)

    print(json.dumps(summary, indent=2))
    if summary["row_count"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
