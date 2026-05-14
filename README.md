# aesthetic-model-training

Train a portable baseline model to score how a derivative image diverges from a
reference — edits, generations, upscales, codecs, renders.

The trained artifact (`baseline_model.json`) is a small, device-agnostic JSON
file that any scoring layer can consume.  The training pipeline itself runs on
CPU with no neural-net or GPU dependency; the model is lightweight enough that
inference runs anywhere.

## Who this is for

Anyone who needs to measure *how* one image differs from another across
iterative or generative workflows:

- Iterative inpainting / image editing (does the latest pass diverge from the source?)
- Generative model outputs vs. a conditioning image
- Encode → decode round-trips (codec, VAE, quantization)
- Upscaling / super-resolution vs. original
- Style transfer vs. content source
- Compression / codec A/B testing
- Render-frame regression vs. a golden reference

The model operates on **image pairs** (reference + derivative).  It outputs
normalized divergence scores in `[0, 1]` across six dimensions:

| Score | Meaning |
|---|---|
| `quality_target` | Structural fidelity to reference; higher = more similar |
| `divergence_target` | Overall divergence; higher = more different |
| `artifact_target` | Perceptual / technical artefact severity |
| `outside_intent_drift_target` | Change outside the masked region of interest |
| `inside_intent_quality_target` | Edit quality inside the masked region |
| `temporal_consistency_target` | Stability vs. an earlier reference in the same group |

## Relationship to image-aesthetic-scoring

[image-aesthetic-scoring](https://github.com/anomalyco/image-aesthetic-scoring)
is a companion library that wraps v1 GPU aesthetic scorers (LAION, PickScore,
HPSv2, FGAesQ) and provides a v2 `score_reference_comparison` runtime API.

The two repos are **fully decoupled**:

- This repo produces `baseline_model.json`.
- The scoring library reads it at runtime via `model_path=` — no imports cross
  between the repos.
- `FEATURE_VERSION` is stamped into `baseline_model.json`; the consumer should
  verify it matches the version in the scoring library before using a trained
  model in production.

Neither repo is required to use the other.  You can train a model here and
consume it in your own inference code without `image-aesthetic-scoring`.

## Requirements

- Python 3.11
- `numpy`, `Pillow`, `scikit-image`, `scipy` (pinned in `pyproject.toml`)
- Linux, macOS, or Windows — pure-Python training, no GPU required

## Installation

### From a clone (development)

```bash
git clone https://github.com/anomalyco/aesthetic-model-training
cd aesthetic-model-training
pip install -e ".[dev]"
```

### From GitHub (no clone needed)

```bash
pip install git+https://github.com/anomalyco/aesthetic-model-training
```

## Quick start — end to end

### 1. Prepare a manifest CSV

```
source_id,reference_path,derivative_path,step_index,intent_state,caption_path
seq_a,/data/ref_a.png,/data/edit_a_0.png,0,A,/data/caption_a.txt
seq_a,/data/ref_a.png,/data/edit_a_1.png,1,B,/data/caption_a.txt
seq_a,/data/ref_a.png,/data/edit_a_2.png,2,A,/data/caption_a.txt
```

See [Manifest schema](#manifest-schema) for all columns.

### 2. Build a JSONL dataset

```bash
python -m aesthetic_scoring.dataset \
    --manifest my_manifest.csv \
    --out dataset.jsonl
```

### 3. Train a baseline model

```bash
python model_dev/train_baseline.py \
    --dataset dataset.jsonl \
    --out model_dir/ \
    --seed 42
```

Two runs with the same `--dataset` and `--seed` produce **byte-identical**
`baseline_model.json`.

### 4. Evaluate

```bash
python model_dev/evaluate_baseline.py \
    --dataset dataset.jsonl \
    --model model_dir/baseline_model.json \
    --out evaluation.json
```

### 5. Use the trained model

#### With image-aesthetic-scoring

```python
from aesthetic_scoring import score_reference_comparison

result = score_reference_comparison(
    reference_path="source.png",
    derivative_path="output.png",
    model_path="model_dir/baseline_model.json",
    mask_policy="custom_intent",
    intent_mask_path="region_mask.png",
)
print(result.divergence_score)
print(result.quality_score)
print(result.regional_breakdown)
```

#### Standalone (no external dependencies)

```python
import dataclasses
import json
import math
from aesthetic_scoring import extract_reference_features

# Load your trained model
model         = json.loads(open("model_dir/baseline_model.json").read())
feature_names = model["feature_names"]
weights       = model["weights"]
intercepts    = model["intercepts"]

# Extract features for one image pair
fv     = extract_reference_features("source.png", "output.png")
fv_dict = dataclasses.asdict(fv)


def _clamp(v):
    return max(0.0, min(1.0, v)) if math.isfinite(v) else 0.5


scores = {}
for target in model["target_keys"]:
    w   = weights[target]
    b   = intercepts[target]
    x   = [fv_dict.get(fn) or 0.0 for fn in feature_names]
    scores[target] = _clamp(b + sum(w[fn] * xi for fn, xi in zip(feature_names, x)))

print(scores)
```

## Python API (programmatic, no CSV)

Use `build_derivation_chain_dataset_from_rows` to build a dataset directly from
Python dicts — useful when your sequence data lives in a database or is generated
programmatically.

```python
from aesthetic_scoring import build_derivation_chain_dataset_from_rows

rows = [
    {
        "source_id":            "seq_a",
        "reference_path":       "/data/ref_a.png",
        "derivative_path":      "/data/edit_a_0.png",
        "step_index":           "0",
        "intent_state":         "A",
        "caption_path":         "/data/caption_a.txt",
        # optional:
        "prior_reference_path": "",
        "intent_mask_path":     "/data/mask_a.png",
        "subject_mask_path":    "",
        "mask_policy":          "custom_intent",
    },
    # ... more rows
]

summary = build_derivation_chain_dataset_from_rows(rows, "dataset.jsonl")
print(summary)
# {"row_count": 1, "source_count": 1, "skipped_count": 0, "error_count": 0}
```

## Manifest schema

### Required columns

| Column | Description |
|---|---|
| `source_id` | Identifier for the sequence; used for per-source normalization |
| `reference_path` | Path to the reference (baseline/source) image |
| `derivative_path` | Path to the derivative image |
| `step_index` | Integer position in the sequence (0-based) |
| `intent_state` | Group/category label for this step (e.g. `A`, `B`, or any string) |
| `caption_path` | Path to a plain-text file describing the intended change |

### Optional columns

| Column | Description |
|---|---|
| `prior_reference_path` | Earlier derivative in the same intent group; enables temporal consistency metrics |
| `intent_mask_path` | PNG mask (0/255 single-channel) defining the region of interest |
| `subject_mask_path` | PNG subject mask for `subject` or `background` policy |
| `mask_policy` | `whole_image` \| `subject` \| `background` \| `custom_intent` \| `none` |

### Notes

- `step_index` and `intent_state` are caller-defined with no built-in semantics.
  Use them however best describes your sequence: `step_index` as edit pass, frame
  number, or variant index; `intent_state` as prompt variant, style bucket, or
  A/B group.
- Per-source normalization is applied across all rows sharing the same
  `source_id`.  Never mix unrelated sequences under one `source_id`.
- `prior_reference_path` is most useful when your pipeline alternates between
  two or more intent states and you want to measure stability across repetitions
  (e.g. pass 0 and pass 2 are both state A — compare them to check for drift).

## Produced artifacts

### `baseline_model.json`

| Key | Description |
|---|---|
| `type` | `"ridge"` |
| `model_version` | Semantic version string (e.g. `"baseline-1.0"`) |
| `feature_version` | Feature extraction version; must match consumer (e.g. `"1.0.0"`) |
| `feature_names` | Ordered list of 24 feature names used for training |
| `target_keys` | Ordered list of 6 target names |
| `weights` | `{target: {feature_name: float}}` |
| `intercepts` | `{target: float}` |
| `alpha` | Ridge regularization strength used |
| `seed` | Random seed (for reproducibility audit) |
| `row_count` | Number of training rows |
| `schema_hash` | SHA-256 prefix of the serialized feature name list |

### `evaluation.json`

| Key | Description |
|---|---|
| `row_count` | Number of rows evaluated |
| `source_count` | Number of distinct `source_id` values |
| `spearman_pass_degradation` | Spearman r between `step_index` and predicted `divergence_target` |
| `mean_same_intent_stability_by_pass` | Mean `temporal_consistency_target` per step |
| `worst_pass_by_source` | Step with highest predicted divergence per source |
| `intent_state_means` | Per-intent-state mean of each predicted target |

## Determinism

Training uses closed-form ridge regression (Cholesky solve).  There is no
iterative solver, no random initialization, and no data shuffling that affects
the result.  Two runs with the same `--dataset` and `--seed` produce
byte-identical `baseline_model.json`.

## Tests

```bash
python -m pytest tests/unit -q
```

~66 tests, CPU-only, completes in under 15 seconds.  No GPU, no network access,
and no external model weights required.

## Feature extraction overview

`extract_reference_features(reference_path, derivative_path, ...)` returns a
`RegionalFeatureVector` with 24+ deterministic metrics:

- **Global**: SSIM, LPIPS proxy, RGB L1 delta, edge retention, HF retention, gradient ratio, banding
- **Mask-aware** (when `intent_mask_path` / `mask_policy` supplied): inside-region and outside-region SSIM, LPIPS proxy, RGB L1, edge retention
- **Boundary**: seam intensity and edge continuity across the mask boundary ring
- **Tile**: mean, worst, variance, and high-risk tile count (64x64 grid)
- **Temporal** (when `prior_reference_path` supplied): SSIM, LPIPS proxy, RGB L1 vs. prior reference

All metrics are deterministic, fp32, numpy-only.  No model weights required.

## Not in scope

- GPU inference, LAION / PickScore / HPSv2 / FGAesQ scorers — see [image-aesthetic-scoring](https://github.com/anomalyco/image-aesthetic-scoring)
- Human labeling or annotation workflows
- Identity, anatomy, or facial-quality scoring (future plugin scope)

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
