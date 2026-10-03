# RetinaGraph Training Pipeline

## Dataset configuration

The training pipeline expects a local dataset root such as:

```text
data/
├── raw/
├── metadata/
└── processed/
```

The path is configured with either:
- `--dataset-path` on `scripts/train.py`
- `RETINAGRAPH_DATASET_PATH`
- `--metadata-file` or `RETINAGRAPH_METADATA_FILE`

The metadata CSV must include at least:
- `image_path`
- `label`
- `dataset_source`
- `image_id`
- `patient_id` (optional but strongly preferred)

## Label handling

The training code accepts integer labels directly and also maps common text labels such as `No DR`, `Mild`, `Moderate`, `Severe`, and `Proliferative` into numeric values.

The default label convention used for the training pipeline is:

- 0 = no DR / healthy
- 1 = DR present / mild
- 2 = moderate
- 3 = severe
- 4 = proliferative

This mapping is intentionally explicit and should be validated against the real dataset before scientific use.

## Splitting

The project performs a reproducible split with a fixed random seed:

- train ratio = 0.70
- validation ratio = 0.15
- test ratio = 0.15

If patient IDs are available, the split is performed at patient level to prevent leakage. In the absence of patient IDs, the split falls back to image-level assignment.

## Graph and training flow

The training pipeline reuses the existing modules in the project:

1. Validate image
2. Preprocess with `ImageProcessor`
3. Extract vessel structure with `FeatureExtractor`
4. Build a PyG graph with `GraphBuilder`
5. Feed graph to `RetinaGAT`
6. Compute `CrossEntropyLoss`
7. Backpropagate with AdamW
8. Save checkpoints in `checkpoints/`

## Reproducibility

The script sets:
- Python `random.seed`
- NumPy seed
- PyTorch seed
- CUDA seed when available

The same dataset, seed, and model config yield deterministic outcomes to the extent permitted by the environment.

## Smoke test

Run a quick smoke test before full training:

```powershell
python scripts/train.py --dataset-path data --metadata-file data/metadata/metadata.csv --epochs 1 --batch-size 2 --max-samples 20
```

This verifies dataset loading, graph construction, batching, GAT forward pass, loss computation, and backpropagation.
