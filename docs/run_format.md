# Run Output Format

## Directory Layout

All outputs for a training run are written under:

```
local_workspace/results/<run_name>/
├── config/
│   ├── input.yaml              # exact copy of the user-supplied run YAML
│   └── resolved.yaml           # fully expanded effective config
├── metadata/
│   ├── run.json                # run status, timing, git state, device
│   ├── environment.json        # Python, CUDA, and package provenance
│   └── config_hash.txt         # sha256 of resolved.yaml
├── logs/
│   └── training.log            # Python logging output
├── checkpoints/
│   ├── ep010.pth               # checkpoint saved every checkpoint_rate epochs
│   ├── ep020.pth
│   └── ...
├── metrics/
│   ├── train.csv               # per-epoch training losses
│   └── validation.csv          # per-epoch validation losses
├── artifacts/
│   ├── output_train/           # generated images for train-split samples
│   ├── output_val/             # generated images for validation-split samples
│   └── output_test/            # generated images for test-split samples (from vs-infer)
├── evaluation/
│   ├── per_image_metrics.csv   # per-image MAE, RMSE, PSNR, SSIM
│   └── summary.csv             # aggregate statistics across the test split
└── comparisons/                # comparison panels produced by vs-compare-panels
```

## File Descriptions

### `config/input.yaml`

Verbatim copy of the YAML file passed to `--config`. Preserved for full
reproducibility - re-running with this file reproduces the same experiment.

### `config/resolved.yaml`

The fully expanded effective configuration after all defaults have been applied
and all derived paths resolved. Differences from `input.yaml` reflect default
values that were not explicitly set by the user.

### `metadata/run.json`

Written at the start of a run and updated on completion or failure.
See [run.json Schema](#runjson-schema) below.

### `metadata/environment.json`

Snapshot of the runtime environment captured at run start:

- Python version and platform
- `torch`, `numpy`, `opencv` package versions
- CUDA availability, CUDA version, GPU name

### `metadata/config_hash.txt`

SHA-256 hash of `config/resolved.yaml`. Two runs with identical hashes used the
same effective configuration.

### `metrics/train.csv` and `metrics/validation.csv`

One row per epoch. Columns include at minimum `epoch`, `loss_g` (generator loss),
and `loss_d` (discriminator loss). Written incrementally during training so
partial results are available if the run is interrupted.

### `checkpoints/ep<NNN>.pth`

PyTorch checkpoint saved every `training.checkpoint_rate` epochs.
`NNN` is zero-padded to three digits (e.g. `ep010.pth`).
The checkpoint contains generator and discriminator state dicts plus the epoch
number. Use `inference.checkpoint_policy: latest` to load the most recent one
automatically.

### `artifacts/output_train/`, `output_val/`, `output_test/`

Generated images produced during training (train/val) and inference (test).
Each file is named after its source patch (e.g. `00512_09216_target_generated.tif`).

### `evaluation/per_image_metrics.csv`

One row per test image. Columns: `sample_id`, `mae`, `rmse`, `psnr`, `ssim`.

### `evaluation/summary.csv`

Aggregate statistics (mean, std, min, max) for each metric across the full
test split.

## run.json Schema

```json
{
  "run_name":       "example_run",
  "status":         "completed",
  "started_at":     "2025-01-15T10:30:00+00:00",
  "completed_at":   "2025-01-15T12:45:00+00:00",
  "git_commit":     "abc1234...",
  "git_dirty":      false,
  "config_hash":    "sha256:e3b0c4...",
  "seed":           42,
  "device":         "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 3090",
  "entrypoint":     "vs-train",
  "package_version": "0.1.0"
}
```

| Field | Description |
|---|---|
| `run_name` | Value of `run_name` from the config |
| `status` | `"running"`, `"completed"`, or `"failed"` |
| `started_at` | UTC ISO 8601 timestamp set at run start |
| `completed_at` | UTC ISO 8601 timestamp set on completion or failure; `null` if still running |
| `git_commit` | Full SHA of HEAD at run start; `null` if not in a git repo |
| `git_dirty` | `true` if there were uncommitted changes; `null` if not in a git repo |
| `config_hash` | SHA-256 of `config/resolved.yaml`; matches `metadata/config_hash.txt` |
| `seed` | Random seed used for the run |
| `device` | PyTorch device string (`"cuda"` or `"cpu"`) |
| `cuda_device_name` | GPU model name; `null` when running on CPU |
| `entrypoint` | CLI command that created the run (e.g. `"vs-train"`) |
| `package_version` | Installed `virtual-staining` package version |
