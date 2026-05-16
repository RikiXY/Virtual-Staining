# Run Output Format

## Local Queues

Queue definitions are stored as YAML files under `config/queues/`. Runtime
queue state is written under `local_workspace/queues/`. Queue execution is
explicitly local, sequential, and single-worker in v1.

Example queue file:

```yaml
name: nightly
continue_on_failure: true
jobs:
  - config_path: ../runs/local/run_a.yaml
    label: baseline
  - config_path: ../runs/local/run_b.yaml
    notes: retry with lower lr
```

Example layout:

```text
config/queues/
├── nightly.yaml
└── local/
    └── my_queue.yaml

local_workspace/queues/
├── nightly.state.json
└── my_queue.state.json
```

Queue state is flattened under `local_workspace/queues/` by queue name. The
state file records queue-level status plus per-job fields such as
`status`, `started_at`, `completed_at`, and `error`.

## Directory Layout

All outputs for a training run are written under:

```
local_workspace/results/<run_name>/
├── config/
│   ├── input.yaml              # exact copy of the user-supplied run YAML
│   └── resolved.yaml           # fully expanded effective config
├── metadata/
│   ├── run.json                # run-level provenance and aggregate stage summary
│   ├── environment.json        # training-stage environment snapshot
│   ├── config_hash.txt         # training-stage sha256 of resolved.yaml
│   ├── events.jsonl            # append-only stage lifecycle log
│   └── stages/
│       ├── train.json          # current train stage state
│       ├── infer.json          # current inference stage state
│       └── evaluate.json       # current evaluation stage state
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
│   ├── per_image_metrics.csv   # per-image metrics for all evaluated test samples
│   ├── summary.csv             # aggregate statistics across the test split
│   └── skipped.csv             # samples that could not be evaluated (optional)
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

Optional composable losses are recorded under top-level `losses.generator` and
`losses.discriminator` lists. In the current registry contract, `ssim` is the
only accepted explicit loss term, it is generator-side only, and its schedule
type is `constant`. Registered losses have a default weight of `0.0`; a term is
active only when it is explicitly listed, `enabled` is `true`, and its current
weight is nonzero. Explicitly listed terms must declare `weight`; unlisted
registry entries remain absent and inactive.

```yaml
losses:
  generator:
    - name: ssim
      weight: 1.0
      enabled: true
      params:
        data_range: 1.0
        window_size: 11
        sigma: 1.5
        channel_mode: rgb
        reduction: mean
      schedule:
        type: constant
  discriminator: []
```

The training SSIM implementation is differentiable PyTorch code. It maps
current training tensors from `[-1, 1]` to `[0, 1]` before computing SSIM, and
uses `ssim_loss = 1 - SSIM(prediction, target)`. MS-SSIM and other structural
losses are not supported by this registry yet.

If `losses` is omitted, the training loop keeps the legacy Pix2Pix objective
defined by `model.gan_loss: bce` and `training.l1_weight`.

### `metadata/run.json`

Stable run-level provenance and aggregate summary. It identifies what the run is
without pretending to be the complete lifecycle record for every stage.
See [run.json Schema](#runjson-schema) below.

### `metadata/environment.json`

Snapshot of the runtime environment captured for the training stage:

- Python version and platform
- `torch`, `numpy`, `opencv` package versions
- CUDA availability, CUDA version, GPU name

### `metadata/config_hash.txt`

SHA-256 hash of `config/resolved.yaml` for the training stage. Inference and
evaluation keep their own stage-scoped hash files (`inference_config_hash.txt`
and `evaluation_config_hash.txt`).

### `metadata/stages/<stage>.json`

Current-state records for each stage. These files are overwritten on rerun and
contain fields such as:

- `stage`
- `status`
- `started_at`
- `completed_at`
- `config_hash`
- stage-specific provenance such as checkpoint path, manifest hash, counts, and output paths

### `metadata/events.jsonl`

Append-only execution log. Each line is a JSON object representing a lifecycle
event such as:

- `stage_started`
- `stage_completed`
- `stage_failed`

Events include `timestamp`, `run_name`, `stage`, `status`, `config_hash`, and
optional `details`.

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

One row per evaluated test image.

| Column | Description |
|---|---|
| `sample_id` | Sample identifier from the manifest |
| `target_path` | Absolute path to the ground-truth image |
| `generated_path` | Absolute path to the generated image |
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `channels` | Number of image channels |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `psnr` | Peak Signal-to-Noise Ratio (dB) |
| `ssim` | Structural Similarity Index |
| `pcc_gray` | Pearson Correlation Coefficient (grayscale) |
| `pcc_r` | Pearson Correlation Coefficient (red channel) |
| `pcc_g` | Pearson Correlation Coefficient (green channel) |
| `pcc_b` | Pearson Correlation Coefficient (blue channel) |
| `pcc_rgb_mean` | Mean PCC across RGB channels |

### `evaluation/summary.csv`

Aggregate statistics (mean, std, min, max) for each metric across the full
test split.

### `evaluation/skipped.csv`

Written when one or more test samples could not be evaluated. Contains one row
per skipped sample.

| Column | Description |
|---|---|
| `sample_id` | Sample identifier from the manifest |
| `reason` | Why the sample was skipped (`missing_generated`, `missing_target`, or an exception message) |
| `target_path` | Absolute path to the ground-truth target image |
| `generated_path` | Absolute path to the expected generated image |

This file is not written when all test samples are evaluated successfully.

## run.json Schema

```json
{
  "run_name":       "example_run",
  "started_at":     "2025-01-15T10:30:00+00:00",
  "git_commit":     "abc1234...",
  "git_dirty":      false,
  "config_hash":    "sha256:e3b0c4...",
  "manifest_path":  "/abs/path/to/manifest.csv",
  "manifest_sha256": "sha256:abcd...",
  "seed":           42,
  "device":         "cuda",
  "cuda_device_name": "NVIDIA GeForce RTX 3090",
  "entrypoint":     "vs-train",
  "package_version": "0.1.0",
  "last_event_at":  "2025-01-15T12:45:00+00:00",
  "stages_present": ["train", "infer", "evaluate"],
  "last_completed_stage": "evaluate"
}
```

| Field | Description |
|---|---|
| `run_name` | Value of `run_name` from the config |
| `started_at` | UTC ISO 8601 timestamp when `run.json` was first bootstrapped |
| `git_commit` | Full SHA of HEAD at run start; `null` if not in a git repo |
| `git_dirty` | `true` if there were uncommitted changes; `null` if not in a git repo |
| `config_hash` | Most recently recorded stable config hash for the run-level bootstrap |
| `manifest_path` | Most recently recorded manifest path associated with the run |
| `manifest_sha256` | Hash of the recorded manifest |
| `seed` | Training seed when training has populated the record |
| `device` | Training or inference device string (`"cuda"` or `"cpu"`) when available |
| `cuda_device_name` | GPU model name; `null` when running on CPU or unknown |
| `entrypoint` | CLI command that first created the run record |
| `package_version` | Installed `virtual-staining` package version |
| `last_event_at` | Timestamp of the most recent appended stage event |
| `stages_present` | Unique set of stage names observed so far in `events.jsonl` |
| `last_completed_stage` | Most recent stage that completed successfully |
