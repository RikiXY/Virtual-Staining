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

Controlled ablation queues can add an optional `ablation` block. Queue
preflight loads each run config, compares the resolved config dictionaries, and
fails before running any job if a difference is not covered by
`variable_fields`. Dot paths are compared against resolved config fields, not
raw YAML text. Use `run_name` as a variable when each ablation arm writes to a
separate run directory.

```yaml
name: loss_ablation
continue_on_failure: false
ablation:
  fixed_fields:
    - model.generator.base_channels
    - training.epochs
  variable_fields:
    - run_name
    - training.losses.generator
    - training.losses.discriminator
    - training.scheduler.name
jobs:
  - config_path: ../runs/local/ablation/baseline.yaml
    label: baseline_l1_adv
  - config_path: ../runs/local/ablation/ssim_only.yaml
    label: ssim_only
```

Ablation summaries are written beside queue state as
`local_workspace/queues/<queue-name>.ablation.summary.json`. The summary lists
jobs, labels, run names, canonical resolved config hashes, declared fixed
values, and declared variable values. Loss lists are compared through resolved
EPIC_11 loss config entries, so omitted default-zero terms are not treated as
active losses.

Example layout:

```text
config/queues/
├── nightly.yaml
└── local/
    └── my_queue.yaml

local_workspace/queues/
├── nightly.state.json
├── loss_ablation.ablation.summary.json
└── my_queue.state.json
```

Queue state is flattened under `local_workspace/queues/` by queue name. The
state file records queue-level status plus per-job fields such as
`status`, `started_at`, `completed_at`, and `error`.

`RunLayout` in `virtual_staining.experiment.run_layout` is the single owner of
run paths. `ExperimentSession` is the only run-stage bootstrap and creates the
directories; layout instances themselves are pure path contracts.

## Directory Layout

All outputs for an experiment run are written under:

```text
local_workspace/results/<run_name>/
├── config/
│   ├── train/
│   │   ├── input.yaml
│   │   └── resolved.yaml
│   ├── infer/
│   │   ├── input.yaml
│   │   └── resolved.yaml
│   └── evaluate/
│       ├── input.yaml
│       └── resolved.yaml
├── metadata/
│   ├── environments/
│   │   ├── train.json
│   │   ├── infer.json
│   │   └── evaluate.json
│   ├── stages/
│   │   ├── train.json
│   │   ├── infer.json
│   │   └── evaluate.json
│   ├── events.jsonl
│   └── run.json
├── logs/
│   └── run.log
├── metrics/
│   └── epochs.csv
├── checkpoints/
│   ├── ep010.pth
│   ├── ep020.pth
│   └── best.json
├── artifacts/
│   ├── output_train/
│   ├── output_val/
│   └── output_test/
└── evaluation/
    ├── per_image_metrics.csv
    ├── summary.csv
    └── skipped.csv
```

`prepare` is dataset-owned: `data/provenance.py` writes dataset
config/environment snapshots, fingerprint, manifest, split, input-hash, and
`dataset_build.json` files under the dataset root. It does not write experiment
`run.json`, `events.jsonl`, or `metadata/stages/prepare.json`.

Run checkpoint metadata uses the neutral `checkpoint_contract.py` and
`checkpoint_selection.py` modules; optimizer, scaler, scheduler, and resume
state remain training-owned. Training progress is a callback event rendered by
the CLI, not terminal output from library code.

## File Descriptions

### `config/input.yaml`

Verbatim copy of the YAML file passed to `--config`. Preserved for full
reproducibility - re-running with this file reproduces the same experiment.

### `config/resolved.yaml`

The fully expanded effective configuration after all defaults have been applied
and all derived paths resolved. Differences from `input.yaml` reflect default
values that were not explicitly set by the user.

Training losses are recorded under `training.losses.generator` and
`training.losses.discriminator` lists. Training requires explicit loss terms. Registered
losses have a default weight of `0.0`; a term is active only when it is
explicitly listed, `enabled` is `true`, and its scheduled current weight is
nonzero. Explicitly listed terms must declare `weight`; unlisted registry
entries remain absent and inactive.

Training-only augmentation is recorded under `training.augmentation`. When
enabled, the training split is virtually expanded in memory; no augmented patch
files are written, and validation/test data keep deterministic preprocessing.

```yaml
training:
  augmentation:
    enabled: false
    expansion_factor: 1
    intensity: light  # light, medium, or strong
```

Optimizer learning-rate schedules are configured under `training.scheduler`.
Omitting the section preserves a flat learning rate. Epoch numbers are
zero-based. `linear_decay` keeps the initial optimizer LR through
`decay_start_epoch`, then decays linearly through the final epoch. Plateau
scheduling steps only after validation, using validation metric columns such as
`loss_G_val`, `val_ssim`, `val_mae`, `val_rmse`, `val_psnr`, `val_pcc_gray`, or
`val_pcc_rgb_mean`. `loss_G_val` depends on the configured training loss terms.

```yaml
training:
  scheduler:
    name: linear_decay
    decay_start_epoch: 50
```

```yaml
training:
  scheduler:
    name: reduce_on_plateau
    monitor: val_ssim
    mode: max
    factor: 0.5
    patience: 5
    min_lr: 0.00002
```

Optimizer LR schedules are separate from `training.losses.*.schedule`, which changes
loss-term weights rather than optimizer learning rates.

Early stopping is configured under `training.early_stopping` and is disabled
when omitted. `patience` counts validation events, not raw epochs, so
`validate_rate` controls how often the monitored value can become stale. Use
validation CSV column names such as `val_ssim`, `val_mae`, `val_rmse`,
`loss_G_val`, `loss_D_val`, or configured `loss_val_*` component columns.

```yaml
training:
  early_stopping:
    monitor: val_ssim
    mode: max
    patience: 15
    min_delta: 0.0
```

Accepted loss names are:

- `adversarial_bce`: generator or discriminator BCE-with-logits adversarial loss.
- `l1`: generator image reconstruction loss.
- `ssim`: generator image structural similarity loss.

```yaml
training:
  losses:
    generator:
      - name: adversarial_bce
        weight: 1.0
      - name: l1
        weight: 25.0
      - name: ssim
        weight: 0.0
        enabled: false
        params:
          mask:
            enabled: false
            source: foreground_mask
            background_weight: 0.25
        schedule:
          type: linear_warmup
          start_epoch: 0
          end_epoch: 5
    discriminator:
      - name: adversarial_bce
        weight: 1.0
```

The baseline objective uses generator `adversarial_bce` with weight `1.0`,
generator `l1` with weight `25.0`, and discriminator `adversarial_bce` with
weight `1.0`.

The training SSIM implementation is differentiable PyTorch code. It maps
current training tensors from `[-1, 1]` to `[0, 1]` before computing SSIM, and
uses `ssim_loss = 1 - SSIM(prediction, target)`. MS-SSIM and other structural
losses are not supported by this registry yet.

Supported schedule types are `constant`, `linear_warmup`, `linear_decay`,
`step`, `cosine`, `turn_on_after_epoch`, and `turn_off_after_epoch`.
`linear_warmup`, `linear_decay`, and `cosine` use `start_epoch` and
`end_epoch`. `step`, `turn_on_after_epoch`, and `turn_off_after_epoch` use
`epoch`; `step` also uses `factor`.

Mask weighting is optional. When `params.mask.enabled` is `true`, the training
dataset must provide a `foreground_mask` tensor for every batch. Missing masks
raise an error instead of being treated as all-foreground. The manifest loads
the exact `foreground_mask_path` written when `masks.save_patch_masks: true`.
The saved patch mask is the aligned target foreground mask.

When configured loss terms are present, `metrics/epochs.csv` adds deterministic
component columns using normalized names:

```text
loss_train_total_generator
loss_train_total_discriminator
loss_train_raw_<role>_<loss_name>
loss_train_weighted_<role>_<loss_name>
loss_train_current_weight_<role>_<loss_name>
loss_val_total_generator
loss_val_total_discriminator
loss_val_raw_<role>_<loss_name>
loss_val_weighted_<role>_<loss_name>
loss_val_current_weight_<role>_<loss_name>
```

For example, configured SSIM writes `loss_train_raw_generator_ssim`,
`loss_train_weighted_generator_ssim`, and
`loss_train_current_weight_generator_ssim`, plus matching validation columns
when validation runs.

### `metadata/run.json`

Run identity and aggregate state only:

```json
{
  "schema_version": 1,
  "run_id": "uuid",
  "run_name": "example_run",
  "created_at": "2025-01-15T10:30:00+00:00",
  "dataset_fingerprint": "sha256:...",
  "last_event_at": "2025-01-15T12:45:00+00:00",
  "stages_present": ["train", "infer", "evaluate"],
  "last_completed_stage": "evaluate"
}
```

The first non-null dataset fingerprint becomes run identity. A later conflicting
fingerprint or run name fails rather than mixing artifacts. Stage config,
device, entrypoint, manifest, git, and package provenance do not belong here.

### `metadata/stages/<stage>.json`

The current stage view is replaced on every attempt. Its normalized shape is:

```json
{
  "schema_version": 1,
  "stage": "train",
  "status": "completed",
  "started_at": "...",
  "completed_at": "...",
  "entrypoint": "vs train",
  "config": {
    "input_path": "config/train/input.yaml",
    "resolved_path": "config/train/resolved.yaml",
    "sha256": "sha256:..."
  },
  "environment_path": "metadata/environments/train.json",
  "dataset": {
    "fingerprint": "sha256:...",
    "manifest_path": ".../manifest.csv",
    "manifest_sha256": "sha256:..."
  },
  "details": {}
}
```

Failed stages add `error_type` and `error`. Stage-specific results remain under
`details`; later `run.result()` calls replace earlier values by key.

### `metadata/events.jsonl`

Append-only events use the same config/environment/dataset/details fields plus
`schema_version`, `timestamp`, `run_id`, `run_name`, `stage`, `event_type`, and
`status`. The event is appended before the stage and run JSON views are replaced.
All local writes are strict; external reporters run only after successful local
writes and cannot invalidate them.

### `metrics/epochs.csv`

One canonical row per training epoch. The header is the deterministic union of
train, validation, configured component-loss, and validation-image columns.
Values use six decimal places; missing or non-finite validation values are blank.
Rows flush after every epoch. Resume requires a matching header and complete
epochs `0..resume_at-1`; stale rows at or after `resume_at` are discarded before
new rows append. Missing, malformed, gapped, duplicate, or incompatible history
fails instead of silently truncating it.

### `checkpoints/ep<NNN>.pth`

PyTorch checkpoint saved every `training.checkpoint_rate` epochs.
`NNN` is zero-padded to three digits (e.g. `ep010.pth`).
The checkpoint contains generator and discriminator state dicts plus the epoch
number. Use `inference.checkpoint_policy: latest` to load the most recent one
automatically.

### `checkpoints/best.json`

Machine-readable checkpoint selection record written during validation. It
records per-metric `best` and ranked `records` for all finite validation
checkpoint metrics available at a checkpointed validation epoch. Each record
includes `rank`, `epoch`, `checkpoint_path`, and `metric_value`, plus
config/loss context when available.

Inference can use `checkpoint_policy: best` with `checkpoint_metric` to load
rank 1 for that metric. `checkpoint_policy: top_k` additionally uses
`checkpoint_rank`.
Checkpoint files are not deleted by this metadata record.

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
