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
    - losses.generator
    - losses.discriminator
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
│   ├── validation.csv          # validation losses and image-quality metrics
│   └── all.csv                 # combined train and validation epoch metrics
├── artifacts/
│   ├── output_train/           # generated images for train-split samples
│   ├── output_val/             # generated images for validation-split samples
│   └── output_test/            # generated images for test-split samples (from vs-infer)
├── evaluation/
│   ├── artifacts.json          # manifest of core evaluation artifacts
│   ├── per_image_metrics.csv   # per-image metrics for all evaluated test samples
│   ├── summary.csv             # aggregate statistics across the test split
│   ├── weak_tail.csv           # weak-tail threshold counts and percentiles (non-empty evaluations)
│   ├── residual_heatmaps.csv   # standalone residual heatmap manifest (optional)
│   ├── residual_heatmaps/      # standalone residual heatmap PNGs (optional)
│   ├── sorted_by_metrics/      # ranked sample file exports from vs-organize (optional)
│   └── skipped.csv             # samples that could not be evaluated (optional)
└── comparisons/                # diagnostic panels produced by vs-render-panels
```

## File Descriptions

### `config/input.yaml`

Verbatim copy of the YAML file passed to `--config`. Preserved for full
reproducibility - re-running with this file reproduces the same experiment.

### `config/resolved.yaml`

The fully expanded effective configuration after all defaults have been applied
and all derived paths resolved. Differences from `input.yaml` reflect default
values that were not explicitly set by the user.

Training losses are recorded under top-level `losses.generator` and
`losses.discriminator` lists. Training requires explicit loss terms. Registered
losses have a default weight of `0.0`; a term is active only when it is
explicitly listed, `enabled` is `true`, and its scheduled current weight is
nonzero. Explicitly listed terms must declare `weight`; unlisted registry
entries remain absent and inactive.

Training-only augmentation is recorded under top-level `augmentation`. When
enabled, the training split is virtually expanded in memory; no augmented patch
files are written, and validation/test data keep deterministic preprocessing.

```yaml
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

Optimizer LR schedules are separate from `losses.*.schedule`, which changes
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
losses:
  generator:
    - name: adversarial_bce
      weight: 1.0
      enabled: true
      schedule:
        type: constant
    - name: l1
      weight: 25.0
      enabled: true
      params:
        reduction: mean
      schedule:
        type: constant
    - name: ssim
      weight: 0.0
      enabled: false
      params:
        data_range: 1.0
        window_size: 11
        sigma: 1.5
        channel_mode: rgb
        reduction: mean
        mask:
          enabled: false
          source: foreground_mask
          foreground_weight: 1.0
          background_weight: 0.25
          ignore_empty_mask: true
      schedule:
        type: linear_warmup
        start_epoch: 0
        end_epoch: 5
  discriminator:
    - name: adversarial_bce
      weight: 1.0
      enabled: true
      schedule:
        type: constant
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
raise an error instead of being treated as all-foreground. Current datasets look
for sidecar patch masks named `<sample_id>_foreground_mask<ext>` in the same
split directory. `vs-prepare` writes those sidecar masks for accepted patches
when `preprocessing.save_masks: true`; the sidecar mask is the aligned target
foreground mask for that patch.

When configured loss terms are present, `metrics/train.csv`,
`metrics/validation.csv`, and `metrics/all.csv` add deterministic component
columns using normalized names:

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

Stable run-level provenance and aggregate summary. It identifies what the run is
without pretending to be the complete lifecycle record for every stage.
See [run.json Schema](#runjson-schema) below.

### `metadata/environment.json`

Snapshot of the runtime environment captured for the training stage:

- Python version and platform
- `torch`, `numpy`, `opencv`, `albumentations` package versions
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

For `evaluate`, output paths include `metrics_csv_path`, `summary_csv_path`,
`weak_tail_csv_path`, `residual_heatmaps_csv_path`, and
`artifact_manifest_path` when the corresponding artifacts are available.

### `metadata/events.jsonl`

Append-only execution log. Each line is a JSON object representing a lifecycle
event such as:

- `stage_started`
- `stage_completed`
- `stage_failed`

Events include `timestamp`, `run_name`, `stage`, `status`, `config_hash`, and
optional `details`.

### `metrics/train.csv`

One row per training epoch. Columns include at minimum `epoch`, `loss_G_train`,
and `loss_D_train`. Written incrementally during training so partial results are
available if the run is interrupted.

### `metrics/validation.csv`

One row per validation epoch. Columns include at minimum `epoch`, `loss_G_val`,
`loss_D_val`, `val_ssim`, `val_mae`, `val_rmse`, `val_psnr`, `val_pcc_gray`,
and `val_pcc_rgb_mean`. Validation image metrics are computed from generated and
target tensors converted from `[-1, 1]` to `[0, 1]`, reusing the same metric
semantics as test-set evaluation. Non-finite aggregates are written as blank
cells.

### `metrics/all.csv`

One row per training epoch with the union of `metrics/train.csv` and
`metrics/validation.csv` columns. Validation columns are blank on epochs where
validation does not run.

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

### `evaluation/artifacts.json`

Canonical manifest of evaluation artifacts. `vs-evaluate` writes the core
manifest after evaluation completes. Secondary utilities append their outputs
when they can identify the source run: `vs-render-panels from-metrics`
registers metric panel artifacts and `vs-organize` registers ranked sample file
exports. If the base file is missing, a secondary utility creates a compatible
manifest containing its own artifacts. The manifest only lists files or
directories that exist at manifest write time. Paths are relative to the run
root when the artifact is under the run directory; artifacts written outside
the run root use absolute paths and set `path_type` to `absolute`.

Top-level fields:

| Field | Description |
|---|---|
| `schema_version` | Manifest schema version. Current value: `1` |
| `created_at` | ISO-8601 timestamp from the manifest creation time, normally evaluation completion |
| `updated_at` | ISO-8601 timestamp from the latest secondary manifest update, when present |
| `path_policy` | Human-readable path policy |
| `artifacts` | List of artifact records |

Artifact record fields:

| Field | Description |
|---|---|
| `stage` | Creation stage, such as `evaluate`, `render_panels`, or `organize` |
| `artifact_type` | Stable artifact type such as `per_image_metrics_csv`, `summary_csv`, `weak_tail_csv`, `skipped_csv`, `metric_histogram`, `metrics_boxplot`, `residual_heatmaps_csv`, `residual_heatmap_png`, `selection_summary`, `comparison_panel`, `diagnostic_image`, `diagnostic_panel`, `organization_summary`, or `ranked_sample_export` |
| `path` | Artifact path using the manifest path policy |
| `path_type` | `run_relative` or `absolute` |
| `metric` | Metric associated with the artifact, or `null` |
| `sample_id` | Sample associated with the artifact, or `null` |
| `description` | Short human-readable description |
| `metadata` | Artifact-specific key/value metadata, including the creating `command` for secondary utility artifacts |

Secondary utility metadata includes enough context for consumers to avoid path
guessing. `render_panels` entries include the command, source run, selected
metrics, kind, rank, and metric values where applicable. `organize` entries
include the command, source run, selected metrics, top-k, export mode,
rank-count, selected file roles, and exported file counts. Rerunning a
secondary utility replaces that utility's prior manifest entries while
preserving artifacts from other stages.

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

### `evaluation/weak_tail.csv`

Weak-tail statistics for thresholded metric failures. Written for non-empty
evaluations. Non-finite metric values are excluded from `finite_count` and from
the `weak_share` denominator, and are reported in `non_finite_count`.
Higher-is-better metrics are weak when their finite value is below the
threshold; lower-is-better metrics are weak when their finite value is above the
threshold.

Default weak-tail thresholds are:

| Metric | Direction | Weak rule |
|---|---|---|
| `ssim` | higher is better | `< 0.60` |
| `psnr` | higher is better | `< 20.0` |
| `mae` | lower is better | `> 0.08` |
| `rmse` | lower is better | `> 0.12` |
| `mse` | lower is better | `> 0.0100` |
| `pcc_gray`, `pcc_rgb_mean`, `pcc_r`, `pcc_g`, `pcc_b` | higher is better | `< 0.80` |

| Column | Description |
|---|---|
| `metric` | Metric name |
| `direction` | `higher_is_better` or `lower_is_better` |
| `weak_rule` | Threshold comparison used for weak samples (`<` or `>`) |
| `threshold` | Weak-tail threshold |
| `count` | Number of metric values present in the input rows |
| `finite_count` | Number of finite metric values used for weak-tail statistics |
| `non_finite_count` | Number of NaN or infinite values excluded from weak-tail statistics |
| `weak_count` | Number of finite values on the weak side of the threshold |
| `weak_share` | `weak_count / finite_count`; `nan` when no finite values exist |
| `worst_value` | Minimum value for higher-is-better metrics, maximum value for lower-is-better metrics |
| `p05`, `p10`, `p90`, `p95` | Percentiles computed over finite values |

### `evaluation/residual_heatmaps.csv`

Written only when `evaluation.save_residual_heatmaps: true`. The evaluator
ranks finite per-image metric rows by `evaluation.residual_heatmap_metric`,
exports the worst `evaluation.residual_heatmap_top_k` samples, and writes each
PNG under `evaluation/residual_heatmaps/`. Defaults are disabled export,
metric `ssim`, and top-k `25`.

| Column | Description |
|---|---|
| `rank` | Worst-case rank for the selected metric |
| `sample_id` | Sample identifier from `per_image_metrics.csv` |
| `metric` | Metric used for ranking |
| `metric_value` | Per-image value used for ranking |
| `target_path` | Absolute path to the ground-truth target image |
| `generated_path` | Absolute path to the generated image |
| `heatmap_path` | Absolute path to the standalone residual heatmap PNG |

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

### `comparisons/<run_a>_vs_<run_b>/<mode>_<metric>/`

Written by single-metric `vs-compare unpaired` and `vs-compare paired` runs.
Existing artifacts such as `comparison_summary.csv`, `group_statistics.csv`,
`paired_sample_deltas.csv`, `summary.json`, `report.txt`, and plot PNGs remain
available. The compare report also writes decision-scoring artifacts that make
the score-based suggestion transparent. These scores are engineering summaries,
not formal statistical conclusions.

Additional unpaired files:

| File | Description |
|---|---|
| `unpaired_decision_breakdown.csv` | Ordered scoring criteria with values for run A and B, direction-aware signed difference, favored label, criterion weight, and per-run score contribution |
| `unpaired_quantile_comparison.csv` | q10/q25/q50/q75/q90 values for both runs, raw B-A deltas, direction-aware signed deltas, and favored label |
| `unpaired_threshold_shares.csv` | Favorable threshold shares for each configured threshold and the share delta favoring A or B |

Additional paired files:

| File | Description |
|---|---|
| `paired_decision_breakdown.csv` | Ordered scoring criteria based on mean/median signed deltas, share improved, q25 signed delta, and worst-tail q10 signed delta |
| `paired_delta_summary.csv` | One-row summary with signed-delta quantiles, relative signed-delta aggregates, improved/worsened/equal shares, score fields, decision strength, reason, and score-based suggestion |

Decision fields use `score_a`, `score_b`, `score_difference`,
`decision_strength`, and `decision_reason`. `decision_strength` is one of
`tie`, `weak`, `moderate`, or `strong`. For all signed deltas, positive means
run B improved after applying the metric direction.

### `comparisons/<run_a>_vs_<run_b>/paired_all_metrics/`

Written by `vs-compare paired --column all`, by `vs-compare paired --metrics
...`, or by `compare.mode: paired` with `column: all` or `metrics` in YAML.
The command aligns `evaluation/per_image_metrics.csv` files by `sample_id` and
writes multi-metric paired delta reports without changing the existing
single-metric comparison artifacts.

| File | Description |
|---|---|
| `paired_sample_deltas_all_metrics.csv` | Wide per-sample report with `<metric>_a`, `<metric>_b`, `<metric>_raw_delta_b_minus_a`, `<metric>_signed_delta`, and `<metric>_winner` columns |
| `paired_metric_delta_summary.csv` | One row per metric with direction, tolerance, matched counts, missing/non-finite counts, improved/worsened/equal counts and shares, and mean/median raw and signed deltas |

`raw_delta_b_minus_a` is always run B minus run A. `signed_delta` is direction
aware: positive means run B improved for that metric, including lower-is-better
metrics such as MAE and RMSE.

### `comparisons/metrics/`

Written by `vs-render-panels from-metrics` or `render_panels.mode:
from_metrics`. The command reads `evaluation/per_image_metrics.csv` and
`evaluation/summary.csv`, then writes rendered panels and diagnostics under
one subdirectory per metric:

```text
comparisons/metrics/
├── metrics_selection_summary.csv
└── <metric>/
    ├── selection_summary.csv
    ├── best_<sample_id>_comparison.png
    ├── median_<sample_id>_comparison.png
    ├── worst_<sample_id>_comparison.png
    └── diagnostics/
```

By default, `top_k: 1` preserves the historical best/median/worst filenames.
When `top_k > 1`, ranked filenames include the rank, for example
`worst_001_<sample_id>_comparison.png`. Optional `metrics` and `kinds` config
fields can restrict which metric names and representative kinds are rendered.
Metric direction and sample tie-breaking use the shared ranked-selection helper
also used by `vs-organize`: SSIM, PSNR, and PCC metrics rank higher values as
better, while MAE, MSE, and RMSE rank lower values as better. Ties use
`sample_id` when available.

| Column | Description |
|---|---|
| `metric` | Metric used for ranking |
| `kind` | Representative kind: `best`, `median`, or `worst` |
| `rank` | Rank within the metric/kind selection |
| `sample_id` | Sample identifier from `per_image_metrics.csv` |
| `metric_value` | Per-image value used for ranking |
| `target_value` | Summary target for the kind (`max`, `median`, or `min` depending on metric direction) |
| `source_path` | Source image path inferred from the sample row |
| `target_path` | Ground-truth target image path |
| `generated_path` | Generated image path |
| `comparison_path` | Saved source/generated/target/MAE-map panel |
| `error_histogram_path` | Saved absolute-error histogram |
| `intensity_overlay_histogram_path` | Saved target/generated intensity overlay histogram |
| `target_vs_generated_scatter_by_channel_path` | Saved target-vs-generated scatter plot |

When run with a source run path, `vs-render-panels from-metrics` appends these
outputs to `evaluation/artifacts.json` using `selection_summary`,
`comparison_panel`, `diagnostic_image`, and `diagnostic_panel` artifact types.

### `evaluation/sorted_by_metrics/`

Written by `vs-organize` or `organize` config sections. This utility is a
ranked file exporter: it places source, target, and generated image files into
metric-ranked folders for manual inspection or downstream collection. It does
not render panels, plots, or diagnostics; use `vs-render-panels`
for visual diagnostics and `vs-evaluate` for dataset evaluation artifacts.

The default output directory is `RUN/evaluation/sorted_by_metrics/`:

```text
evaluation/sorted_by_metrics/
├── organization_summary.csv
└── <metric>/
    ├── best/
    ├── worst/
    └── all_ranked/             # optional, when include_all_ranked is enabled
```

Ranking uses the shared ranked-selection helper also used by
`vs-render-panels`: SSIM, PSNR, and PCC metrics rank higher values as better,
while MAE, MSE, and RMSE rank lower values as better. Ties use `sample_id`
when available. File placement mode is controlled by `mode` and may be
`hardlink`, `symlink`, or `copy`.

`organization_summary.csv` has one row per metric/kind export:

| Column | Description |
|---|---|
| `metric` | Metric used for ranking |
| `kind` | Export kind: `best`, `worst`, or `all_ranked` |
| `rank_count` | Number of ranked samples selected for that kind |
| `export_mode` | File placement mode: `hardlink`, `symlink`, or `copy` |
| `selected_file_roles` | Comma-separated exported roles, such as `generated,target,source` |
| `output_dir` | Directory containing the exported files for this metric/kind |
| `files_exported` | Number of files placed for this metric/kind |

When `vs-organize` is run with `--run-path`, a run config, or a
`--metrics-csv` path under `RUN/evaluation/per_image_metrics.csv`, it appends
`organization_summary` and `ranked_sample_export` entries to
`evaluation/artifacts.json`. Standalone `--metrics-csv` usage outside a run
still writes the export folders and summary CSV, but manifest registration is
skipped because there is no run root to anchor relative paths.

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
