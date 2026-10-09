# Run Output Format

## Local Queues

Queue definitions are stored as YAML files under `config/queues/`. Runtime
queue state is written under `local_workspace/queues/`. Queue execution is
explicitly local, sequential, and single-worker in v1. Every queue and ablation key
is documented in [`config/queues/example.yaml`](../config/queues/example.yaml) and
[`config/queues/example_ablation.yaml`](../config/queues/example_ablation.yaml).

Every job resolves with its explicit `stages`, or the full `prepare, train, infer,
evaluate` sequence when omitted. All configurations and ablation comparisons pass
before any stage starts, even with `continue_on_failure: true`. Configuration preflight
reads YAML only; it does not validate assets or freeze consumed data. Queue state still
records preflight failure and leaves other jobs pending. Execution validates and binds
its own input snapshots. Job paths are queue-directory-relative; paths inside run YAML
retain their existing bases.

See [operation-specific jobs](../config/queues/example_operations.yaml) and
[default full-run jobs](../config/queues/example_full.yaml).

Ablation summaries are written beside queue state as
`local_workspace/queues/<queue-name>.ablation.summary.json`. The summary lists
jobs, labels, run names, canonical resolved config hashes, declared fixed
values, and declared variable values. Loss lists are compared through resolved
loss config entries, so omitted default-zero terms are not treated as
active losses. Every requested fixed or variable dot path must exist in every resolved job. An absent/inapplicable field fails preflight with the job, path
and stages; it is never a meaningful ablation choice. Choose applicable fields or separate
operations into queues. Prepare-only summaries omit `run_name` when absent. Existing
ablation `config_hash` values identify loss-order-normalized JSON for comparison, not
tracked resolved-YAML bytes; that established comparison format is unchanged.

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

Run configuration options belong to the exhaustive
[Pix2Pix](../config/runs/example.yaml) and
[CycleGAN](../config/runs/example_cyclegan.yaml) YAML references. Python extensions are
covered in [Library Stage API](library_api.md#extending-with-explicit-definitions).

Selected-operation requirements and snapshot reconstruction are documented in the
[configuration API](library_api.md#selected-operation-configuration). Resolved YAML
snapshots record explicit resolution stages in a deterministic comment; that comment
participates in the config hash. Authored `input.yaml` is preserved byte for byte.
Preparation has no tracked run identity and keeps these artifacts under the dataset.

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
│   ├── consumed_data/
│   │   ├── train/{snapshot.json,rows.csv}
│   │   ├── infer/{snapshot.json,rows.csv}
│   │   └── evaluate/{snapshot.json,rows.csv}
│   ├── produced_data/
│   │   └── infer/{snapshot.json,rows.csv}
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
    ├── evaluation_metadata.json
    ├── per_image_metrics.csv      # paired protocol
    ├── summary.csv                # paired protocol
    ├── summary_<unit>.csv         # paired protocol, unit = set | specimen | patient
    ├── coverage.csv               # paired protocol: one row per requested sample
    ├── evaluation_result.json     # paired protocol: metric identities, statuses, counts
    ├── unpaired_image_statistics.csv    # unpaired protocol
    └── unpaired_feature_comparison.csv  # unpaired protocol
```

Cross-run comparisons are written separately under
`local_workspace/results/comparisons/<A>_vs_<B>/<mode>_<metric>/`.
Preparation writes [dataset-local artifacts](dataset_format.md#prepared-layout),
not experiment `run.json`, `events.jsonl`, or `metadata/stages/prepare.json`.

## File Descriptions

### `config/<stage>/input.yaml`

Verbatim copy of the YAML file passed to `--config` for that stage.

### `config/<stage>/resolved.yaml`

The effective configuration with parsed values and defaults, serialized with sorted
YAML keys. Its hash identifies these bytes only; see [Reproducibility](reproducibility.md).

`vs config resolve --config ... --stages train` prints these exact bytes without running
a stage, and `vs config check --config ... --stages train` prints their SHA-256 as
`config_sha256` (the stage's `config_hash`). Supply the same ordered stages as execution,
e.g. `--stages train infer` for that selected pipeline. Without stages these commands
perform unscoped inspection, with a different snapshot identity and no execution assurance.
The deterministic stage-context comment is part of the canonical bytes, not a schema key.

Losses, augmentation, schedulers, and early stopping retain their effective values in
this snapshot; their configuration semantics are documented in the
[run YAML references](../config/runs/example.yaml).

### `metadata/run.json`

Run identity and aggregate state only (schema version 2; version 1 files, which held a
single incidental `dataset_fingerprint`, are rejected - start a new run directory):

```json
{
  "schema_version": 2,
  "run_id": "uuid",
  "run_name": "example_run",
  "created_at": "2025-01-15T10:30:00+00:00",
  "training_data": {
    "snapshot_id": "sha256:...",
    "membership_sha256": "sha256:...",
    "hash_policy": "content"
  },
  "last_event_at": "2025-01-15T12:45:00+00:00",
  "stages_present": ["train", "infer", "evaluate"],
  "last_completed_stage": "evaluate"
}
```

`training_data` is the consumed-data identity of the first bound training attempt and
is part of run identity. A later training attempt, including a `resume`, whose train or
validation membership, split, domain role, group metadata, or verified content differs
fails before training starts; use a new `run_name`. Inference and evaluation inputs are
stage-specific and are never required to equal the training inputs. Stage config,
device, entrypoint, manifest, git, and package provenance do not belong here.

### `metadata/stages/<stage>.json`

The current stage view is replaced on every attempt (schema version 2). Its normalized
shape is:

```json
{
  "schema_version": 2,
  "stage": "infer",
  "status": "completed",
  "started_at": "...",
  "completed_at": "...",
  "entrypoint": "vs infer",
  "config": {
    "input_path": "config/infer/input.yaml",
    "resolved_path": "config/infer/resolved.yaml",
    "sha256": "sha256:..."
  },
  "environment_path": "metadata/environments/infer.json",
  "consumed_data": {
    "snapshot_id": "sha256:...",
    "membership_sha256": "sha256:...",
    "hash_policy": "content",
    "content_verified": true,
    "row_count": 42,
    "metadata_path": "metadata/consumed_data/infer/snapshot.json",
    "rows_path": "metadata/consumed_data/infer/rows.csv"
  },
  "produced_data": {"snapshot_id": "sha256:...", "...": "..."},
  "details": {}
}
```

`consumed_data` references the frozen inventory of files this stage read (see
[Consumed-data snapshots](#consumed-data-snapshots)); rows are never inlined.
`produced_data` is set only by inference, after its outputs are written. Failed stages
add `error_type` and `error`. Stage-specific results remain under `details`; later
results replace earlier values by key.

A stage-start record binds its config, environment, and consumed-data identity. Failure
before inputs are bound records `consumed_data: null` and the original error; a snapshot
already written remains evidence of the failed attempt. Metadata is a local
single-writer store: concurrent writers to one run are not coordinated.

### `metadata/events.jsonl`

Append-only events use the same config/environment/consumed/produced/details fields plus
`schema_version`, `timestamp`, `run_id`, `run_name`, `stage`, `event_type`, and
`status`. The event is appended before the stage and run JSON views are replaced.
All local writes are strict; external reporters run only after successful local
writes and cannot invalidate them.

### Consumed-data snapshots

A consumed-data snapshot answers *which exact assets did this tracked stage consume?*
It is distinct from the dataset fingerprint, which answers *what dataset did
preparation build?* (see [Dataset Format](dataset_format.md)). Snapshot membership
matches the inputs supplied to the stage computation.

`snapshot.json` (snapshot schema version 1) binds the SHA-256 of `rows.csv`.
Each file is published atomically, but the pair is not a transaction; readers reject
mismatched metadata and rows after an interrupted update.
Row columns:

```text
root,locator,role,domain,split,sample_id,set_id,specimen_id,patient_id,status,size,sha256
```

- `root` names a binding (`dataset`, `generated`, `output`) and `locator` is a
  normalized path relative to it. Absolute, traversing, or root-escaping locators and
  escaping symlinks are rejected. Absolute roots appear only in `root_binding`.
- `role` is `input`, `target`, `mask`, `generated`, or `reference`; `domain` is the
  modality or domain name.
- `sample_id` is set only where the adapter has an explicit correspondence (manifest
  samples). Independent unpaired-domain images have none.
- `status` is `present` or `missing` (a requested evaluation file that was absent).

Identity: `membership_sha256` hashes the canonically sorted rows; `snapshot_id` also
covers the adapter, hash policy, selection (modalities/domains, splits), and context
(for inference: method, direction, checkpoint SHA-256, and a digest of the generation
configuration). Neither depends on timestamps, absolute mount points, CSV row order, or
directory traversal order.

Hash policy (`data.hash_policy`):

- `content` (default): every consumed file is SHA-256 hashed and must remain stable
  while hashed. This is the mode for a content-identical freeze. A hashing failure never falls back.
- `membership`: records locators, sizes, and semantic metadata only.
  `content_verified: false` and an explicit limitation mark it as unverified.

Duplicate and leakage checks: the same resolved file, an alias of it (symlink or
hard link), or identical verified bytes in two different splits fail. Identical content
within one split is listed under `duplicates` for review, never deduplicated. Byte
identity is only asserted under `content`; `membership` still detects same-file aliases.

Held-out assets: training also observes the held-out test assets of its selected data
contract - the test manifest records' selected inputs, targets, and masks (when used) for
paired training, and the domain A and B `test` collections resolved from the same
`data.domains` specs for unpaired training - under the same hash policy. They take part
in the file, content, and group leakage checks but are not consumed rows and do not
enter `snapshot_id`; the metadata's `validation_context` records their row count,
splits, and membership digest.

Biological groups (`data.group_validation`): `auto` (default) validates split
independence at the strongest unit for which every asset has an ID
(`patient` > `specimen` > `set`); an explicit unit requires complete IDs for it. Any
observed group ID that appears in more than one split fails, including held-out test
assets during training. The same group may appear across modalities or domains within
one split. The result is stored under
`group_validation` (`validated` with its `unit`, `unavailable`, or `not_applicable`
for single-split snapshots). Paired stages take IDs from the prepared
`manifests/slide_sets.csv`; unpaired `data.domains` collections take them only from an
optional `data.group_metadata` CSV sidecar with columns
`path,domain,split,set_id,specimen_id,patient_id`; entries for unselected paths are
ignored. Without group IDs, training requires an explicit `group_validation:
unavailable`, which is persisted with a limitation and makes no patient, specimen, or
set independence claim. `unavailable` is not a validation-off switch: any supplied ID
that appears in more than one split still fails. The one exception is a prepared
patch-level split (`split.unit: patch`, read from `metadata/split_assignment.csv`;
unpaired preparation also verifies the selected canonical group sidecar and assignment
against its build record)
trained under explicit `unavailable`: its groups span splits by construction, so the
shared group counts are recorded under `group_validation` with a limitation instead of
failing. `auto` or an explicit unit still fails on the same data. IDs are never
inferred from filenames or directories.

These are file-provenance checks only. Distinct SHA-256 digests do not prove
biological independence, re-encoded or near-duplicate images are not detected, and no
clinical or biological validity follows from them.

Per stage:

- **train** (paired): the files each train/val manifest record supplies - selected
  inputs, every selected target (`domain` is the target name), and each target's
  foreground mask only when a mask loss requires it - with set/specimen/patient IDs.
  Augmentation expansion is configuration, not extra rows.
  `sources` records the manifest SHA-256 and dataset fingerprint.
- **train** (unpaired): train/val membership of domain A (`input`) and domain B
  (`target`). The seeded epoch draw is a sampling policy (stage `details`), not a
  correspondence, so no pairs are recorded.
- **infer**: only the files fed to the predictor - the selected inputs for Pix2Pix
  and CycleGAN `A_to_B`, the held-out target for `B_to_A`. These are exactly the
  image files inference opens; the other side of each record is never read. After writing, the
  `produced_data` snapshot lists one `generated` row per `(sample_id, output_name)` with
  the output name as `domain` and its content identity, and references the consumed
  snapshot and checkpoint.
- **evaluate** (paired): one `reference` and one `generated` row per evaluated
  `(sample_id, output_name)` pair, sharing that `sample_id` and naming the output as
  `domain` - the explicit correspondence the evaluator used.
  Missing generated files keep `status=missing`; evaluated/excluded counts stay in
  stage `details` and `coverage.csv`.
- **evaluate** (unpaired): the generated and reference collections, without
  correspondence; `selection.reference_spec` records the reference collection used.

Evaluation lineage: `details.generated_producer` (also in
`evaluation_metadata.json`) is `linked` only when the consumed generated files match
this run's completed inference `produced_data` by locator, size, and content; it then
names the inference output and input snapshots, checkpoint SHA-256, and direction.
Otherwise it is `unlinked` with `missing`, `extra`, and `changed` locators or a
direction mismatch, or `external` when no tracked inference exists. External
predictions remain valid evaluation inputs; no checkpoint origin is fabricated for
them, and a matching directory path alone never establishes a link.

### `metrics/epochs.csv`

One canonical row per training epoch. The header is the deterministic union of
train, validation, configured component-loss, and validation-image columns.
Values use six decimal places; missing or non-finite validation values are blank.
Training loss/component columns use `step_mean`: the unweighted mean over the
epoch's optimization steps, so a smaller final batch counts as much as a full one.
Validation loss/component columns are likewise means over validation batches, while
validation image metrics (`val_ssim__HE`, `val_mae__PAS`, ...) are means over individual
images, skipping non-finite values. Validation-image columns are those the method
declares: Pix2Pix reports `val_ssim__<output>`, `val_psnr__<output>`, `val_mae__<output>`,
`val_rmse__<output>`, `val_pcc_rgb_mean__<output>` and `val_pcc_gray__<output>` for every
model output (no column averages outputs); CycleGAN reports none.
Rows flush after every epoch. Resume requires a matching header and complete
epochs `0..resume_at-1`; stale rows at or after `resume_at` are discarded before
new rows append. Missing, malformed, gapped, duplicate, or incompatible history
fails instead of silently truncating it.

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

Pix2Pix reports every reconstruction term per output as `<loss_name>__<output>`; adversarial
terms and the totals stay joint. For example, configured SSIM with `model.outputs: [HE,
PAS]` writes `loss_train_raw_generator_ssim__HE`, `loss_train_raw_generator_ssim__PAS`,
and the matching `weighted` and `current_weight` columns, plus validation columns when
validation runs. `raw` is that output's term, `weighted` its contribution to `loss_G`
(`current_weight * raw / M`), and `current_weight` the scheduled term weight. The same
loss parameters apply to every output; this training mean is not an evaluation score.
CycleGAN objective components sum the two domain/direction contributions.

### `checkpoints/ep<NNN>.pth`

PyTorch checkpoint saved every `training.checkpoint_rate` epochs.
`NNN` is zero-padded to three digits (e.g. `ep010.pth`).
Checkpoints use format version 4, the only supported format; v3, earlier and
unversioned checkpoints are rejected (retrain with current code). The payload is
topology-neutral:

| Key | Content |
|---|---|
| `format_version` | `4` |
| `epoch` | Completed epoch; resume starts at `epoch + 1` |
| `method` | Registered method `name`; `implementation` (`version`, `source`) of that definition; `pairing`; ordered `inputs`; ordered `outputs`; `prediction_directions`; method-level reconstruction `options`; and `components`, each a registered component identity (`name`, `version`, `source`) with its normalized constructor `options` |
| `image_size` | `[width, height]` |
| `normalization` | Model-I/O normalization contract |
| `config_hash` | Resolved-config hash of the writing stage, or `null` (provenance only) |
| `state` | Method-owned state from `state_dict()`: models, `optimization` policy, optimizers, AMP scalers, schedulers, and CycleGAN replay pools |

Checkpoints are read onto the CPU with PyTorch's restricted `weights_only=True`
unpickler, so only tensors and primitive containers are accepted; there is no
unrestricted fallback. This narrows arbitrary-code deserialization exposure but is
not a resource sandbox. State reaches the execution device through normal model and
optimizer restoration.

Names in `method` are compared with the definitions the caller supplied; they are never
imported. A checkpoint naming a method or component that is not registered is rejected
before anything is built, and v4 checkpoints written before explicit definition identity
existed (no `method.implementation`) are rejected rather than converted. The resolved
config hash is recorded for provenance only, so output, evaluation, or reporting paths do
not affect compatibility.

Resume and inference require compatible semantic metadata and method state before
restoration. If restoration fails after validation, the runtime may be partially
restored and must be discarded.

`state.optimization` records, per optimizer role, the configured optimizer policy
(class, initial `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`) and the
scheduler policy (`null` when none; `linear_decay` adds `decay_start_epoch` and the
`epochs` basis of its decay horizon; `reduce_on_plateau` adds `monitor`, `mode`,
`factor`, `patience`, `min_lr`). Resume requires the current configuration to match
exactly. Changing optimizer or scheduler policy, including `training.epochs` under
`linear_decay`, is a warm start rather than a resume and is rejected. Scheduler-decayed
learning rates are restored from optimizer state and are not compared. `config_hash`
stays provenance only.

Resume restores model, optimizer, AMP scaler, scheduler, and replay-pool state (with
its RNG) as of a completed epoch and continues at `epoch + 1`. Global Python, NumPy,
and torch RNG, DataLoader shuffling and worker RNG, and augmentation RNG are **not**
checkpointed, so a resumed run is not guaranteed to reproduce the stochastic
trajectory of an uninterrupted run.

A failed or interrupted save never exposes partial checkpoint bytes or replaces an
existing checkpoint. Selection considers only successfully published checkpoints.

### `checkpoints/best.json`

Machine-readable checkpoint selection record (schema version 2) written during
validation. It records per-metric `mode`, `best` and ranked `records` for all finite
checkpoint metrics the method ranks at a checkpointed validation epoch; each method
declares its metrics and their `min`/`max` direction. Each record includes `rank`,
`epoch`, `checkpoint_path`, and `metric_value`, plus `config_hash` and the method's
optional JSON `objective_metadata` (the built-ins store their resolved `training.losses`)
when available. Version 1 files, which stored a built-in `loss_config`, are rejected;
start a new run directory.

Inference can use `checkpoint_policy: best` with `checkpoint_metric` to load
rank 1 for that metric. `checkpoint_policy: top_k` additionally uses
`checkpoint_rank`.
Checkpoint files are not deleted by this metadata record.

### Generated images

`artifacts/output_train/`, `output_val/`, and `output_test/`:

Inference writes test predictions to `output_test/`; training previews use `output_val/`.
The Trainer creates `output_train/`, but built-in methods currently write no images there.
Each inference artifact is identified by `(sample_id, output_name)` and written as
`<output_dir>/<output_name>/<sample_id>_generated.<ext>`, one file per predicted output, so
all Pix2Pix outputs and both CycleGAN directions share one output directory without
collisions and every path inverts to its pair:

```text
HE/00512_09216_generated.tif          # Pix2Pix output HE
PAS/00512_09216_generated.tif         # Pix2Pix output PAS
stained/00512_09216_generated.tif     # CycleGAN A_to_B predicts domain B
label_free/00512_09216_generated.tif  # CycleGAN B_to_A predicts domain A
```

Pix2Pix validation previews in `output_val/` are `epoch<e>_batch<b>_input.tif`
(the first configured input) plus `..._output__<output>.tif` and
`..._target__<output>.tif` per output. CycleGAN writes
`epoch<e>_batch<b>_preview.tif` grids of real A, fake B, real B, and fake A.

`vs infer-images --recursive` uses the same layout and preserves the input's relative
directory structure (before `<output_name>`) under the output root, so equal filenames
in different source folders do not collide. An explicit output file is accepted only for
a one-output model; several outputs need an output directory. Inputs that would map to the same output file (for example `a.png` and
`a.tif` with `--output-format png`) are rejected before prediction. An existing output
file is replaced atomically only after its prediction succeeds. Without `--output`,
files go to `inference.output_dir` or the run's `artifacts/output_single/` (one file)
or `artifacts/output_images/` (directories). Library callers without run defaults
must pass an output path (see [`library_api.md`](library_api.md#direct-predictor-inference)).

Full-resolution WSI outputs are pyramidal BigTIFFs with the shared input pixel dimensions.
MPP is copied from source metadata only when every input provides compatible known
values (relative tolerance `1e-4`); conflicting known values fail before prediction,
even if another input lacks calibration. Missing calibration leaves MPP unknown, never zero or inferred from
pixel counts. A TIFF has one resolution unit, so both output axes remain unknown if only
one axis is known. Every output shares the same grid and MPP; this does not certify
biological registration. Operational requirements are in
[Direct predictor inference](library_api.md#direct-predictor-inference).

## Evaluation outputs

A successful tracked evaluation writes `evaluation/evaluation_metadata.json` (`schema_version` 3)
recording `method`, `training_pairing`, `evaluation_protocol`, `inference_direction`
(`null` for Pix2Pix), `source_domains`, `reference_domains` (the predicted outputs),
`generated_dir`, `counts`,
the written `artifacts`, the `consumed_data` reference and `generated_producer` lineage,
and `pairwise_metrics_available`: `true` for `paired`, `false` for `unpaired`. Unpaired
metadata also records the feature definitions and explicit `limitations`. For the paired
protocol, `artifacts.evaluation_result` points to `evaluation_result.json`, which owns
the metric and coverage semantics. Switching protocol removes the other protocol's stale
reports (including every `*_histogram.png`) from the output directory. Earlier schema
versions are not read or migrated.

The **paired** protocol writes `per_image_metrics.csv`, `summary.csv`, `coverage.csv`,
`evaluation_result.json`, and grouped `<unit>_metrics.csv` / `summary_<unit>.csv` for
`set`, `specimen`, and `patient` (bootstrap confidence intervals). Every aligned test
record yields one explicit pair per predicted output, `(sample_id, output_name)`: the
record's real image of that output (the target for Pix2Pix and CycleGAN `A_to_B`, the
domain-A input for CycleGAN `B_to_A`) against the generated artifact of that output.
Every report keeps `output_name`; no statistic, plot or ranking averages different
outputs. Strict/permissive input failures apply per pair, so a missing `PAS` prediction
never hides a present `HE` prediction. With `save_graphs: true` it also writes one
`<output>__<metric>_histogram.png` per output and requested metric and
`metrics_boxplot.png` (one box per output and metric), over finite values only.

The **unpaired** protocol (CycleGAN default, any one-output method opt-in; rejected for
a model with several simultaneous outputs, since one reference collection has no
unambiguous per-output contract) compares the active direction's generated images of
its one output with the real `test` collection of
`evaluation.reference_collection`, else of the reference domain from `data.domains`. No
pairs are formed. Each image is reduced to per-image RGB mean/std and luminance mean/std:

- `unpaired_image_statistics.csv`: `collection` (`generated` or `reference`), `path`,
  and the per-image features.
- `unpaired_feature_comparison.csv`: per feature, descriptive statistics of both
  collections, Wasserstein distance, and a two-sample KS statistic and p-value.
- `unpaired_feature_distributions.png`: optional plot when `save_graphs: true`.

These are low-order appearance/distribution diagnostics only. They carry no pairwise
fidelity metrics and no ranking of generated against real images, and they do not
establish sample-level fidelity, biological correctness, or clinical validity. They do
not use `evaluation.metrics`.

### Metric result statuses

Every requested metric of every evaluated image has exactly one status:

| Status | Value | Example |
|---|---|---|
| `finite` | a finite number | MAE of two images |
| `positive_infinity` | `inf` | PSNR of identical images |
| `undefined` | none, with a reason | PCC when either image is constant; any metric over an empty valid region |
| `unavailable` | none, with a reason | SSIM for an image smaller than its 7 px window |

A negative infinity, NaN or any other value outside these states is an evaluator
defect and fails the evaluation; nothing is silently normalized.

### `evaluation/per_image_metrics.csv`

Paired protocol only. One row per evaluated `(sample_id, output_name)` pair. The base
columns are `sample_id`, `output_name`, `set_id`, `target_path` (the real image of that
output), `generated_path`, `width`, `height`, `channels`
(plus `support_path` when valid-region support was used). Then, for each requested
metric `<m>` in request order:

| Column | Description |
|---|---|
| `<m>` | Finite value, `inf` for `positive_infinity`, empty for `undefined`/`unavailable` |
| `<m>_status` | One of the statuses above |
| `<m>_reason` | Why the value is undefined or unavailable; otherwise empty |
| `<m>_support_count` | With valid-region support only: valid pixels used |
| `<m>_support_fraction` | With valid-region support only: valid pixels / all pixels |

The built-in metrics (all over the full RGB image in [0, 1], all channels):

| Metric | Description | Better |
|---|---|---|
| `mae`, `mse`, `rmse` | Mean absolute / mean squared / root mean squared error | lower |
| `psnr` | `20 log10(1 / sqrt(mse))` dB | higher |
| `ssim` | scikit-image `structural_similarity` with `data_range=1`, `channel_axis=2`, `win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`, `K1=0.01`, `K2=0.03` | higher |
| `pcc_gray` | Pearson correlation of BT.601 luminance | higher |
| `pcc_r`, `pcc_g`, `pcc_b` | Pearson correlation per channel (not in the default request) | higher |
| `pcc_rgb_mean` | Mean of the defined per-channel PCCs | higher |

Evaluation SSIM is fixed as above and is independent of the training SSIM loss.

### `evaluation/summary.csv`

One row per output and requested metric (`output_name`, `metric`): `count` (evaluated
images of that output), `finite_count`,
`positive_infinity_count`, `undefined_count`, `unavailable_count` (these four sum to
`count`), and `finite_mean`, `finite_median`, `finite_std`, `finite_min`, `finite_max`
computed from finite values only (empty when there are none).

### Grouped summaries

`<unit>_metrics.csv` has one row per output and group (`output_name`, `unit`,
`group_id`, `patch_count`) with `<m>_finite_count` and `<m>_finite_mean` per requested
metric. `summary_<unit>.csv` bootstraps one output's groups with replacement:
`output_name`, `resampling_unit`, `metric`, `group_count` (groups
with a finite mean), `finite_mean` and `ci95_low`/`ci95_high`. Groups come from the
supplied `set_id` and the slide-set `specimen_id`/`patient_id`; a level is skipped when
any row lacks it. Each metric has its own seeded resampling stream, so adding a metric
never changes another metric's interval. Outputs of different stains are never averaged
together.

### `evaluation/coverage.csv`

One row per requested `(sample_id, output_name)` pair, the single source of coverage truth:

| Column | Description |
|---|---|
| `sample_id`, `output_name`, `set_id` | Pair and slide-set identifiers |
| `status` | `evaluated`, `excluded` (permissive) or `failed` (strict) |
| `reason` | Stable code: `missing_target`, `missing_generated`, `missing_support`, `unreadable_<role>`, `unsupported_<role>_mode`, `shape_mismatch`, `malformed_support`, `support_shape_mismatch` |
| `detail` | Human-readable message |
| `target_path`, `generated_path`, `support_path` | Input paths (`support_path` empty without support) |

It is written for known per-pair input failures, including strict-mode failure or zero
evaluable samples; those failures publish no new paired metric reports. Invalid requests
and unexpected evaluator errors may fail before coverage is written.

### `evaluation/evaluation_result.json`

The paired result contract (`schema_version` 1): the ordered `metrics` with their
resolved identity (`name`, `version`, `source`, `options`, `applicability`, `input`,
`supports_valid_region`, `higher_is_better`, and `presentation` `thresholds` /
`plot_range`), the `statuses`, `input_failures`, `valid_region_support`, and `counts`
(`requested` = `evaluated` + `excluded` + `failed`). Strict JSON: non-finite numbers
never appear.

`vs compare` compares one model output at a time: paired comparisons align rows by
`(sample_id, output_name)` (each key must be unique), and a CSV holding several outputs
requires `--output-name` (library: `CompareRequest.output_name`); no cross-output mean
or winner is computed. `vs organize` ranks each output's rows separately (one
subdirectory per output) and `vs panels` selects representative cases per
`<output>/<metric>`.

`vs organize`, `vs compare` and `vs panels` read ranking directions,
presentation thresholds and plot ranges from this file next to the CSV, else from the
built-in definition of the same name. For any other metric they require an explicit
direction (`--direction METRIC=higher|lower`, `--higher-is-better`/`--lower-is-better`)
and use data-driven plot ranges and no thresholds. Thresholds are presentation
heuristics for colouring and share statistics only; they are not biological,
diagnostic, clinical or scientific acceptance criteria.

## Model Bundles

`vs export-model` (or `applications.export_model.export_model_bundle`) copies selected
checkpoints of one tracked run, with the configuration needed to interpret them, into
a small versioned directory. It is a utility, not a pipeline stage, and it is not a new
checkpoint format: the bundled checkpoints are the run's v4 files, byte for byte.

```text
<bundle>/
├── bundle.json
├── checkpoints/
│   └── ep010.pth                     # each selected physical checkpoint, once
├── config/
│   ├── input.yaml                    # exact copy of config/train/input.yaml
│   └── resolved.yaml                 # exact copy of config/train/resolved.yaml
└── metadata/
    └── training_environment.json     # exact copy of metadata/environments/train.json
```

### Source run contract

The source must be a tracked run with `config/train/input.yaml`,
`config/train/resolved.yaml`, `metadata/environments/train.json` and `checkpoints/`;
none may be (or pass through) a symlink. The resolved training config must resolve with the same method and component
definitions used for reconstruction. Every selected checkpoint's
`config_hash` must equal the SHA-256 of that tracked resolved config; a missing or
different hash is rejected (the binding is never inferred from file location).

Selectors (repeatable):

| Selector | Source |
|---|---|
| explicit (`--checkpoint PATH`) | A regular, non-symlink file directly in the run's `checkpoints/`; relative paths are relative to it. `..`, absolute paths elsewhere, nested paths and symlinks are rejected |
| `latest` (`--latest`) | Newest `epNNN.pth` checkpoint |
| `best` (`--best METRIC`) | Rank 1 of `METRIC` in `checkpoints/best.json` |
| `top_k` (`--top-k METRIC RANK`) | Rank `RANK` of `METRIC` in `checkpoints/best.json` |

Ranked selections preserve metric, rank, value, mode and epoch from `best.json`, without
re-ranking. Selected checkpoints must satisfy the current checkpoint contract and match
the tracked config hash and supplied definitions. Selections of the same physical file,
including hard links, share one bundled copy.

### `bundle.json` (schema version 1)

Strict JSON (no `NaN`/`Infinity`); every path is relative to the bundle root.

| Key | Content |
|---|---|
| `schema_version` | `1` |
| `notice` | Local-artifact and non-redistribution notice |
| `config` | `input` and `resolved`, each `{path, sha256, role}`; roles `training_input` / `training_resolved` |
| `environment` | `{path, sha256}` of the copied training environment snapshot |
| `requirements` | `methods` and `components`: sorted `{name, source, version}` a reader must supply as definitions, derived from the checkpoints' metadata |
| `checkpoints` | Unique checkpoints sorted by path: `path`, `sha256`, and the v4 payload's `format_version`, `epoch`, `config_hash`, `method` (name, implementation, pairing, inputs, outputs, prediction directions, options, components), `image_size`, `normalization` embedded verbatim |
| `selections` | One record per request, in request order: `policy`, `metric`, `rank`, `metric_value`, `mode` (all `null` for `latest`/`explicit`), `epoch`, `checkpoint` (bundle-relative path) |

The source run directory is not recorded, and no Python module path or source code is
bundled.

### Publication and verification

Verification checks the index schema, bundle-contained relative paths (no `..` or
symlinks), every SHA-256, and checkpoint compatibility with the bundled resolved config,
its hash, and the supplied definitions. Export publishes only a fully verified bundle.
The destination must not exist or lie inside the source run, including through symlinks.
The source and any existing destination, including one created concurrently, remain
untouched on failure. Export requires filesystem support for atomic no-replace
publication on Linux, macOS, or Windows.

### Portability semantics

A bundle stays verifiable and reconstructable after it is moved and the original run
and dataset directories are gone; see the
[Python reconstruction example](library_api.md#exporting-model-bundles).
The bundled configs are unmodified research provenance; their `dataset_root`,
`results_path`, `run_name` and manifest paths still name the original locations, so
the resolved config is not a runnable reproduction of training once those are gone.

A successful export does not imply permission to redistribute the weights or configs.
Rights and privacy approval, attribution, release naming and publication are outside
this exporter.
