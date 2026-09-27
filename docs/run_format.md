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
loss config entries, so omitted default-zero terms are not treated as
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

`RunLayout` in `virtual_staining.experiment.run_layout` owns one run's paths.
`ResultsLayout` owns shared cross-run comparison paths under
`local_workspace/results/comparisons/`. `ExperimentSession` is the only run-stage
bootstrap and creates the directories; layout instances themselves are pure path
contracts.

## Method-Specific Configuration

A run selects one of the two built-in methods. Complete examples are
[`config/runs/example.yaml`](../config/runs/example.yaml) (Pix2Pix) and
[`config/runs/example_cyclegan.yaml`](../config/runs/example_cyclegan.yaml) (CycleGAN).

### `method`

```yaml
method:
  name: pix2pix          # pix2pix (default) | cyclegan
  replay_buffer_size: 50 # cyclegan only; default 50
```

`replay_buffer_size` is the per-domain number of previously generated images kept in the
`fake_A` / `fake_B` replay pools. Once a pool is full, each new fake is shown to the
discriminator directly or, with probability 0.5, swapped for a stored one. `0` disables
the pools. Pool contents and their RNG state are checkpointed. Setting the field for
Pix2Pix is an error.

### `data`

```yaml
data:
  pairing: unpaired            # paired (default) | unpaired
  domains:                     # unpaired only
    label_free: domains/label_free
    stained: "prepared/{split}/stained/**/*.tif"
```

Pix2Pix requires `pairing: paired` (the default) and trains from the prepared manifest;
`domains` must then be omitted. CycleGAN requires `pairing: unpaired` and exactly two
`domains`, keyed by `model.inputs[0]` (domain A) and `model.target` (domain B). Each entry
is either a directory containing `train/`, `val/`, and `test/` (searched recursively) or a
path/glob containing the literal `{split}`. Relative entries resolve from `dataset_root`.
The two collections are independent: an epoch has `max(len(A), len(B))` samples, the
shorter domain wraps around, and the domain-B draw is a deterministic function of the
seed, epoch, and index.

Provenance fields apply to both pairings:

```yaml
data:
  hash_policy: content        # content (default) | membership
  group_validation: auto      # auto (default) | patient | specimen | set | unavailable
  group_metadata: groups.csv  # unpaired only: path,domain,split,set_id,specimen_id,patient_id
```

Domain collections must resolve inside `dataset_root` so their locators stay portable.
See [Consumed-data snapshots](#consumed-data-snapshots).

### `model.generator`

| `architecture` | Method | Fields |
|---|---|---|
| `concat_unet` (default) | Pix2Pix only | `base_channels`, `norm` (default `batch`), `dropout`, `bilinear` |
| `resnet` | CycleGAN only | `base_channels`, `blocks` (default 9), `norm` (must be `instance`) |

The architecture is fixed by the method; it is not a free choice. Both methods use
`model.discriminator` (`ndf`, `norm`, `use_sigmoid`) for their PatchGAN discriminators:
conditional on the concatenated inputs for Pix2Pix, unconditional for CycleGAN. CycleGAN
takes exactly one `model.inputs` entry and needs `image_size` dimensions that are
multiples of 4 and at least 8.

### `inference.direction`

```yaml
inference:
  direction: A_to_B   # cyclegan only: A_to_B (default) | B_to_A
```

`A_to_B` translates `model.inputs[0]` into `model.target`; `B_to_A` translates the
reverse with the second generator of the same checkpoint. Pix2Pix rejects the field.
CycleGAN validation reports training-objective losses only, so CycleGAN checkpoint,
scheduler, and early-stopping monitors must be `loss_G_val` or `loss_val_*` columns,
not `val_*` image metrics.

### `evaluation.protocol`

```yaml
evaluation:
  protocol: unpaired  # paired | unpaired
```

The default follows the method: Pix2Pix -> `paired`, CycleGAN -> `unpaired`.
`unpaired` is available only for CycleGAN. CycleGAN may explicitly select `paired` when an
aligned held-out test manifest exists; `data.domains` collections are never treated as
pairs. See [Evaluation outputs](#evaluation-outputs).

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
    ├── skipped.csv                # paired protocol, when applicable
    ├── unpaired_image_statistics.csv    # unpaired protocol
    └── unpaired_feature_comparison.csv  # unpaired protocol
```

Cross-run comparisons are written separately under
`local_workspace/results/comparisons/<A>_vs_<B>/<mode>_<metric>/`; `ResultsLayout`
owns this shared results root rather than either individual `RunLayout`.

`applications.prepare` orchestrates dataset-local config and environment snapshots
through the generic experiment snapshot helpers. `data/provenance.py` owns dataset
fingerprints and source-file hashing. Preparation does not write experiment
`run.json`, `events.jsonl`, or `metadata/stages/prepare.json`.

Run checkpoints use the method-aware v4 `checkpoint_contract.py` and
`checkpoint_selection.py` modules; model, optimizer, scaler, scheduler, and (for
CycleGAN) replay-pool state are method-owned and persisted opaquely through
`state_dict()`. Training progress is a callback event rendered by
the CLI, not terminal output from library code. `ProgressUpdate` carries raw
data: monotonic `elapsed_seconds`/`eta_seconds` and a wall-clock
`estimated_end`, formatted only by `training/progress.py` helpers.

## File Descriptions

### `config/input.yaml`

Verbatim copy of the YAML file passed to `--config`. Preserved for full
reproducibility - re-running with this file reproduces the same experiment.

### `config/resolved.yaml`

The fully expanded effective configuration after all defaults have been applied
and all derived paths resolved. Differences from `input.yaml` reflect default
values that were not explicitly set by the user.

Training losses are recorded under `training.losses.generator` and
`training.losses.discriminator` lists. Training requires explicit loss terms. A term is
active only when it is explicitly listed, `enabled` is `true`, and its scheduled current
weight is nonzero. Explicitly listed terms must declare `weight`; unlisted losses remain
absent and inactive. Weights, schedule factors, mask weights, and SSIM numeric
parameters must be finite; NaN and infinity are rejected.

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

Accepted loss names depend on the method; a name from the other method's set is
rejected.

Pix2Pix:

- `adversarial_bce`: generator or discriminator BCE-with-logits adversarial loss.
- `l1`: generator image reconstruction loss.
- `ssim`: generator image structural similarity loss.

CycleGAN:

- `adversarial_lsgan`: least-squares adversarial loss; required for the generator and
  the discriminator.
- `cycle_l1`: generator cycle-consistency L1 (`A -> B -> A` and `B -> A -> B`); required.
- `identity_l1`: optional generator identity L1 (each generator applied to real images of
  its own output domain).

```yaml
training:
  losses:
    generator:
      - name: adversarial_lsgan
        weight: 1.0
      - name: cycle_l1
        weight: 10.0
      - name: identity_l1
        weight: 5.0
    discriminator:
      - name: adversarial_lsgan
        weight: 1.0
```

CycleGAN also requires `training.augmentation.enabled: false`.

The Pix2Pix example below shows the SSIM options:

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

The Pix2Pix baseline objective uses generator `adversarial_bce` with weight `1.0`,
generator `l1` with weight `25.0`, and discriminator `adversarial_bce` with
weight `1.0`.

The training SSIM implementation is differentiable PyTorch code. It maps
current training tensors from `[-1, 1]` to `[0, 1]` before computing SSIM, and
uses `ssim_loss = 1 - SSIM(prediction, target)`. MS-SSIM and other structural
losses are not built in.

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
`run.result()` calls replace earlier values by key.

Stage lifecycle: the session first snapshots config and environment; the application
then resolves its exact inputs once, and `bind_inputs` persists the consumed-data
snapshot before `stage_started` is published and reporters start (their start
metadata carries the config hash and consumed snapshot ID). If configuration or input
resolution fails first, a single `stage_failed` event and a failed stage record are
written with `consumed_data: null` and the original error; reporters are not started.
A snapshot written before a later failure is kept as evidence of what the failed
attempt consumed. Metadata is a local single-writer store: concurrent writers to one
run are not coordinated.

### `metadata/events.jsonl`

Append-only events use the same config/environment/consumed/produced/details fields plus
`schema_version`, `timestamp`, `run_id`, `run_name`, `stage`, `event_type`, and
`status`. The event is appended before the stage and run JSON views are replaced.
All local writes are strict; external reporters run only after successful local
writes and cannot invalidate them.

### Consumed-data snapshots

A consumed-data snapshot answers *which exact assets did this tracked stage consume?*
It is distinct from the dataset fingerprint, which answers *what dataset did
preparation build?* (see [Dataset Format](dataset_format.md)). Each tracked stage
resolves its inputs once, builds the snapshot from those same objects, and passes the
same sequence to the computation, so membership cannot change between provenance and
consumption.

`snapshot.json` (snapshot schema version 1) and `rows.csv` are published atomically;
`snapshot.json` binds the SHA-256 of `rows.csv`, so a mismatched pair is rejected.
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

- `content` (default): every consumed file is SHA-256 hashed; the file is stat-ed
  before and after reading and a file that changes while hashed fails the stage. This
  is the mode for a content-identical freeze. A hashing failure never falls back.
- `membership`: records locators, sizes, and semantic metadata only.
  `content_verified: false` and an explicit limitation mark it as unverified.

Duplicate and leakage checks: the same resolved file, an alias of it (symlink or
hard link), or identical verified bytes in two different splits fail. Identical content
within one split is listed under `duplicates` for review, never deduplicated. Byte
identity is only asserted under `content`; `membership` still detects same-file aliases.

Held-out assets: training also observes the held-out test assets of its selected data
contract - the test manifest records' selected inputs, target, and mask (when used) for
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
patch-level split (`split.unit: patch`, read from `metadata/split_assignment.csv`)
trained under explicit `unavailable`: its groups span splits by construction, so the
shared group counts are recorded under `group_validation` with a limitation instead of
failing. `auto` or an explicit unit still fails on the same data. IDs are never
inferred from filenames or directories.

These are file-provenance checks only. Distinct SHA-256 digests do not prove
biological independence, re-encoded or near-duplicate images are not detected, and no
clinical or biological validity follows from them.

Per stage:

- **train** (paired): the files each train/val manifest record supplies - selected
  inputs, target, and the foreground mask only when a mask loss requires it - with
  set/specimen/patient IDs. Augmentation expansion is configuration, not extra rows.
  `sources` records the manifest SHA-256 and dataset fingerprint.
- **train** (unpaired): train/val membership of domain A (`input`) and domain B
  (`target`). The seeded epoch draw is a sampling policy (stage `details`), not a
  correspondence, so no pairs are recorded.
- **infer**: only the files fed to the predictor - the selected inputs for Pix2Pix
  and CycleGAN `A_to_B`, the held-out target for `B_to_A`. These are exactly the
  image files inference opens; the other side of each record is never read. After writing, the
  `produced_data` snapshot lists each `generated` output with its `sample_id` and
  content identity and references the consumed snapshot and checkpoint.
- **evaluate** (paired): one `reference` and one `generated` row per evaluated
  sample, sharing its `sample_id` - the explicit correspondence the evaluator used.
  Missing generated files keep `status=missing`; evaluated/skipped counts stay in
  stage `details` and `skipped.csv`.
- **evaluate** (unpaired): the generated and reference collections, without
  correspondence.

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
validation image metrics (`val_ssim`, `val_mae`, ...) are means over individual
images, skipping non-finite values.
Rows flush after every epoch. Resume requires a matching header and complete
epochs `0..resume_at-1`; stale rows at or after `resume_at` are discarded before
new rows append. Missing, malformed, gapped, duplicate, or incompatible history
fails instead of silently truncating it.

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
| `method` | `name`, `pairing`, ordered `inputs`, ordered `outputs`, `prediction_directions`, and method-owned `components` metadata |
| `image_size` | `[width, height]` |
| `normalization` | Model-I/O normalization contract |
| `config_hash` | Resolved-config hash of the writing stage, or `null` (provenance only) |
| `state` | Method-owned state from `state_dict()`: models, `optimization` policy, optimizers, AMP scalers, schedulers, and CycleGAN replay pools |

Checkpoints are read onto the CPU with PyTorch's restricted `weights_only=True`
unpickler, so only tensors and primitive containers are accepted; there is no
unrestricted fallback. This narrows arbitrary-code deserialization exposure but is
not a resource sandbox. State reaches the execution device through normal model and
optimizer restoration.

Resume and inference validate every semantic field before method state is touched.
Each method then preflights its own state before mutating anything: exact state
groups and roles, model state keys, tensor shapes and dtypes, optimizer parameter
groups and per-parameter state shapes, AMP scaler state, scheduler presence and
state keys, and (CycleGAN) replay pools. Inference checks the selected generator the
same way. Malformed or incompatible state raises `CheckpointCompatibilityError`
naming the offending field. If restoration fails after preflight, the runtime is
reported as partially restored and must be discarded.

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

Saving writes a hidden `.ep<NNN>.pth.*.tmp` file in the same directory, fsyncs it,
reads it back and validates it under the contract, and only then atomically renames
it to `ep<NNN>.pth`. A failed or interrupted save never exposes partial bytes or
replaces an existing checkpoint, and `latest`, `best`, and `top_k` selection only
ever see published files. Use `inference.checkpoint_policy: latest` to load the most
recent one automatically.

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
Each file is named after its source patch. Pix2Pix outputs use the `_target_generated`
suffix; CycleGAN outputs carry the translation direction, so both directions can share
one output directory without collisions:

```text
00512_09216_target_generated.tif   # Pix2Pix
00512_09216_A_to_B_generated.tif   # CycleGAN, inference.direction: A_to_B
00512_09216_B_to_A_generated.tif   # CycleGAN, inference.direction: B_to_A
```

`vs infer-images --recursive` uses the same suffixes and preserves the input's relative
directory structure under the output root, so equal filenames in different source folders
do not collide.

## Evaluation outputs

Every evaluation writes `evaluation/evaluation_metadata.json` recording `method`,
`training_pairing`, `evaluation_protocol`, `inference_direction` (`null` for Pix2Pix),
`source_domains`, `reference_domain`, `generated_dir`, `counts`, the written `artifacts`,
the `consumed_data` reference and `generated_producer` lineage,
and `pairwise_metrics_available`: `true` for `paired`, `false` for `unpaired`. Unpaired
metadata also records the feature definitions and explicit `limitations`. Switching
protocol removes the other protocol's stale reports from the output directory.

The **paired** protocol writes `per_image_metrics.csv`, `summary.csv`, `skipped.csv` when
applicable, and grouped `<unit>_metrics.csv` / `summary_<unit>.csv` for `set`,
`specimen`, and `patient` (bootstrap confidence intervals). Each generated image is
compared with its aligned manifest reference: the target for Pix2Pix and CycleGAN
`A_to_B`, the domain-A input for CycleGAN `B_to_A`.

The **unpaired** protocol (CycleGAN) compares the active direction's generated images
with the real `test` collection of the reference domain from `data.domains`. No pairs
are formed. Each image is reduced to per-image RGB mean/std and luminance mean/std:

- `unpaired_image_statistics.csv`: `collection` (`generated` or `reference`), `path`,
  and the per-image features.
- `unpaired_feature_comparison.csv`: per feature, descriptive statistics of both
  collections, Wasserstein distance, and a two-sample KS statistic and p-value.
- `unpaired_feature_distributions.png`: optional plot when `save_graphs: true`.

These are low-order appearance/distribution diagnostics only. They carry no pairwise
fidelity metrics and no ranking of generated against real images, and they do not
establish sample-level fidelity, biological correctness, or clinical validity.

### `evaluation/per_image_metrics.csv`

Paired protocol only. One row per evaluated test image.

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
