# Run Output Format

## Local Queues

Queue definitions are stored as YAML files under `config/queues/`. Runtime
queue state is written under `local_workspace/queues/`. Queue execution is
explicitly local, sequential, and single-worker in v1. Every queue and ablation key
is documented in [`config/queues/example.yaml`](../config/queues/example.yaml) and
[`config/queues/example_ablation.yaml`](../config/queues/example_ablation.yaml).

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

A run selects one of the two built-in methods. The annotated references
[`config/runs/example.yaml`](../config/runs/example.yaml) (Pix2Pix) and
[`config/runs/example_cyclegan.yaml`](../config/runs/example_cyclegan.yaml) (CycleGAN)
document every supported option; `config/runs/minimal_pix2pix.yaml` and
`config/runs/minimal_cyclegan.yaml` are the same experiments with defaults omitted.

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
`domains`, keyed by `model.inputs[0]` (domain A) and `model.outputs[0]` (domain B). Each entry
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

### `model.inputs` and `model.outputs`

```yaml
model:
  inputs: [AF, LF]     # N ordered named RGB inputs
  outputs: [PAS, HE]   # M ordered named RGB outputs; [HE] for one output
```

Both are required, ordered, non-empty lists of unique machine identifiers matching
`[A-Za-z][A-Za-z0-9_-]*` (names are never sanitized), and they must be disjoint. One
output is the one-item case of the same representation; the superseded singular
`model.target` (and `preprocessing.inputs.target_modality`) is rejected with a pointer to
the plural field. With a `preprocessing` section, `model.inputs` must be a subset of
`preprocessing.inputs.modalities` and `model.outputs` a subset of
`preprocessing.inputs.target_modalities`, each in any order; model order is
authoritative. Pix2Pix accepts any M; CycleGAN requires exactly one input and one
output.

### `model.generator`

| `architecture` | Method | Fields |
|---|---|---|
| `concat_unet` (default) | Pix2Pix only | `base_channels`, `norm` (default `batch`), `dropout`, `bilinear` |
| `resnet` | CycleGAN only | `base_channels`, `blocks` (default 9), `norm` (must be `instance`) |

The architecture is fixed by the method; it is not a free choice. Both methods use
`model.discriminator` (`ndf`, `norm`, `use_sigmoid`) for their PatchGAN discriminators:
one joint discriminator conditioned on all inputs and scoring all outputs together
(`3*N + 3*M` channels) for Pix2Pix, unconditional per domain for CycleGAN. The Pix2Pix
generator has `3*N` input and `3*M` output channels, split back into the named outputs.
Channel counts are derived from `model.inputs`/`model.outputs`, never component options.
CycleGAN takes exactly one `model.inputs` and one `model.outputs` entry and needs
`image_size` dimensions that are multiples of 4 and at least 8.

### `inference.direction`

```yaml
inference:
  direction: A_to_B   # cyclegan only: A_to_B (default) | B_to_A
```

`A_to_B` translates `model.inputs[0]` into `model.outputs[0]`; `B_to_A` translates the
reverse with the second generator of the same checkpoint. The two directions are
alternative predictions of one checkpoint, each producing exactly one named output
(`model.outputs[0]` for `A_to_B`, `model.inputs[0]` for `B_to_A`), never two
simultaneous outputs. Pix2Pix rejects the field.
CycleGAN validation reports training-objective losses only, so CycleGAN checkpoint,
scheduler, and early-stopping monitors must be `loss_G_val` or `loss_val_*` columns,
not `val_*` image metrics.

### `evaluation.protocol`

```yaml
evaluation:
  protocol: unpaired  # paired | unpaired
```

The default follows the method's training pairing: Pix2Pix -> `paired`, CycleGAN ->
`unpaired`. The protocol chooses how evaluation inputs are read, never how the model was
trained. `paired` is the normal/default protocol for paired-training methods. CycleGAN
may explicitly select `paired` when an aligned held-out test manifest exists;
`data.domains` collections are never treated as pairs.

`unpaired` compares independent generated and real reference collections and is
method-independent. The reference collection is resolved as:

1. `evaluation.reference_collection`, when set;
2. otherwise the reference domain's `data.domains` entry (CycleGAN normally has one);
3. otherwise the configuration is rejected.

```yaml
evaluation:
  protocol: unpaired
  reference_collection: reference/stained  # unpaired only
```

`reference_collection` uses the `data.domains` spec semantics: a directory holding
`test/` (searched recursively) or a path/glob containing the literal `{split}`; relative
values resolve against `dataset_root`. It is rejected with the paired protocol, is never
inferred from the paired manifest, and `data.group_metadata` is not applied to it. See
[Evaluation outputs](#evaluation-outputs).

### `evaluation.metrics` and `evaluation.input_failures`

```yaml
evaluation:
  metrics:              # optional; omitted = mae, mse, rmse, psnr, ssim, pcc_gray, pcc_rgb_mean
    - name: mae
    - name: ssim
    - name: my_metric   # registered in Python; see docs/library_api.md
      options: {scale: 2.0}
  input_failures: strict  # strict (default) | permissive
```

Both apply to the paired protocol only; the unpaired protocol rejects an explicit
`metrics` list and `permissive`. `metrics` is an ordered list of `{name, options}`
mappings resolved once, before any image is read, against the metric definitions the
caller supplied (`builtin_definitions()` by default). Unknown or repeated names, unknown
or malformed options, and unknown keys fail preflight. Only requested metrics are
computed and reported, in request order. The resolved config records `metrics` only when
it was configured.

`input_failures` covers known input problems: a missing or unreadable target or
generated file, a non-RGB image, a target/generated shape mismatch, and a missing,
non-binary or mismatched valid-region support mask. `strict` records them in
`coverage.csv` and fails the stage. `permissive` excludes those samples, records the
reasons and continues, but still fails when nothing could be evaluated. Metric
implementation, backend and programming errors always fail the stage in either mode.

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

`vs config resolve --config ...` prints these exact bytes without running a stage, and
`vs config check` prints their SHA-256 as `config_sha256` (the stage's `config_hash`).

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
    intensity: light          # light, medium, or strong
    photometric_inputs: []    # resolved; see below
```

One sampled geometry realization (resize, flips, `RandomRotate90`, affine) is applied to
every selected input, every selected target and every target mask; images keep
continuous interpolation and masks use nearest-neighbour. Targets and masks never get
photometric transforms. `light` has none (`photometric_inputs` resolves to `[]`, and a
non-empty list with `light` is rejected); for `medium`/`strong` each input in
`photometric_inputs` gets its own photometric stream seeded from `training.seed` and a
SHA-256 digest of its name. Omitted, it resolves to the preparation reference input
when that input is selected, else to the first selected model input; the resolved
config records the effective list. Entries must be unique selected `model.inputs`.
Because every preset includes `RandomRotate90`, `enabled: true` requires a square
`image_size`; disabled augmentation keeps non-square support. Results repeat only under
the same complete execution setup (seed, worker count, library versions); no
worker-count-independent or exact-resume replay is promised. CycleGAN requires
`enabled: false`.

Optimizer learning-rate schedules are configured under `training.scheduler`.
Omitting the section preserves a flat learning rate. Epoch numbers are
zero-based. `linear_decay` keeps the initial optimizer LR through
`decay_start_epoch`, then decays linearly through the final epoch. Plateau
scheduling steps only after validation, using validation metric columns such as
`loss_G_val` or a per-output Pix2Pix column `val_<metric>__<output>` (`val_ssim__HE`,
`val_mae__PAS`, with `<metric>` one of `ssim`, `psnr`, `mae`, `rmse`, `pcc_gray`,
`pcc_rgb_mean`). `loss_G_val` depends on the configured training loss terms.

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
    monitor: val_ssim__HE
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
validation CSV column names such as `val_ssim__HE`, `val_mae__PAS`,
`loss_G_val`, `loss_D_val`, or configured `loss_val_*` component columns. A Pix2Pix
monitor that names an output must name one of `model.outputs`. With one output the
default monitor is `val_ssim__<output>`; with several outputs `monitor` is required,
because choosing which output decides early stopping is the user's decision.

```yaml
training:
  early_stopping:
    monitor: val_ssim__HE
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
dataset must provide a `foreground_mask` tensor for every model output in every batch
(`masks.foreground_mask.<output>`). Missing masks raise an error naming the output
instead of being treated as all-foreground. The manifest loads the exact
`foreground_mask__<target>` paths written when `masks.save_patch_masks: true`; each saved
patch mask is that target's own aligned foreground mask, and an output's loss only ever
uses its own mask.

Pix2Pix composes one joint adversarial term with, for every configured reconstruction
term, `weight * mean(term(prediction[o], target[o]) for o in model.outputs)`. The
arithmetic mean over outputs is deliberate: one output keeps its exact scale and adding
outputs does not inflate the reconstruction magnitude. The schedule applies after the
mean, the loss parameters apply uniformly to all outputs, and this training mean is not
an evaluation score.

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
(`current_weight * raw / M`), and `current_weight` the scheduled term weight.

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

### `artifacts/output_train/`, `output_val/`, `output_test/`

Generated images produced during training (train/val) and inference (test). Every
generated artifact is identified by `(sample_id, output_name)` and written as
`<output_dir>/<output_name>/<sample_id>_generated.<ext>`, one file per model output, so
all Pix2Pix outputs and both CycleGAN directions share one output directory without
collisions and every path inverts to its pair:

```text
HE/00512_09216_generated.tif          # Pix2Pix output HE
PAS/00512_09216_generated.tif         # Pix2Pix output PAS
stained/00512_09216_generated.tif     # CycleGAN A_to_B predicts domain B
label_free/00512_09216_generated.tif  # CycleGAN B_to_A predicts domain A
```

Validation previews in `output_val/` are `epoch<e>_batch<b>_input.tif` plus
`..._output__<output>.tif` and `..._target__<output>.tif` per output.

`vs infer-images --recursive` uses the same layout and preserves the input's relative
directory structure (before `<output_name>`) under the output root, so equal filenames
in different source folders do not collide. An explicit output file is accepted only for
a one-output model; several outputs need an output directory. Inputs that would map to the same output file (for example `a.png` and
`a.tif` with `--output-format png`) are rejected before prediction. An existing output
file is replaced atomically only after its prediction succeeds. Without `--output`,
files go to `inference.output_dir` or the run's `artifacts/output_single/` (one file)
or `artifacts/output_images/` (directories). Library callers without run defaults
must pass an output path (see [`library_api.md`](library_api.md#direct-predictor-inference)).

Full-resolution WSI outputs are pyramidal BigTIFFs with the input's pixel dimensions.
For a single WSI input, known source MPP is preserved. For multi-input WSI, output
MPP is preserved only when every input provides compatible known MPP. Conflicting
known values fail; if any input lacks calibration, the shared output MPP remains
unknown (never zero and never inferred from pixel counts). All outputs share the input
grid and MPP and come from one tile traversal. The run first requires at least
`M x width x height x 15` bytes of free scratch space next to the outputs for M outputs.

## Evaluation outputs

Every evaluation writes `evaluation/evaluation_metadata.json` (`schema_version` 3)
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

It is written even when the evaluation fails, and no other paired report is published
then.

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
none may be (or pass through) a symlink. The resolved training config is parsed with
`RunConfig.from_yaml(..., definitions)`, so it must resolve through the same method and
component definitions as normal reconstruction. Every selected checkpoint's
`config_hash` must equal the SHA-256 of that tracked resolved config; a missing or
different hash is rejected (the binding is never inferred from file location).

Selectors:

| Selector | Source |
|---|---|
| explicit (`--checkpoint PATH`) | A regular, non-symlink file directly in the run's `checkpoints/`; relative paths are relative to it. `..`, absolute paths elsewhere, nested paths and symlinks are rejected |
| `latest` (`--latest`) | `checkpoint_selection` latest-checkpoint resolution |
| `best` (`--best METRIC`) | Rank 1 of `METRIC` in `checkpoints/best.json` |
| `top_k` (`--top-k METRIC RANK`) | Rank `RANK` of `METRIC` in `checkpoints/best.json` |

Ranked selections read metric, rank, value, mode and epoch from `best.json`; nothing is
re-ranked or re-scored. Every selected file is read with the weights-only
`read_checkpoint`, checked with `Definitions.require_checkpoint`, validated against
`config.method.definition.checkpoint_identity(config)`, bound to the config hash and
hashed before anything is written. Unsupported, corrupt, unversioned and pre-v4
checkpoints fail through the normal checkpoint policy. Selectors that resolve to the
same physical file (same device and inode, including hard links) share one bundled copy.

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

The bundle is built in a hidden `.<name>.*.staging` directory beside the destination
and verified there with `verify_model_bundle`: exact index schema, every path strictly
inside the bundle with no absolute path, `..` or symlink, every SHA-256, and every
checkpoint re-read and re-validated against the bundled resolved config, its config
hash and the supplied definitions. Only then is the staging directory renamed to the
destination with an atomic no-replace rename (`renameat2(RENAME_NOREPLACE)` on Linux,
`renamex_np(RENAME_EXCL)` on macOS, `MoveFileExW` without replace on Windows), so a
destination created concurrently is never replaced; other platforms are refused. The
destination must not exist and must not lie inside the source run
(directly or through a symlink). On failure only the staging directory is removed; the
source run and any existing destination are never modified.

### Portability semantics

A bundle stays verifiable and reconstructable after it is moved and the original run
and dataset directories are gone: `RunConfig.from_yaml(bundle/"config/resolved.yaml",
definitions)` plus `load_inference_generator(config, RunLayout(bundle), device,
bundle/"checkpoints/<file>")` rebuilds the prediction network for new inference inputs.
The bundled configs are unmodified research provenance; their `dataset_root`,
`results_path`, `run_name` and manifest paths still name the original locations, so
the resolved config is not a runnable reproduction of training once those are gone.

A successful export does not imply permission to redistribute the weights or configs.
Rights and privacy approval, attribution, release naming and publication are outside
this exporter.
