# Architecture

## Layer Model

The package is organised in three layers. Dependencies flow downward only -
upper layers may import from lower layers, never the reverse.

| Layer | Description | Examples |
|---|---|---|
| **Library** | Reusable package code with explicit, testable I/O boundaries. Some modules are pure helpers; others are side-effecting services. | `metrics.py`, `utils/`, `config/`, `experiment/`, `models/`, `data/`, `training/`, `inference/`, `evaluation/` |
| **Application** | Use-case orchestrators that wire core modules together | `applications/` |
| **Adapter** | Entry points that translate CLI arguments into application calls | `cli/` |

## Package Map

| Package | Responsibility |
|---|---|
| `metrics.py` | The evaluation metric contract (`MetricDefinition`, `MetricResult`, request resolution, grouped computation) and the built-in numerical definitions with their directions and presentation-only thresholds |
| `definitions.py` | The torch-free extension seam: `MethodDefinition`, `ComponentDefinition`, and the immutable `Definitions` set (methods, components, evaluation metrics) callers supply explicitly |
| `checkpoint_contract.py` | Topology-neutral v4 checkpoint payload and strict method-aware compatibility validation |
| `checkpoint_selection.py` | Neutral `best.json` ranking, policy, and metric-direction selection |
| `loss_definitions.py` | Canonical built-in loss definitions: name, allowed roles, supported methods, parameter contract and validation, and primitive tensor math, shared by the built-in definitions and runtimes |
| `utils/` | Shared primitives: artifact naming, image dimensions, and image I/O helpers |
| `config/` | Framework-common YAML-facing dataclasses and strict parsers, `RunConfig.from_mapping` resolution against supplied definitions, and reusable option blocks (losses, LR scheduler) that definitions parse |
| `experiment/` | Canonical `RunLayout` for one run, `ResultsLayout` for shared comparisons, stage snapshots, run metadata, manifest/config hashing, and environment snapshots |
| `models/` | Network implementations (`ConcatUNetGenerator`, `ResnetGenerator`, `PatchGANDiscriminator`), their `ComponentDefinition`s (`components.py`), and the model-I/O normalization contract; no training state |
| `data/` | Canonical `DatasetLayout`, slide sets, paired manifests, unpaired domain collections, dataset building, registration, dataset-owned provenance/fingerprints, and the versioned consumed/produced-data snapshot format (`consumption.py`) |
| `methods/` | The built-in definitions (`builtin.py`: `Pix2PixDefinition`, `CycleGANDefinition`, `builtin_definitions()`) and runtimes (`Pix2PixMethod`, `CycleGANMethod`), each owning its options, topology, optimizers, losses, checkpoint state and inference-only generator |
| `training/` | The `TrainingMethodRuntime` protocol, method-agnostic `Trainer`, generic `MethodCheckpointManager`, validation, history, the Pix2Pix configured-loss evaluator, and callback-driven progress events |
| `inference/` | Checkpoint resolution, definition-driven inference model construction, prediction-direction resolution, generic single/directory/tiled/WSI inference, and output naming |
| `evaluation/` | Paired evaluation of a resolved metric request (input-failure coverage, valid-region support, per-image reports, summaries, `evaluation_result.json`), unpaired collection diagnostics, diagnostic plots, representative selection, and comparison panels |
| `applications/` | User-visible stage lifecycle owners and infer-images runtime composition; no `argparse` |
| `cli/` | The `argparse` entrypoint, terminal rendering, and thin adapters over `applications/` |

## Translation Methods

Every method, built-in or external, is a registered `MethodDefinition`, and every
network architecture a registered `ComponentDefinition` (`definitions.py`). Callers pass
an immutable `Definitions` set explicitly; `builtin_definitions()` is the default and
holds Pix2Pix (paired, named N-input -> one-target) and CycleGAN (unpaired, one domain
A <-> one domain B) plus the `concat_unet`, `resnet` and `patchgan` components. There
is no dynamic import, `class_path` loading, or plugin discovery, and checkpoint
metadata is compared with, never used to import, definitions.

`RunConfig.from_mapping` (used by `from_yaml`) parses the framework-common fields,
resolves `method.name`, splits the shared `method`/`model`/`training` sections into
framework keys and the keys the definition declares in `owned_keys`, and lets the
definition parse its options. Generic config knows no method, loss, metric catalogue
or architecture: the definition validates its options and cross-section rules,
declares its pairing and prediction directions, and supplies the direction and
validity of `training.early_stopping.monitor` and `inference.checkpoint_metric`. The
built-ins keep the documented `model.generator`, `model.discriminator`, `training.lr_*`,
`beta*`, `scheduler`, `losses` and `method.replay_buffer_size` spelling as keys they own;
an external method owns `method.options` and needs none of them.

The six built-in losses (`adversarial_bce`, `l1`, `ssim` for Pix2Pix; `adversarial_lsgan`,
`cycle_l1`, `identity_l1` for CycleGAN) are each defined once in `loss_definitions.py`.
Config parsing derives accepted names, roles, parameters, and method compatibility from
those definitions, and runtimes take primitive math (BCE/LSGAN adversarial, L1, SSIM,
foreground-mask weighting) from them. Objective composition stays method-owned: Pix2Pix's
`ConfiguredLossEvaluator` (one per method instance, shared by training and validation)
applies primitives to the conditional discriminator logits and generated image, while
`CycleGANMethod` decides which directional tensors each term compares, sums the A/B
directions, and requires its active adversarial and cycle terms. Loss-weight schedules
remain `LossScheduleConfig` in `config/losses.py`; definitions do not own weights.

`training/runtime.py` defines the `TrainingMethodRuntime` protocol the generic training
code consumes: `name`, the history schema (`metric_names`, optional per-term
`loss_names`/`component_total_names`, and extra `validation_metric_names`), `step()` /
`validate()` returning named `MethodMetrics`, checkpoint-selection metrics and modes,
scheduler stepping, learning rates, the `checkpoint_identity()` built by its definition,
optional JSON `objective_metadata()` for `best.json`, and opaque `state_dict()` /
`load_state_dict()`. It exposes no generator, discriminator, optimizer, loss-config, or
model-count accessors.

- `Trainer` owns the epoch loop, validation cadence, `epochs.csv` history, checkpoint
  cadence and `best.json` ranking, resume, and early stopping. It does not know which
  networks a method trains.
- `MethodCheckpointManager` wraps the runtime's `state_dict()` in the v4 payload from
  `checkpoint_contract.py` and validates the definition name/version/source, component
  identities and options, I/O names, directions, image size, and normalization before
  calling `load_state_dict()`. It assumes no fixed number of models
  or optimizers. Only v4 is accepted; older or unversioned payloads are rejected, with no
  migration path.
- `Pix2PixMethod` owns the ConcatUNet generator, conditional PatchGAN discriminator, their
  optimizers, AMP scalers, schedulers, and the configured BCE/L1/SSIM objective.
- `CycleGANMethod` owns `G_A_to_B`, `G_B_to_A`, `D_A`, `D_B`, joint generator and
  discriminator optimizers, scalers, schedulers, CycleGAN weight initialization, the LSGAN
  / cycle L1 / identity L1 objective, and the `fake_A` / `fake_B` replay pools including
  their RNG state.

`applications/train.py` builds either manifest-backed paired datasets or
`data/unpaired.py` domain datasets according to `data.pairing`, asks the selected
definition for the training runtime, then hands it to the `Trainer`.

For inference, `inference/runner.py` resolves the checkpoint through the shared selection
policy, rejects checkpoints naming unregistered definitions, validates the definition's
checkpoint identity, and calls `build_inference_model`, which constructs only the
prediction network (no optimizer, scheduler, objective, discriminator or replay pool).
CycleGAN returns a `CycleGANInferenceAdapter` wrapping the generator for
`inference.direction`, so every loaded model accepts the same named-input mapping. Single-image, directory, tiled, and WSI
inference in `inference/single.py` and the `vs infer` test-split loop are method-agnostic
apart from choosing the input names and the direction-aware output filename
(`utils/artifacts.py`).

Evaluation protocol selection belongs to `applications/evaluate.py`; it defaults to the
run's `data.pairing`. `paired` maps aligned manifest records to references and reuses `evaluation/evaluator.py`
and `metrics.py`; `unpaired` (default for CycleGAN) collects the active direction's
generated images and an independent real reference collection
(`evaluation.reference_collection`, else the reference domain's `data.domains` entry) and
delegates to `evaluation/unpaired.py`; it is independent of the training pairing. The paired metric
request (`evaluation.metrics`, default the built-in set) is resolved from the caller's
`Definitions` during config resolution. Standalone evaluation metrics and method-owned
training metrics are separate: a `MethodDefinition` declares its own validation and
checkpoint metrics; Pix2Pix explicitly reuses built-in metric definitions for its
`val_*` columns, and CycleGAN reports none.

## Purity and I/O Boundaries

The library layer is intentionally mixed:

- some modules are pure or mostly pure helpers
- some modules are side-effecting services that read/write files, logs, checkpoints, or outputs

Typical examples:

| Kind | Example | Notes |
|---|---|---|
| Pure helper | `metrics.py` | Metric computations over arrays |
| I/O helper | `utils/image_io.py` | Reads/writes image files |
| Mostly pure indexing/data model | `data/dataset.py` | Dataset indexing and manifest-backed lookup |
| Dataset orchestration | `data/builder.py` | Coordinates slide-set processing and writes manifests, metadata and provenance |
| Registration and warping | `data/alignment/` | Identity/SIFT policy, affine estimation, diagnostics, coordinate conversion and image/mask warping |
| Slide-set processing | `data/slide_set_processor.py` | Computes masks, delegates alignment and writes patches for one set; returns `SetBuildResult` and closes readers |
| Side-effecting training service | `training/trainer.py` | Training loop, checkpoint and epoch-history writes into a supplied `RunLayout`; an optional tracked session receives epoch metrics |
| Side-effecting inference service | `inference/runner.py`, `inference/single.py` | Reusable model loading and prediction plus single-image output writing |
| Side-effecting evaluation service | `evaluation/` runners/report writers | Metrics computation plus report/CSV output |

The architectural boundary is not “no I/O in library code.” The actual rule is:

- reusable package code should keep I/O explicit and testable
- orchestration belongs in `applications/`

Each stage is also usable as a standalone library primitive from its natural inputs,
without a tracked run; see [`library_api.md`](library_api.md).

The `ExperimentSession` owns each train/infer/evaluate lifecycle: stage snapshots,
strict local metadata writes, and best-effort reporter callbacks. Applications decide
what a stage consumes: they resolve inputs once, build a consumed-data snapshot from
those objects, and bind it with `session.bind_inputs()` before the stage starts; the
session never infers stage inputs from which dataset files happen to exist. `RunLayout` owns
one run's paths; `ResultsLayout` owns shared cross-run comparisons under
`results/comparisons`. `applications.prepare` orchestrates dataset-local config and
environment snapshots through the generic experiment snapshot helpers. Preparation
is dataset-owned, writes dataset fingerprints and source hashes, and emits no
experiment run events. Dataset provenance lives in `data/provenance.py`; run
provenance lives in `experiment/snapshots.py`.
`applications/train.py` builds the method runtime and datasets and hands them to the reusable `Trainer`;
its `ProgressUpdate` callback is silent unless an adapter supplies a reporter.
The CLI supplies terminal rendering, while application/library callers remain
presentation-neutral. Infer-images runtime creation belongs to `applications/`;
`inference/single.py` accepts an already-loaded `InferenceRuntime` (a caller-owned
predictor plus an explicit `PredictionContract`: ordered input names, tile size,
same-grid single RGB output, [-1, 1] range, optional direction) or a factory for one.
This is the only image-inference transport. `applications/infer_images.py` builds
the predictor from a checkpoint through the method definition, fills in the same
contract and optional checkpoint provenance, and delegates to it. The manifest
`applications/infer.py` loop keeps its own provenance-owning loop but calls the same
`predict_batch`, which enforces the same-grid output check. Transport never names a
method, network topology, optimizer or loss.

Within training, `trainer.py` owns epoch orchestration, `validator.py` owns validation
inference, `preview.py` owns the optional validation preview sink (methods hand it
detached semantic tensors; `applications/train.py` injects the default TIFF writer),
`history.py` owns metric CSV persistence, `checkpoints.py` persists opaque
method-owned state through the generic `MethodCheckpointManager`,
`checkpoint_contract.py` owns the topology-neutral v4 contract, and
`checkpoint_selection.py` owns `best.json` ranking and resolution. Evaluation keeps
plot primitives in `diagnostics.py`, representative-row policy in `selection.py`,
and composed image layouts in `panels.py`.

## Architectural Rules

These constraints are enforced by convention and checked in code review:

- **No `argparse` outside `cli/`** - application and core modules accept typed
  dataclasses, not raw CLI strings.
- **Core and application modules use `logging`**, never `print`, so callers can
  suppress or redirect output.
- **No `sys.exit()` outside `cli/`** - applications raise exceptions; the CLI
  layer converts them to exit codes.

The current direct package dependencies (excluding self-imports) are:

```text
cli -> applications, metrics, training
applications -> config, data, definitions, evaluation, experiment, inference, metrics,
                models, split_contract, training, utils
inference -> checkpoint_contract, checkpoint_selection, config, data, experiment, models,
             utils
methods -> checkpoint_contract, checkpoint_selection, config, definitions,
           loss_definitions, metrics, models, training
training -> checkpoint_contract, checkpoint_selection, config, experiment,
            loss_definitions, metrics, models
definitions -> checkpoint_selection (+ type-only/lazy: checkpoint_contract, config, metrics,
               training)
evaluation -> config, metrics, utils
experiment -> config, data, utils
data -> config, split_contract, utils
models -> config, definitions
checkpoint_contract -> models
config -> checkpoint_selection, definitions, loss_definitions, split_contract, utils
          (+ one lazy import of methods.builtin for the default definition set and
          type-only/lazy imports of metrics for metric requests)
loss_definitions -> config
checkpoint_selection, metrics, split_contract, utils -> (none)
```

`tests/architecture/test_package_dependencies.py` resolves absolute, relative,
nested, and `TYPE_CHECKING` imports with the standard library and enforces the
boundaries that matter: no library package imports `applications` or `cli`;
`utils`, `metrics`, and `split_contract` stay leaves; `config` imports no runtime
domain; CLI command modules call only `applications`; and the registration boundary
below. `training` and `inference` never import `methods`: concrete methods depend on
the generic layers, not the reverse. `tests/architecture/test_method_definitions.py`
additionally checks that config resolution loads no method runtime or torch, that the
`Trainer` loads no loss configuration, that only definition owners name the built-in
methods, and that no import-string or discovery mechanism exists in these layers.

Registration is implemented entirely in `data/alignment/`: `models.py` defines
`AlignmentImage`, immutable `AlignmentResult`, `RegistrationDiagnostics`, and `AlignmentError`;
`registration.py` owns identity/declared-alignment policy, SIFT/RANSAC, validation
and diagnostics; `warping.py` owns affine application and coordinate conversion.
The package exports only those four types, `identity_alignment`,
`resolve_alignment`, `warp_aligned_patch`, and `warp_aligned_mask_patch`.
SIFT helpers are private. `preprocessing.py` retains general mask generation/sampling
and patch filtering, with no registration dependency.

Every result matrix maps moving full-resolution `(x, y)` into reference
full-resolution `(x, y)`. Array shapes are `(height, width)`; output sizes are
`(width, height)`. Registration normalizes whole-image masks to preview geometry
with nearest neighbors, halves previews for estimation, then compensates for
both images' per-axis scales using the actual SIFT input dimensions, including
resize rounding. Mask IoU describes estimation-space overlap and is diagnostic,
with no rejection threshold.
Serialized keypoint fields retain their existing `src`/`tgt` names for dataset
metadata compatibility; the implementation uses reference/moving terminology.

Reader-backed warping accepts an already-open reader's `read_region` callback:
alignment computes inverse bounds and the local matrix, IO reads the requested
region. Opening, backend selection, cleanup and `skip_set` remain outside
alignment. Dependency tests enforce `models <- warping <- registration`, forbid
alignment's dependencies on orchestration/training/inference/evaluation, and
require the processor to use the public alignment API.

## Configuration Policy

| Data type | Format | Example path |
|---|---|---|
| User experiment config | YAML | `config/runs/example.yaml` |
| Run identity and stage events | JSON/JSONL | `results/<run>/metadata/run.json`, `events.jsonl` |
| Stage snapshots and environment | YAML/JSON | `results/<run>/config/<stage>/`, `metadata/environments/` |
| Per-epoch training losses | CSV | `results/<run>/metrics/epochs.csv` |
| Per-image evaluation metrics | CSV | `results/<run>/evaluation/per_image_metrics.csv` |
| Dataset manifest and fingerprint | CSV/JSON | `datasets/<name>/manifests/manifest.csv`, `metadata/dataset_fingerprint.json` |

See [`docs/run_format.md`](run_format.md) for the full run output directory layout
and file schemas.
