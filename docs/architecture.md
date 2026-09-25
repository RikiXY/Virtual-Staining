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
| `metrics.py` | Image metric computations, directions, quality thresholds, and validation image metric names |
| `checkpoint_contract.py` | Topology-neutral v4 checkpoint payload and strict method-aware compatibility validation |
| `checkpoint_selection.py` | Neutral `best.json` ranking, policy, and metric-direction selection |
| `utils/` | Shared primitives: artifact naming, image dimensions, and image I/O helpers |
| `config/` | Sole owner of YAML-facing dataclasses and strict parsers for every config section |
| `experiment/` | Canonical `RunLayout` for one run, `ResultsLayout` for shared comparisons, stage snapshots, run metadata, manifest/config hashing, and environment snapshots |
| `models/` | Network implementations (`ConcatUNetGenerator`, `ResnetGenerator`, `PatchGANDiscriminator`), factories, and the model-I/O normalization contract; no training state |
| `data/` | Canonical `DatasetLayout`, slide sets, paired manifests, unpaired domain collections, dataset building, registration, and dataset-owned provenance/fingerprints |
| `methods/` | The two built-in method runtimes (`Pix2PixMethod`, `CycleGANMethod`), each owning its topology, optimizers, losses, checkpoint state, component metadata, and inference loader; `registry.py` selects one from `method.name` |
| `training/` | The `TrainingMethodRuntime` protocol, method-agnostic `Trainer`, generic `MethodCheckpointManager`, validation, history, loss configuration/registry, and callback-driven progress events |
| `inference/` | Checkpoint resolution, method dispatch to the method-owned loaders, CycleGAN direction resolution, generic single/directory/tiled/WSI inference, and output naming |
| `evaluation/` | Paired per-image metrics and grouped summaries, unpaired collection diagnostics, diagnostic plots, representative selection, and comparison panels |
| `applications/` | User-visible stage lifecycle owners and infer-images runtime composition; no `argparse` |
| `cli/` | The `argparse` entrypoint, terminal rendering, and thin adapters over `applications/` |

## Translation Methods

Two methods are built in: Pix2Pix (paired, named N-input -> one-target) and CycleGAN
(unpaired, one domain A <-> one domain B). `methods/registry.py` maps `method.name` to
`Pix2PixMethod` or `CycleGANMethod` with an explicit branch; there is no dynamic import,
`class_path` loading, or plugin discovery. `RunConfig` validates the method-specific
combination of data pairing, generator architecture, losses, inference direction, and
evaluation protocol before any runtime is built.

`training/runtime.py` defines the `TrainingMethodRuntime` protocol the generic training
code consumes: method identity (`name`, `pairing`, `input_names`, `output_names`,
`prediction_directions`), `step()` / `validate()` returning named `MethodMetrics`,
checkpoint-selection metrics and modes, scheduler stepping, learning rates, component
metadata, and opaque `state_dict()` / `load_state_dict()`. It exposes no generator,
discriminator, optimizer, or model-count accessors.

- `Trainer` owns the epoch loop, validation cadence, `epochs.csv` history, checkpoint
  cadence and `best.json` ranking, resume, and early stopping. It does not know which
  networks a method trains.
- `MethodCheckpointManager` wraps the runtime's `state_dict()` in the v4 payload from
  `checkpoint_contract.py` and validates method identity, I/O names, image size, and
  normalization before calling `load_state_dict()`. It assumes no fixed number of models
  or optimizers. Only v4 is accepted; older or unversioned payloads are rejected, with no
  migration path.
- `Pix2PixMethod` owns the ConcatUNet generator, conditional PatchGAN discriminator, their
  optimizers, AMP scalers, schedulers, and the configured BCE/L1/SSIM objective.
- `CycleGANMethod` owns `G_A_to_B`, `G_B_to_A`, `D_A`, `D_B`, joint generator and
  discriminator optimizers, scalers, schedulers, CycleGAN weight initialization, the LSGAN
  / cycle L1 / identity L1 objective, and the `fake_A` / `fake_B` replay pools including
  their RNG state.

`applications/train.py` builds either manifest-backed paired datasets or
`data/unpaired.py` domain datasets according to `data.pairing`, then hands the runtime to
the `Trainer`.

For inference, `inference/runner.py` resolves the checkpoint through the shared selection
policy and dispatches to `load_pix2pix_inference_generator` or
`load_cyclegan_inference_generator`. The CycleGAN loader returns a
`CycleGANInferenceAdapter` wrapping the generator for `inference.direction`, so every
loaded model accepts the same named-input mapping. Single-image, directory, tiled, and WSI
inference in `inference/single.py` and the `vs infer` test-split loop are method-agnostic
apart from choosing the input names and the direction-aware output filename
(`utils/artifacts.py`).

Evaluation protocol selection belongs to `applications/evaluate.py`: `paired` (default for
Pix2Pix) maps aligned manifest records to references and reuses `evaluation/evaluator.py`
and `metrics.py`; `unpaired` (default for CycleGAN) collects the active direction's
generated images and the real test collection of the reference domain and delegates to
`evaluation/unpaired.py`. Method code contains no evaluation metrics.

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
| Side-effecting training service | `training/trainer.py` | Training loop, checkpoint and epoch-history writes; the active session owns run metadata/logging |
| Side-effecting inference service | `inference/runner.py`, `inference/single.py` | Reusable model loading and prediction plus single-image output writing |
| Side-effecting evaluation service | `evaluation/` runners/report writers | Metrics computation plus report/CSV output |

The architectural boundary is not “no I/O in library code.” The actual rule is:

- reusable package code should keep I/O explicit and testable
- orchestration belongs in `applications/`
The `ExperimentSession` owns each train/infer/evaluate lifecycle: stage snapshots,
strict local metadata writes, and best-effort reporter callbacks. `RunLayout` owns
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
`inference/single.py` accepts an already-loaded `InferenceRuntime`.

Within training, `trainer.py` owns epoch orchestration, `validator.py` owns validation
inference, `history.py` owns metric CSV persistence, `checkpoints.py` persists opaque
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
cli -> applications, metrics
applications -> config, data, evaluation, experiment, inference, methods, metrics,
                models, split_contract, training, utils
inference -> checkpoint_selection, config, data, experiment, methods, models, utils
methods -> checkpoint_contract, checkpoint_selection, config, models, training
training -> checkpoint_contract, checkpoint_selection, config, experiment, metrics, models
evaluation -> metrics, utils
experiment -> config, data, utils
data -> config, split_contract, utils
models -> config
checkpoint_contract -> models
checkpoint_selection -> metrics
config -> checkpoint_selection, split_contract, utils
metrics, split_contract, utils -> (none)
```

`tests/architecture/test_package_dependencies.py` resolves absolute, relative,
nested, and `TYPE_CHECKING` imports with the standard library and enforces the
boundaries that matter: no library package imports `applications` or `cli`;
`utils`, `metrics`, and `split_contract` stay leaves; `config` imports no runtime
domain; CLI command modules call only `applications`; and the registration boundary
below. `training` never imports `methods`: concrete methods depend on the generic
training layer, not the reverse.

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
