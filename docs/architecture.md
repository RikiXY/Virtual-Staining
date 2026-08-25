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
| `metrics.py` | Image metric computations, directions, and quality thresholds |
| `utils/` | Shared primitives: image dimensions and image I/O helpers |
| `config/` | YAML loading and validation, typed config dataclasses, per-section accessors |
| `experiment/` | `RunPaths`, strict `LocalRunStore`/`ExperimentSession` persistence, reporter seam, and environment snapshots |
| `models/` | `ConcatUNetGenerator` (ordered early RGB concatenation), internal `UNetGenerator`, `PatchGANDiscriminator`, and model config |
| `data/` | `SlideAsset`, `SlideSet`, v3 `DatasetManifest`, `ManifestRecord`, and `DatasetBuilder` |
| `training/` | Training mechanics: `Trainer`, validation, metric history, losses, per-step logic, checkpoint state and selection |
| `inference/` | Reusable checkpoint loading, device selection, named-input prediction, single-image workflows, and output naming |
| `evaluation/` | Set evaluation, diagnostic plots, representative selection, comparison panels, summary statistics, ranking utilities |
| `applications/` | User-visible stage lifecycle owners (`prepare.py`, `train.py`, `infer.py`, `evaluate.py`) and other use cases - no `argparse` |
| `cli/` | The `argparse` entrypoint and helpers for `vs <command>` - thin adapters over `applications/` |

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
| Side-effecting preprocessing service | `data/builder.py` | Builds datasets, writes patches/manifests/metadata |
| Side-effecting training service | `training/trainer.py` | Training loop, checkpoint and epoch-history writes; the active session owns run metadata/logging |
| Side-effecting inference service | `inference/runner.py`, `inference/single.py` | Reusable model loading and prediction plus single-image output writing |
| Side-effecting evaluation service | `evaluation/` runners/report writers | Metrics computation plus report/CSV output |

The architectural boundary is not “no I/O in library code.” The actual rule is:

- reusable package code should keep I/O explicit and testable
- orchestration belongs in `applications/`
- CLI translation belongs in `cli/`

The `ExperimentSession` owns each train/infer/evaluate lifecycle: stage snapshots,
strict local metadata writes, and best-effort reporter callbacks. Preparation is
dataset-owned and writes only dataset provenance. Training model construction and
dataset wiring terminate in the reusable `Trainer`; test-set inference delegates
checkpoint loading, device selection, transforms, and batch prediction to `inference/`.

Within training, `trainer.py` owns epoch orchestration, `validator.py` owns validation
inference, `history.py` owns metric CSV persistence, `checkpoints.py` owns model/training
state, and `checkpoint_selection.py` owns `best.json` ranking and resolution. Evaluation
keeps plot primitives in `diagnostics.py`, representative-row policy in `selection.py`,
and composed image layouts in `panels.py`.

## Architectural Rules

These constraints are enforced by convention and checked in code review:

- **No `argparse` outside `cli/`** - application and core modules accept typed
  dataclasses, not raw CLI strings.
- **Core and application modules use `logging`**, never `print`, so callers can
  suppress or redirect output.
- **No `sys.exit()` outside `cli/`** - applications raise exceptions; the CLI
  layer converts them to exit codes.

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
