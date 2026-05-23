# Architecture

## Layer Model

The package is organised in three layers. Dependencies flow downward only -
upper layers may import from lower layers, never the reverse.

| Layer | Description | Examples |
|---|---|---|
| **Library** | Reusable package code with explicit, testable I/O boundaries. Some modules are pure helpers; others are side-effecting services. | `utils/`, `config/`, `experiment/`, `reporting/`, `models/`, `data/`, `training/`, `inference/`, `evaluation/` |
| **Application** | Use-case orchestrators that wire core modules together | `applications/` |
| **Adapter** | Entry points that translate CLI arguments into application calls | `cli/` |

## Package Map

| Package | Responsibility |
|---|---|
| `utils/` | Shared primitives: image dimensions, image I/O helpers, pixel-level metric utilities |
| `config/` | YAML loading and validation, typed config dataclasses, per-section accessors |
| `experiment/` | Run concept: `RunPaths` (directory layout), `RunContext`, `RunMetadata` (run-level provenance), stage/event metadata helpers, environment snapshots |
| `reporting/` | `Reporter` protocol with `NullReporter`, `LoggingReporter`, and `ConsoleReporter` implementations |
| `models/` | `UNetGenerator`, `PatchGANDiscriminator`, model factory, model config dataclass |
| `data/` | `DatasetManifest`, `ManifestRecord`, `DatasetBuilder` (preprocessing pipeline), `PatchDataset` |
| `training/` | `Trainer`, training runner, augmentation, per-step logic, adversarial and L1 losses, checkpoint I/O |
| `inference/` | `Predictor`, inference runner, output writers |
| `evaluation/` | Per-image metrics, `Evaluator`, summary statistics, comparison panels, ranking utilities |
| `applications/` | Use-case orchestrators (`train.py`, `infer.py`, `evaluate.py`, ...) - no `argparse` |
| `cli/` | `argparse` entrypoints (`vs-prepare`, `vs-train`, `vs-infer`, `vs-evaluate`, ...) - thin adapters over `applications/` |

## Purity and I/O Boundaries

The library layer is intentionally mixed:

- some modules are pure or mostly pure helpers
- some modules are side-effecting services that read/write files, logs, checkpoints, or outputs

Typical examples:

| Kind | Example | Notes |
|---|---|---|
| Pure helper | `utils/metrics.py` | Metric computations over arrays/tensors |
| I/O helper | `utils/image_io.py` | Reads/writes image files |
| Mostly pure indexing/data model | `data/dataset.py` | Dataset indexing and manifest-backed lookup |
| Side-effecting preprocessing service | `data/builder.py` | Builds datasets, writes patches/manifests/metadata |
| Side-effecting training service | `training/trainer.py` | Training loop, checkpoint/log/metric writes |
| Side-effecting inference service | `inference/predictor.py`, `inference/runner.py` | Model execution plus output writing at runner boundary |
| Side-effecting evaluation service | `evaluation/` runners/report writers | Metrics computation plus report/CSV output |

The architectural boundary is not “no I/O in library code.” The actual rule is:

- reusable package code should keep I/O explicit and testable
- orchestration belongs in `applications/`
- CLI translation belongs in `cli/`

## Architectural Rules

These constraints are enforced by convention and checked in code review:

- **No `argparse` outside `cli/`** - application and core modules accept typed
  dataclasses, not raw CLI strings.
- **No `print()` outside `cli/` and `reporting/`** - structured progress output
  goes through the `Reporter` protocol; all other modules use `logging`.
- **No `sys.exit()` outside `cli/`** - applications raise exceptions; the CLI
  layer converts them to exit codes.
- **Core and application modules use `logging`**, never `print`, so callers can
  suppress or redirect output.

## Configuration Policy

| Data type | Format | Example path |
|---|---|---|
| User experiment config | YAML | `config/runs/example.yaml` |
| Run metadata (provenance + aggregate stage summary) | JSON | `results/<run>/metadata/run.json` |
| Environment provenance | JSON | `results/<run>/metadata/environment.json` |
| Per-epoch training losses | CSV | `results/<run>/metrics/metrics.csv` |
| Per-image evaluation metrics | CSV | `results/<run>/evaluation/per_image_metrics.csv` |
| Dataset manifest | CSV | `datasets/<name>/manifests/manifest.csv` |

See [`docs/run_format.md`](run_format.md) for the full run output directory layout
and file schemas.
