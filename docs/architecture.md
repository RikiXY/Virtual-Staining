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
| `checkpoint_contract.py` | Neutral v3 checkpoint format, model-I/O normalization, and generator/discriminator metadata validation |
| `checkpoint_selection.py` | Neutral `best.json` ranking, policy, and metric-direction selection |
| `utils/` | Shared primitives: artifact naming, image dimensions, and image I/O helpers |
| `config/` | Sole owner of YAML-facing dataclasses and strict parsers for every config section |
| `experiment/` | Canonical `RunLayout` for one run, `ResultsLayout` for shared comparisons, stage snapshots, run metadata, manifest/config hashing, and environment snapshots |
| `models/` | Model factory, model-I/O normalization contract, and generator/discriminator implementations |
| `data/` | Canonical `DatasetLayout`, slide sets, manifests, dataset building, and dataset-owned provenance/fingerprints |
| `training/` | Training mechanics, validation, history, losses, resume state, and callback-driven progress events |
| `inference/` | Reusable checkpoint loading and runtime inference; application code owns runtime composition |
| `evaluation/` | Set evaluation, diagnostic plots, representative selection, comparison panels, and summaries |
| `applications/` | User-visible stage lifecycle owners and infer-images runtime composition; no `argparse` |
| `cli/` | The `argparse` entrypoint, terminal rendering, and thin adapters over `applications/` |

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
The `ExperimentSession` owns each train/infer/evaluate lifecycle: stage snapshots,
strict local metadata writes, and best-effort reporter callbacks. `RunLayout` owns
one run's paths; `ResultsLayout` owns shared cross-run comparisons under
`results/comparisons`. `applications.prepare` orchestrates dataset-local config and
environment snapshots through the generic experiment snapshot helpers. Preparation
is dataset-owned, writes dataset fingerprints and source hashes, and emits no
experiment run events. Dataset provenance lives in `data/provenance.py`; run
provenance lives in `experiment/snapshots.py`.
Training model construction and dataset wiring terminate in the reusable `Trainer`;
its `ProgressUpdate` callback is silent unless an adapter supplies a reporter.
The CLI supplies terminal rendering, while application/library callers remain
presentation-neutral. Infer-images runtime creation belongs to `applications/`;
`inference/single.py` accepts an already-loaded `InferenceRuntime`.

Within training, `trainer.py` owns epoch orchestration, `validator.py` owns validation
inference, `history.py` owns metric CSV persistence, `checkpoints.py` owns model/training
state, `checkpoint_contract.py` owns the neutral v3 contract, and
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

The library graph is an enforced direct-edge DAG:

```text
cli -> applications, cli, metrics
applications -> checkpoint_contract, checkpoint_selection, config, data, evaluation,
                experiment, inference, metrics, models, training, utils
config -> config, checkpoint_selection, metrics, utils
checkpoint_contract -> models
data -> config, data, utils
models -> config, models
experiment -> config, data, experiment
training -> checkpoint_contract, checkpoint_selection, config, experiment, metrics,
            models, training, utils
inference -> checkpoint_contract, checkpoint_selection, config, data, experiment,
             inference, models, utils
evaluation -> config, evaluation, metrics, utils
utils -> utils
```

`tests/architecture/test_package_dependencies.py` resolves absolute, relative,
nested, and `TYPE_CHECKING` imports with the standard library. It reports the
source file and illegal import, and verifies the allowlist topologically sorts.

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
