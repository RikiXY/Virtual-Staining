# Architecture

## Layer Model

Dependencies flow from adapters to applications to library packages; library code
never imports applications or the CLI.

| Layer | Responsibility |
|---|---|
| Library | Reusable computation and services with explicit I/O boundaries |
| `applications/` | Use-case composition, input selection, and tracked stage lifecycles |
| `cli/`, `ui/` | Command-line and browser adapters over application services |

Library services may read and write files. Standalone stages consume their natural
inputs without requiring a tracked run; their public boundaries are documented in
[Library Stage API](library_api.md).

## Package Map

| Package or contract | Semantic owner |
|---|---|
| `config/` | Framework configuration and resolution against explicit definitions |
| `definitions.py` | Method, component, and metric registration boundary |
| `data/` | Dataset layout, paired manifests, unpaired collections, preparation, registration, and data provenance |
| `models/` | Network components and model-I/O normalization; no training state |
| `methods/` | Built-in method options, topology, objectives, optimization, and restorable state |
| `training/` | Method-independent epochs, validation cadence, history, checkpointing, and early stopping |
| `inference/` | Definition-driven model loading and shared image/directory/tiled/WSI prediction |
| `evaluation/` | Paired reports, unpaired diagnostics, grouping, comparisons, and panels |
| `experiment/` | Run and comparison layouts, tracked sessions, config/environment snapshots, and events |
| `checkpoint_contract.py`, `checkpoint_selection.py` | Checkpoint compatibility and selection semantics |
| `metrics.py`, `loss_definitions.py` | Evaluation metric definitions and built-in loss primitives respectively |
| `utils/`, `split_contract.py` | Shared low-level utilities and split vocabulary |

The browser adapter calls `applications.api.ApplicationService`; it shares config
resolution, checkpoint validation, prediction transport, and evaluation services with
the CLI. Its patch catalog reconstructs the supported built-in method through its
definition. Plotting uses a non-interactive backend and a shared lock for worker threads.

## Translation Methods

Methods and network components enter through explicit definitions. Generic configuration
resolves framework fields while each definition validates its own options and
cross-section rules. This boundary is torch-free; config resolution does not load a
training runtime. The built-in definition set is a default, not a dependency of the
generic training or inference layers. Checkpoint metadata identifies supplied definitions
and never imports code. Extension contracts and examples belong to
[Library Stage API](library_api.md#extending-with-explicit-definitions).

Concrete methods depend on generic training and inference contracts, never the reverse.
A method owns its topology, objective composition, optimization, prediction directions,
and opaque checkpoint state. The Trainer owns the epoch and history lifecycle without
assuming a GAN, a fixed number of networks, or how named image channels are packed.
Inference constructs only the prediction network and shares one image-path transport
between caller-owned predictors and checkpoint-backed models.

Loss primitives are shared, but their composition is method-owned. Evaluation metric
definitions and method-owned validation/checkpoint metrics remain separate contracts.
The application selects the evaluation protocol independently of training pairing.
Paired reports retain output identity; unpaired diagnostics describe collection
appearance and do not establish sample-level fidelity. Persisted score meanings are in
[Run Output Format](run_format.md#evaluation-outputs).

## Purity and I/O Boundaries

Applications select the actual stage inputs and bind their provenance; sessions do not
infer consumption from files that happen to exist. `ExperimentSession` owns tracked
train/infer/evaluate lifecycles, local metadata, and best-effort reporters. Preparation
is dataset-owned and emits no experiment run events. Dataset paths belong to
`DatasetLayout`, run paths to `RunLayout`, and shared comparison paths to `ResultsLayout`.
See [dataset provenance](dataset_format.md#prepared-layout) and
[consumed-data snapshots](run_format.md#consumed-data-snapshots) for their distinct identities.

Registration geometry, estimation, independent QC, and inverse resampling belong to
`data/alignment/`, which depends on no preparation or run configuration. Callers own
readers, cleanup, explicit reference selection, and failure policy. Existing preparation
adapts its coordinate declarations to candidate requests; it does not certify biological
correspondence or apply study QC. See the [alignment API](library_api.md#registration-and-resampling).

Model export is an application utility, not a pipeline stage. It uses the existing
configuration, checkpoint, and inference contracts; it introduces no separate model
loader or registry.

## Architectural Rules

- `argparse`, `sys.exit()`, and terminal presentation belong only in `cli/`.
  Applications accept typed inputs and raise exceptions; library and application
  diagnostics use `logging`. Progress remains silent unless a caller supplies a reporter.
- Library packages never import `applications/` or `cli/`; command adapters delegate
  to applications.
- `training/` and `inference/` never import concrete `methods/`.
- `config/` does not depend on runtime domains; definitions provide the extension boundary.
- `utils/`, `metrics.py`, and `split_contract.py` remain dependency leaves within the package.

Configuration options live in the [annotated run YAMLs](../config/runs/example.yaml);
persisted schemas live in [Run Output Format](run_format.md) and
[Dataset Format](dataset_format.md).
