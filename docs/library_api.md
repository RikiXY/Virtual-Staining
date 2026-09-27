# Library Stage API

Every stage can be called directly as a library primitive from the inputs it
actually consumes. The YAML/application workflow (`vs prepare`, `vs train`,
`vs infer`, `vs evaluate`, `vs run`) is the convenience path that wires them
together and adds tracked-run provenance; it is not required to use any single
stage. This page only covers the standalone boundaries; see
[`architecture.md`](architecture.md) for package layers and
[`run_format.md`](run_format.md) for configuration and output layouts.

| Stage | Standalone boundary | Natural inputs | Outputs / side effects | Tracked alternative | Important limitation |
|---|---|---|---|---|---|
| Prepare | `DatasetBuilder(config, slide_sets).run_all()` (`data/builder.py`) | `PreprocessingConfig` + explicit `SlideSet` tuple | Patches, manifest, manifest metadata, slide-set metadata, split assignment and dataset fingerprint under `config.dataset_root` | `applications.prepare` / `vs prepare`: resolves `SlideSet`s from the YAML inventory, snapshots config and sources, reuses unchanged datasets | Writes only under `dataset_root`; the inventory CSV is read only by `resolve_slide_sets`, not by the builder |
| Train | `Trainer(config, run_paths, method, train_loader, val_loader, device)` then `.train(seed)` (`training/trainer.py`) | `TrainingConfig` + `TrainingMethodRuntime` + train/val `DataLoader`s + output `RunLayout`; optional `progress_reporter`, `preview_sink`, `benchmark_recorder`, `config_hash`, `experiment_session` | `metrics/epochs.csv`, `checkpoints/ep*.pth`, `checkpoints/best.json`, training/validation output dirs under the `RunLayout` root; `.resume()` reads checkpoints there | `applications.train` / `vs train`: builds loaders from the manifest or domain collections, binds the consumed-data snapshot and passes its real `ExperimentSession` | The loaders are the data boundary; the Trainer never reads manifests, dataset roots or preparation outputs. The runtime comes from `config.method.definition.build_training_runtime(config, device, seed=...)` and supplies its own checkpoint identity |
| Infer | `run_image_path_inference(runtime_factory, named_paths, output_path)` (`inference/single.py`) | A `RuntimeFactory` returning an `InferenceRuntime` + named input files or directories + explicit output path | Generated images at the output path (or at the runtime's default output dir when none is given) | `applications.infer` / `vs infer`: test-split manifest inference with consumed/produced snapshots; `applications.infer_images` builds the runtime from a run's checkpoint | Accepts an injected runtime factory only; `inference.runner.load_inference_generator` builds the network through the selected method definition. A general predictor/spatial-output boundary is separate future work |
| Evaluate | `evaluate_samples(samples, output_dir)` / `evaluate_pair(target, generated)` (`evaluation/evaluator.py`) | Explicit `EvaluationSample` records or a pair of image paths + output directory | `per_image_metrics.csv`, `summary.csv`, `skipped.csv` in `output_dir`; `evaluate_pair` writes nothing | `applications.evaluate` / `vs evaluate`: resolves records from the manifest or run outputs and records evaluation provenance | Uses the standard metric set; grouped summaries and producer linking stay in the application |

## Notes

- **The boundaries are deliberately not symmetrical.** Each primitive takes what
  its computation needs; there is no universal stage interface.
- **Preparation is useful, not mandatory.** A prepared manifest is a data
  contract: a compatible externally produced current-schema dataset (manifest,
  manifest metadata, referenced files, and slide-set grouping metadata when group
  validation needs it) can be trained on directly without a `vs prepare` run or
  dataset fingerprint. Manifest validation, path safety, split checks,
  biological-group leakage checks and content hashing still apply; the consumed-data
  snapshot bound by the train application becomes the training identity.
- **Standalone calls do not claim tracked provenance.** Direct `DatasetBuilder`,
  `Trainer` without `experiment_session`, `run_image_path_inference` and
  `evaluate_samples` / `evaluate_pair` never create an `ExperimentSession`, `run.json`,
  stage records, events or consumed-data snapshots. Their ordinary output files are
  still real side effects. When a tracked application calls the same primitive it
  owns the session and records the provenance.
- **Trainer without a session** still writes `epochs.csv` and checkpoints; per-epoch
  metrics are only additionally forwarded to reporters when a session is supplied.
  `config_hash` is optional and, when omitted, checkpoints and `best.json` simply
  carry no config hash.
- Importing these primitives does not initialize CUDA, load a model, open WSI
  readers, import UI packages or inspect project roots.

`tests/architecture/test_standalone_stages.py` exercises each boundary from a fresh
temporary directory and asserts that unrelated predecessor paths (manifest, run
metadata, checkpoints, inventory) are never opened.

## Extending with explicit definitions

A new image-translation method or network architecture is supplied as Python
definitions and selected by name in the run configuration:

```python
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.applications.train import train
from virtual_staining.methods.builtin import builtin_definitions

from my_package.methods import TINY_CONV, TINY_RESIDUAL, TinyReconstruction

definitions = builtin_definitions().extend(
    methods=[TinyReconstruction()],
    components=[TINY_CONV, TINY_RESIDUAL],
)
config = RunConfig.from_yaml("my_run.yaml", definitions)  # or RunConfig.from_mapping(raw, definitions)
train(config, Path("my_run.yaml"))
```

```yaml
method:
  name: tiny_reconstruction
  options:
    architecture: tiny_conv
    learning_rate: 0.001
model:
  inputs: [source]
  target: target
```

- **Registration is explicit Python.** `Definitions` is an immutable value; `extend()`
  returns a new set and rejects any duplicate method or component name instead of
  replacing it. `builtin_definitions()` is the default set (Pix2Pix, CycleGAN and the
  `concat_unet`, `resnet`, `patchgan` components), and the built-ins are registered
  through exactly this mechanism.
- **The stock CLI contains the built-ins only.** There is no plugin discovery: no entry
  points, directory scanning, import strings, or `class_path` in YAML.
- **Checkpoints never import code.** A v4 checkpoint records the registered method and
  component names, their `version`/`source`, normalized options, I/O names, directions,
  image size and normalization. Loading compares them with definitions the caller
  already supplied and fails with `DefinitionNotAvailableError` when a named definition
  is missing, before anything is built.
- **Resolution order.** `RunConfig.from_mapping` parses the framework-common fields,
  resolves `method.name` in the supplied definitions, and only then lets the selected
  definition validate the keys it owns (`owned_keys`; `method.options` by default). YAML
  loading uses the same path. Unknown keys fail everywhere. The resolved config records
  the method name and its option spelling; `MethodConfig.definition` is a live
  reference and is never serialized.
- **Contract: named N RGB inputs -> one RGB output**, normalized to [-1, 1], matching
  `model.inputs` / `model.target`. N-to-M translation, other output kinds, generalized
  metric registration and registration backends are separate work.
- **Ownership.** A `MethodDefinition` owns its options and their validation, its
  pairing, prediction directions, the validation metrics it ranks and their direction
  (`checkpoint_metrics`, `monitor_mode`), training-runtime construction, inference-only
  model construction, and its checkpoint identity. A `ComponentDefinition` owns one
  architecture's option parser and factory; the method decides which components it
  accepts and how they compose. The shared `Trainer` owns the epoch, validation,
  checkpoint, `best.json` and history lifecycle; the shared inference layer owns
  single/directory/tiled/WSI transport.

Public modules for extension code: `virtual_staining.definitions` (`MethodDefinition`,
`ComponentDefinition`, `Component`, `ComponentContext`, `ResolutionContext`,
`Definitions`, `DefinitionNotAvailableError`), `virtual_staining.methods.builtin`
(`builtin_definitions`), `virtual_staining.training.runtime` (`TrainingMethodRuntime`,
`MethodMetrics`), `virtual_staining.checkpoint_contract` (`CheckpointIdentity`,
`ValidatedCheckpoint`, `CheckpointCompatibilityError`), `virtual_staining.config`
(`reject_unknown_keys`, `parse_bool_strict`), `virtual_staining.config.run.RunConfig`,
`virtual_staining.training.trainer.Trainer`, `virtual_staining.inference.runner`
(`load_inference_generator`) and `virtual_staining.inference.single`
(`run_image_path_inference`, `InferenceRuntime`). The application entry points
`applications.train.train`, `applications.infer.infer` and
`applications.infer_images.infer_images(..., definitions=...)` /
`applications.pipeline.run_stages(..., definitions=...)` accept an externally resolved
configuration or definition set.

A method's training runtime reports `MethodMetrics`: `losses` holds its objective
scalars (`metric_names`, written as `<name>_train`/`<name>_val`), `image` holds further
validation scalars (`validation_metric_names`), and the per-term component maps are
optional. `objective_metadata()` may return JSON-compatible objective provenance for
`best.json`. `tests/external_method/` is a complete non-GAN example (one network, one
optimizer, an L1 objective, a custom `val_abs_bias` checkpoint metric, two registered
architectures) that uses only these modules.
