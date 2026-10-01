# External Python consumer

This separate Python package is executable engineering extensibility evidence for
MEXINA. It is not scientific, model-quality, registration-benchmark or clinical
validation. It contains no framework changes and imports no repository test helpers.

From the repository checkout, using the managed dependencies:

```bash
nix develop -c uv run env PYTHONPATH=examples/external_consumer CUDA_VISIBLE_DEVICES= python -P -m mexina_external_example.demo /tmp/mexina-demo-new
nix develop -c uv run pytest -q tests/external_consumer
```

The demo destination must not already exist. The integration test copies this entire
directory into `/tmp`, uses an unrelated working directory, and runs all seven
`unittest` checks there with `python -P`. Only that copied consumer directory is added
to `PYTHONPATH`; `virtual_staining` comes from the managed environment's normal
installation. A subprocess audit hook rejects reads/listings anywhere in the original
checkout except the installed framework sources and `.venv`. No original datasets,
configs, tests, results or `local_workspace` may be accessed. CUDA is disabled for the
CPU fixture. The Nix shell's bare Python does not include project dependencies; use
`uv run` as above.

To run a copied consumer independently with an environment where MEXINA is installed:

```bash
cd /path/to/copied/external_consumer
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= python -P -m unittest discover -s checks -v
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES= python -P -m mexina_external_example.demo /tmp/mexina-demo-new
```

No packaging metadata is needed for this small, importable directory package.

## Pieces and evidence

- `mexina_external_example/definitions.py` supplies immutable definitions explicitly:
  `builtin_definitions().extend(methods=[Reconstruction()], components=[TINY_CONV,
  TINY_RESIDUAL], metrics=[SCALED_MAX_ERROR])`. External definitions are explicitly
  supplied in Python; stock CLI plugin discovery is not provided. YAML contains only
  supplied definition names and validated scalar options, never executable imports.
- `method.py` concatenates ordered RGB `LF` and `AF` tensors, predicts `HE` with a
  tanh output, and trains one network with one Adam optimizer and L1 reconstruction.
  `tiny_conv` has a 3×3 feature convolution and 1×1 output convolution;
  `tiny_residual` adds a 3×3 residual feature block. Both have a validated `width`
  option (default 4). The method-owned `loss_reconstruction_val` ranks checkpoints.
  No discriminator, GAN fields, scheduler or training-metric registry is involved.
- `metric.py` defines `scaled_max_error = scale * max(abs(target - generated))` on
  float HWC RGB `[0, 1]` arrays. It returns `MetricResult`, validates a finite positive
  numeric scale, and persists name/version/source/options in evaluation metadata.
  The independent known-pixel pair yields `4/255` at scale 2; built-in MAE runs beside it.
- `registration.py` supplies a deterministic affine translation of moving pixel centres
  by `(-2, -1)` directly to explicit reference `LF`. Actual `DatasetBuilder` preparation
  resamples both `AF` and `HE`. Every written patch is checked against source pixels
  at `(x+2, y+1)`, and every asset's canonical alignment result is read back from
  `manifests/slide_sets.csv`. This analytic fixture is not a production backend;
  QC remains unassessed and no accuracy claim or threshold is supplied.
- `demo.py` shows independent preparation from explicit `SlideSet`s (no inventory),
  training from caller-owned tensors/loaders and `RunLayout`, in-memory predictor
  inference on actual image paths, and evaluation of explicit reference pairs.
  Preparation leaves source bytes unchanged and checks output paths under the dataset
  root. Standalone training creates checkpoints/history without tracked run metadata.
- The composed function prepares 12 patches across three distinct sets (4 per split),
  then calls existing train, infer and evaluate applications. It reconstructs the
  checkpoint through `load_inference_generator` and checks all produced artifacts.
  There is no additional stage engine. This composition uses standalone preparation;
  only train/infer/evaluate have tracked application provenance.
- `checks/test_contracts.py` also exercises both architectures, actual second-epoch
  resume including optimizer restoration, deterministic config round-trips, missing
  definitions, duplicate registrations, strict options, changed checkpoint identities,
  predictor name/shape/finiteness/range failures and strict/permissive evaluation
  coverage. Metadata naming an importable trap module is rejected without importing it.
  Checkpoints remain current-format, weights-only loaded by the existing framework.

The demo writes `evidence.json` with artifact paths and observed custom metric values.
Other outputs are `training/standalone_run/{checkpoints,metrics}`, predictor PNGs,
`explicit_evaluation/report/{per_image_metrics.csv,summary.csv,coverage.csv,evaluation_result.json}`,
and the composed dataset manifests, checkpoints, generated images and evaluation reports.
Results are engineering observations on tiny deterministic fixtures, not quality scores.

## Public framework imports

These are the complete framework imports of the consumer implementation. The supported
boundaries are described in `docs/library_api.md`, with canonical dataset artifacts in
`docs/dataset_format.md` and path ownership in `docs/architecture.md`.

| Module | Symbols |
| --- | --- |
| `virtual_staining.applications.evaluate` | `evaluate` |
| `virtual_staining.applications.infer` | `infer` |
| `virtual_staining.applications.train` | `train` |
| `virtual_staining.checkpoint_contract` | `CheckpointCompatibilityError`, `CheckpointIdentity`, `ValidatedCheckpoint` |
| `virtual_staining.config` | `reject_unknown_keys` |
| `virtual_staining.config.data` | `PreprocessingConfig` |
| `virtual_staining.config.run` | `RunConfig` |
| `virtual_staining.data.alignment` | `AlignmentResult`, `AlignmentTransform`, `RegistrationAttempt`, `RegistrationBackend`, `RegistrationRuntime` |
| `virtual_staining.data.builder` | `DatasetBuilder` |
| `virtual_staining.data.slide_sets` | `SlideAsset`, `SlideSet` |
| `virtual_staining.definitions` | `Component`, `ComponentContext`, `ComponentDefinition`, `Definitions`, `MethodDefinition`, `ResolutionContext` |
| `virtual_staining.evaluation.evaluator` | `EvaluationSample`, `evaluate_pair`, `evaluate_samples` |
| `virtual_staining.experiment.run_layout` | `RunLayout` |
| `virtual_staining.inference.runner` | `load_inference_generator` |
| `virtual_staining.inference.single` | `InferenceRuntime`, `PredictionContract`, `run_image_path_inference` |
| `virtual_staining.methods.builtin` | `builtin_definitions` |
| `virtual_staining.metrics` | `MetricDefinition`, `MetricResult`, `resolve_metrics` |
| `virtual_staining.training.runtime` | `MethodMetrics` |
| `virtual_staining.training.trainer` | `Trainer` |

The checks additionally import public `DefinitionNotAvailableError` from
`virtual_staining.definitions` and `EvaluationCoverageError` from
`virtual_staining.evaluation.evaluator`. An AST check rejects test/private imports and
checks the implementation import list against this table.

## Limits

The example method deliberately supports exactly one output; MEXINA itself supports
ordered named N-input/M-output mappings. RGB, same-grid prediction and `[-1, 1]` values
are enforced by the existing inference transport. This fixture has tiny CPU networks,
synthetic PNGs, no WSI benchmark, no independent biological groups and no scientific
registration evidence. Reported registration success means the analytic transform was
applied, not that it passed scientific QC. A second contributor has not independently
executed this example; that evidence is unverified.
