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
| Infer | `run_image_path_inference(runtime, named_paths, output_path)` (`inference/single.py`) | An `InferenceRuntime` (caller-constructed predictor + `PredictionContract` + device), or a factory returning one, + named input files or directories + output path | Generated images at the output path (or at the runtime's default output dir, if it has one) | `applications.infer` / `vs infer`: test-split manifest inference with consumed/produced snapshots; `applications.infer_images` builds the runtime from a run's checkpoint | Named RGB inputs -> one RGB output on the same pixel grid only; see [Direct predictor inference](#direct-predictor-inference) |
| Evaluate | `evaluate_samples(samples, output_dir, metrics=..., input_failures=...)` / `evaluate_pair(target, generated, metrics=..., support_path=...)` (`evaluation/evaluator.py`) | Explicit `EvaluationSample` records (optional `support_path`) or a pair of image paths + output directory; an optional resolved metric request | `per_image_metrics.csv`, `summary.csv`, `coverage.csv`, `evaluation_result.json` in `output_dir`; `evaluate_pair` writes nothing | `applications.evaluate` / `vs evaluate`: resolves records from the manifest or run outputs and records evaluation provenance | Default request is the built-in default metric set; grouped summaries and producer linking stay in the application; see [Evaluation metrics](#evaluation-metrics) |

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

## Direct predictor inference

An already constructed predictor runs through the same file/directory/resize/tile/WSI
transport as checkpoint-backed inference, without a checkpoint, `RunConfig`, run
directory, manifest or `ExperimentSession`:

```python
from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    run_image_path_inference,
)

model = MyModel().to(device).eval()  # caller-owned and caller-prepared
runtime = InferenceRuntime(
    predictor=model,
    contract=PredictionContract(input_names=("AF", "LF"), image_size=(256, 256)),
    device=device,
)
run_image_path_inference(runtime, {"AF": af_path, "LF": lf_path}, Path("out/sample.png"))
run_image_path_inference(runtime, {"AF": af_dir, "LF": lf_dir}, Path("out/batch"), recursive=True)
```

- **Contract.** `PredictionContract` declares the ordered `input_names`, the
  predictor/tile input `image_size` as `(width, height)`, `output_semantics`
  (only `"same_grid_rgb"`), `value_range` (only `(-1, 1)`) and an optional
  `artifact_direction` used for output names. The predictor is any callable taking
  `{name: (N, 3, H, W) float tensor in [-1, 1]}` in contract order and returning one
  `(N, 3, H, W)` float tensor in [-1, 1] on exactly the input grid. It needs no
  `input_names` attribute. Transport converts images to that range and back.
- **Validation before publication.** Every prediction is checked before it is
  accumulated or written: one tensor (tuples/dicts are rejected), same batch size,
  exactly 3 channels, the same height and width as the input tile, floating dtype,
  finite values, and values within [-1, 1] (±1e-3). Nothing is cropped, padded,
  resized, selected or clamped to make a bad output fit. N-to-M translation, several
  outputs, and scalar or segmentation outputs are not supported.
- **Ownership.** The caller owns the predictor and the device. Transport only calls it
  under `torch.no_grad` (with CUDA autocast on CUDA devices) after moving the inputs
  to `runtime.device`. It never moves, rebuilds, switches the train/eval mode of, or
  closes the predictor, and the runtime stays usable afterwards. The checkpoint
  adapter puts its model in eval mode when it builds it.
- **Provenance is optional.** `InferenceRuntime.checkpoint_path` and
  `predictor_identity` default to `None`, and results report them as they are. The
  checkpoint adapter fills in the real checkpoint path and the method name.
- **Output paths.** Without `default_single_output_dir` /
  `default_directory_output_dir`, every call needs an explicit output path. A call
  without one fails before any prediction and never writes to the working directory.
  `run_image_path_inference` also accepts a zero-argument factory. Directory pairing
  is checked before the factory is called, so a checkpoint is only loaded when the
  inputs are valid.
- **Modes.** `auto` runs one pass when the input already has the contract size and
  tiles otherwise. `resize` resizes the input to `image_size` and writes an output at
  that size, with no claim about the source's physical resolution. `tile` uses
  `image_size` tiles with stride `image_size - tile_overlap`. The last tile is anchored
  at the image edge, partial tiles are padded with white, all inputs are read at the
  same coordinates, overlaps are averaged with equal weight, and only the unpadded
  region contributes. `tile_overlap` must be smaller than both tile dimensions.
- **Outputs.** A single-file output is written to a hidden temporary file next to
  the destination and atomically renamed over it, replacing any existing file. A
  failed run leaves an existing output untouched. An output may not overwrite an
  input. In directory mode, two inputs that map to the same output name (for example
  `a.png` and `a.tif` with `output_format="png"`) are rejected before prediction.
  Names are `<stem>_target_generated<ext>`, or `<stem>_<direction>_generated<ext>`
  when a direction is set.
- **WSI.** When every input opens with OpenSlide and tiling is needed, inputs are read
  region by region. The output has exactly the shared input pixel dimensions and is
  written as a pyramidal BigTIFF by libvips. MPP is copied only from source metadata,
  per axis. For a single WSI input, known source MPP is preserved. For multi-input
  WSI, output MPP is preserved only when every input provides compatible known MPP
  (relative tolerance 1e-4). Conflicting known values fail before prediction, even
  when another input lacks calibration. If any input lacks calibration, the shared
  output MPP remains unknown (not zero). A TIFF has one resolution unit, so an output
  with only one known axis is published with both axes unknown. MPP is never derived
  from pixel counts, and the same-grid output is a pixel-grid contract, not proof
  that the inputs are biologically registered. Before
  prediction, the free space on the output's filesystem must cover at least
  `width x height x 3 x 5` bytes (float32 accumulator plus raw RGB). The compressed
  TIFF needs space on top of that estimate. This is disk-backed scratch space, not
  zero-disk execution, and there is no resumable WSI job. Scratch files live in a
  temporary directory next to the output and are always removed. The TIFF is reopened
  and its geometry, pyramid levels, MPP and sample pixels are checked (headers, not a
  full reread) before it atomically replaces the destination.

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
  `model.inputs` / `model.target`. N-to-M translation, other output kinds and
  registration backends are separate work. Evaluation metrics are supplied the same
  way; see [Evaluation metrics](#evaluation-metrics).
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
(`builtin_definitions`), `virtual_staining.metrics` (`MetricDefinition`,
`MetricResult`, `ResolvedMetric`, `resolve_metrics`, `BUILTIN_METRICS`),
`virtual_staining.evaluation.evaluator` (`evaluate_samples`, `evaluate_pair`,
`EvaluationSample`, `EvaluationInputError`, `EvaluationCoverageError`), `virtual_staining.training.runtime` (`TrainingMethodRuntime`,
`MethodMetrics`), `virtual_staining.checkpoint_contract` (`CheckpointIdentity`,
`ValidatedCheckpoint`, `CheckpointCompatibilityError`), `virtual_staining.config`
(`reject_unknown_keys`, `parse_bool_strict`), `virtual_staining.config.run.RunConfig`,
`virtual_staining.training.trainer.Trainer`, `virtual_staining.inference.runner`
(`load_inference_generator`) and `virtual_staining.inference.single`
(`run_image_path_inference`, `InferenceRuntime`, `PredictionContract`). The application entry points
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

## Inspecting and checking configs

`virtual_staining.applications.config_authoring` is the one read-only seam for
authoring and checking a run config (`vs config resolve` / `vs config check` call it):

```python
from virtual_staining.applications.config_authoring import (
    inspect_run_mapping, inspect_run_yaml, preflight, write_config_yaml,
)

inspection = inspect_run_mapping(raw, definitions=my_definitions)  # or inspect_run_yaml(path, ...)
inspection.authored_yaml     # the caller's mapping, key order and every valid field kept
inspection.resolved_yaml     # RunConfig.to_dict(), byte-identical to a tracked resolved.yaml
inspection.resolved_sha256   # the config hash a tracked stage records for this config
inspection.origins           # {"training.losses.generator[0].weight": "supplied", ...}

report = preflight(inspection.config, ["prepare", "train"], depth="assets")
report.valid                 # False only when a check is "invalid"
write_config_yaml(inspection.authored_yaml, Path("run.yaml"))  # FileExistsError if present
```

- Resolution always goes through `RunConfig.from_mapping`; `definitions` defaults to the
  built-in set, and external methods/components keep their options in both forms.
- `authored` is the caller's mapping (plain dicts/lists), never `to_dict()`. A starter is
  one of the committed `config/runs/minimal_*.yaml` files; there is no generated starter.
- `origins` is explanatory only: `supplied` for leaves present in the authored mapping
  (including a value its owner normalized), `defaulted` for owner-filled leaves.
- `preflight(config, stages, depth=...)` checks the stages in the given order. `config`
  depth inspects no path. `assets` depth reuses the stage owners read-only
  (`resolve_slide_sets`, manifest validation, `resolve_domain_collections`,
  `validate_groups`, `resolve_inference_checkpoint`, the evaluation protocol and
  generated-file naming). Check statuses: `valid`, `invalid`, `planned` (an earlier
  selected stage produces the input; not verified), `unverified` (deliberately not
  established), `not_applicable`.
- Preflight never opens a session, runs a stage, hashes or decodes a file, deserializes a
  checkpoint, builds a model or probes a device, and writes nothing. `content_verified`
  is always `false`, and a report does not freeze or lock its inputs; tracked execution
  re-resolves, re-validates, and snapshots what it consumes.

## Exporting model bundles

`virtual_staining.applications.export_model` packages selected checkpoints of one
tracked run as a portable local bundle (format: [`run_format.md`](run_format.md#model-bundles)):

```python
from virtual_staining.applications.export_model import (
    ExportCheckpointSelection,
    export_model_bundle,
    verify_model_bundle,
)

bundle = export_model_bundle(
    Path("local_workspace/results/my_run"),
    Path("bundles/my_run"),
    [
        ExportCheckpointSelection("best", metric="val_abs_bias"),
        ExportCheckpointSelection("top_k", metric="val_abs_bias", rank=2),
        ExportCheckpointSelection("latest"),
        ExportCheckpointSelection("explicit", checkpoint_path=Path("ep010.pth")),
    ],
    definitions,  # optional; defaults to builtin_definitions()
)

# Later, anywhere the bundle was moved to:
verify_model_bundle(moved, definitions)
config = RunConfig.from_yaml(moved / "config" / "resolved.yaml", definitions)
model, _ = load_inference_generator(config, RunLayout(moved), device, moved / "checkpoints" / "ep010.pth")
```

- **Explicit providers.** External methods and components are exported and verified
  only with their `Definitions` supplied; without them export, verification and
  reconstruction fail with `DefinitionNotAvailableError`. `vs export-model` knows the
  built-ins only. `bundle.json` records each required definition's `name`, `source` and
  `version`, never code to import.
- **Owners reused.** Selection goes through `checkpoint_selection`, reading and
  validation through `checkpoint_contract` and the method definition's checkpoint
  identity, configuration through `RunConfig.from_yaml`, hashing through
  `utils.hashing`. There is no bundle-specific loader: reconstruction is the normal
  `load_inference_generator` with an explicit checkpoint path.
- **Verification.** `export_model_bundle` runs `verify_model_bundle` on the staged
  bundle before publishing it and returns the verified `ModelBundle` (`root`, `index`,
  `config`). Call `verify_model_bundle` again before using a bundle received from
  elsewhere.
- **Not a distribution decision.** The bundled configs are exact research provenance
  and may name private local paths. Export does not decide whether weights may be
  redistributed.

## Evaluation metrics

Paired evaluation computes an explicit, ordered request of metric definitions. A
`MetricDefinition` is immutable and owns its `name` (also the output column),
`version`, `source`, an `evaluator` (metrics sharing one evaluator callable form a
group computed once per image pair), a strict `parse_options(raw, field)` returning
JSON-compatible options, `higher_is_better` (`None` when ranking is meaningless),
`supports_valid_region`, and presentation-only `thresholds` / `plot_range`. Every
evaluator receives two float `H x W x 3` RGB arrays in [0, 1] on the same grid; inputs
are never resized, cropped, clipped, rescaled or channel-converted.

```python
import numpy as np

from virtual_staining.config.run import RunConfig
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.metrics import MetricDefinition, MetricResult


def max_error(target, generated, support, requested):
    scale = requested["max_error"]["scale"]
    return {"max_error": MetricResult.of(scale * float(np.abs(target - generated).max()))}


def parse(raw, field):
    if set(raw) - {"scale"}:
        raise ValueError(f"{field} has unknown keys {sorted(set(raw) - {'scale'})}")
    return {"scale": float(raw.get("scale", 1.0))}


definitions = builtin_definitions().extend(
    metrics=[MetricDefinition("max_error", "1", "my_package", max_error, False, parse)]
)
config = RunConfig.from_yaml("my_run.yaml", definitions)  # evaluation.metrics may name it
```

- **Requests.** `evaluation.metrics` (or `resolve_metrics(request, definitions.metrics)`
  for the library path) is an ordered list of `{name, options}` mappings, resolved once
  before anything is read. Unknown or duplicate names, unknown or malformed options and
  non-JSON options fail. Omitted, it is the built-in default set `mae`, `mse`, `rmse`,
  `psnr`, `ssim`, `pcc_gray`, `pcc_rgb_mean`; `pcc_r`, `pcc_g`, `pcc_b` are also
  built in. Only requested evaluator groups run, and only requested outputs are
  reported, so a custom metric gets its per-image columns, summary row, grouped columns,
  histogram and result-metadata entry without any report code changes.
- **Results.** Evaluators return one `MetricResult` per requested name with status
  `finite`, `positive_infinity`, `undefined` or `unavailable` (the last two carry a
  reason). `MetricResult.of(x)` classifies a number and rejects NaN and negative
  infinity. Missing outputs, non-`MetricResult` values and invalid states raise
  `MetricEvaluatorError`. Built-in PSNR of identical images is `positive_infinity`, PCC of
  constant data is `undefined`, and SSIM (fixed scikit-image parameters: 7x7 uniform
  window, sample covariance, `K1=0.01`, `K2=0.03`, `data_range=1`) is `unavailable` for
  images smaller than 7 px rather than recomputed with another window.
- **Coverage.** Only known input problems (`EvaluationInputError`: missing, unreadable
  or non-RGB file, shape mismatch, bad support) are per-sample coverage events.
  `input_failures="strict"` (default) writes `coverage.csv` and raises
  `EvaluationCoverageError`; `"permissive"` excludes those samples. No samples, no
  metrics, or zero evaluated samples never produce a result. Any other exception,
  including a metric bug, propagates in both modes.
- **Valid-region support.** `EvaluationSample.support_path` (for every sample or none)
  restricts the support-capable metrics (built-in `mae`, `mse`, `rmse`, `psnr`) to the
  valid pixels, all RGB channels, and adds `<m>_support_count` /
  `<m>_support_fraction` columns. The mask must be explicitly binary (PNG mode `1`, or
  mode `L` holding only 0 and 255) on exactly the image grid; soft masks are rejected,
  never thresholded. An empty region yields `undefined`. Requests with support and a
  metric without support (SSIM, PCC) fail before reading any file. Support is
  evaluation input only: it is not a tissue mask, registration confidence or
  preparation mask, and none is discovered or generated; the configured `evaluate`
  stage does not use one.
- **Result metadata.** `evaluation_result.json` records each resolved identity,
  direction and presentation metadata. Downstream ranking (`vs organize`,
  `vs compare`, `vs panels`) takes directions from it, else from a built-in
  definition of the same name, else from explicit caller input, and fails otherwise.
- **Training metrics are separate.** A `MethodDefinition` owns its validation and
  checkpoint metrics (`validation_metric_names`, `checkpoint_metrics`); they never need a
  `MetricDefinition`. Pix2Pix explicitly reuses the built-in `ssim`, `psnr`, `mae`,
  `rmse`, `pcc_rgb_mean` and `pcc_gray` definitions for its `val_*` columns
  (`PIX2PIX_VALIDATION_METRICS`); CycleGAN reports none.
- **Unpaired diagnostics** (`evaluation/unpaired.py`,
  `evaluate_unpaired_collections(generated_paths, reference_paths, output_dir, ...)`)
  compare per-image RGB/luminance feature distributions of two independent collections
  and need no model, checkpoint, method or target pair. They are appearance
  diagnostics, not sample-level fidelity metrics, and do not use metric definitions.
