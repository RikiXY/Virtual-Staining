# Library Stage API

Every stage can be called directly as a library primitive from the inputs it
actually consumes. The YAML/application workflow (`vs prepare`, `vs train`,
`vs infer`, `vs evaluate`, `vs run`) is the convenience path that wires them
together and adds tracked-run provenance; it is not required to use any single
stage. This page only covers the standalone boundaries; see
[`architecture.md`](architecture.md) for package layers and
[`run_format.md`](run_format.md) for persisted output formats.

| Stage | Standalone boundary | Inputs and side effects |
|---|---|---|
| Prepare | `data.builder.DatasetBuilder(config, slide_sets).run_all()` | `PreprocessingConfig` and explicit `SlideSet` tuple; writes the [prepared dataset](dataset_format.md#prepared-layout) under `config.dataset_root` without reading an inventory |
| Train | `training.trainer.Trainer(config, run_paths, method, train_loader, val_loader, device).train(seed)` | `TrainingConfig`, output `RunLayout`, `TrainingMethodRuntime`, and train/val `DataLoader`s; writes history, checkpoints and optional previews; `.resume()` reads checkpoints |
| Infer | `inference.single.run_image_path_inference(runtime, named_paths, output_path)` | Predictor runtime or factory plus named files/directories; writes generated images; see [Direct predictor inference](#direct-predictor-inference) |
| Evaluate | `evaluation.evaluator.evaluate_samples(samples, output_dir, metrics=..., input_failures=...)` / `evaluate_pair(target, generated, metrics=..., support_path=...)` | Explicit `EvaluationSample(sample_id, output_name, set_id, target_path, generated_path)` records (optional `support_path`, unique sample/output pairs) or two image paths; the sample API writes [paired reports](run_format.md#evaluation-outputs), while `evaluate_pair` writes nothing |

The matching `applications.prepare`, `applications.train`, `applications.infer`, and
`applications.evaluate` entry points add tracked provenance and resolve stage inputs.
Grouped evaluation summaries and producer linking belong to the tracked application.
`applications.infer_images` supplies a checkpoint-backed runtime for image-path inference.

## Notes

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
- **Trainer** consumes caller-supplied loaders, without reading manifests or dataset
  roots. Its runtime is built by `config.method.definition.build_training_runtime(config,
  device, seed=...)`. Optional integrations are `progress_reporter`, `preview_sink`,
  `benchmark_recorder`, `config_hash`, and `experiment_session`. Without a session it
  still writes `epochs.csv` and checkpoints; per-epoch metrics are additionally
  forwarded to reporters when a session is supplied.
  `config_hash` is optional and, when omitted, checkpoints and `best.json` simply
  carry no config hash. `resume(checkpoint)` restores state and returns the next epoch;
  the caller passes that value to `train(seed, start_epoch=...)`.
- Importing these primitives does not initialize CUDA, load a model, open WSI
  readers, import UI packages or inspect project roots.

## Direct predictor inference

An already constructed predictor runs through the same file/directory/resize/tile/WSI
transport as checkpoint-backed inference, without a checkpoint, `RunConfig`, run
directory, manifest or `ExperimentSession`:

```python
from pathlib import Path

from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    run_image_path_inference,
)

model = MyModel().to(device).eval()  # caller-owned and caller-prepared
runtime = InferenceRuntime(
    predictor=model,
    contract=PredictionContract(
        input_names=("AF", "LF"), output_names=("HE", "PAS"), image_size=(256, 256)
    ),
    device=device,
)
run_image_path_inference(runtime, {"AF": af_path, "LF": lf_path}, Path("out"))
run_image_path_inference(runtime, {"AF": af_dir, "LF": lf_dir}, Path("out/batch"), recursive=True)
```

- **Contract.** `PredictionContract` takes non-empty tuples of unique `input_names` and
  `output_names`, a positive `(width, height)` `image_size`, `output_semantics`
  (only `"same_grid_rgb"`), and `value_range` (only `(-1, 1)`). Output names must match
  `[A-Za-z][A-Za-z0-9_-]*`. The predictor accepts an ordered
  `{name: (N, 3, H, W) float tensor}` mapping and returns the same shape for each
  declared output as `{output_name: tensor}`, in exactly the declared order, on the
  input grid. Values must be finite and within `[-1, 1]` (tolerance `1e-3`).
  One output is a one-item mapping;
  no predictor `input_names` attribute is required. Transport converts image values
  to this range and back. Predictions are checked before accumulation or writing;
  invalid outputs are never repaired by resizing, selection, or clamping. Scalar and
  segmentation outputs are unsupported.
- **Inputs.** Paths must all be files or all be directories. Supported extensions are
  `.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff`. Named inputs must already be
  spatially registered and have identical pixel dimensions. Directory batches match
  exact relative paths including extensions; `recursive=True` includes subdirectories.
- **Ownership.** The caller owns the predictor and the device. Transport only calls it
  under `torch.no_grad` (with CUDA autocast on CUDA devices) after moving the inputs
  to `runtime.device`. It never moves, rebuilds, switches the train/eval mode of, or
  closes the predictor, and the runtime stays usable afterwards. The checkpoint
  adapter puts its model in eval mode when it builds it.
- **Provenance is optional.** `InferenceRuntime.checkpoint_path` and
  `predictor_identity` default to `None`, and results report them as they are. The
  checkpoint adapter fills in the real checkpoint path and the method name.
- **Output paths.** A single output file is accepted only for a one-output contract;
  several outputs require a directory. Without `default_single_output_dir` /
  `default_directory_output_dir`, each call requires an explicit output path before prediction.
  `run_image_path_inference` also accepts a zero-argument runtime factory; directory
  pairing is checked before it is called. Naming, collision checks, replacement
  guarantees, and WSI metadata are specified in [Generated images](run_format.md#generated-images).
  An output may not overwrite an input.
- **Modes.** `auto` uses one pass at the contract size and tiles otherwise. `resize`
  writes at `image_size`, with no claim about source physical resolution. `tile`
  preserves input dimensions, using overlap in pixels (`0 <= tile_overlap <` both
  tile dimensions), shared input coordinates, equal-weight overlap averaging, and
  white padding outside the image; padding does not contribute to the output.
- **WSI.** When every input opens with OpenSlide and tiling is needed, inference reads
  regions and writes pyramidal BigTIFFs through libvips; output must be `.tif` or `.tiff`.
  Each output filesystem needs at least `M × width × height × 15` bytes of free scratch
  space for M outputs, plus the compressed TIFFs. Scratch is cleaned up after the call;
  WSI jobs are not resumable.

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
  outputs: [target]
```

- **Registration is explicit Python.** `Definitions` is an immutable value; `extend()`
  returns a new set and rejects any duplicate method or component name instead of
  replacing it. `builtin_definitions()` is the default set (Pix2Pix, CycleGAN and the
  `concat_unet`, `resnet`, `patchgan` components), and the built-ins are registered
  through exactly this mechanism.
- **The stock CLI contains the built-ins only.** There is no plugin discovery: no entry
  points, directory scanning, import strings, or `class_path` in YAML.
- **Checkpoints never import code.** The caller must supply every referenced method
  and component definition. Persisted identities and compatibility rules are in the
  [checkpoint contract](run_format.md#checkpointsepnnnpth).
- **Options.** A method validates the config keys it declares in `owned_keys`
  (`method.options` by default). Unknown keys fail. Resolved configs preserve method
  options; the live `MethodConfig.definition` reference is not serialized.
- **Prediction and batches.** `build_inference_model` returns a module satisfying the
  [named RGB prediction contract](#direct-predictor-inference), with names from
  `prediction_inputs(config, direction)` and `prediction_outputs(config, direction)`
  (defaulting to `model.inputs` / `model.outputs`). A method may restrict output count
  in `validate`. Paired batches use the [named sample contract](#named-runtime-samples).
  Evaluation extensions are described in [Evaluation metrics](#evaluation-metrics).
- **Ownership.** A `MethodDefinition` owns its options and their validation, its
  pairing, prediction directions, the validation metrics it ranks and their direction
  (`checkpoint_metrics`, `monitor_mode`), training-runtime construction, inference-only
  model construction, and its checkpoint identity. A `ComponentDefinition` owns one
  architecture's option parser and factory; the method decides which components it
  accepts and how they compose. The shared `Trainer` owns the epoch, validation,
  checkpoint, `best.json` and history lifecycle; the shared inference layer owns
  single/directory/tiled/WSI transport.

Public extension modules (relative to `virtual_staining`):

| Module | Public API |
|---|---|
| `definitions` | `MethodDefinition`, `ComponentDefinition`, `Component`, `ComponentContext`, `ResolutionContext`, `Definitions`, `DefinitionNotAvailableError` |
| `methods.builtin` | `builtin_definitions` |
| `metrics` | `MetricDefinition`, `MetricResult`, `ResolvedMetric`, `resolve_metrics`, `BUILTIN_METRICS` |
| `evaluation.evaluator` | `evaluate_samples`, `evaluate_pair`, `EvaluationSample`, `EvaluationInputError`, `EvaluationCoverageError` |
| `training.runtime` | `TrainingMethodRuntime`, `MethodMetrics` |
| `checkpoint_contract` | `CheckpointIdentity`, `ValidatedCheckpoint`, `CheckpointCompatibilityError` |
| `config` / `config.run` | `reject_unknown_keys`, `parse_bool_strict` / `RunConfig` |
| `training.trainer` | `Trainer` |
| `inference.runner` | `load_inference_generator` |
| `inference.single` | `run_image_path_inference`, `InferenceRuntime`, `PredictionContract` |

The application entry points
`applications.train.train`, `applications.infer.infer` and
`applications.infer_images.infer_images(..., definitions=...)` /
`applications.pipeline.run_stages(..., definitions=...)` accept an externally resolved
configuration or definition set.

The `TrainingMethodRuntime` protocol supplies `step()` / `validate()`, scheduling,
learning rates, checkpoint identity, and `state_dict()` / `load_state_dict()`.
It reports `MethodMetrics`: `losses` holds its objective
scalars (`metric_names`, written as `<name>_train`/`<name>_val`), `image` holds further
validation scalars (`validation_metric_names`), and the per-term component maps are
optional. `objective_metadata()` may return JSON-compatible objective provenance for
`best.json`. [`tests/external_method/`](../tests/external_method/) contains a complete
non-GAN extension example.

## Inspecting and checking configs

`virtual_staining.applications.config_authoring` exposes config inspection and
preflight for Python and `vs config resolve` / `vs config check`:

```python
from pathlib import Path

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
write_config_yaml(inspection.authored_yaml, Path("run.yaml"))  # never overwrites
```

`definitions` defaults to the built-in set; supplied external options are preserved in
both authored and resolved forms. `authored` retains the caller's mapping and key order;
`resolved_yaml` and `resolved_sha256` match tracked config snapshots. `origins` marks
leaves `supplied` (even if normalized) or `defaulted`; it is explanatory only.

`preflight(config, stages, depth=...)` respects the given stage order. `config` depth
inspects no asset paths. `assets` adds read-only path, schema, membership, group, and
checkpoint-selection checks. Results are `valid`, `invalid`, `planned` (an earlier
selected stage produces the input, not yet verified), `unverified`, or `not_applicable`.
`report.valid` means no check is `invalid`; it does not mean all inputs were verified.
Preflight does not hash or decode files, deserialize checkpoints, build models, probe
devices, run stages, or write artifacts. `content_verified` is always `false`.
Execution validates and snapshots inputs again; preflight does not freeze them or
establish scientific validity. `write_config_yaml` never replaces an existing file.

## Authoring the slide-set inventory

`virtual_staining.applications.inventory_authoring` builds the canonical wide
`inputs/slide_sets.csv` from explicit asset mappings (`vs inventory preview|write` call
it); no `RunConfig` is needed:

```python
from pathlib import Path

from virtual_staining.applications.inventory_authoring import (
    InventoryRequest, preview_inventory, render_inventory_csv, write_inventory,
)

request = InventoryRequest(
    dataset_root=Path("DATASET"),
    inputs=(("LF", "raw/LF"), ("AF", "raw/AF/**/*.svs")),   # ordered (name, spec)
    targets=(("HE", "raw/HE"), ("PAS", "raw/PAS")),         # ordered (name, spec)
    reference="LF",
    input_masks=(("AF", "masks/AF"),),   # optional
    target_masks=(("HE", "masks/HE"),),  # optional, per target
    metadata=None,                       # optional CSV joined on its `key` column
    key_rule="relative-path",            # or "relative-stem"
)
preview = preview_inventory(request)     # read-only; opens no image
preview.valid, preview.matched_count
preview.issues                           # all discovered authoring problems
render_inventory_csv(preview)            # the exact bytes write_inventory publishes
write_inventory(preview)                 # -> DATASET/inputs/slide_sets.csv
```

`preview.valid` means no authoring issues were found. Only valid, unchanged previews
can be written. Matching, metadata, alignment, masks, output paths, and no-overwrite
guarantees belong to the
[inventory authoring contract](dataset_format.md#authoring-the-inventory).

The CSV loader is
`virtual_staining.data.slide_sets.load_slide_set_inventory(path, dataset_root, *,
modalities, reference_modality, target_modalities)`; `resolve_slide_sets(config)`
resolves the inventory configured for preparation.

## Exporting model bundles

`virtual_staining.applications.export_model` packages selected checkpoints of one
tracked run as a portable local bundle (format: [`run_format.md`](run_format.md#model-bundles)):

```python
from pathlib import Path

from virtual_staining.applications.export_model import (
    ExportCheckpointSelection,
    export_model_bundle,
    verify_model_bundle,
)
from virtual_staining.experiment.run_layout import RunLayout
from virtual_staining.inference.runner import load_inference_generator

bundle = export_model_bundle(
    Path("local_workspace/results/my_run"),
    Path("bundles/my_run"),
    [ExportCheckpointSelection("latest")],
    definitions,  # optional; defaults to builtin_definitions()
)

# Later, anywhere the bundle was moved to:
bundle = verify_model_bundle(Path("moved_bundle"), definitions)
checkpoint = bundle.root / bundle.index["checkpoints"][0]["path"]
model, _ = load_inference_generator(
    bundle.config, RunLayout(bundle.root), device, checkpoint
)
```

`ExportCheckpointSelection` also accepts `"best"` with `metric`, `"top_k"` with
`metric` and `rank`, or `"explicit"` with `checkpoint_path`.

External methods and components require their `Definitions` for export, verification,
and reconstruction; the CLI supports built-ins only. Export returns a verified
`ModelBundle` with `root`, `index`, and resolved `config`. `verify_model_bundle` returns
the same type without building a model and should be called before using a bundle from
elsewhere. Source requirements, publication guarantees, portability, and redistribution
limits are specified in [Model Bundles](run_format.md#model-bundles).

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

- **Requests.** `resolve_metrics(request, definitions.metrics)` accepts an ordered list
  of unique `{name, options}` mappings. Options must be valid for the definition and
  JSON-compatible. Omission uses the [default run request](../config/runs/example.yaml).
  Only requested metrics run and receive report entries.
- **Results.** Evaluators return one `MetricResult` per requested name, following the
  [metric status contract](run_format.md#metric-result-statuses). `MetricResult.of(x)`
  classifies numeric results; NaN, negative infinity, missing results, and invalid
  states fail evaluation. Built-in numerical definitions are documented with
  [per-image metrics](run_format.md#evaluationper_image_metricscsv).
- **Coverage.** `input_failures="strict"` (default) records known input problems in
  `coverage.csv` and fails; `"permissive"` excludes those samples. Empty sample or metric
  requests and zero evaluated samples cannot produce a result. Metric implementation
  and programming errors propagate in both modes.
- **Valid-region support.** `EvaluationSample.support_path` (for every sample or none)
  restricts the support-capable metrics (built-in `mae`, `mse`, `rmse`, `psnr`) to the
  valid pixels, all RGB channels, and adds `<m>_support_count` /
  `<m>_support_fraction` columns. The mask must be explicitly binary (mode `1`, or
  mode `L` holding only 0 and 255) on exactly the image grid; soft masks are rejected,
  never thresholded. An empty region yields `undefined`. Requests with support and a
  metric without support (SSIM, PCC) fail before reading any file. Support is
  evaluation input only: it is not a tissue mask, registration confidence or
  preparation mask, and none is discovered or generated; the configured `evaluate`
  stage does not use one.
- **Result metadata.** Metric identity, ranking direction, and presentation semantics
  persist in [evaluation_result.json](run_format.md#evaluationevaluation_resultjson).
- **Training metrics are separate.** A `MethodDefinition` owns its validation and
  checkpoint metrics; these do not require a `MetricDefinition`.
- **Unpaired diagnostics** (`evaluation/unpaired.py`,
  `evaluate_unpaired_collections(generated_paths, reference_paths, output_dir, ...)`)
  compare per-image RGB/luminance feature distributions of two independent collections
  and need no model, checkpoint, method or target pair. They are appearance
  diagnostics, not sample-level fidelity metrics, and do not use metric definitions.

## Named runtime samples

`virtual_staining.data.dataset.PairedManifestDataset` returns named mappings in the
selected order. With the training transforms, samples have this tensor structure:

```python
{
    "inputs": {"LF": lf_tensor, "AF": af_tensor},
    "targets": {"PAS": pas_tensor, "HE": he_tensor},
    "masks": {"foreground_mask": {"PAS": pas_mask, "HE": he_mask}},
}
```

Without a transform, images and masks are PIL images. `include_foreground_mask=True`
requires a mask for every selected target; otherwise `masks` is `{}`. The training
application sets this flag when a configured loss needs masks. With training transforms,
each target has its own `1HW` mask, never another target's mask, and collation gives
`NCHW` images and `N1HW` masks under the same names. Model configuration
selects `model.inputs` from the manifest input modalities and `model.outputs` from its
target modalities, each in any order; model order is authoritative.

## Registration and resampling

`virtual_staining.data.alignment` exposes `ImageGeometry`, `GridGeometry`,
`AlignmentTransform`, `AlignmentImage`, `SpatialEvidence`, `RegistrationRequest`,
`AlignmentResult`, `RegistrationAttempt`, `RegistrationRuntime`, `RegistrationResources`,
`RegistrationFailure`, `FailureCategory`, `QCPolicy`, and `QCDecision`.
`GridGeometry.resized_crop()` binds
an explicitly specified crop/resize to native pixel centres; `AlignmentTransform.from_estimated()`
composes both grid maps with the estimated forward transform. `map_points()`,
`inverse()` and `then()` use explicit moving/reference frames. The
[persisted contract](dataset_format.md#persisted-alignment-geometry-and-results)
defines coordinates and serialization.

`resolve_alignment(reference, moving, request)` returns a direct identity or SIFT
candidate and backend outcome, without QC acceptance. Same-coordinate-frame requests
permit identity only and require equal native shape and compatible known per-axis
MPP. `ImageGeometry.validate_shared_frame(moving)` applies the established
reference-first `np.isclose(..., rtol=0.01)` comparison; unknown spacing is permitted
without inference. Contradictory shared-frame declarations raise `AlignmentError`
in identity construction, supplied results, and direct QC. Identity candidates under
other relationships are not globally restricted to equal shapes.
Same-section modality/restaining requests permit identity,
similarity and affine. Serial sections permit spatial association, not dense
correspondence. Unknown relationships require an explicit bounded diagnostic region;
non-corresponding assets cannot register. Explicit restrictions may narrow these
permissions. Existing alignment declarations never establish biological relationships.

Every result has a typed attempt; backend failures carry
`result.attempt.failure.category`, optional `subcode`, and `message` instead of requiring
exception-text parsing. SIFT distinguishes insufficient content, extraction, matching,
optimizer and geometry failures; unexpected handled exceptions use `internal_error`.
Runtime/attempt records accept caller-supplied identity and evidence without creating
runs, collecting resources, or scheduling retries. The current backends record known
execution details and timing; unavailable measurements, hashes and determinism remain
null. See the persisted contract for the result version and record fields.

`evaluate_alignment_qc(candidate, request, policy, ...)` evaluates supplied independent
landmarks, tissue-support and observation-validity evidence against caller-supplied
thresholds. Empty policies and required missing evidence yield `insufficient_evidence`.
Landmark improvement compares held-out landmark RMS against identity. Acceptance is
scoped to the declared purpose and does not upgrade a correspondence declaration;
backend scores are not QC evidence. No scientific acceptance thresholds are supplied.

`warp_aligned_patch(image_or_read_region, transform, ..., max_source_pixels=...)`
returns a `WarpedPatch` with the image, geometric validity and separately nullable
support/observation-validity values and known masks. It inverse-maps reference pixel
centres, subdivides source reads to the supplied pixel budget, and borrows already-open
readers without closing them. Linear image interpolation accounts conservatively for
all validity contributors; nearest interpolation preserves labels. Out-of-bounds
samples are geometrically invalid. Boolean `SpatialEvidence` maps carry their own
asset and grid: true means specimen content for tissue support, or usable observation
for validity. Outside their grid evidence is unknown. Foreground/loss masks remain
separate and use `warp_aligned_mask_patch()` with their explicit grid.

### Injecting registration into preparation

Standalone preparation accepts
`DatasetBuilder(config, slide_sets, *, registration_backend=backend).run_all()`.
`SlideSetProcessor(config, slide_set, assigned_split=None, *, registration_backend=backend)`
and `applications.prepare.prepare(config, config_path, *, registration_backend=backend)`
accept the same dependency; the latter retains dataset reuse.

Construct the public `virtual_staining.data.alignment.RegistrationBackend` with
`RegistrationBackend(register, identifier, version, options=None, qc_disposition=None)`.
`register(reference: AlignmentImage, moving: AlignmentImage, request: RegistrationRequest)`
returns an `AlignmentResult` with the supplied request and canonical geometry matching
those actual images. Each moving input and target is called directly against the explicit
reference; the reference itself always receives built-in identity. Inventory declarations
and `alignment.mode` still determine the requested transform permission. A callback cannot
escalate an identity request or substitute another asset's geometry. Its result, including
reason, runtime, typed failure and nullable QC, is retained in set metadata.

```python
from virtual_staining.data.alignment import RegistrationBackend
from virtual_staining.data.builder import DatasetBuilder

backend = RegistrationBackend(
    register=my_registration_callable,
    identifier="laboratory_registration",
    version="1",
    options={"calibration_revision": "2026-09", "independent_qc_revision": "3"},
    qc_disposition={"rejected": "skip_set", "insufficient_evidence": "error"},
)
result = DatasetBuilder(config, slide_sets, registration_backend=backend).run_all()
```

The caller owns deterministic execution and must include every relevant backend option,
QC policy and external evidence revision/digest in its JSON identity. Options are copied
into a frozen JSON snapshot; callable representations are never used. This record and
QC disposition participate in the existing dataset fingerprint. A supplied
`fingerprint_metadata` must have the same registration record. Per-asset execution results
remain separate from that configuration identity.

The callback may attach independently evaluated QC using `evaluate_alignment_qc()` and
an explicit `QCPolicy`, passing support/validity through `SpatialEvidence`. Preparation
never derives that evidence from foreground masks. `qc_disposition` maps `accepted`,
`rejected`, `insufficient_evidence`, or `unassessed` (`qc is None`) to `continue`, `skip_set`,
or `error`. Unspecified outcomes continue without changing their QC status. An explicit
QC action takes precedence over `alignment.on_failure`; a hard QC error raises
`data.slide_set_processor.AlignmentQCError` with its `result`. Returned typed failures,
including `interrupted`, follow `alignment.on_failure` without losing their category.
A raised `KeyboardInterrupt` propagates, and owned partial sample writes are removed.
No manifest is published when an exception interrupts the build.

With no injection, preparation retains its identity/SIFT selection and absent QC.
No scientific thresholds, biological relationships, or acceptance claims are added.
