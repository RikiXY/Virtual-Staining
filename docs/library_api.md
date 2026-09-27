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
| Train | `Trainer(config, run_paths, method, train_loader, val_loader, device, image_size=...)` then `.train(seed)` (`training/trainer.py`) | `TrainingConfig` + `TrainingMethodRuntime` + train/val `DataLoader`s + output `RunLayout` + image size; optional `progress_reporter`, `preview_sink`, `benchmark_recorder`, `config_hash`, `experiment_session` | `metrics/epochs.csv`, `checkpoints/ep*.pth`, `checkpoints/best.json`, training/validation output dirs under the `RunLayout` root; `.resume()` reads checkpoints there | `applications.train` / `vs train`: builds loaders from the manifest or domain collections, binds the consumed-data snapshot and passes its real `ExperimentSession` | The loaders are the data boundary; the Trainer never reads manifests, dataset roots or preparation outputs. Constructing a method runtime is method-specific (current methods build from a `RunConfig`) |
| Infer | `run_image_path_inference(runtime_factory, named_paths, output_path)` (`inference/single.py`) | A `RuntimeFactory` returning an `InferenceRuntime` + named input files or directories + explicit output path | Generated images at the output path (or at the runtime's default output dir when none is given) | `applications.infer` / `vs infer`: test-split manifest inference with consumed/produced snapshots; `applications.infer_images` builds the runtime from a run's checkpoint | Accepts an injected runtime factory only; a general predictor/provider reconstruction boundary is separate future work |
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
