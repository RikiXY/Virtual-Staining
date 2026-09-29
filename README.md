# Virtual Staining

Experimental, reproducible image-translation framework focused on virtual staining of
histopathology images: generating stained-looking images from label-free microscopy inputs
(and vice versa).

Two translation methods are built in and share one training, checkpoint, inference, and run
infrastructure:

- **Pix2Pix** (reference method) - paired training on aligned patches; named
  N-input -> one-target translation; ConcatUNet generator and conditional PatchGAN
  discriminator; configurable BCE / L1 / SSIM losses.
- **CycleGAN** - one source domain A <-> one target domain B; unpaired training from two
  independent image collections; two ResNet generators and two unconditional PatchGAN
  discriminators (LSGAN, cycle L1, optional identity L1, replay pools); `A_to_B` and
  `B_to_A` inference from the same checkpoint.

The method is selected with `method.name` in the run config. The stock CLI ships only
these two built-in methods. Python callers can register further methods and network
components explicitly (see [`docs/library_api.md`](docs/library_api.md#extending-with-explicit-definitions));
there is no plugin discovery, and translation is always to exactly one target.

## CLI Commands

| Command | Purpose |
|---|---|
| `vs prepare` | Build the patch dataset from full-size slide sets |
| `vs run` | Run the complete pipeline or selected stages |
| `vs train` | Train the configured built-in method |
| `vs infer` | Run inference on the test split (CycleGAN: in the configured direction) |
| `vs infer-images` | Run inference on one image file or a directory of images |
| `vs evaluate` | Paired image metrics or unpaired collection diagnostics for a run, or metrics for one image pair |
| `vs compare` | Compare metric distributions across runs |
| `vs convert` | Convert TIFF images to OpenSlide-compatible pyramidal BigTIFFs |
| `vs panels` | Build source / generated / target comparison panels |
| `vs organize` | Organise run outputs |
| `vs export-model` | Export selected run checkpoints as a portable local model bundle |
| `vs queue` | Execute full or staged runs sequentially from a queue file |
| `vs config` | Print the resolved config, or check it (optionally read-only asset preflight) without running a stage |
| `vs inventory` | Preview or write the raw paired slide-set inventory from explicit asset mappings |
| `vs status` | Check required Python/native dependencies, system memory, and GPU support |

## Quick Start

```bash
# 1. Enter the Nix environment
nix develop

# 2. Install the mandatory Python dependencies
uv sync --frozen

# 3. Copy and edit a run config (minimal starter; config/runs/example.yaml is the
#    fully annotated reference of every option)
cp config/runs/minimal_pix2pix.yaml config/runs/local/my_run.yaml

# 4. Check it without running anything (add --assets for read-only input checks)
vs config check --config config/runs/local/my_run.yaml --stages prepare train infer evaluate

# 5. Run the full pipeline
vs run --config config/runs/local/my_run.yaml
```

The supported runtime is the Nix development shell. `uv sync --frozen` installs
all mandatory Python dependencies, including OpenSlide Python and pyvips; no WSI
extra is needed. The shell supplies native OpenSlide and libvips on Linux and
macOS, including library search paths for Python FFI loading. Run pipeline and
development commands inside this shell (or with `nix develop -c ...`).

`uv run vs status` checks required Python imports and native WSI library usability.
A missing or broken required dependency produces a failing status. NVIDIA drivers,
CUDA devices, and GPU availability are optional; a CPU-only runtime can be healthy.

### Development commands

```bash
make sync
make format
make lint
make typecheck
make test
make qa
make clean
```

Or call the CLI directly:

```bash
vs run --config config/runs/local/my_run.yaml
vs run --config config/runs/local/my_run.yaml --stages train infer evaluate
vs status
```

Convert one or more large TIFFs-or a whole directory recursively-without loading them fully
into memory. Directory inputs keep their relative layout under the output directory:

```bash
vs convert raw/source.tif raw/target.tif --output-dir converted
vs convert raw/slides --output-dir converted
```

Export selected checkpoints of a finished run, with their exact tracked training
configs, as a verified local bundle that can be moved and reconstructed without the
original run or dataset:

```bash
vs export-model \
  --run-path local_workspace/results/my_run \
  --output local_workspace/bundles/my_run \
  --best val_ssim --top-k val_ssim 2 --latest
```

Selectors are `--checkpoint PATH`, `--latest`, `--best METRIC` and
`--top-k METRIC RANK` (repeatable). The CLI knows the built-in methods only; external
methods are exported from Python with explicit definitions. A bundle is a local
research artifact, not a redistribution approval; see
[`docs/run_format.md`](docs/run_format.md#model-bundles).

Evaluate one generated image without adding another top-level command:

```bash
vs evaluate --pair target.png target_generated.png --output-dir evaluation
```

Run inference on one image or a directory. Multi-input models take one named
path per configured modality; paths must already be spatially registered and
have identical pixel dimensions.

For one image per modality:

```bash
vs infer-images \
  --config config/runs/local/my_run.yaml \
  --input AF=examples/sample_af.png \
  --input LF=examples/sample_lf.png \
  --output local_workspace/results/my_run/sample.png
```

For directory batches, matching files must have exactly the same relative
paths, including extensions. Recursive subdirectories are preserved:

```bash
vs infer-images \
  --config config/runs/local/my_run.yaml \
  --input AF=examples/af \
  --input LF=examples/lf \
  --recursive \
  --output local_workspace/results/my_run/example_outputs
```

Single-input models retain the shorthand `--input PATH`. CycleGAN runs consume the
domain selected by `inference.direction` (`model.inputs[0]` for `A_to_B`,
`model.target` for `B_to_A`). Generated names carry the direction
(`tile_A_to_B_generated.png`, `tile_B_to_A_generated.png`), so both directions can
share one output root; Pix2Pix outputs keep the `_target_generated` suffix.

`vs infer-images` accepts `.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff`.
It defaults to `--mode auto`: patch-sized inputs use the standard single-patch
path, while larger images are processed tile-by-tile and saved at the original
size. Use `--mode resize` to force the resizing of the whole input to
`image_size`. Use `--output-format png` to force a common output format
for directory batches. To run an in-memory model without a checkpoint or run
config, see [`docs/library_api.md`](docs/library_api.md#direct-predictor-inference).

Queue multiple full or partial pipeline runs locally:

```yaml
# config/queues/nightly.yaml
name: nightly
continue_on_failure: true
jobs:
  - config_path: ../runs/local/run_a.yaml
    label: baseline
  - config_path: ../runs/local/run_b.yaml
    stages: [train, infer, evaluate]
    notes: retry with lower lr
```

```bash
vs queue --queue config/queues/nightly.yaml
```

Omit `stages` to run the full `prepare`, `train`, `infer`, `evaluate`
sequence. When `stages` is present, allowed values are `prepare`, `train`,
`infer`, and `evaluate`, executed in the order listed.

Queue definitions live under `config/queues/`. Personal queue YAMLs can live
under `config/queues/local/`. Queue runtime state is written under
`local_workspace/queues/`, separate from the committed queue definitions.
State files are flat in that directory, for example
`local_workspace/queues/nightly.state.json`.

For controlled ablations, add an optional `ablation` block to the queue. The
queue preflight compares resolved configs and fails before training if a field
differs outside the declared `variable_fields`. Summary metadata is written to
`local_workspace/queues/<queue-name>.ablation.summary.json`.
[`config/queues/example.yaml`](config/queues/example.yaml) and
[`config/queues/example_ablation.yaml`](config/queues/example_ablation.yaml) document
every queue and ablation key.

## Configuration

All experiment parameters live in a single YAML file. Start from a short starter,
[`config/runs/minimal_pix2pix.yaml`](config/runs/minimal_pix2pix.yaml) or
[`config/runs/minimal_cyclegan.yaml`](config/runs/minimal_cyclegan.yaml). The annotated
references [`config/runs/example.yaml`](config/runs/example.yaml) (Pix2Pix) and
[`config/runs/example_cyclegan.yaml`](config/runs/example_cyclegan.yaml) (CycleGAN) are
the same experiments with every supported option, default, and path base written out. A condensed Pix2Pix config:

```yaml
dataset_root: local_workspace/datasets/your_sample
results_path: local_workspace/results
run_name: your_run_name
image_size: [256, 256]

model:
  inputs: [autofluorescence, label_free]
  target: H&E
  generator: {architecture: concat_unet, base_channels: 64, norm: batch, dropout: false, bilinear: false}
  discriminator: {ndf: 64, norm: instance, use_sigmoid: false}

preprocessing:
  inputs:
    inventory: inputs/slide_sets.csv
    modalities: [autofluorescence, label_free]
    reference: label_free
    target_modality: H&E
  masks: {generation: if_missing, strategy: connected_components, scale: 0.25}
  alignment: {mode: auto, method: affine_sift}
  filtering: {foreground: {enabled: true, policy: reference, min_ratio: 0.25}}
  split: {unit: patient, train: 0.80, val: 0.10, test: 0.10, seed: 42}
  io: {tiled: true, backend: auto}
training:
  batch_size: 8
  epochs: 100
  lr_g: 0.0002
  seed: 42
  augmentation:
    enabled: false
    expansion_factor: 1
    intensity: light
  losses:
    generator:
      - name: l1
        weight: 25.0
    discriminator:
      - name: adversarial_bce
        weight: 1.0

inference:
  checkpoint_policy: latest   # or: checkpoint_path: checkpoints/ep099.pth

evaluation:
  save_graphs: true
```

A CycleGAN run instead sets `method.name: cyclegan`, `data.pairing: unpaired` with one
image collection per domain under `data.domains`, a `resnet` generator, and the
`adversarial_lsgan` / `cycle_l1` / `identity_l1` losses; see the CycleGAN example.

Experiment commands accept YAML configuration directly through `--config`.

### Inspecting and checking a config

```bash
vs config resolve --config my_run.yaml                      # resolved YAML on stdout
vs config resolve --config my_run.yaml --output resolved.yaml   # never overwrites
vs config check --config my_run.yaml                        # config only; no assets needed
vs config check --config my_run.yaml --stages prepare train infer evaluate --assets
```

The *authored* config is what you wrote; the *resolved* config is every effective value
after the owners filled their defaults, exactly the `config/<stage>/resolved.yaml` a
tracked stage records (same bytes, same SHA-256, printed as `config_sha256`). Each
resolved field is either *supplied* by you or *defaulted* by its owner
(`inspect_run_yaml`/`inspect_run_mapping` in
[`docs/library_api.md`](docs/library_api.md#inspecting-and-checking-configs)).

`check` without `--assets` only resolves the config, so it works on a machine without
the data. `--assets` adds read-only checks of the selected stages' inputs (inventory,
masks, manifest, domain collections, supplied group metadata, checkpoint selection,
expected generated files) in the given stage order. An input that an earlier selected
stage produces is reported `planned`, which is not verified. Nothing is hashed, decoded,
loaded, or written (`content_verified: false`), no content-level leakage or scientific
validity is claimed, and a passing check is not a frozen input snapshot: running the
stages repeats every required validation and freezes what they actually consume.

### Authoring the paired slide-set inventory

```bash
vs inventory preview --dataset-root DATASET \
  --input LF=raw/LF --input 'AF=raw/AF/**/*.svs' \
  --target-modality HE --target raw/HE --reference LF
vs inventory write ...   # publishes DATASET/inputs/slide_sets.csv; never overwrites
```

Files are matched across the explicit input/target mappings by their path relative to
each mapping (`--key relative-stem` ignores the final extension). `preview` lists every
incomplete, duplicate, or colliding key and writes nothing; `write` publishes only a valid
preview after the canonical loader has read it back. Alignment of non-reference inputs,
patient/specimen IDs, and masks come only from an optional `--metadata` CSV and mask
mappings; nothing is inferred from names or image content. Unpaired (CycleGAN)
`data.domains` do not use this inventory. See
[`docs/dataset_format.md`](docs/dataset_format.md#authoring-the-inventory).

See [`docs/run_format.md`](docs/run_format.md) for the method-specific config fields and
run output layout, [`docs/architecture.md`](docs/architecture.md) for package boundaries, and
[`docs/reproducibility.md`](docs/reproducibility.md) for canonical config snapshots and hashes.
To call a single stage from Python without the YAML pipeline, see
[`docs/library_api.md`](docs/library_api.md).

## Qualitative Results

Pix2Pix results. Each panel compares source patch, generated target, and real target.

From label-free to H&E staining:
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to label-free:
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Package Structure

- `metrics.py` - evaluation metric definitions, result statuses and built-in metrics
- `utils/` - shared primitives: dimensions and image I/O
- `config/` - YAML loading, validation, typed config sections
- `experiment/` - run paths, metadata, stage lifecycle, and environment snapshots
- `models/` - network implementations (ConcatUNet and ResNet generators, PatchGAN
  discriminator) and the model-I/O normalization contract
- `methods/` - the two built-in method runtimes (Pix2Pix, CycleGAN): topology, optimizers,
  losses, method-owned checkpoint state, and inference loaders
- `data/` - paired manifests, unpaired domain collections, dataset builder, preprocessing
- `training/` - method-agnostic `Trainer`, validation loop, history, loss config, and the
  generic v4 `MethodCheckpointManager`
- `inference/` - checkpoint resolution, method dispatch, single/directory/tiled inference,
  output naming
- `evaluation/` - paired image metrics, unpaired collection diagnostics, plots, summaries,
  panels, ranking
- `applications/` - stage lifecycle owners (`prepare`, `train`, `infer`, `evaluate`) and other use cases
- `cli/` - thin argparse entrypoints delegating to `applications/`

See [`docs/architecture.md`](docs/architecture.md) for the full description and layer boundaries.

## Repository Structure

```text
Virtual-Staining/
├── config/
│   ├── queues/                 # queue YAMLs (example.yaml, example_ablation.yaml references)
│   └── runs/                   # run YAMLs (minimal_*.yaml starters, example*.yaml references)
├── docs/
│   ├── assets/                 # qualitative result images
│   ├── notebooks/
│   └── reports/
├── examples/                   # example input images
├── local_workspace/
│   ├── datasets/               # input paired samples (gitignored)
│   ├── queues/                 # queue state files (gitignored except .gitkeep)
│   └── results/                # run outputs (gitignored)
├── tests/                      # pytest suite grouped by subsystem
│   ├── applications/
│   ├── architecture/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
│   ├── methods/
│   ├── models/
│   ├── smoke/
│   ├── training/
│   └── utils/
├── virtual_staining/           # installable package
│   ├── metrics.py              # metric definitions and built-in metrics
│   ├── applications/           # use-case orchestrators
│   ├── cli/                    # argparse entry points
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
│   ├── methods/
│   ├── models/
│   ├── training/
│   └── utils/
├── Makefile
├── flake.nix
├── pyproject.toml
└── uv.lock
```

## Development

```bash
# Inside nix develop shell:
make qa
# Equivalent to:
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run --group dev pytest -m "not slow"
```

The test layout is documented in [`tests/README.md`](tests/README.md).

### Pre-commit hooks

Install the hooks once per clone:

```bash
nix develop -c pre-commit install
```

Run them manually across the repository:

```bash
nix develop -c pre-commit run --all-files
```

Other useful commands:

```bash
make format       # apply ruff formatting
make lint         # ruff lint check
make typecheck    # pyright only
make test         # pytest only
make sync         # reinstall from uv.lock
uv lock           # re-resolve dependencies
```

## Data Split Caveat

The default split is **patch-level**: train, validation, and test patches may be drawn
from the same slide. For independent generalization evidence, configure `split.unit` as
`set`, `specimen`, or `patient`; spatial-block splitting is not implemented.

## Method

- **Shared framework** - one generic `Trainer` drives a method runtime; checkpoints use a
  single method-aware v4 format with opaque method-owned state; inference, run metadata,
  and provenance are shared. Details: [`docs/architecture.md`](docs/architecture.md).
- **Preprocessing** (paired data) - tissue masking, feature-based affine alignment of
  target to reference, patch extraction with foreground and white-area quality filters.
- **Pix2Pix** (reference method) - conditional GAN on aligned pairs: ConcatUNet generator
  over the concatenated named inputs, conditional PatchGAN discriminator, adversarial BCE
  plus L1 (optional SSIM) losses.
- **CycleGAN** (alternative method) - unpaired A <-> B translation: two ResNet generators,
  two unconditional PatchGAN discriminators, LSGAN adversarial, cycle-consistency L1 and
  optional identity L1 losses, fake-image replay pools.
- **Evaluation** - two distinct protocols:
  - *paired* (Pix2Pix default; CycleGAN opt-in) - per-image MAE, MSE, RMSE, PSNR, SSIM,
    and PCC (or any requested metric, including ones supplied in Python) against aligned
    references, with explicit result statuses, input coverage and set/specimen/patient
    summaries. Requires an aligned held-out test manifest.
  - *unpaired* (CycleGAN default; any method opt-in) - compares the generated collection
    with an independent real reference collection through per-image RGB/luminance
    feature distributions. No pairs are formed and no pairwise fidelity metric is
    reported. Methods without a `data.domains` reference collection (e.g. Pix2Pix) set
    `evaluation.reference_collection`; this changes evaluation only, not training.

### Scientific scope

Paired image metrics are meaningful only against spatially aligned references. CycleGAN's
unpaired diagnostics compare low-order appearance statistics of two image collections;
they do not measure sample-level fidelity and do not establish biological correctness or
clinical validity. Nothing in this repository is validated for clinical use.

## License

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
