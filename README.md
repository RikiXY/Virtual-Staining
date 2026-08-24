# Virtual Staining

Research portfolio project for virtual staining of histopathology images using a Pix2Pix conditional GAN.

The pipeline trains a paired image-to-image translation model on aligned histology patch pairs, enabling
generation of virtually stained images from label-free microscopy inputs (and vice versa).

## CLI Commands

| Command | Purpose |
|---|---|
| `vs prepare` | Build the patch dataset from full-size slide sets |
| `vs run` | Run the complete pipeline or selected stages |
| `vs train` | Train the Pix2Pix model |
| `vs infer` | Run inference on the test split |
| `vs infer-images` | Run inference on one image file or a directory of images |
| `vs evaluate` | Evaluate a configured run or one image pair |
| `vs compare` | Compare metric distributions across runs |
| `vs convert` | Convert TIFF images to OpenSlide-compatible pyramidal BigTIFFs |
| `vs panels` | Build source / generated / target comparison panels |
| `vs organize` | Organise run outputs |
| `vs queue` | Execute full or staged runs sequentially from a queue file |
| `vs status` | Check dependencies, system memory, OpenSlide, and GPU support |

## Quick Start

```bash
# 1. Enter the Nix environment
nix develop

# 2. Install dependencies
uv sync --frozen

# 3. Copy and edit the example run config
cp config/runs/example.yaml config/runs/local/my_run.yaml

# 4. Run the full pipeline
vs run --config config/runs/local/my_run.yaml
```

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

Convert one or more large TIFFs—or a whole directory recursively—without loading them fully
into memory. Directory inputs keep their relative layout under the output directory:

```bash
vs convert raw/source.tif raw/target.tif --output-dir converted
vs convert raw/slides --output-dir converted
```

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

Single-input models retain the shorthand `--input PATH`.

`vs infer-images` accepts `.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff`.
It defaults to `--mode auto`: patch-sized inputs use the standard single-patch
path, while larger images are processed tile-by-tile and saved at the original
size. Use `--mode resize` to force the resizing of the whole input to
`image_size`. Use `--output-format png` to force a common output format
for directory batches.

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

## Configuration

All experiment parameters live in a single YAML file. Copy
[`config/runs/example.yaml`](config/runs/example.yaml) and edit it:

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

Experiment commands accept YAML configuration directly through `--config`.

See [`docs/architecture.md`](docs/architecture.md) for the full config schema,
[`docs/run_format.md`](docs/run_format.md) for run output layout, and
[`docs/reproducibility.md`](docs/reproducibility.md) for canonical config snapshots and hashes.

## Qualitative Results

Each panel compares source patch, generated target, and real target.

From label-free to H&E staining:
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to label-free:
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Package Structure

- `metrics.py` - image metric computations and metric metadata
- `utils/` - shared primitives: dimensions and image I/O
- `config/` - YAML loading, validation, typed config sections
- `experiment/` - run paths, metadata, stage lifecycle, and environment snapshots
- `models/` - UNetGenerator, PatchGANDiscriminator, model config
- `data/` - dataset, manifest, builder, preprocessing pipeline
- `training/` - training mechanics: Trainer, steps, losses, validation, checkpoints
- `inference/` - reusable model loading, prediction, single-image workflows, output naming
- `evaluation/` - evaluator, plots, summaries, panels, ranking
- `applications/` - stage lifecycle owners (`prepare`, `train`, `infer`, `evaluate`) and other use cases
- `cli/` - thin argparse entrypoints delegating to `applications/`

See [`docs/architecture.md`](docs/architecture.md) for the full description and layer boundaries.

## Repository Structure

```text
Virtual-Staining/
├── config/
│   ├── queues/                 # queue YAML files (example.yaml template)
│   └── runs/                   # run YAML files (example.yaml template)
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
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
│   ├── models/
│   ├── smoke/
│   ├── training/
│   └── utils/
├── virtual_staining/           # installable package
│   ├── metrics.py              # metric computations and metadata
│   ├── applications/           # use-case orchestrators
│   ├── cli/                    # argparse entry points
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
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
uv run --group dev pytest
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

- **Preprocessing** - tissue masking, feature-based affine alignment of target to source,
  patch extraction with foreground and white-area quality filters.
- **Model** - Pix2Pix conditional GAN: U-Net generator with skip connections and
  PatchGAN discriminator, trained with adversarial loss + L1 reconstruction loss.
- **Evaluation** - per-image MAE, RMSE, PSNR, SSIM; summary statistics; optional
  comparison panels with difference maps.

## License

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
