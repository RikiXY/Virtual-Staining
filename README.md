# Virtual Staining

Research portfolio project for virtual staining of histopathology images using a Pix2Pix conditional GAN.

The pipeline trains a paired image-to-image translation model on aligned histology patch pairs, enabling
generation of virtually stained images from label-free microscopy inputs (and vice versa).

## CLI Commands

| Command | Purpose |
|---|---|
| `vs-prepare` | Build the patch dataset from full-size image pairs |
| `vs-train` | Train the Pix2Pix model |
| `vs-infer` | Run inference on the test split |
| `vs-evaluate` | Evaluate generated images with MAE, RMSE, PSNR, SSIM |
| `vs-compare` | Compare metric distributions across runs |
| `vs-compare-panels` | Build source / generated / target comparison panels |
| `vs-evaluate-single` | Evaluate a single image pair |
| `vs-organize` | Organise run outputs |

## Quick Start

```bash
# 1. Enter the Nix environment
nix develop

# 2. Install dependencies
uv sync --frozen

# 3. Copy and edit the example run config
cp config/runs/example.yaml config/runs/my_run.yaml

# 4. Prepare dataset
vs-prepare --config config/runs/my_run.yaml

# 5. Train
vs-train --config config/runs/my_run.yaml

# 6. Run inference
vs-infer --config config/runs/my_run.yaml

# 7. Evaluate
vs-evaluate --config config/runs/my_run.yaml
```

### Makefile shortcuts

```bash
make dataset        CONFIG=config/runs/my_run.yaml
make train          CONFIG=config/runs/my_run.yaml
make infer          CONFIG=config/runs/my_run.yaml
make evaluate       CONFIG=config/runs/my_run.yaml
make compare        ARGS="--run-path local_workspace/results/my_run"
make compare-panels ARGS="from-metrics --run-path local_workspace/results/my_run"
```

Or run the full sequence in one command:

```bash
make complete-run CONFIG=config/runs/my_run.yaml
```

## Configuration

All experiment parameters live in a single YAML file. Copy
[`config/runs/example.yaml`](config/runs/example.yaml) and edit it:

```yaml
dataset_root: local_workspace/datasets/your_sample
results_path: local_workspace/results
run_name: your_run_name

# [width, height] - e.g. [320, 256] means 320 px wide, 256 px tall
image_size: [256, 256]

preprocessing:
  source_name: label_free.tif
  target_name: stained.tif
  train_ratio: 0.80
  val_ratio: 0.05
  test_ratio: 0.15
  min_foreground_ratio: 0.25
  seed: 42

training:
  batch_size: 8
  epochs: 100
  lr_g: 0.0002
  l1_weight: 25.0
  seed: 42

inference:
  checkpoint_policy: latest   # or: checkpoint: checkpoints/ep099.pth

evaluation:
  save_graphs: true
```

The `CONFIG` variable is the only Make argument accepted for experiment targets.
Put dataset paths, run names, image sizes, epochs, seeds, and checkpoint selection
in the YAML - not in Make variables.

See [`docs/architecture.md`](docs/architecture.md) for the full config schema and
[`docs/run_format.md`](docs/run_format.md) for run output layout.

## Qualitative Results

Each panel compares source patch, generated target, and real target.

From label-free to H&E staining:
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to label-free:
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Package Structure

- `utils/` - shared primitives: dimensions, image I/O, metrics helpers
- `config/` - YAML loading, validation, typed config sections
- `experiment/` - run concept: RunPaths, RunContext, RunMetadata, environment snapshots
- `reporting/` - Reporter protocol with Null, Logging, and Console implementations
- `models/` - UNetGenerator, PatchGANDiscriminator, factory, model config
- `data/` - dataset, manifest, builder, preprocessing pipeline
- `training/` - Trainer, runner, steps, losses, checkpoint management
- `inference/` - Predictor, runner, output writers
- `evaluation/` - metrics, evaluator, summaries, panels, ranking
- `applications/` - use-case orchestrators (no argparse)
- `cli/` - thin argparse entrypoints delegating to `applications/`

See [`docs/architecture.md`](docs/architecture.md) for the full description and layer boundaries.

## Repository Structure

```text
Virtual-Staining/
├── config/
│   └── runs/                   # run YAML files (example.yaml template)
├── docs/
│   ├── assets/                 # qualitative result images
│   ├── notebooks/
│   └── reports/
├── examples/                   # example input images
├── local_workspace/
│   ├── datasets/               # input paired samples (gitignored)
│   └── results/                # run outputs (gitignored)
├── tests/                      # pytest test suite
├── virtual_staining/           # installable package
│   ├── applications/           # use-case orchestrators
│   ├── cli/                    # argparse entry points
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiment/
│   ├── inference/
│   ├── models/
│   ├── reporting/
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

Other useful commands:

```bash
make format       # apply ruff formatting
make lint         # ruff lint check
make check-types  # pyright only
make test         # pytest only
make sync         # reinstall from uv.lock
uv lock           # re-resolve dependencies
```

## Data Split Caveat

The default split is **patch-level**: train, validation, and test patches are all drawn
from the same slide. Metrics on `dataset_test/` measure same-slide internal validation,
not independent generalization. For generalizability evidence, use a slide-level,
patient-level, or spatial-block split strategy - the current pipeline does not implement
these.

## Method

- **Preprocessing** - tissue masking, feature-based affine alignment of target to source,
  patch extraction with foreground and white-area quality filters.
- **Model** - Pix2Pix conditional GAN: U-Net generator with skip connections and
  PatchGAN discriminator, trained with adversarial loss + L1 reconstruction loss.
- **Evaluation** - per-image MAE, RMSE, PSNR, SSIM; summary statistics; optional
  comparison panels with difference maps.

## License

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
