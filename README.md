# Virtual Staining

Research portfolio project for virtual staining of histopathology images using a Pix2Pix conditional GAN.

The pipeline trains a paired image-to-image translation model on aligned histology patch pairs, enabling
generation of virtually stained images from label-free microscopy inputs (and vice versa).

## CLI Commands

| Command | Purpose |
|---|---|
| `vs-prepare` | Build the patch dataset from full-size image pairs |
| `vs-complete-run` | Run prepare, train, infer, and evaluate in sequence |
| `vs-run-queue` | Execute multiple full runs sequentially from a queue file |
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
cp config/runs/example.yaml config/runs/local/my_run.yaml

# 4. Run the full pipeline
vs-complete-run --config config/runs/local/my_run.yaml
```

### Makefile shortcuts

```bash
make dataset        CONFIG=config/runs/local/my_run.yaml
make train          CONFIG=config/runs/local/my_run.yaml
make infer          CONFIG=config/runs/local/my_run.yaml
make evaluate       CONFIG=config/runs/local/my_run.yaml
make complete-run   CONFIG=config/runs/local/my_run.yaml
make run-queue      QUEUE=config/queues/example.yaml
make compare        CONFIG=config/runs/local/my_run.yaml
make compare-panels CONFIG=config/runs/local/my_run.yaml
```

Or call the CLI directly:

```bash
vs-complete-run --config config/runs/local/my_run.yaml
```

Queue multiple full runs locally:

```yaml
# config/queues/nightly.yaml
name: nightly
continue_on_failure: true
jobs:
  - config_path: ../runs/local/run_a.yaml
    label: baseline
  - config_path: ../runs/local/run_b.yaml
    notes: retry with lower lr
```

```bash
vs-run-queue --queue config/queues/nightly.yaml
```

Queue definitions live under `config/queues/`. Personal queue YAMLs can live
under `config/queues/local/`. Queue runtime state is written under
`local_workspace/queues/`, separate from the committed queue definitions.
State files are flat in that directory, for example
`local_workspace/queues/nightly.state.json`.

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
  checkpoint_policy: latest   # or: checkpoint_path: checkpoints/ep099.pth

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
make check-types  # pyright only
make test         # pytest only
make sync         # reinstall from uv.lock
uv lock           # re-resolve dependencies
```

## Data Split Caveat

The default split is **patch-level**: train, validation, and test patches are all drawn
from the same slide. Metrics on `splits/test/` measure same-slide internal validation,
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
