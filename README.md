# Virtual-Staining

Paired virtual staining pipeline for histopathology images.

This repository implements a paired virtual staining workflow that combines classical image processing and deep learning. Starting from paired full-size histology images, the pipeline builds an aligned patch dataset, trains a Pix2Pix-style conditional GAN, runs test inference, and provides integrated evaluation and comparison tools.

The learning stage currently uses a **U-Net generator** and a **PatchGAN discriminator**, trained with **adversarial loss + L1 reconstruction loss**.

## Overview

The project follows this workflow:

1. load a paired full-size sample (`source.tif` / `target.tif`)
2. generate tissue masks
3. align the target image to the source reference
4. extract aligned paired patches
5. build `dataset_train`, `dataset_val`, and `dataset_test`
6. train the model
7. run test inference on held-out patches
8. evaluate generated results with integrated metrics and comparison panels

In addition to preprocessing and training, the repository also includes:

- a metric evaluation tool for **MAE, RMSE, PSNR, and SSIM**
- a comparison tool to generate **visual panels and diagnostic plots**

## Features

- paired preprocessing pipeline for full-size histology images
- tissue mask generation for source and target images
- feature-based alignment of target images to source references
- patch extraction from aligned tissue regions
- automatic train/validation/test split generation
- Pix2Pix-style conditional GAN training and inference
- Nix-based development environment with `uv`
- simple `Makefile` commands for common workflows

## Requirements

- **Python 3.11**

Main runtime dependencies:

- OpenCV
- NumPy
- Pillow
- PyTorch
- torchvision
- Matplotlib

Dependencies are declared in `pyproject.toml` and the development shell is defined in `flake.nix`.

### Dependency pinning

`pyproject.toml` declares minimum compatibility bounds (e.g. `torch>=2.0`). The exact versions used in development are pinned in `uv.lock`, which records every package and its hash. To reproduce the exact environment:

```bash
make sync
```

`uv lock` re-resolves against the bounds in `pyproject.toml` and updates `uv.lock`. Run it when you intentionally want to upgrade dependencies.

## Development Environment

Recommended setup:

```bash
nix develop
make sync
```

The Nix shell provides the base tools:

- Python 3.11
- `uv`
- `make`

Project, test, lint, and type-check dependencies are managed by `uv` from
`pyproject.toml` and pinned in `uv.lock`.

Useful development commands:

```bash
make format
make test
make qa
make clean
uv lock
```

## Quick Start

The `Makefile` is the recommended entry point for the standard workflow.

### 1. Prepare the input folder

Create a dataset folder inside `local_workspace/datasets/` containing one paired full-size sample:

```text
local_workspace/
├── datasets/
│   └── your_sample/
│       ├── source.tif
│       └── target.tif
└── results/
```

### 2. Enter the development environment

```bash
nix develop
make sync
```

The README examples below assume you are already inside the Nix shell so that `uv` is available.

### 3. Create a run YAML

Use [`config/runs/example.yaml`](config/runs/example.yaml) as the starting point
for a complete run configuration:

```bash
cp config/runs/example.yaml config/runs/my_run.yaml
```

Edit the YAML and put every experiment value there: dataset path, source/target
image names, shared image size, preprocessing settings, split ratios, filtering
thresholds, run name, training hyperparameters, checkpoint selection, and
evaluation options. Paths under the dataset and run directory are derived by
default, so the dataset and run name do not need to be repeated.
The example also shows commented optional overrides for training split dirs,
inference dirs, and evaluation dirs.

```yaml
dataset_root: local_workspace/datasets/your_sample
results_path: local_workspace/results
run_name: your_run_name

image_size: [256, 256]  # [width, height]
```

Size values follow the `[width, height]` convention throughout the codebase — `image_size[0]` is width and `image_size[1]` is height.  Square defaults hide any ordering ambiguity, so non-square sizes such as `[320, 256]` (320 px wide, 256 px tall) rely on this convention being honoured.

The Makefile intentionally accepts only `CONFIG` for experiment execution. Do not pass `DATASET`, `RUN_NAME`, `IMAGE_SIZE`, `EPOCHS`, `SEED`, `CHECKPOINT`, or similar run settings to `make`.

### 4. Run the pipeline

Build the paired patch dataset:

```bash
make dataset CONFIG=config/runs/my_run.yaml
```

Train the model, run test inference, and evaluate the outputs:

```bash
make train CONFIG=config/runs/my_run.yaml
make infer CONFIG=config/runs/my_run.yaml
make evaluate CONFIG=config/runs/my_run.yaml
```

Or run the full sequence in one command:

```bash
make complete-run CONFIG=config/runs/my_run.yaml
```

Set `inference.checkpoint` or `inference.checkpoint_policy` in the YAML before
running inference. The example uses `checkpoint_policy: latest`.

The main outputs for a run are written under:

```text
local_workspace/results/<run_name>/
```

To inspect the available commands and flags:

```bash
make help
uv run python src/prepare_dataset.py --help
uv run python src/pix2pix.py --help
uv run python tools/evaluate_generation.py --help
uv run python tools/make_comparison.py --help
```

## Common Commands

Use `make` for the standard workflow. Experiment parameters belong in the run YAML, not in Make variables.

### `make dataset`

Builds `dataset_train/`, `dataset_val/`, and `dataset_test/` from one full-size paired sample.

Inputs:

- `CONFIG`, pointing to a run YAML with `dataset_root` and a `preprocessing:` section

Example:

```bash
make dataset CONFIG=config/runs/my_run.yaml
```

### `make train`

Starts a training run and writes logs, checkpoints, validation outputs, and metadata into `results_path/run_name`.

Inputs:

- `CONFIG`, pointing to a run YAML with `dataset_root`, `results_path`, `run_name`, and a `training:` section

Example:

```bash
make train CONFIG=config/runs/my_run.yaml
```

### `make infer`

Runs inference on `dataset_test/` using a trained checkpoint and writes generated outputs to `output_test/`.

Inputs:

- `CONFIG`, pointing to a run YAML with `inference.checkpoint` or `inference.checkpoint_policy`

Example:

```bash
make infer CONFIG=config/runs/my_run.yaml
```

### `make evaluate`

Computes MAE, RMSE, PSNR, and SSIM by comparing `dataset_test/` against `output_test/`. The evaluation tool can also save summary CSV files and aggregate plots.

Example:

```bash
make evaluate CONFIG=config/runs/my_run.yaml
```

### `make complete-run`

Runs `dataset`, `train`, `infer`, and `evaluate` sequentially using the same run YAML.

Example:

```bash
make complete-run CONFIG=config/runs/my_run.yaml
```

## Qualitative Results

The figures below show example qualitative outputs included for documentation.

Each panel compares:

- source patch
- generated target
- real target

From label free to H&E staining.
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to label free.
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Main Pipeline

### 1. Dataset preparation

`src/prepare_dataset.py` builds the paired dataset from full-size images.

It:

- computes tissue masks
- aligns the target image to the source image
- extracts paired patches
- creates `dataset_train/`, `dataset_val/`, and `dataset_test/`

### 2. Training and test inference

`src/pix2pix.py` provides two CLI modes:

- `train`
- `test`

Training creates a run directory with logs, checkpoints, validation outputs, and metadata. Test inference loads a checkpoint and generates predictions for the test split.

### 3. Evaluation and visual comparison

`tools/evaluate_generation.py` evaluates generated images against target images and can save:

- per-image metrics
- summary CSV files
- skipped samples
- optional plots

`tools/make_comparison.py` creates:

- source / generated / target comparison panels
- MAE difference maps
- optional diagnostic plots
- representative comparisons selected from evaluation results

## Repository Structure

```text
Virtual-Staining/
├── config/
│   └── runs/               # run YAML files (example.yaml template)
├── docs/
│   ├── assets/             # qualitative result images
│   ├── notebooks/
│   └── reports/
├── examples/               # example input images
├── local_workspace/
│   ├── datasets/           # input paired samples (gitignored)
│   └── results/            # run outputs (gitignored)
├── src/
│   ├── prepare_dataset.py  # dataset preparation entry point
│   └── pix2pix.py          # training and inference entry point
├── tests/                  # pytest test suite
├── tools/
│   ├── compare_distributions.py
│   ├── evaluate_generation.py
│   ├── make_comparison.py
│   └── organize_by_metrics.py
├── virtual_staining/       # shared library package
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── Makefile
├── flake.nix
├── pyproject.toml
├── uv.lock
└── README.md
```

## Advanced CLI Examples

These are the direct Python equivalents of the Makefile experiment targets.

### Prepare the dataset

```bash
uv run python src/prepare_dataset.py \
  --config config/runs/my_run.yaml
```

### Train the model

```bash
uv run python src/pix2pix.py train \
  --config config/runs/my_run.yaml
```

### Run test inference

```bash
uv run python src/pix2pix.py test \
  --config config/runs/my_run.yaml
```

### Evaluate generated results

```bash
uv run python tools/evaluate_generation.py dataset \
  --config config/runs/my_run.yaml
```

### Create representative comparison panels

```bash
uv run python tools/make_comparison.py from-metrics \
  --run-path local_workspace/results/your_run
```

### Create a single comparison panel

```bash
uv run python tools/make_comparison.py single \
  --source-image local_workspace/datasets/your_sample/dataset_test/00512_09216_source.tif \
  --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif \
  --target-image local_workspace/datasets/your_sample/dataset_test/00512_09216_target.tif \
  --with-diagnostics
```

## Notes

- `src/prepare_dataset.py` is the preprocessing entry point
- `src/pix2pix.py` handles training, validation, checkpoints, and test inference
- `tools/evaluate_generation.py` handles metric evaluation
- `tools/make_comparison.py` handles visual comparison and diagnostics
- [`config/runs/example.yaml`](config/runs/example.yaml) documents the run
  schema used by `dataset`, `train`, `infer`, and `evaluate`

## Input and Output Conventions

### Preprocessing input

Expected full-size input files:

- `source.tif`
- `target.tif`

### Training and test patch pairs

Expected paired patch naming:

- `00000_00000_source.tif`
- `00000_00000_target.tif`

Only prefix-matched pairs are used by the dataset loader.

### Typical generated artifacts

- tissue masks for both input images
- aligned target image artifacts
- extracted paired patches in `subimages/`
- dataset splits in `dataset_train/`, `dataset_val/`, and `dataset_test/`
- validation predictions
- saved checkpoints
- test predictions

## Method

The pipeline combines deterministic image processing with paired supervised deep learning.

### Preprocessing

- mask generation isolates tissue regions and reduces background-heavy areas
- alignment registers the target image to the source reference using feature-based matching and affine transformation
- patch extraction divides aligned images into fixed-size patches while filtering mostly background regions

### Learning model

- Pix2Pix-style conditional GAN trained on paired histology patches
- U-Net-like generator with skip connections
- PatchGAN discriminator operating on conditional image pairs
- adversarial supervision combined with reconstruction loss

This setup is suitable for paired image-to-image translation, where the generated target image should remain structurally consistent with the input source image.

## Limitations

The project is usable, but still experimental in several respects.

- The workflow is CLI-driven and uses a run YAML for standard experiment settings.
- The repository includes historical material alongside the current workflow.
- Dependency management is present, but still relatively lightweight.

### Data split and generalization

The default train/validation/test split is **patch-level**: all splits are drawn
from patches of the same slide. Metrics reported on `dataset_test/` therefore
measure same-slide internal validation, not independent generalization. A model
that performs well under this setting may still fail on slides or patients it
has never seen, because nearby patches from the same tissue share texture,
staining conditions, and artifacts.

Interpreting same-slide patch metrics as evidence of generalization is
discouraged. Stronger evidence of generalization requires one of the following:

- **slide-level split** — hold out entire slides for the test set
- **patient-level split** — hold out all slides from certain donors or cases
- **spatial-block split** — partition each slide into non-overlapping spatial
  blocks and assign whole blocks to each split

The current pipeline does not implement these strategies. If your evaluation
goal is generalizability to new tissue samples or new patients, the split
strategy must be adapted before drawing conclusions from the reported metrics.

## Future Improvements

Possible directions for improvement:

- centralized configuration for paths and hyperparameters
- stronger reproducibility and dataset bookkeeping
- clearer separation between preprocessing, training, and evaluation modules
- expanded evaluation and visualization utilities
- more consistent CLI design across scripts

## Academic Context

This project was developed in a university and research-oriented setting focused on:

- virtual staining
- computational histopathology
- paired image-to-image translation

It reflects both the methodological importance of aligned supervision and the practical engineering work required to turn full-size microscopy images into training-ready paired datasets.

## License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
