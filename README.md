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

`make lock` re-resolves against the bounds in `pyproject.toml` and updates `uv.lock`. Run it when you intentionally want to upgrade dependencies.

## Development Environment

Recommended setup:

```bash
nix develop
make sync
```

The Nix shell provides:

- Python 3.11
- `uv`
- `make`
- `ruff`
- `pyright`

Useful development commands:

```bash
make lint
make format-check
make check-types
make check
make lock
make clean
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

`DATASET` must match the folder name under `local_workspace/datasets/`.

### 2. Enter the development environment

```bash
nix develop
make sync
```

The README examples below assume you are already inside the Nix shell so that `uv` is available.

### 3. Create the Makefile config file

```bash
cp .env.make.example .env.make
```

Edit `.env.make` and set the dataset you want to use and the run directory you want to create:

```make
DATASET=your_sample
RUN_NAME=your_run_name
```

Example:

```make
DATASET=inv_512
RUN_NAME=inv_P-512_L1-50
```

### 4. Run the pipeline

Build the paired patch dataset:

```bash
make prepare-dataset
```

If your full-size files use different names, override them explicitly:

```bash
make prepare-dataset SOURCE_NAME=label_free.tif TARGET_NAME=he_stain.tif
```

Train the model, run test inference, and evaluate the outputs:

```bash
make train
make test
make evaluate
```

Or run the full sequence in one command:

```bash
make run-all
```

Useful one-off overrides:

```bash
make train EPOCHS=10 SEED=123
make test CHECKPOINT=local_workspace/results/your_run_name/checkpoints/ep010.pth
make run-all RUN_NAME=debug_run L1=37
```

If `CHECKPOINT` is omitted, `make test` looks for the highest `ep*.pth` file under:

```text
local_workspace/results/<RUN_NAME>/checkpoints/
```

The main outputs for a run are written under:

```text
local_workspace/results/<RUN_NAME>/
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

Use `make` for the standard workflow. Use the raw Python CLIs when you need advanced options not exposed by the `Makefile`.

### `make prepare-dataset`

Builds `dataset_train/`, `dataset_val/`, and `dataset_test/` from one full-size paired sample.

Inputs:

- `DATASET` from `.env.make` or the command line
- full-size input files inside `local_workspace/datasets/<DATASET>/`
- optional overrides such as `SOURCE_NAME`, `TARGET_NAME`, `IMAGE_SIZE`, `GRID_MOVEMENT`, `MARGIN`, and `SAVE_MASKS`

Example:

```bash
make prepare-dataset \
  DATASET=your_sample \
  SOURCE_NAME=source.tif \
  TARGET_NAME=target.tif \
  SAVE_MASKS=1 \
  IMAGE_SIZE="512 512" \
  GRID_MOVEMENT="512 512"
```

### `make train`

Starts a training run and writes logs, checkpoints, validation outputs, and metadata into `local_workspace/results/<RUN_NAME>/`.

Inputs:

- `DATASET`
- `RUN_NAME`
- optional overrides such as `EPOCHS`, `SEED`, and `L1`

Example:

```bash
make train DATASET=your_sample RUN_NAME=your_run_name EPOCHS=100 L1=50
```

### `make test`

Runs inference on `dataset_test/` using a trained checkpoint and writes generated outputs to `output_test/`.

Inputs:

- `DATASET`
- `RUN_NAME`
- optional `CHECKPOINT`

Examples:

```bash
make test
make test CHECKPOINT=local_workspace/results/your_run_name/checkpoints/ep050.pth
```

### `make evaluate`

Computes MAE, RMSE, PSNR, and SSIM by comparing `dataset_test/` against `output_test/`. The evaluation tool can also save summary CSV files and aggregate plots.

Example:

```bash
make evaluate
```

### `make run-all`

Runs `train`, `test`, and `evaluate` sequentially using the current `DATASET` and `RUN_NAME`.

Example:

```bash
make run-all DATASET=your_sample RUN_NAME=debug_run EPOCHS=10
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
├── archive/
├── docs/
├── examples/
├── local_workspace/
├── src/
│   ├── json/
│   ├── prepare_dataset.py
│   └── pix2pix.py
├── tools/
│   ├── evaluate_generation.py
│   └── make_comparison.py
├── Makefile
├── flake.nix
├── pyproject.toml
├── README.md
└── TASKS.md
```

## Advanced CLI Examples

Use these when you want more control than the `Makefile` exposes.

### Prepare the dataset

```bash
uv run python src/prepare_dataset.py \
  --path local_workspace/datasets/your_sample \
  --source-name source.tif \
  --target-name target.tif \
  --image-size 512 512 \
  --grid-movement 512 512 \
  --save-masks \
  --margin 200 \
  --seed 42 \
  --lang en
```

### Train the model

```bash
uv run python src/pix2pix.py train \
  --dataset-root local_workspace/datasets/your_sample \
  --run-name your_run \
  --results-path local_workspace/results \
  --epochs 100 \
  --batch-size 8 \
  --image-size 512 512 \
  --l1-weight 50 \
  --seed 42
```

### Run test inference

```bash
uv run python src/pix2pix.py test \
  --dataset-root local_workspace/datasets/your_sample \
  --run-path local_workspace/results/your_run \
  --checkpoint local_workspace/results/your_run/checkpoints/ep099.pth
```

### Evaluate generated results

```bash
uv run python tools/evaluate_generation.py dataset \
  --target-dir local_workspace/datasets/your_sample/dataset_test \
  --generated-dir local_workspace/results/your_run/output_test \
  --save-graphs
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

- The workflow is CLI-driven, but configuration is only partially centralized.
- The repository includes historical material alongside the current workflow.
- Dependency management is present, but still relatively lightweight.

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
