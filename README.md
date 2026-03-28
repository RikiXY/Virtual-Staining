# Virtual-Staining

Paired virtual staining pipeline for histopathology images.

This repository implements a paired virtual staining workflow that combines classical image processing and deep learning. Starting from paired full-size histology images, the pipeline builds an aligned patch dataset, trains a Pix2Pix-style conditional GAN, runs test inference, and provides integrated evaluation and comparison tools.

The learning stage currently uses a **U-Net generator** and a **PatchGAN discriminator**, trained with **adversarial loss + L1 reconstruction loss**.

## Overview

The project follows this workflow:

1. load a paired full-size sample (`source` / `target`);
2. generate tissue masks;
3. align the target image to the source reference;
4. extract aligned paired patches;
5. build `dataset_train`, `dataset_val`, and `dataset_test`;
6. train the model;
7. run test inference on held-out patches;
8. evaluate generated results with integrated metrics and comparison panels.

In addition to preprocessing and training, the repository also includes:

- a metric evaluation tool for **MAE, RMSE, PSNR and SSIM**
- a comparison tool to generate **visual panels and diagnostic plots**

## Quick Start

### 1. Prepare the input folder

Create a sample folder inside `local_workspace/` containing a paired full-size sample:

```text
local_workspace/
├── datasets/
│  └── your_sample/
│     ├── source.tif
│     └── target.tif
└── results/
```

### 2. Enter the development environment and sync dependencies

```bash
nix develop
make sync
```

### 3. Build the paired dataset

```bash
export DATASET=your_sample
make prepare-dataset SOURCE_NAME=source.tif TARGET_NAME=target.tif
```

Use `make help` to inspect available `prepare-dataset` options, or see script-level options with:

```bash
uv run python src/prepare_dataset.py --help
```

### 4. Train, test, and evaluate (Makefile workflow)

Set dataset and run once, then use `make` targets:

```bash
export DATASET=your_sample
export RUN_NAME=your_run_name

make train
make test
make evaluate
```

Useful optional overrides:

```bash
make train EPOCHS=10 SEED=123
make test CHECKPOINT=local_workspace/results/your_run_name/checkpoints/ep010.pth
```

If `CHECKPOINT` is not provided, `make test` automatically picks the highest `ep*.pth` found in `local_workspace/results/<RUN_NAME>/checkpoints/`.

Training outputs, checkpoints, and test predictions are written under `local_workspace/results/<RUN_NAME>/`.

## Qualitative Results

The figures below show example qualitative outputs included for documentation.

Each panel compares:

- source patch
- generated target
- real target

From Label free to H&E staining.
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to Label free.
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Main Pipeline

### 1. Dataset preparation

`src/prepare_dataset.py` builds the paired dataset from full-size images.

It:

- computes tissue masks;
- aligns the target image to the source image;
- extracts paired patches;
- creates `dataset_train/`, `dataset_val/`, and `dataset_test/`

### 2. Training and test inference

`src/pix2pix.py` provides two CLI modes:

- `train`
- `test`

Training creates a run directory with logs, checkpoints, validation outputs and metadata.  
Test inference loads a checkpoint and generates predictions for the test split.

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

## Example Workflow

The commands below show a typical end-to-end usage example.

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
python src/pix2pix.py train \
  --dataset-root local_workspace/datasets/your_sample \
  --run-name your_run \
  --results-path local_workspace/results \
  --epochs 100 \
  --batch-size 8 \
  --l1-lambda 25 \
  --seed 42
```

### Run test inference

```bash
python src/pix2pix.py test \
  --dataset-root local_workspace/datasets/your_sample \
  --run-path local_workspace/results/your_run \
  --checkpoint local_workspace/results/your_run/checkpoints/ep099.pth
```

### Evaluate generated results

```bash
python tools/evaluate_generation.py dataset \
  --target-dir local_workspace/datasets/your_sample/dataset_test \
  --generated-dir local_workspace/results/your_run/output_test \
  --save-graphs
```

### Create representative comparison panels

```bash
python tools/make_comparison.py \
  --from-metrics \
  --run-path local_workspace/results/your_run
```

### Create a single comparison panel

```bash
python tools/make_comparison.py \
  --source-image local_workspace/datasets/your_sample/dataset_test/source.tif \
  --generated-image local_workspace/results/your_run/output_test/generated.tif \
  --target-image local_workspace/datasets/your_sample/dataset_test/target.tif \
  --with-diagnostics
```

## Notes

- `prepare_dataset.py` is the preprocessing entry point
- `pix2pix.py` handles training, validation, checkpoints and test inference
- `evaluate_generation.py` handles metric evaluation
- `make_comparison.py` handles visual comparison and diagnostics

For detailed options, use:

```bash
uv run python src/prepare_dataset.py --help
uv run python src/pix2pix.py --help
uv run python src/pix2pix.py train --help
uv run python src/pix2pix.py test --help
uv run python tools/evaluate_generation.py --help
uv run python tools/make_comparison.py --help
```

**Typical outputs**

- `mask_lf.tif`
- `mask_st.tif`
- `aligned_stained.tif`
- `aligned_mask_st.tif`
- `subimages/`
- `dataset_train/`
- `dataset_val/`
- `dataset_test/`

When `--save-masks` is enabled, patch-level mask files are also saved inside `subimages/`.

### `src/pix2pix.py`

Learning and inference stage of the pipeline.

**What it does**

- loads paired patch datasets using the naming convention `*_label_free.tif` and `*_stained.tif`;
- trains a Pix2Pix-style conditional GAN;
- validates the model during training;
- saves checkpoints;
- runs test-time inference from a selected checkpoint.

**CLI usage**

```bash
uv run python src/pix2pix.py train --dataset-root <dataset_root>/ --run-name <run_name> --results-path local_workspace/results --epochs 100 --seed 42
uv run python src/pix2pix.py test --dataset-root <dataset_root>/ --run-path local_workspace/results/<run_name> --checkpoint <checkpoint_path>
```

To see all available options:

```bash
uv run python src/pix2pix.py --help
uv run python src/pix2pix.py train --help
uv run python src/pix2pix.py test --help
```

**Practical note**: logs, checkpoints, validation outputs, and test outputs are automatically created inside `local_workspace/results/<run_name>/`.

## Input and Outputs

**Expected preprocessing input**:

- `label_free.tif`
- `stained.tif`

**Expected training/testing pairs**:

- `00000_00000_label_free.tif`
- `00000_00000_stained.tif`

Only valid prefix-matched pairs are used by the dataset loader.

**Typical generated artifacts**:

- masks for both input images;
- aligned stained image artifacts;
- extracted paired patches in `subimages/`;
- dataset splits in `dataset_train/`, `dataset_val/`, and `dataset_test/`.
- validation predictions;
- saved checkpoints;
- test predictions.

## Method

The pipeline combines deterministic image processing with paired supervised deep learning.

**Preprocessing**

- mask generation isolates tissue regions and reduces background-dominated areas;
- alignment registers the stained image to the label-free reference using feature-based matching and affine transformation;
- patch extraction divides aligned images into fixed-size patches while filtering mostly background regions.

**Learning model:**

- Pix2Pix-style conditional GAN trained on paired histology patches;
- U-Net-like generator with skip connections;
- PatchGAN discriminator on conditional image pairs;
- adversarial supervision combined with reconstruction loss.

This setup is suitable for paired image-to-image translation, where the generated stained image should remain structurally consistent with the input label-free image.

## Requirements

The project currently targets:

- **Python 3.11**

Main libraries used in the codebase include:

- OpenCV
- NumPy
- Pillow
- PyTorch
- torchvision
- Matplotlib

Recommended setup and dev workflow:

```bash
nix develop
make sync
make lint
make format-check
make check-types
make check
```

Nix notes:

- `nix develop` enters the pinned development shell from `flake.nix`.
- The shell provides Python 3.11, `uv`, `make`, `ruff`, and `pyright`.

Lock refresh:

```bash
make lock
```

Cleanup:

```bash
make clean
```

## Detailed Usage

**Preprocessing:**

```bash
export DATASET=liver
make prepare-dataset SOURCE_NAME=source.tif TARGET_NAME=target.tif SAVE_MASKS=1 PREPARE_LANG=en
```

**Training:**

```bash
export DATASET=liver
export RUN_NAME=liver_run
make train
```

**Testing/Inference:**

```bash
make test
```

Optional explicit checkpoint:

```bash
make test CHECKPOINT=local_workspace/results/liver_run/checkpoints/ep042.pth
```

**Evaluation:**

```bash
make evaluate
```

## Limitations/Current State

The project is structured and usable, but still experimental in several practical respects.

- The workflow is now CLI-driven, but configuration is still only partially centralized.
- The repository includes historical material alongside the current main workflow.
- Dependency management is minimal and not yet tightly pinned.

## Future Improvements

Planned or plausible directions for improvement include:

- centralized configuration for paths and hyperparameters;
- more consistent CLI design and centralized configuration;
- stronger reproducibility and dataset bookkeeping;
- more modular separation of preprocessing, training, and evaluation code;
- expanded evaluation and visualization utilities.

## Academic Context

This project was developed in a university and research-oriented setting focused on **virtual staining**, **computational histopathology**, and **paired image-to-image translation**. It reflects both the methodological importance of aligned supervision and the practical engineering work required to transform full-size microscopy images into training-ready paired data.
