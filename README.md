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
├── requirements.txt
├── README.md
└── TASKS.md
```

## Example Workflow

The commands below show a typical end-to-end usage example.

### Prepare the dataset

```bash
python src/prepare_dataset.py \
  --path local_workspace/datasets/your_sample \
  --source-name source.tif \
  --target-name target.tif \
  --image-size 512 512 \
  --grid-movement 512 512 \
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
python src/prepare_dataset.py --help
python src/pix2pix.py --help
python src/pix2pix.py train --help
python src/pix2pix.py test --help
python tools/evaluate_generation.py --help
python tools/make_comparison.py --help
```
