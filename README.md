# Virtual-Staining

Virtual staining pipeline for histopathology images, combining classical image processing and Pix2Pix-based image-to-image translation.

## Overview

This repository implements a computational histopathology workflow for transforming paired **label-free** tissue images into **virtually stained** images. The project is structured as a full supervised pipeline rather than as an isolated neural network experiment: it prepares coherent input-target pairs through preprocessing, mask generation, image alignment, patch extraction, and dataset splitting, then trains a Pix2Pix-style model on the resulting paired patches.

The supervised learning stage relies on real stained images as targets. Because the method depends on paired data, spatial alignment between the label-free and stained views is a critical preprocessing step, not an optional refinement.

## Why This Repository Matters

Virtual staining is relevant in computational pathology because it explores a digital alternative, complement, or preprocessing aid to conventional chemical staining workflows. In supervised settings, however, the quality of the learned mapping depends heavily on the quality of the paired data. This repository addresses that practical requirement by combining:

- foreground isolation through mask generation;
- classical registration of stained and label-free images;
- patch-based dataset preparation for paired learning;
- conditional GAN training with a Pix2Pix-style architecture.

As a result, the repository is useful both as an experimental research codebase and as a structured end-to-end project in computational imaging and deep learning.

## Current Pipeline

The current workflow is centered on two main scripts in [`repo/scripts/`](./repo/scripts):

1. Provide a folder containing a paired full-size input sample, typically `label_free.tif` and `stained.tif`.
2. Run [`repo/scripts/ollie_wan_kenobi.py`](./repo/scripts/ollie_wan_kenobi.py) to generate foreground masks.
3. Use the same script to align the stained image to the label-free reference.
4. Extract aligned patches from tissue regions.
5. Split the extracted paired patches into `train`, `val`, and `test`.
6. Run [`repo/scripts/pix2pix.py`](./repo/scripts/pix2pix.py) in training mode to train the model on paired patches.
7. During training, validation outputs and checkpoints are saved.
8. Run the same script in test mode to generate virtual staining predictions on the test split.

In short:

- `ollie_wan_kenobi.py` handles preprocessing and dataset creation.
- `pix2pix.py` handles model training, validation, checkpoints, and test inference.

## Repository Structure

The repository contains both the current executable workflow and older project material accumulated during development. The main operational path is concentrated in `repo/scripts/`.

```text
Virtual-Staining/
├── repo/
│   ├── scripts/
│   │   ├── ollie_wan_kenobi.py
│   │   ├── pix2pix.py
│   │   └── json/
│   │       ├── help.json
│   │       ├── messages.json
│   │       └── p2p_settings.json
│   ├── template/
│   ├── *.ipynb / *.pdf / *.md
│   └── build_pdf.*
├── Appunti/
├── Materiale/
├── Locale/
├── requirements.txt
└── README.md
```

### Main folders and files

- `repo/`
  Contains the current project package, including the main scripts, configuration JSON files, and supporting project material.
- `repo/scripts/`
  Main executable code for preprocessing and model training/testing.
- `repo/scripts/ollie_wan_kenobi.py`
  Integrated preprocessing pipeline for mask generation, alignment, patch extraction, and dataset split creation.
- `repo/scripts/pix2pix.py`
  Pix2Pix training and inference script for paired histology patches.
- `repo/scripts/json/`
  Support files used by the scripts, including CLI/help messages and some model block settings.
- `Appunti/`
  Historical notes, notebooks, and earlier development scripts.
- `Materiale/`, `Locale/`
  Project material and local data/output areas used during experimentation.

Top-level folders such as `Appunti/` and parts of `Materiale/` remain useful as historical and experimental context, but they are not the main entry point for the current workflow.

## Main Scripts

### `repo/scripts/ollie_wan_kenobi.py`

This is the integrated preprocessing script for building a paired dataset from full-size histology images.

**Role**

- loads a folder containing `label_free.tif` and `stained.tif`;
- generates foreground masks for both images;
- estimates an affine alignment of the stained image onto the label-free reference;
- extracts subimages from aligned tissue regions;
- saves patch pairs and creates `train`, `val`, and `test` splits.

**Input expectations**

- one directory per sample or processing run;
- inside that directory:
  - `label_free.tif`
  - `stained.tif`

**Typical outputs**

- `mask_lf.tif`
- `mask_st.tif`
- `aligned_stained.tif`
- `aligned_mask_st.tif`
- `subimages/`
- `train/`
- `val/`
- `test/`

**CLI role**

The script exposes a simple CLI with:

- a required input path;
- an optional random seed;
- an optional `--save_masks` flag;
- an optional `--lang {en,it}` switch for messages.

### `repo/scripts/pix2pix.py`

This script handles the neural network stage of the project.

**Role**

- loads paired patch datasets;
- trains a Pix2Pix-style conditional GAN;
- runs validation during training;
- saves checkpoints;
- performs test-time inference from a saved checkpoint.

**Dataset convention**

The dataset is based on paired filenames such as:

- `00000_00000_label_free.tif`
- `00000_00000_stained.tif`

The script expects paired files sharing the same prefix and different suffixes.

**Train vs test**

- `train` mode: trains the generator and discriminator, logs losses, validates, and saves checkpoints.
- `test` mode: loads a checkpoint and generates virtual staining outputs for test images.

**Current practical limitation**

Some internal paths are still defined directly in the script, including dataset folders and checkpoint locations. In practice, users may still need to edit those variables before running training or inference on a new setup.

## How the Method Works

The method combines deterministic image processing with paired supervised deep learning.

### 1. Mask generation and foreground selection

The preprocessing script estimates binary masks to isolate relevant tissue regions and reduce the influence of large background areas. The current implementation uses thresholding, connected components, contour filling, and repeated grid-based mask estimation to retain useful foreground regions.

### 2. Image alignment

The stained image is aligned to the label-free image before patch extraction. This is essential because supervised image-to-image translation requires spatially coherent input-target pairs. The script uses classical feature-based registration with CLAHE-enhanced grayscale images, SIFT feature detection, brute-force matching, Euclidean-distance filtering, and affine transformation estimation.

### 3. Patch extraction

Once alignment is complete, the pipeline crops the images into fixed-size patches. Foreground masks are used to discard regions that are mostly background, improving the quality of the training data and reducing useless pairs.

### 4. Train/validation/test split

The extracted paired patches are shuffled and divided into `train`, `val`, and `test` subsets. The current preprocessing script creates these splits directly from the extracted patch pairs.

### 5. Paired supervised learning with Pix2Pix

The learning stage treats the label-free patch as input and the stained patch as target. The generator learns to synthesize a stained-like image from the label-free image, while the discriminator evaluates whether the generated result is consistent with the target distribution in a conditional setting.

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

The repository includes a minimal [`requirements.txt`](./requirements.txt), but dependency management is not yet fully polished or tightly pinned. Depending on the environment, some packages used by the scripts may need to be installed manually.

## Expected Input Data

The project expects **paired histology images**.

### For preprocessing

You should provide a folder containing:

- `label_free.tif`
- `stained.tif`

These are full-size paired images of the same sample, where:

- `label_free.tif` is the unstained or label-free image used as model input;
- `stained.tif` is the stained counterpart used as supervision target.

### For model training and testing

The Pix2Pix script expects paired patches named with a shared prefix and suffixes such as:

- `xxxxx_yyyyy_label_free.tif`
- `xxxxx_yyyyy_stained.tif`

Only filenames that form valid pairs are used by the dataset loader.

## Quick Start

For users who want the shortest path:

### 1. Prepare paired full-size input images

Create a folder containing:

```text
your_sample/
├── label_free.tif
└── stained.tif
```

### 2. Run preprocessing

```bash
python repo/scripts/ollie_wan_kenobi.py your_sample --lang en --seed 42
```

### 3. Train the model

Before training, check the dataset and checkpoint paths configured inside `repo/scripts/pix2pix.py`.

```bash
python repo/scripts/pix2pix.py train
```

### 4. Run test inference

After setting the desired checkpoint path in the script:

```bash
python repo/scripts/pix2pix.py test
```

## Detailed Usage

### Preprocessing with `ollie_wan_kenobi.py`

Basic usage:

```bash
python repo/scripts/ollie_wan_kenobi.py <path>
```

Example:

```bash
python repo/scripts/ollie_wan_kenobi.py Materiale/Locale/liver --lang en --seed 42 --save_masks
```

What it does:

- loads `label_free.tif` and `stained.tif` from the given folder;
- computes masks for both images;
- aligns the stained image to the label-free image;
- extracts patch pairs;
- saves the extracted dataset into `train`, `val`, and `test`.

Available CLI options:

```bash
python repo/scripts/ollie_wan_kenobi.py path [--seed SEED] [--save_masks] [--lang {en,it}]
```

### Training with `pix2pix.py`

Training mode:

```bash
python repo/scripts/pix2pix.py train
```

Current behavior:

- loads paired images from configured training and validation folders;
- trains a U-Net-like generator and PatchGAN discriminator;
- logs progress and losses;
- saves validation outputs;
- saves checkpoints at configured intervals.

Important note:

- dataset root paths, checkpoint paths, and some training settings are still hard-coded in the script and may need to be adjusted manually before execution.

### Testing / inference with `pix2pix.py`

Test mode:

```bash
python repo/scripts/pix2pix.py test
```

Current behavior:

- loads a configured checkpoint;
- scans the configured test folder for `*_label_free.tif` files;
- generates corresponding virtual staining predictions;
- saves outputs to the configured output directory.

Important note:

- the checkpoint path is currently defined inside the script and should be verified before running inference.

## Outputs

After preprocessing with `ollie_wan_kenobi.py`, the sample folder typically contains:

- generated masks:
  - `mask_lf.tif`
  - `mask_st.tif`
- aligned artifacts:
  - `aligned_stained.tif`
  - `aligned_mask_st.tif`
- extracted patches:
  - `subimages/`
- dataset splits:
  - `train/`
  - `val/`
  - `test/`

After training with `pix2pix.py`, the workflow can generate:

- log files;
- validation predictions;
- model checkpoints;
- training preview outputs, depending on the configured paths.

After test inference with `pix2pix.py`, the workflow generates:

- predicted virtually stained images for the test set.

## Model

The learning component is a **Pix2Pix-style conditional GAN** trained on paired histology patches.

- **Generator**: a U-Net-like encoder-decoder architecture with skip connections.
- **Discriminator**: a PatchGAN-style discriminator operating on conditional image pairs.
- **Training objective**: adversarial supervision combined with pixel-level reconstruction loss.

This design is appropriate for paired image-to-image translation, where the goal is to synthesize a stained image while preserving the spatial structure present in the label-free input.

## Limitations / Current State

The repository is structured and usable, but still experimental in several practical aspects.

- Some dataset paths and checkpoint paths are still configured directly inside the code.
- Configuration is only partially centralized.
- The repository contains historical folders, notes, and experimental material alongside the current main workflow.
- Dependency specification is minimal and may require manual environment setup.
- The preprocessing and training scripts are functional entry points, but the codebase is still evolving rather than packaged as a finalized tool.

## Future Improvements

Plausible next steps for the repository include:

- centralized configuration for paths and hyperparameters;
- cleaner CLI unification across preprocessing and model scripts;
- stronger reproducibility controls and dataset bookkeeping;
- clearer separation between active code, generated artifacts, and historical material;
- more modular packaging of preprocessing, training, and evaluation code;
- expanded evaluation, visualization, and reporting utilities.

## Academic Context

This project was developed in a university and research-oriented context focused on **virtual staining**, **computational histopathology**, and **paired image-to-image translation**. It reflects both the methodological requirements of the domain, especially the need for coherent aligned supervision, and the practical engineering work needed to turn full-size microscopy images into usable learning data.
