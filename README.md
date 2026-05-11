# Virtual-Staining

Paired virtual staining pipeline for histopathological image experiments.

This repository prepares aligned source/target patch datasets, trains a Pix2Pix-style model, runs test inference, evaluates generated images, and produces qualitative and run-to-run comparison outputs. The learning pipeline currently uses a U-Net generator with a PatchGAN discriminator.

The current dataset split is patch-level as implemented in `src/prepare_dataset.py`; it is not a slide-level or patient-level split.

<!-- Future documentation image placeholders:
- docs/assets/pipeline_overview.png
- docs/assets/example_comparison.png
-->
From Label free to H&E staining.
![Qualitative results](docs/assets/LabelFree-to-Stained_qualitative_result_2.png)

From H&E staining to Label free.
![Qualitative results](docs/assets/Stained-to-LabelFree_qualitative_result_2.png)

## Workflow

1. Prepare a paired patch dataset from full-size source and target images.
2. Train Pix2Pix on `dataset_train/` and validate on `dataset_val/`.
3. Run inference on `dataset_test/`.
4. Evaluate generated images against real targets.
5. Generate comparison panels or compare metric distributions across runs.

## Repository Layout

```text
Virtual-Staining/
├── src/
│   ├── prepare_dataset.py
│   └── pix2pix.py
├── tools/
│   ├── evaluate_generation.py
│   ├── make_comparison.py
│   ├── compare_distributions.py
│   └── organize_by_metrics.py
├── docs/
├── examples/
├── local_workspace/
│   ├── datasets/
│   └── results/
└── requirements.txt
```

`archive/` contains historical material and is not part of the current workflow.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build for GPU training.

## Expected Data Layout

Input dataset directory:

```text
local_workspace/datasets/YOUR_DATASET/
├── source.tif
└── target.tif
```

Prepared dataset:

```text
local_workspace/datasets/YOUR_DATASET/
├── dataset_train/
├── dataset_val/
├── dataset_test/
└── discarded_patches/
```

Run directory:

```text
local_workspace/results/YOUR_RUN/
├── run_config.json
├── logs/
├── checkpoints/
├── output_val/
├── output_test/
├── evaluation/
└── comparisons/
```

Patch naming conventions:

```text
<sample_id>_source.tif
<sample_id>_target.tif
<sample_id>_target_generated.tif
```

## Quick Start

Prepare the dataset:

```bash
python src/prepare_dataset.py --path local_workspace/datasets/your_dataset --source-name source.tif --target-name target.tif
```

Train:

```bash
python src/pix2pix.py train --dataset-root local_workspace/datasets/your_dataset --run-name your_run --epochs 100
```

Run test inference:

```bash
python src/pix2pix.py test --dataset-root local_workspace/datasets/your_dataset --run-path local_workspace/results/your_run --checkpoint local_workspace/results/your_run/checkpoints/ep099.pth
```

Evaluate:

```bash
python tools/evaluate_generation.py dataset --target-dir local_workspace/datasets/your_dataset/dataset_test --generated-dir local_workspace/results/your_run/output_test
```

Create representative comparison panels:

```bash
python tools/make_comparison.py from-metrics --run-path local_workspace/results/your_run
```

See [workflow examples](workflow_examples.md) for longer command recipes and optional arguments.

## Tools

- `src/prepare_dataset.py`: masks, registration, patch extraction, patch-level train/val/test split.
- `src/pix2pix.py`: Pix2Pix training and test inference.
- `tools/evaluate_generation.py`: per-image metrics, summary statistics, skipped sample report, optional plots.
- `tools/make_comparison.py`: single-case and metric-selected comparison panels.
- `tools/compare_distributions.py`: paired or unpaired comparison of metric CSVs across runs.
- `tools/organize_by_metrics.py`: best/worst folders ranked by metric values.

Evaluation metrics include `mae`, `mse`, `rmse`, `psnr`, `ssim`, `pcc_gray`, and `pcc_rgb_mean`.

## Reproducibility Notes

- Use `--seed` for dataset preparation and training.
- Training stores key run metadata in `run_config.json`.
- For rigorous comparisons, keep the dataset, checkpoint, code version, CLI arguments, and evaluation CSVs together.
- CUDA, PyTorch, OpenCV, and multiprocessing behavior may still vary across environments.

## CLI Help

Use `-h` or `--help` for complete options:

```bash
python src/prepare_dataset.py --help
python src/pix2pix.py train --help
python src/pix2pix.py test --help
python tools/evaluate_generation.py --help
python tools/make_comparison.py --help
python tools/compare_distributions.py --help
python tools/organize_by_metrics.py --help
```
