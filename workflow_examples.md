# Workflow Examples

These examples use the current command-line interfaces. Replace dataset, run, and checkpoint names with paths from your own experiment.

For complete options, run each script with `-h` or `--help`.

<!-- Future documentation image placeholders:
- docs/assets/pipeline_overview.png
- docs/assets/example_comparison.png
-->

## Dataset Preparation

Prepare a paired patch dataset from one source/target full-size image pair:

```bash
python src/prepare_dataset.py \
  --path local_workspace/datasets/your_dataset \
  --source-name source.tif \
  --target-name target.tif \
  --image-size 256 256 \
  --grid-movement 256 256 \
  --margin 200 \
  --seed 42 
```

The preparation script writes `dataset_train/`, `dataset_val/`, `dataset_test/`, `discarded_patches/`, masks, and aligned target images. The split is patch-level.

## Training

```bash
python src/pix2pix.py train \
  --dataset-root local_workspace/datasets/your_dataset \
  --run-name your_run \
  --results-path local_workspace/results \
  --epochs 100 \
  --batch-size 8 \
  --image-size 256 256 \
  --l1-weight 25 \
  --ssim-weight 1 \
  --seed 42
```

Training reads `dataset_train/` and `dataset_val/`, then writes logs, checkpoints, validation previews, and `run_config.json`.

## Test Inference

```bash
python src/pix2pix.py test \
  --dataset-root local_workspace/datasets/your_dataset \
  --run-path local_workspace/results/your_run \
  --checkpoint local_workspace/results/your_run/checkpoints/ep099.pth \
  --image-size 256 256
```

Generated images are written to `local_workspace/results/your_run/output_test/` with names ending in `_target_generated.tif`.

## Evaluation

Evaluate all matching target/generated pairs:

```bash
python tools/evaluate_generation.py dataset \
  --target-dir local_workspace/datasets/your_dataset/dataset_test \
  --generated-dir local_workspace/results/your_run/output_test \
  --save-graphs
```

Evaluate a single pair:

```bash
python tools/evaluate_generation.py single \
  --target-image local_workspace/datasets/your_dataset/dataset_test/00512_09216_target.tif \
  --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif
```

Dataset evaluation writes `per_image_metrics.csv`, `summary.csv`, and `skipped.csv` under the run `evaluation/` directory unless `--output-dir` is provided.

## Comparison Panels

Generate representative panels from evaluation CSVs:

```bash
python tools/make_comparison.py from-metrics \
  --run-path local_workspace/results/your_run
```

Generate a single source/generated/target panel:

```bash
python tools/make_comparison.py single \
  --source-image local_workspace/datasets/your_dataset/dataset_test/00512_09216_source.tif \
  --generated-image local_workspace/results/your_run/output_test/00512_09216_target_generated.tif \
  --target-image local_workspace/datasets/your_dataset/dataset_test/00512_09216_target.tif \
  --with-diagnostics
```

## Comparing Runs

Use paired comparison only when both runs were evaluated on the same sample IDs and compatible test data:

```bash
python tools/compare_distributions.py paired \
  --run-a local_workspace/results/run_a \
  --run-b local_workspace/results/run_b \
  --column ssim
```

Use unpaired comparison for independent metric distributions:

```bash
python tools/compare_distributions.py unpaired \
  --run-a local_workspace/results/run_a \
  --run-b local_workspace/results/run_b \
  --column ssim
```

## Organizing Outputs by Metric

```bash
python tools/organize_by_metrics.py \
  --run-path local_workspace/results/your_run \
  --top-k 20
```

The default output directory is `local_workspace/results/your_run/evaluation/sorted_by_metrics/`.
