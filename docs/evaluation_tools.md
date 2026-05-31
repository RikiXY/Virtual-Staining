# Evaluation Tools

The evaluation CLI has one dataset evaluator and several downstream inspection
utilities. Use `vs-evaluate` to create dataset-level metrics first; use the
other commands to inspect, compare, render, or export artifacts from those
metrics.

| Tool | Use When | Main Input | Main Output | Does Not Do |
|---|---|---|---|---|
| `vs-evaluate` | Evaluate a run's generated test images against targets. This is the only dataset-level evaluator. | Run YAML with `evaluation` settings, test manifest, generated images under `RUN/artifacts/output_test/` by default. | `RUN/evaluation/per_image_metrics.csv`, `summary.csv`, `weak_tail.csv`, optional plots, optional residual heatmaps, and `artifacts.json`. | Compare runs, render panels, or export ranked file folders. |
| `vs-evaluate-single` | Check one target/generated pair during debugging. | `--target-image` and `--generated-image`; optional `--output-dir`. | `individual_cases/<sample_id>_evaluation.csv` under the inferred or supplied evaluation directory. | Evaluate directories, read run config, or produce dataset summaries. |
| `vs-compare` | Compare metric distributions or paired per-sample metrics across runs. | Two `per_image_metrics.csv` files or a run config `compare` section. | Reports under `comparisons/<run_a>_vs_<run_b>/`, including paired or unpaired CSV summaries. | Compute image metrics or render qualitative panels. |
| `vs-render-panels` | Render visual diagnostics for selected source/generated/target samples. | A source/generated/target image triple, or `RUN/evaluation/per_image_metrics.csv` plus `summary.csv` through `from-metrics`. | Panel PNGs, diagnostic PNGs, selection summaries under `RUN/comparisons/metrics/`, and manifest entries when run-scoped. | Compare metric distributions or export raw ranked file sets. |
| `vs-organize` | Export ranked sample files for manual review or collection. | `RUN/evaluation/per_image_metrics.csv` via `--run-path`, `--metrics-csv`, or an `organize` config section. | Ranked source/target/generated file folders and `organization_summary.csv` under `RUN/evaluation/sorted_by_metrics/`, plus manifest entries when run-scoped. | Compute metrics or render diagnostic panels. |

## Common Flow

```bash
vs-evaluate --config config/runs/local/my_run.yaml
vs-render-panels --config config/runs/local/my_run.yaml
vs-organize --config config/runs/local/my_run.yaml
vs-compare --config config/runs/local/my_run.yaml
```

For one-off case debugging, call the single-pair utility directly:

```bash
vs-evaluate-single \
  --target-image local_workspace/datasets/my_run/splits/test/00512_09216_target.tif \
  --generated-image local_workspace/results/my_run/artifacts/output_test/00512_09216_target_generated.tif
```

Downstream apps and reports should consume the generated CSV, JSON, and image
artifacts instead of guessing paths. The artifact manifest lives at
`RUN/evaluation/artifacts.json`; see [Run Output Format](run_format.md) for the
manifest schema and run directory layout.
