from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.io import (
    collect_image_files,
    extract_single_sample_id,
)
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.reports import (
    build_metric_row,
    write_single_case_csv,
    write_skipped_csv,
)
from virtual_staining.evaluation.summaries import write_summary_csv
from virtual_staining.experiment.run_paths import RunPaths


def infer_default_output_dir(generated_path: str | Path) -> Path:
    """Tries to derive results/NAME_RUN/evaluation from a path inside the run."""
    path = Path(generated_path).resolve()
    base = path.parent if path.is_file() else path
    parts = base.parts

    if "results" not in parts:
        raise ValueError(
            "Could not infer output directory from generated path. Expected the generated "
            "data to be inside a path like .../results/NAME_RUN/... Please provide "
            "--output-dir explicitly."
        )

    results_index = parts.index("results")

    if results_index + 1 >= len(parts):
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    run_dir = Path(*parts[: results_index + 2])

    if run_dir.name == "results":
        raise ValueError(
            "Could not infer NAME_RUN from generated path. Expected a path like "
            ".../results/NAME_RUN/... Please provide --output-dir explicitly."
        )

    if run_dir.parent.name != "results":
        raise ValueError(
            "Could not infer a valid run directory inside results/. Please provide "
            "--output-dir explicitly."
        )

    return run_dir / "evaluation"


def resolve_output_dir(output_dir: str | None, generated_path: str | Path) -> Path:
    """Uses the explicit output path if provided, otherwise tries to infer it."""
    if output_dir is not None:
        return Path(output_dir)
    return infer_default_output_dir(generated_path)


@dataclass
class SingleEvalResult:
    target: str | Path
    generated: str | Path
    metrics: dict[str, float]
    shape: tuple[int, int, int]
    single_case_csv: Path


@dataclass
class DatasetEvalResult:
    target_files: dict[str, Path]
    generated_files: dict[str, Path]
    per_image_rows: list[dict[str, object]]
    skipped_rows: list[dict[str, str]]
    output_dir: Path
    per_image_csv: Path
    summary_csv: Path
    skipped_csv: Path
    plot_paths: list[Path] = field(default_factory=list)


def run_single(args: argparse.Namespace) -> SingleEvalResult:
    """Runs the complete flow for the single mode."""
    from virtual_staining.evaluation.metrics import evaluate_pair

    sample_id = extract_single_sample_id(args.target, args.generated)
    metrics, shape = evaluate_pair(args.target, args.generated)

    output_dir = resolve_output_dir(args.output_dir, args.generated)
    individual_cases_dir = output_dir / "individual_cases"
    individual_cases_dir.mkdir(parents=True, exist_ok=True)

    row = build_metric_row(sample_id, args.target, args.generated, shape, metrics)
    single_case_csv = individual_cases_dir / f"{sample_id}_evaluation.csv"
    write_single_case_csv(row, single_case_csv)

    return SingleEvalResult(
        target=args.target,
        generated=args.generated,
        metrics=metrics,
        shape=shape,
        single_case_csv=single_case_csv,
    )


def run_dataset(args: argparse.Namespace) -> DatasetEvalResult:
    """Runs the complete flow for the dataset mode."""
    config = RunConfig.from_yaml(Path(args.config).resolve())
    eval_cfg = config.evaluation
    run_root = config.project.run_root
    paths = RunPaths(run_root)
    target_dir = (
        eval_cfg.target_dir if eval_cfg and eval_cfg.target_dir else config.project.dataset_test_dir
    )
    generated_dir = (
        eval_cfg.generated_dir if eval_cfg and eval_cfg.generated_dir else paths.output_test_dir
    )
    output_dir = (
        eval_cfg.output_dir if eval_cfg and eval_cfg.output_dir else run_root / "evaluation"
    )
    save_graphs = eval_cfg.save_graphs if eval_cfg else False

    target_files = collect_image_files(target_dir, "_target", "Target")
    generated_files = collect_image_files(generated_dir, "_target_generated", "Generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids = sorted(set(target_files) | set(generated_files))
    skipped_rows: list[dict[str, str]] = []
    paired_samples: list[tuple[Path, Path, str]] = []

    for sample_id in all_sample_ids:
        target_path = target_files.get(sample_id)
        generated_path = generated_files.get(sample_id)

        if target_path is None:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_target",
                    "target_path": "",
                    "generated_path": str(generated_path),
                }
            )
            continue

        if generated_path is None:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": "missing_generated",
                    "target_path": str(target_path),
                    "generated_path": "",
                }
            )
            continue

        paired_samples.append((target_path, generated_path, sample_id))

    evaluation = evaluate_pairs(paired_samples, output_dir)
    per_image_rows = evaluation.rows
    skipped_rows.extend(evaluation.skipped_rows)

    per_image_csv = output_dir / "per_image_metrics.csv"
    skipped_csv = output_dir / "skipped.csv"
    summary_csv = write_summary_csv(
        per_image_rows,
        output_dir,
        num_targets_found=len(target_files),
        num_generated_found=len(generated_files),
        num_pairs_evaluated=len(per_image_rows),
        num_skipped=len(skipped_rows),
    )
    write_skipped_csv(skipped_rows, skipped_csv)

    plot_paths: list[Path] = []
    if save_graphs and per_image_rows:
        plot_paths = save_dataset_plots(per_image_rows, output_dir)

    return DatasetEvalResult(
        target_files=target_files,
        generated_files=generated_files,
        per_image_rows=per_image_rows,
        skipped_rows=skipped_rows,
        output_dir=output_dir,
        per_image_csv=per_image_csv,
        summary_csv=summary_csv,
        skipped_csv=skipped_csv,
        plot_paths=plot_paths,
    )
