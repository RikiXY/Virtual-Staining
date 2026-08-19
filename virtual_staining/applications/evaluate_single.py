from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from virtual_staining.config.run import RunConfig
from virtual_staining.evaluation.evaluator import evaluate_pairs
from virtual_staining.evaluation.io import (
    build_evaluation_pairs,
    collect_image_files,
    extract_single_sample_id,
)
from virtual_staining.evaluation.plotting import save_dataset_plots
from virtual_staining.evaluation.reports import (
    build_metric_row,
    write_single_case_csv,
    write_skipped_csv,
)
from virtual_staining.evaluation.summaries import metric_value, write_summary_csv
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.utils.metrics import DEFAULT_METRICS

__all__ = [
    "DEFAULT_METRICS",
    "DatasetEvalResult",
    "SingleEvalResult",
    "evaluate_dataset",
    "evaluate_pair",
    "metric_value",
]


@dataclass(frozen=True)
class _EvaluateRequest:
    target_dir: Path
    generated_dir: Path
    output_dir: Path
    sample_id: str | None  # None → dataset mode
    save_graphs: bool = False


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


def evaluate_pair(
    target_path: Path,
    generated_path: Path,
    output_dir: Path | None = None,
) -> SingleEvalResult:
    """Evaluate one user-selected target/generated image pair."""
    return _run_single(
        _EvaluateRequest(
            target_dir=target_path.parent,
            generated_dir=generated_path.parent,
            output_dir=output_dir or _infer_default_output_dir(generated_path),
            sample_id=extract_single_sample_id(target_path, generated_path),
        )
    )


def evaluate_dataset(config_path: Path) -> DatasetEvalResult:
    """Evaluate the generated dataset selected by a run config."""
    config = RunConfig.from_yaml(config_path.resolve())
    eval_cfg = config.evaluation
    paths = RunPaths(config.project.run_root)
    return _run_dataset(
        _EvaluateRequest(
            target_dir=(
                eval_cfg.target_dir
                if eval_cfg and eval_cfg.target_dir
                else config.project.split_dir("test")
            ),
            generated_dir=(
                eval_cfg.generated_dir
                if eval_cfg and eval_cfg.generated_dir
                else paths.output_test_dir
            ),
            output_dir=(
                eval_cfg.output_dir
                if eval_cfg and eval_cfg.output_dir
                else config.project.run_root / "evaluation"
            ),
            sample_id=None,
            save_graphs=eval_cfg.save_graphs if eval_cfg else False,
        )
    )


def _infer_default_output_dir(generated_path: str | Path) -> Path:
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


def _run_single(request: _EvaluateRequest) -> SingleEvalResult:
    from virtual_staining.evaluation.metrics import evaluate_pair

    assert request.sample_id is not None
    target_files = collect_image_files(request.target_dir, "_target", "Target")
    generated_files = collect_image_files(request.generated_dir, "_target_generated", "Generated")

    if request.sample_id not in target_files:
        raise ValueError(
            f"Sample '{request.sample_id}' not found in target dir {request.target_dir}"
        )
    if request.sample_id not in generated_files:
        raise ValueError(
            f"Sample '{request.sample_id}' not found in generated dir {request.generated_dir}"
        )

    target_path = target_files[request.sample_id]
    generated_path = generated_files[request.sample_id]
    metrics, shape = evaluate_pair(target_path, generated_path)

    individual_cases_dir = request.output_dir / "individual_cases"
    individual_cases_dir.mkdir(parents=True, exist_ok=True)

    row = build_metric_row(request.sample_id, target_path, generated_path, shape, metrics)
    single_case_csv = individual_cases_dir / f"{request.sample_id}_evaluation.csv"
    write_single_case_csv(row, single_case_csv)

    return SingleEvalResult(
        target=target_path,
        generated=generated_path,
        metrics=metrics,
        shape=shape,
        single_case_csv=single_case_csv,
    )


def _run_dataset(request: _EvaluateRequest) -> DatasetEvalResult:
    target_files = collect_image_files(request.target_dir, "_target", "Target")
    generated_files = collect_image_files(request.generated_dir, "_target_generated", "Generated")
    request.output_dir.mkdir(parents=True, exist_ok=True)

    paired_samples, skipped_ids = build_evaluation_pairs(request.target_dir, request.generated_dir)
    skipped_rows: list[dict[str, str]] = [
        {
            "sample_id": sid,
            "reason": "missing_target" if sid not in target_files else "missing_generated",
            "target_path": str(target_files.get(sid, "")),
            "generated_path": str(generated_files.get(sid, "")),
        }
        for sid in skipped_ids
    ]

    evaluation = evaluate_pairs(paired_samples, request.output_dir)
    per_image_rows = evaluation.rows
    skipped_rows.extend(evaluation.skipped_rows)

    per_image_csv = request.output_dir / "per_image_metrics.csv"
    skipped_csv = request.output_dir / "skipped.csv"
    summary_csv = write_summary_csv(
        per_image_rows,
        request.output_dir,
        num_targets_found=len(target_files),
        num_generated_found=len(generated_files),
        num_pairs_evaluated=len(per_image_rows),
        num_skipped=len(skipped_rows),
    )
    write_skipped_csv(skipped_rows, skipped_csv)

    plot_paths: list[Path] = []
    if request.save_graphs and per_image_rows:
        plot_paths = save_dataset_plots(per_image_rows, request.output_dir)

    return DatasetEvalResult(
        target_files=target_files,
        generated_files=generated_files,
        per_image_rows=per_image_rows,
        skipped_rows=skipped_rows,
        output_dir=request.output_dir,
        per_image_csv=per_image_csv,
        summary_csv=summary_csv,
        skipped_csv=skipped_csv,
        plot_paths=plot_paths,
    )
