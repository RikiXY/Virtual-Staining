from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from virtual_staining.evaluation.io import (
    collect_image_files,
    extract_single_sample_id,
)
from virtual_staining.evaluation.reports import (
    build_metric_row,
    write_single_case_csv,
)
from virtual_staining.evaluation.summaries import metric_value
from virtual_staining.metrics import DEFAULT_METRICS

__all__ = [
    "DEFAULT_METRICS",
    "SingleEvalResult",
    "evaluate_pair",
    "metric_value",
]


@dataclass(frozen=True)
class _EvaluateRequest:
    target_dir: Path
    generated_dir: Path
    output_dir: Path
    sample_id: str


@dataclass
class SingleEvalResult:
    target: str | Path
    generated: str | Path
    metrics: dict[str, float]
    shape: tuple[int, int, int]
    single_case_csv: Path


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
    from virtual_staining.evaluation.evaluator import evaluate_pair

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
