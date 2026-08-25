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
from virtual_staining.experiment.run_layout import RunLayout
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
    try:
        return RunLayout.from_artifact_path(Path(generated_path)).evaluation_dir
    except ValueError:
        raise ValueError(
            "Could not infer output directory from generated path. "
            "Please provide --output-dir explicitly."
        ) from None


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

    row = build_metric_row(
        request.sample_id,
        target_path,
        generated_path,
        shape,
        metrics,
        set_id=request.sample_id,
    )
    single_case_csv = individual_cases_dir / f"{request.sample_id}_evaluation.csv"
    write_single_case_csv(row, single_case_csv)

    return SingleEvalResult(
        target=target_path,
        generated=generated_path,
        metrics=metrics,
        shape=shape,
        single_case_csv=single_case_csv,
    )
