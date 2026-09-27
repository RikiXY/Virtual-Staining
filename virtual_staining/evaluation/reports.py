"""Paired evaluation reports: per-image metric rows, coverage and the result metadata.

Per-image columns are generated from the resolved metric request. For each metric
``<m>`` the row carries ``<m>`` (the number: finite value, ``inf`` for positive infinity,
empty for undefined/unavailable), ``<m>_status`` (one of ``METRIC_STATUSES``) and
``<m>_reason``; with valid-region support also ``<m>_support_count`` and
``<m>_support_fraction``. ``evaluation_result.json`` next to the CSVs records the resolved
metric identities, their ranking directions and the coverage counts.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    METRIC_STATUSES,
    MetricDefinition,
    MetricResult,
    ResolvedMetric,
)

PER_IMAGE_METRICS_CSV = "per_image_metrics.csv"
COVERAGE_CSV = "coverage.csv"
EVALUATION_RESULT_JSON = "evaluation_result.json"
EVALUATION_RESULT_SCHEMA_VERSION = 1
COVERAGE_FIELDNAMES = [
    "sample_id",
    "set_id",
    "status",
    "reason",
    "detail",
    "target_path",
    "generated_path",
    "support_path",
]
_BASE_FIELDNAMES = [
    "sample_id",
    "set_id",
    "target_path",
    "generated_path",
    "width",
    "height",
    "channels",
]


def metric_fieldnames(metric_names: Sequence[str], *, support: bool) -> list[str]:
    fields = [*_BASE_FIELDNAMES, *(["support_path"] if support else [])]
    for name in metric_names:
        fields += [name, f"{name}_status", f"{name}_reason"]
        if support:
            fields += [f"{name}_support_count", f"{name}_support_fraction"]
    return fields


def build_metric_row(
    sample_id: str,
    target_path: str | Path,
    generated_path: str | Path,
    shape: tuple[int, int, int],
    results: Mapping[str, MetricResult],
    set_id: str,
    support_path: str | Path | None = None,
) -> dict[str, object]:
    height, width, channels = shape
    row: dict[str, object] = {
        "sample_id": sample_id,
        "set_id": set_id,
        "target_path": str(target_path),
        "generated_path": str(generated_path),
        "width": width,
        "height": height,
        "channels": channels,
    }
    if support_path is not None:
        row["support_path"] = str(support_path)
    for name, result in results.items():
        row[name] = "" if result.value is None else result.value
        row[f"{name}_status"] = result.status
        row[f"{name}_reason"] = result.reason or ""
        if support_path is not None:
            row[f"{name}_support_count"] = result.support_count
            row[f"{name}_support_fraction"] = result.support_fraction
    return row


def write_per_image_metrics_csv(
    rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str], output_path: str | Path
) -> Path:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_coverage_csv(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    """One row per requested sample: ``evaluated``, ``excluded`` or ``failed`` with a reason."""
    return write_per_image_metrics_csv(rows, COVERAGE_FIELDNAMES, output_path)


def write_evaluation_result(
    output_dir: Path,
    metrics: Sequence[ResolvedMetric],
    *,
    counts: Mapping[str, int],
    input_failures: str,
    valid_region_support: bool,
) -> Path:
    payload = {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "statuses": list(METRIC_STATUSES),
        "input_failures": input_failures,
        "valid_region_support": valid_region_support,
        "counts": dict(counts),
        "metrics": [metric.identity() for metric in metrics],
        "artifacts": {
            "per_image_metrics_csv": PER_IMAGE_METRICS_CSV,
            "summary_csv": "summary.csv",
            "coverage_csv": COVERAGE_CSV,
        },
    }
    path = output_dir / EVALUATION_RESULT_JSON
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class MetricInfo:
    """What a downstream report may know about one metric column."""

    name: str
    higher_is_better: bool | None
    thresholds: tuple[float, ...] = ()
    plot_range: tuple[float, float] | None = None

    @classmethod
    def from_definition(cls, definition: MetricDefinition) -> MetricInfo:
        return cls(
            definition.name,
            definition.higher_is_better,
            definition.thresholds,
            definition.plot_range,
        )


def recorded_metrics(csv_path: Path) -> tuple[MetricInfo, ...] | None:
    """Metrics recorded in ``evaluation_result.json`` beside ``csv_path``; None if absent."""
    path = csv_path.parent / EVALUATION_RESULT_JSON
    if not path.is_file():
        return None
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        EVALUATION_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported evaluation result metadata at {path}; expected schema_version "
            f"{EVALUATION_RESULT_SCHEMA_VERSION}"
        )
    infos: list[MetricInfo] = []
    for entry in payload["metrics"]:
        presentation = entry.get("presentation") or {}
        plot_range = presentation.get("plot_range")
        infos.append(
            MetricInfo(
                name=entry["name"],
                higher_is_better=entry["higher_is_better"],
                thresholds=tuple(presentation.get("thresholds") or ()),
                plot_range=(float(plot_range[0]), float(plot_range[1])) if plot_range else None,
            )
        )
    return tuple(infos)


def metric_info(csv_path: Path, metric: str) -> MetricInfo | None:
    """Recorded metadata for ``metric``, else the built-in definition, else None (unknown).

    A metric absent from recorded metadata is an error: the CSV's own result contract
    does not describe it.
    """
    recorded = recorded_metrics(csv_path)
    if recorded is not None:
        for info in recorded:
            if info.name == metric:
                return info
        raise ValueError(
            f"Metric {metric!r} is not recorded in {csv_path.parent / EVALUATION_RESULT_JSON}; "
            f"recorded: {[info.name for info in recorded]}"
        )
    definition = BUILTIN_METRIC_DEFINITIONS.get(metric)
    return MetricInfo.from_definition(definition) if definition is not None else None


def ranking_direction(csv_path: Path, metric: str, explicit: bool | None = None) -> bool:
    """Return whether higher is better; never guessed for an unknown metric."""
    if explicit is not None:
        return explicit
    info = metric_info(csv_path, metric)
    if info is None:
        raise ValueError(
            f"Ranking direction of {metric!r} is unknown: {csv_path} has no "
            f"{EVALUATION_RESULT_JSON} and {metric!r} is not a built-in metric. "
            "Pass the direction explicitly."
        )
    if info.higher_is_better is None:
        raise ValueError(f"Metric {metric!r} declares no ranking direction")
    return info.higher_is_better
