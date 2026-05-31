from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from virtual_staining.utils.metrics import is_higher_better_metric

SelectionKind: TypeAlias = Literal["best", "median", "worst"]
SUPPORTED_SELECTION_KINDS: tuple[SelectionKind, ...] = ("best", "median", "worst")


@dataclass(frozen=True)
class RankedSample:
    """A metric-ranked sample row plus selection metadata."""

    metric: str
    kind: SelectionKind
    rank: int
    sample_id: str
    metric_value: float
    target_value: float
    row: dict[str, Any]


def _metric_value(row: Mapping[str, object], metric: str) -> float | None:
    if metric not in row:
        return None
    value = row[metric]
    if value is None or value == "":
        return None
    if isinstance(value, str | int | float):
        try:
            metric_value = float(value)
        except ValueError:
            return None
        return None if math.isnan(metric_value) else metric_value
    raise TypeError(f"Metric '{metric}' must be a scalar value, got {type(value).__name__}.")


def _sample_id(row: Mapping[str, object], index: int) -> str:
    sample_id = row.get("sample_id")
    if sample_id is None or sample_id == "":
        return f"row_{index:06d}"
    return str(sample_id)


def _validate_kinds(kinds: Sequence[str]) -> tuple[SelectionKind, ...]:
    invalid_kinds = sorted(set(kinds) - set(SUPPORTED_SELECTION_KINDS))
    if invalid_kinds:
        raise ValueError(f"Unsupported selection kinds: {', '.join(invalid_kinds)}")
    return tuple(cast(SelectionKind, kind) for kind in kinds)


def _rankable_rows(
    rows: Sequence[Mapping[str, object]],
    metric: str,
) -> list[tuple[float, str, dict[str, Any]]]:
    rankable: list[tuple[float, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        metric_value = _metric_value(row, metric)
        if metric_value is None:
            continue
        rankable.append((metric_value, _sample_id(row, index), dict(row)))
    return rankable


def select_ranked_samples(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    kinds: Sequence[str] = SUPPORTED_SELECTION_KINDS,
    top_k: int = 1,
    median_value: float | None = None,
    finite_only: bool = False,
) -> dict[SelectionKind, list[RankedSample]]:
    """Select ranked best, median-band, and worst samples for one metric.

    The selector is deliberately independent from plotting and file placement.
    Rows are sorted by metric direction, with ``sample_id`` used as the final
    deterministic tie-breaker when available.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    resolved_kinds = _validate_kinds(kinds)
    higher_is_better = is_higher_better_metric(metric)
    rankable = _rankable_rows(rows, metric)
    if finite_only:
        rankable = [item for item in rankable if math.isfinite(item[0])]

    if not rankable:
        return {kind: [] for kind in resolved_kinds}

    values = [metric_value for metric_value, _, _ in rankable]
    target_values: dict[SelectionKind, float] = {
        "best": max(values) if higher_is_better else min(values),
        "worst": min(values) if higher_is_better else max(values),
        "median": median_value if median_value is not None else float(statistics.median(values)),
    }
    selected: dict[SelectionKind, list[RankedSample]] = {}

    for kind in resolved_kinds:
        if kind == "best":
            ordered = sorted(
                rankable,
                key=lambda item: (
                    -item[0] if higher_is_better else item[0],
                    item[1],
                ),
            )
        elif kind == "worst":
            ordered = sorted(
                rankable,
                key=lambda item: (
                    item[0] if higher_is_better else -item[0],
                    item[1],
                ),
            )
        elif kind == "median":
            target = target_values["median"]
            ordered = sorted(
                rankable,
                key=lambda item: (
                    abs(item[0] - target),
                    item[0],
                    item[1],
                ),
            )
        else:
            raise ValueError(f"Unsupported selection kind: {kind}")

        selected[kind] = [
            RankedSample(
                metric=metric,
                kind=kind,
                rank=rank,
                sample_id=sample_id,
                metric_value=metric_value,
                target_value=target_values[kind],
                row=row,
            )
            for rank, (metric_value, sample_id, row) in enumerate(ordered[:top_k], start=1)
        ]

    return selected
