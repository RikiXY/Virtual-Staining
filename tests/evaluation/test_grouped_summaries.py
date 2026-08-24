from __future__ import annotations

import csv
from pathlib import Path

from virtual_staining.evaluation.summaries import write_grouped_summaries
from virtual_staining.metrics import DEFAULT_METRICS


def _row(set_id: str, value: float) -> dict[str, object]:
    return {"set_id": set_id, **{metric: value for metric in DEFAULT_METRICS}}


def test_grouped_summaries_average_patches_before_units(tmp_path: Path) -> None:
    rows = [_row("P1", 1.0), _row("P1", 3.0), _row("P2", 10.0)]
    sets = {
        "P1": {"patient_id": "PT1", "specimen_id": "SP1"},
        "P2": {"patient_id": "PT2", "specimen_id": "SP2"},
    }
    paths = write_grouped_summaries(
        rows,
        sets,
        tmp_path,
        bootstrap_iterations=100,
        bootstrap_seed=7,
    )
    assert {path.name for path in paths} == {
        "set_metrics.csv",
        "summary_set.csv",
        "specimen_metrics.csv",
        "summary_specimen.csv",
        "patient_metrics.csv",
        "summary_patient.csv",
    }
    with (tmp_path / "set_metrics.csv").open(newline="", encoding="utf-8") as handle:
        grouped = list(csv.DictReader(handle))
    assert float(grouped[0]["mae"]) == 2.0
    assert float(grouped[1]["mae"]) == 10.0


def test_grouped_summaries_skip_incomplete_biological_levels(tmp_path: Path) -> None:
    paths = write_grouped_summaries(
        [_row("P1", 1.0)],
        {"P1": {"patient_id": "", "specimen_id": ""}},
        tmp_path,
        bootstrap_iterations=0,
        bootstrap_seed=0,
    )
    assert [path.name for path in paths] == ["set_metrics.csv", "summary_set.csv"]
