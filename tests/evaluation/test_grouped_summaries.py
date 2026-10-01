from __future__ import annotations

import csv
from pathlib import Path

from virtual_staining.evaluation.reports import build_metric_row
from virtual_staining.evaluation.summaries import write_grouped_summaries
from virtual_staining.metrics import MetricResult


def _row(
    set_id: str, value: float, output: str = "HE", **overrides: MetricResult
) -> dict[str, object]:
    results = {"mae": MetricResult.of(value), "custom": MetricResult.of(value), **overrides}
    return build_metric_row("s", output, "t.png", "g.png", (8, 8, 3), results, set_id=set_id)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_grouped_summaries_average_finite_patches_of_the_requested_metrics(
    tmp_path: Path,
) -> None:
    rows = [
        _row("P1", 1.0),
        _row("P1", 3.0, custom=MetricResult.undefined("no value")),
        _row("P2", 10.0),
    ]
    sets = {
        "P1": {"patient_id": "PT1", "specimen_id": "SP1"},
        "P2": {"patient_id": "PT2", "specimen_id": "SP2"},
    }
    paths = write_grouped_summaries(
        rows, ["mae", "custom"], sets, tmp_path, bootstrap_iterations=100, bootstrap_seed=7
    )

    assert {path.name for path in paths} == {
        "set_metrics.csv",
        "summary_set.csv",
        "specimen_metrics.csv",
        "summary_specimen.csv",
        "patient_metrics.csv",
        "summary_patient.csv",
    }
    grouped = _read(tmp_path / "set_metrics.csv")
    assert list(grouped[0]) == [
        "output_name",
        "unit",
        "group_id",
        "patch_count",
        "mae_finite_count",
        "mae_finite_mean",
        "custom_finite_count",
        "custom_finite_mean",
    ]
    assert (float(grouped[0]["mae_finite_mean"]), float(grouped[1]["mae_finite_mean"])) == (
        2.0,
        10.0,
    )
    assert (grouped[0]["custom_finite_count"], grouped[0]["custom_finite_mean"]) == ("1", "1.0")
    summary = _read(tmp_path / "summary_patient.csv")
    assert [(row["resampling_unit"], row["metric"]) for row in summary] == [
        ("patient", "mae"),
        ("patient", "custom"),
    ]
    assert summary[0]["group_count"] == "2"
    assert float(summary[0]["finite_mean"]) == 6.0


def test_grouped_summary_ci_does_not_depend_on_other_requested_metrics(tmp_path: Path) -> None:
    rows = [_row("P1", 1.0), _row("P2", 3.0), _row("P3", 8.0)]
    sets = {name: {"patient_id": name, "specimen_id": name} for name in ("P1", "P2", "P3")}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    write_grouped_summaries(
        rows, ["mae"], sets, tmp_path / "a", bootstrap_iterations=50, bootstrap_seed=1
    )
    write_grouped_summaries(
        rows, ["custom", "mae"], sets, tmp_path / "b", bootstrap_iterations=50, bootstrap_seed=1
    )

    only = _read(tmp_path / "a" / "summary_set.csv")[0]
    both = [row for row in _read(tmp_path / "b" / "summary_set.csv") if row["metric"] == "mae"]
    assert (only["ci95_low"], only["ci95_high"]) == (both[0]["ci95_low"], both[0]["ci95_high"])


def test_grouped_summaries_skip_incomplete_biological_levels(tmp_path: Path) -> None:
    paths = write_grouped_summaries(
        [_row("P1", 1.0)],
        ["mae"],
        {"P1": {"patient_id": "", "specimen_id": ""}},
        tmp_path,
        bootstrap_iterations=0,
        bootstrap_seed=0,
    )
    assert [path.name for path in paths] == ["set_metrics.csv", "summary_set.csv"]


def test_grouped_summaries_keep_output_identity(tmp_path: Path) -> None:
    rows = [
        _row("P1", 1.0, "PAS"),
        _row("P1", 5.0, "HE"),
        _row("P2", 3.0, "PAS"),
        _row("P2", 7.0, "HE"),
    ]
    sets = {"P1": {"patient_id": "PT1"}, "P2": {"patient_id": "PT2"}}

    write_grouped_summaries(
        rows, ["mae"], sets, tmp_path, bootstrap_iterations=50, bootstrap_seed=0
    )

    grouped = _read(tmp_path / "set_metrics.csv")
    assert [(row["output_name"], row["group_id"], row["mae_finite_mean"]) for row in grouped] == [
        ("PAS", "P1", "1.0"),
        ("PAS", "P2", "3.0"),
        ("HE", "P1", "5.0"),
        ("HE", "P2", "7.0"),
    ]
    summary = _read(tmp_path / "summary_patient.csv")
    assert [(row["output_name"], row["finite_mean"]) for row in summary] == [
        ("PAS", "2.0"),
        ("HE", "6.0"),
    ]
