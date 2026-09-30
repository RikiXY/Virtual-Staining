from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from virtual_staining.applications.compare import CompareRequest, compare
from virtual_staining.evaluation.statistics import (
    UnpairedGroupStats,
    _choose_paired_better_label,
    align_paired_frames,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
    load_metric_values,
)

# ---------------------------------------------------------------------------
# compute_unpaired_group_stats
# ---------------------------------------------------------------------------


def test_unpaired_group_stats_mean_median() -> None:
    values = np.array([0.1, 0.5, 0.9])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.5], higher_is_better=True)
    assert stats.n == 3
    assert stats.mean == pytest.approx(np.mean(values))
    assert stats.median == pytest.approx(np.median(values))


def test_unpaired_group_stats_higher_is_better_share() -> None:
    values = np.array([0.8, 0.9, 0.6])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.75], higher_is_better=True)
    # 2 of 3 values >= 0.75
    assert stats.threshold_shares["ge_0.75"] == pytest.approx(2 / 3)


def test_unpaired_group_stats_lower_is_better_share() -> None:
    values = np.array([0.1, 0.2, 0.5])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.3], higher_is_better=False)
    # 2 of 3 values <= 0.3
    assert stats.threshold_shares["le_0.30"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compute_unpaired_comparison
# ---------------------------------------------------------------------------


def _make_groups(
    a_vals: list[float],
    b_vals: list[float],
    higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray, UnpairedGroupStats, UnpairedGroupStats]:
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    ga = compute_unpaired_group_stats(a, "A", thresholds=[0.5], higher_is_better=higher_is_better)
    gb = compute_unpaired_group_stats(b, "B", thresholds=[0.5], higher_is_better=higher_is_better)
    return a, b, ga, gb


def test_unpaired_comparison_favors_higher_group() -> None:
    a, b, ga, gb = _make_groups([0.5, 0.6, 0.55], [0.8, 0.85, 0.9], higher_is_better=True)
    comparison = compute_unpaired_comparison(a, b, ga, gb, higher_is_better=True)
    assert comparison.mean_favors == "B"
    assert comparison.median_favors == "B"


def test_unpaired_comparison_returns_statistics() -> None:
    a, b, ga, gb = _make_groups([0.4, 0.5, 0.6], [0.7, 0.8, 0.9], higher_is_better=True)
    comparison = compute_unpaired_comparison(a, b, ga, gb, higher_is_better=True)
    assert comparison.ks_statistic >= 0.0
    assert 0.0 <= comparison.ks_pvalue <= 1.0
    assert comparison.wasserstein_between_groups >= 0.0


# ---------------------------------------------------------------------------
# _choose_paired_better_label
# ---------------------------------------------------------------------------


def test_paired_better_label_positive_delta() -> None:
    assert _choose_paired_better_label(0.05, 0.04, 0.8, 0.2, "A", "B") == "B"


def test_paired_better_label_negative_delta() -> None:
    assert _choose_paired_better_label(-0.05, -0.04, 0.2, 0.8, "A", "B") == "A"


def test_paired_better_label_zero() -> None:
    assert _choose_paired_better_label(0.0, 0.0, 0.4, 0.4, "A", "B") == "tie"


def test_paired_better_label_uses_majority_of_signals() -> None:
    assert _choose_paired_better_label(-0.01, 0.03, 0.75, 0.25, "A", "B") == "B"


# ---------------------------------------------------------------------------
# compute_paired_summary
# ---------------------------------------------------------------------------


def _merged(a_vals: list[float], b_vals: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"value_a": a_vals, "value_b": b_vals})


def test_paired_summary_b_better() -> None:
    merged = _merged([0.5, 0.6, 0.7], [0.8, 0.9, 0.95])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    assert summary.better_label == "B"
    assert summary.share_b_better == pytest.approx(1.0)
    assert summary.share_a_better == pytest.approx(0.0)


def test_paired_summary_equal_within_tolerance() -> None:
    merged = _merged([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.01, higher_is_better=True)
    assert summary.share_equal == pytest.approx(1.0)
    assert summary.better_label == "tie"


def test_paired_summary_pair_count() -> None:
    merged = _merged([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    assert summary.n_pairs == 4


# ---------------------------------------------------------------------------
# align_paired_frames
# ---------------------------------------------------------------------------


def test_align_paired_frames_inner_join(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id,ssim\nimg1,0.8\nimg2,0.7\nimg3,0.6\n", encoding="utf-8")
    csv_b.write_text("sample_id,ssim\nimg1,0.9\nimg3,0.85\n", encoding="utf-8")  # img2 missing

    merged = align_paired_frames(csv_a, csv_b, "sample_id", "ssim")

    assert len(merged) == 2
    assert set(merged["sample_id"]) == {"img1", "img3"}


def test_align_paired_frames_raises_on_empty_join(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id,ssim\nimg1,0.8\n", encoding="utf-8")
    csv_b.write_text("sample_id,ssim\nimg2,0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No paired samples"):
        align_paired_frames(csv_a, csv_b, "sample_id", "ssim")


def _two_output_csv(path: Path, values: dict[tuple[str, str], float]) -> Path:
    lines = ["sample_id,output_name,ssim"]
    lines += [f"{sample},{output},{value}" for (sample, output), value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_align_paired_frames_uses_sample_and_output_identity(tmp_path: Path) -> None:
    csv_a = _two_output_csv(
        tmp_path / "a.csv",
        {("img1", "HE"): 0.8, ("img1", "PAS"): 0.1, ("img2", "HE"): 0.7, ("img2", "PAS"): 0.2},
    )
    csv_b = _two_output_csv(
        tmp_path / "b.csv",
        {("img2", "PAS"): 0.4, ("img1", "PAS"): 0.3, ("img1", "HE"): 0.9, ("img2", "HE"): 0.6},
    )

    merged = align_paired_frames(csv_a, csv_b, "sample_id", "ssim", output_name="PAS")

    # Row order differs between the CSVs; the composite key, not position, aligns them.
    assert set(merged["output_name"]) == {"PAS"}
    by_sample = merged.set_index("sample_id")
    assert (by_sample.loc["img1", "value_a"], by_sample.loc["img1", "value_b"]) == (0.1, 0.3)
    assert (by_sample.loc["img2", "value_a"], by_sample.loc["img2", "value_b"]) == (0.2, 0.4)


def test_comparisons_never_pool_several_outputs(tmp_path: Path) -> None:
    values = {("img1", "HE"): 0.8, ("img1", "PAS"): 0.1}
    csv_a = _two_output_csv(tmp_path / "a.csv", values)
    csv_b = _two_output_csv(tmp_path / "b.csv", values)

    with pytest.raises(ValueError, match="select one output_name"):
        align_paired_frames(csv_a, csv_b, "sample_id", "ssim")
    with pytest.raises(ValueError, match="select one output_name"):
        load_metric_values(csv_a, "ssim")
    with pytest.raises(ValueError, match="no rows for output 'IHC'"):
        load_metric_values(csv_a, "ssim", "IHC")
    assert list(load_metric_values(csv_a, "ssim", "HE")) == [0.8]


def test_align_paired_frames_rejects_duplicate_composite_keys(tmp_path: Path) -> None:
    csv_a = _two_output_csv(tmp_path / "a.csv", {("img1", "HE"): 0.8})
    csv_a.write_text(csv_a.read_text() + "img1,HE,0.7\n", encoding="utf-8")
    csv_b = _two_output_csv(tmp_path / "b.csv", {("img1", "HE"): 0.9})

    with pytest.raises(ValueError, match="duplicate"):
        align_paired_frames(csv_a, csv_b, "sample_id", "ssim", output_name="HE")


def test_compare_application_writes_one_output_comparison(tmp_path: Path) -> None:
    rows = {
        (f"img{i}", output): 0.1 * i + (0.5 if output == "HE" else 0.0)
        for i in range(4)
        for output in ("HE", "PAS")
    }
    csv_a = _two_output_csv(tmp_path / "a.csv", rows)
    csv_b = _two_output_csv(
        tmp_path / "b.csv",
        {key: value + 0.01 * int(key[0][-1]) for key, value in rows.items()},
    )
    request = CompareRequest(
        mode="paired", csv_a=csv_a, csv_b=csv_b, output_dir=tmp_path / "out", column="ssim"
    )

    with pytest.raises(ValueError, match="select one output_name"):
        compare(request)
    result = compare(replace(request, output_name="PAS"))

    assert result.paired_summary is not None and result.paired_summary.n_pairs == 4
    deltas = pd.read_csv(tmp_path / "out" / "paired_sample_deltas.csv")
    assert set(deltas["output_name"]) == {"PAS"}
