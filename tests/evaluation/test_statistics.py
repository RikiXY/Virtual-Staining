from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from virtual_staining.applications.compare import CompareRequest, compare
from virtual_staining.evaluation.statistics import (
    UnpairedGroupStats,
    align_paired_frames,
    align_paired_metric_frames,
    build_paired_decision_breakdown_rows,
    build_paired_delta_summary_row,
    build_paired_multi_metric_delta_reports,
    build_unpaired_quantile_comparison_rows,
    choose_paired_better_label,
    classify_decision_strength,
    common_language_effect_size,
    compute_paired_summary,
    compute_unpaired_comparison,
    compute_unpaired_group_stats,
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


def test_common_language_effect_size_favors_expected_run() -> None:
    assert common_language_effect_size(
        np.array([0.1, 0.2]),
        np.array([0.8, 0.9]),
        higher_is_better=True,
    ) == pytest.approx(1.0)
    assert common_language_effect_size(
        np.array([0.8, 0.9]),
        np.array([0.1, 0.2]),
        higher_is_better=False,
    ) == pytest.approx(1.0)
    assert common_language_effect_size(
        np.array([0.5]),
        np.array([0.5]),
        higher_is_better=True,
    ) == pytest.approx(0.5)


def test_unpaired_quantile_comparison_respects_metric_direction() -> None:
    higher_rows = build_unpaired_quantile_comparison_rows(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.6, 0.7, 0.8]),
        label_a="A",
        label_b="B",
        higher_is_better=True,
    )
    lower_rows = build_unpaired_quantile_comparison_rows(
        np.array([0.6, 0.7, 0.8]),
        np.array([0.1, 0.2, 0.3]),
        label_a="A",
        label_b="B",
        higher_is_better=False,
    )

    assert [row["quantile"] for row in higher_rows] == ["q10", "q25", "q50", "q75", "q90"]
    assert all(
        isinstance(row["signed_delta"], int | float) and row["signed_delta"] > 0
        for row in higher_rows
    )
    assert all(row["favors"] == "B" for row in higher_rows)
    assert all(
        isinstance(row["signed_delta"], int | float) and row["signed_delta"] > 0
        for row in lower_rows
    )
    assert all(row["favors"] == "B" for row in lower_rows)


def test_decision_strength_classification() -> None:
    assert classify_decision_strength(0.0, 5.0) == "tie"
    assert classify_decision_strength(1.0, 5.0) == "weak"
    assert classify_decision_strength(2.0, 5.0) == "moderate"
    assert classify_decision_strength(4.0, 5.0) == "strong"


def test_unpaired_decision_scoring_reports_strength_and_counter_signals() -> None:
    a, b, ga, gb = _make_groups(
        [0.80, 0.82, 0.84, 0.86, 0.88],
        [0.70, 0.90, 0.91, 0.92, 0.93],
        higher_is_better=True,
    )
    comparison = compute_unpaired_comparison(a, b, ga, gb, higher_is_better=True)

    assert comparison.better_label == "B"
    assert comparison.decision_strength == "moderate"
    assert comparison.score_b > comparison.score_a
    assert comparison.common_language_b_better > 0.5
    assert ga.decision_counter_count > 0
    assert "score-based advantage" in comparison.decision_reason


# ---------------------------------------------------------------------------
# choose_paired_better_label
# ---------------------------------------------------------------------------


def test_paired_better_label_positive_delta() -> None:
    assert choose_paired_better_label(0.05, 0.04, 0.8, 0.2, "A", "B") == "B"


def test_paired_better_label_negative_delta() -> None:
    assert choose_paired_better_label(-0.05, -0.04, 0.2, 0.8, "A", "B") == "A"


def test_paired_better_label_zero() -> None:
    assert choose_paired_better_label(0.0, 0.0, 0.4, 0.4, "A", "B") == "tie"


def test_paired_better_label_uses_majority_of_signals() -> None:
    assert choose_paired_better_label(-0.01, 0.03, 0.75, 0.25, "A", "B") == "B"


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


def test_paired_delta_summary_includes_quantiles_and_relative_deltas() -> None:
    merged = _merged([1.0, 2.0, 4.0, 8.0, 16.0], [2.0, 2.0, 6.0, 7.0, 20.0])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    signed_delta = np.array([1.0, 0.0, 2.0, -1.0, 4.0])

    assert summary.signed_delta_q10 == pytest.approx(np.quantile(signed_delta, 0.10))
    assert summary.signed_delta_q25 == pytest.approx(np.quantile(signed_delta, 0.25))
    assert summary.signed_delta_q50 == pytest.approx(np.quantile(signed_delta, 0.50))
    assert summary.signed_delta_q75 == pytest.approx(np.quantile(signed_delta, 0.75))
    assert summary.signed_delta_q90 == pytest.approx(np.quantile(signed_delta, 0.90))
    assert summary.relative_delta_count == 5

    row = build_paired_delta_summary_row(summary, metric="ssim", higher_is_better=True)
    assert row["signed_delta_q10"] == pytest.approx(summary.signed_delta_q10)
    assert row["decision_strength"] == summary.decision_strength


def test_paired_decision_breakdown_rows_are_deterministic() -> None:
    merged = _merged([0.5, 0.6, 0.7], [0.8, 0.7, 0.9])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    rows = build_paired_decision_breakdown_rows(summary)

    assert [row["criterion"] for row in rows] == [
        "mean_signed_delta",
        "median_signed_delta",
        "share_improved",
        "q25_signed_delta",
        "worst_tail_q10_signed_delta",
    ]
    assert all(row["mode"] == "paired" for row in rows)


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


def test_align_paired_metric_frames_aligns_by_sample_id(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text(
        "sample_id,ssim,mae\nimg1,0.8,0.10\nimg2,0.7,0.20\nimg3,0.6,0.30\n",
        encoding="utf-8",
    )
    csv_b.write_text(
        "sample_id,ssim,mae\nimg3,0.7,0.25\nimg1,0.9,0.05\n",
        encoding="utf-8",
    )

    merged = align_paired_metric_frames(csv_a, csv_b, "sample_id", ("ssim", "mae"))

    assert list(merged["sample_id"]) == ["img1", "img3"]
    assert list(merged["ssim_a"]) == [0.8, 0.6]
    assert list(merged["ssim_b"]) == [0.9, 0.7]
    assert list(merged["mae_a"]) == [0.10, 0.30]
    assert list(merged["mae_b"]) == [0.05, 0.25]


def test_align_paired_metric_frames_raises_for_missing_metric(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id,ssim\nimg1,0.8\n", encoding="utf-8")
    csv_b.write_text("sample_id,ssim\nimg1,0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Metric columns not found"):
        align_paired_metric_frames(csv_a, csv_b, "sample_id", ("ssim", "mae"))


def test_build_paired_multi_metric_delta_reports_respects_metric_direction() -> None:
    merged = pd.DataFrame(
        {
            "sample_id": ["better", "worse", "equal"],
            "ssim_a": [0.70, 0.80, 0.90],
            "ssim_b": [0.80, 0.70, 0.9005],
            "mae_a": [0.20, 0.10, 0.30],
            "mae_b": [0.10, 0.20, 0.3005],
        }
    )

    samples, summary = build_paired_multi_metric_delta_reports(
        merged,
        ("ssim", "mae"),
        sample_id_column="sample_id",
        label_a="A",
        label_b="B",
        tolerance=0.001,
    )

    assert list(samples["sample_id"]) == ["better", "worse", "equal"]
    assert samples.loc[0, "ssim_raw_delta_b_minus_a"] == pytest.approx(0.10)
    assert samples.loc[0, "ssim_signed_delta"] == pytest.approx(0.10)
    assert samples.loc[0, "ssim_winner"] == "B"
    assert samples.loc[1, "ssim_winner"] == "A"
    assert samples.loc[2, "ssim_winner"] == "equal"

    assert samples.loc[0, "mae_raw_delta_b_minus_a"] == pytest.approx(-0.10)
    assert samples.loc[0, "mae_signed_delta"] == pytest.approx(0.10)
    assert samples.loc[0, "mae_winner"] == "B"
    assert samples.loc[1, "mae_winner"] == "A"
    assert samples.loc[2, "mae_winner"] == "equal"

    by_metric = {row["metric"]: row for row in summary.to_dict("records")}
    assert by_metric["ssim"]["improved_count"] == 1
    assert by_metric["ssim"]["worsened_count"] == 1
    assert by_metric["ssim"]["equal_count"] == 1
    assert by_metric["mae"]["direction"] == "lower_is_better"
    assert by_metric["mae"]["improved_share"] == pytest.approx(1 / 3)


def test_paired_compare_writes_multi_metric_delta_reports(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    output_dir = tmp_path / "comparison"
    csv_a.write_text(
        "sample_id,ssim,mae\nimg1,0.70,0.20\nimg2,0.80,0.10\n",
        encoding="utf-8",
    )
    csv_b.write_text(
        "sample_id,ssim,mae\nimg1,0.80,0.10\nimg2,0.70,0.20\n",
        encoding="utf-8",
    )

    result = compare(
        CompareRequest(
            mode="paired",
            csv_a=csv_a,
            csv_b=csv_b,
            label_a="A",
            label_b="B",
            column="all",
            output_dir=output_dir,
            higher_is_better=True,
            bins=30,
            min_value=0.0,
            max_value=1.0,
            thresholds=(),
            tolerance=0.0,
            sample_id_column="sample_id",
            metrics=("ssim", "mae"),
        )
    )

    assert result.paired_sample_deltas_all_metrics_csv == (
        output_dir / "paired_sample_deltas_all_metrics.csv"
    )
    assert result.paired_metric_delta_summary_csv == (
        output_dir / "paired_metric_delta_summary.csv"
    )
    samples = pd.read_csv(output_dir / "paired_sample_deltas_all_metrics.csv")
    summary = pd.read_csv(output_dir / "paired_metric_delta_summary.csv")

    assert list(samples["sample_id"]) == ["img1", "img2"]
    assert samples.loc[0, "ssim_winner"] == "B"
    assert samples.loc[0, "mae_winner"] == "B"
    assert set(summary["metric"]) == {"ssim", "mae"}


def test_unpaired_compare_writes_decision_reports(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    output_dir = tmp_path / "unpaired"
    csv_a.write_text(
        "sample_id,ssim\nimg1,0.70\nimg2,0.75\nimg3,0.80\n",
        encoding="utf-8",
    )
    csv_b.write_text(
        "sample_id,ssim\nimg4,0.85\nimg5,0.90\nimg6,0.95\n",
        encoding="utf-8",
    )

    result = compare(
        CompareRequest(
            mode="unpaired",
            csv_a=csv_a,
            csv_b=csv_b,
            label_a="A",
            label_b="B",
            column="ssim",
            output_dir=output_dir,
            higher_is_better=True,
            bins=30,
            min_value=0.0,
            max_value=1.0,
            thresholds=(0.80,),
            tolerance=0.0,
            sample_id_column="sample_id",
        )
    )

    assert result.unpaired_decision_breakdown_csv == (
        output_dir / "unpaired_decision_breakdown.csv"
    )
    assert result.unpaired_quantile_comparison_csv == (
        output_dir / "unpaired_quantile_comparison.csv"
    )
    assert result.unpaired_threshold_shares_csv == (output_dir / "unpaired_threshold_shares.csv")
    assert (output_dir / "comparison_summary.csv").is_file()

    decision_rows = pd.read_csv(output_dir / "unpaired_decision_breakdown.csv")
    threshold_rows = pd.read_csv(output_dir / "unpaired_threshold_shares.csv")
    quantile_rows = pd.read_csv(output_dir / "unpaired_quantile_comparison.csv")
    summary_json = (output_dir / "summary.json").read_text(encoding="utf-8")
    report_text = (output_dir / "report.txt").read_text(encoding="utf-8")

    assert list(decision_rows["criterion"]) == [
        "mean",
        "median",
        "worst_tail_q10",
        "threshold_mean",
        "common_language_effect",
    ]
    assert list(threshold_rows["threshold_name"]) == ["ge_0.80"]
    assert list(quantile_rows["quantile"]) == ["q10", "q25", "q50", "q75", "q90"]
    assert "decision_strength" in summary_json
    assert "Score-based suggestion" in report_text


def test_paired_compare_writes_decision_reports(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    output_dir = tmp_path / "paired"
    csv_a.write_text(
        "sample_id,mae\nimg1,0.20\nimg2,0.30\nimg3,0.40\n",
        encoding="utf-8",
    )
    csv_b.write_text(
        "sample_id,mae\nimg1,0.10\nimg2,0.20\nimg3,0.50\n",
        encoding="utf-8",
    )

    result = compare(
        CompareRequest(
            mode="paired",
            csv_a=csv_a,
            csv_b=csv_b,
            label_a="A",
            label_b="B",
            column="mae",
            output_dir=output_dir,
            higher_is_better=False,
            bins=30,
            min_value=0.0,
            max_value=1.0,
            thresholds=(0.08,),
            tolerance=0.0,
            sample_id_column="sample_id",
        )
    )

    assert result.paired_decision_breakdown_csv == output_dir / "paired_decision_breakdown.csv"
    assert result.paired_delta_summary_csv == output_dir / "paired_delta_summary.csv"
    assert (output_dir / "comparison_summary.csv").is_file()

    decision_rows = pd.read_csv(output_dir / "paired_decision_breakdown.csv")
    delta_summary = pd.read_csv(output_dir / "paired_delta_summary.csv")
    report_text = (output_dir / "report.txt").read_text(encoding="utf-8")

    assert list(decision_rows["criterion"]) == [
        "mean_signed_delta",
        "median_signed_delta",
        "share_improved",
        "q25_signed_delta",
        "worst_tail_q10_signed_delta",
    ]
    assert delta_summary.loc[0, "direction"] == "lower_is_better"
    assert "signed_delta_q10" in delta_summary.columns
    assert "Score-based suggestion" in report_text
