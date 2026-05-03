from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from virtual_staining.evaluation.statistics import (
    compute_unpaired_group_stats,
    compute_unpaired_comparison,
    compute_paired_summary,
    align_paired_frames,
    choose_paired_better_label,
)


# ---------------------------------------------------------------------------
# compute_unpaired_group_stats
# ---------------------------------------------------------------------------

def test_unpaired_group_stats_mean_median():
    values = np.array([0.1, 0.5, 0.9])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.5], higher_is_better=True)
    assert stats.n == 3
    assert stats.mean == pytest.approx(np.mean(values))
    assert stats.median == pytest.approx(np.median(values))


def test_unpaired_group_stats_higher_is_better_share():
    values = np.array([0.8, 0.9, 0.6])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.75], higher_is_better=True)
    # 2 of 3 values >= 0.75
    assert stats.threshold_shares["ge_0.75"] == pytest.approx(2 / 3)


def test_unpaired_group_stats_lower_is_better_share():
    values = np.array([0.1, 0.2, 0.5])
    stats = compute_unpaired_group_stats(values, "A", thresholds=[0.3], higher_is_better=False)
    # 2 of 3 values <= 0.3
    assert stats.threshold_shares["le_0.30"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compute_unpaired_comparison
# ---------------------------------------------------------------------------

def _make_groups(a_vals, b_vals, higher_is_better=True):
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    ga = compute_unpaired_group_stats(a, "A", thresholds=[0.5], higher_is_better=higher_is_better)
    gb = compute_unpaired_group_stats(b, "B", thresholds=[0.5], higher_is_better=higher_is_better)
    return a, b, ga, gb


def test_unpaired_comparison_favors_higher_group():
    a, b, ga, gb = _make_groups([0.5, 0.6, 0.55], [0.8, 0.85, 0.9], higher_is_better=True)
    cmp = compute_unpaired_comparison(a, b, ga, gb, higher_is_better=True)
    assert cmp.mean_favors == "B"
    assert cmp.median_favors == "B"


def test_unpaired_comparison_returns_statistics():
    a, b, ga, gb = _make_groups([0.4, 0.5, 0.6], [0.7, 0.8, 0.9], higher_is_better=True)
    cmp = compute_unpaired_comparison(a, b, ga, gb, higher_is_better=True)
    assert cmp.ks_statistic >= 0.0
    assert 0.0 <= cmp.ks_pvalue <= 1.0
    assert cmp.wasserstein_between_groups >= 0.0


# ---------------------------------------------------------------------------
# choose_paired_better_label
# ---------------------------------------------------------------------------

def test_paired_better_label_positive_delta():
    assert choose_paired_better_label(0.05, "A", "B") == "B"


def test_paired_better_label_negative_delta():
    assert choose_paired_better_label(-0.05, "A", "B") == "A"


def test_paired_better_label_zero():
    assert choose_paired_better_label(0.0, "A", "B") == "tie"


# ---------------------------------------------------------------------------
# compute_paired_summary
# ---------------------------------------------------------------------------

def _merged(a_vals, b_vals):
    return pd.DataFrame({"value_a": a_vals, "value_b": b_vals})


def test_paired_summary_b_better():
    merged = _merged([0.5, 0.6, 0.7], [0.8, 0.9, 0.95])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    assert summary.better_label == "B"
    assert summary.share_b_better == pytest.approx(1.0)
    assert summary.share_a_better == pytest.approx(0.0)


def test_paired_summary_equal_within_tolerance():
    merged = _merged([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.01, higher_is_better=True)
    assert summary.share_equal == pytest.approx(1.0)
    assert summary.better_label == "tie"


def test_paired_summary_pair_count():
    merged = _merged([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
    summary = compute_paired_summary(merged, "A", "B", tolerance=0.0, higher_is_better=True)
    assert summary.n_pairs == 4


# ---------------------------------------------------------------------------
# align_paired_frames
# ---------------------------------------------------------------------------

def test_align_paired_frames_inner_join(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id,ssim\nimg1,0.8\nimg2,0.7\nimg3,0.6\n")
    csv_b.write_text("sample_id,ssim\nimg1,0.9\nimg3,0.85\n")  # img2 missing

    merged = align_paired_frames(csv_a, csv_b, "sample_id", "ssim")

    assert len(merged) == 2
    assert set(merged["sample_id"]) == {"img1", "img3"}


def test_align_paired_frames_raises_on_empty_join(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id,ssim\nimg1,0.8\n")
    csv_b.write_text("sample_id,ssim\nimg2,0.9\n")

    with pytest.raises(ValueError, match="No paired samples"):
        align_paired_frames(csv_a, csv_b, "sample_id", "ssim")
