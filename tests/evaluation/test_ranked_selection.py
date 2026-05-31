from __future__ import annotations

from virtual_staining.evaluation.selection import RankedSample, select_ranked_samples


def _sample_ids(samples: list[RankedSample]) -> list[str]:
    return [sample.sample_id for sample in samples]


def test_ranked_selection_uses_higher_is_better_direction_and_sample_id_ties() -> None:
    rows = [
        {"sample_id": "c", "ssim": "0.90"},
        {"sample_id": "a", "ssim": "0.10"},
        {"sample_id": "b", "ssim": "0.90"},
    ]

    selected = select_ranked_samples(rows, "ssim", kinds=("best", "worst"), top_k=2)

    assert _sample_ids(selected["best"]) == ["b", "c"]
    assert _sample_ids(selected["worst"]) == ["a", "b"]
    assert [sample.rank for sample in selected["best"]] == [1, 2]


def test_ranked_selection_uses_lower_is_better_direction() -> None:
    rows = [
        {"sample_id": "low", "mae": "0.10"},
        {"sample_id": "mid", "mae": "0.50"},
        {"sample_id": "high", "mae": "0.90"},
    ]

    selected = select_ranked_samples(rows, "mae", kinds=("best", "worst"), top_k=2)

    assert _sample_ids(selected["best"]) == ["low", "mid"]
    assert _sample_ids(selected["worst"]) == ["high", "mid"]


def test_ranked_selection_median_band_is_deterministic() -> None:
    rows = [
        {"sample_id": "z_exact", "ssim": "0.50"},
        {"sample_id": "c_low", "ssim": "0.40"},
        {"sample_id": "a_low", "ssim": "0.40"},
        {"sample_id": "b_high", "ssim": "0.60"},
    ]

    selected = select_ranked_samples(
        rows,
        "ssim",
        kinds=("median",),
        top_k=4,
        median_value=0.50,
    )

    assert _sample_ids(selected["median"]) == ["z_exact", "a_low", "c_low", "b_high"]
    assert [sample.target_value for sample in selected["median"]] == [0.50] * 4


def test_ranked_selection_preserves_path_columns() -> None:
    rows = [
        {
            "sample_id": "case",
            "mae": "0.10",
            "source_path": "/tmp/source.png",
            "target_path": "/tmp/target.png",
            "generated_path": "/tmp/generated.png",
        }
    ]

    selected = select_ranked_samples(rows, "mae", kinds=("best",), top_k=1)

    assert selected["best"][0].row["source_path"] == "/tmp/source.png"
    assert selected["best"][0].row["target_path"] == "/tmp/target.png"
    assert selected["best"][0].row["generated_path"] == "/tmp/generated.png"
