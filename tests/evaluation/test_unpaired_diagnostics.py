from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.image_helpers import write_rgb_image
from virtual_staining.evaluation.unpaired import (
    COMPARISON_FIELDS,
    FEATURE_NAMES,
    compare_feature_distributions,
    evaluate_unpaired_collections,
    image_features,
)


def _write_array(path: Path, pixels: list[list[tuple[int, int, int]]]) -> Path:
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)
    return path


def test_image_features_match_known_values(tmp_path: Path) -> None:
    path = _write_array(
        tmp_path / "x.png", [[(0, 255, 0), (255, 255, 0)], [(0, 255, 0), (255, 255, 0)]]
    )

    features = image_features(path)

    assert tuple(features) == FEATURE_NAMES
    assert features["mean_r"] == pytest.approx(0.5)
    assert features["mean_g"] == pytest.approx(1.0)
    assert features["mean_b"] == pytest.approx(0.0)
    assert features["std_r"] == pytest.approx(0.5)
    assert features["std_g"] == pytest.approx(0.0)
    assert features["std_b"] == pytest.approx(0.0)
    # Y is 0.587 on the red-free pixels and 0.886 on the others.
    assert features["mean_luminance"] == pytest.approx((0.587 + 0.886) / 2)
    assert features["std_luminance"] == pytest.approx(0.299 / 2)
    assert image_features(path) == features


def test_compare_feature_distributions_reports_descriptives_and_distances() -> None:
    generated = {name: [0.0, 1.0, 2.0] for name in FEATURE_NAMES}
    reference = {name: [10.0, 11.0, 12.0, 13.0] for name in FEATURE_NAMES}

    rows = compare_feature_distributions(generated, reference)

    assert [row["feature"] for row in rows] == list(FEATURE_NAMES)
    row = rows[0]
    assert set(row) == set(COMPARISON_FIELDS)
    assert row["generated_count"] == 3
    assert row["generated_mean"] == pytest.approx(1.0)
    assert row["generated_std"] == pytest.approx(1.0)
    assert row["generated_median"] == pytest.approx(1.0)
    assert (row["generated_min"], row["generated_max"]) == (0.0, 2.0)
    assert row["reference_count"] == 4
    assert row["reference_mean"] == pytest.approx(11.5)
    assert row["reference_std"] == pytest.approx(np.std([10, 11, 12, 13], ddof=1))
    assert row["reference_median"] == pytest.approx(11.5)
    assert (row["reference_min"], row["reference_max"]) == (10.0, 13.0)
    assert row["wasserstein_distance"] == pytest.approx(10.5)
    assert row["ks_statistic"] == pytest.approx(1.0)
    # Exact two-sided p for fully separated samples: 2 / C(7, 3).
    assert row["ks_pvalue"] == pytest.approx(2 / 35)


def test_no_ranking_fields_in_comparison_schema() -> None:
    forbidden = {"better", "winner", "favored", "quality_score", "fidelity_score"}
    assert not any(part in field for field in COMPARISON_FIELDS for part in forbidden)


def test_evaluate_unpaired_collections_writes_reports(tmp_path: Path) -> None:
    generated = [
        write_rgb_image(tmp_path / "g" / "b.png", size=(8, 8), color=(10, 20, 30)),
        write_rgb_image(tmp_path / "g" / "a.png", size=(16, 4), color=(40, 50, 60)),
    ]
    reference = [write_rgb_image(tmp_path / "r" / f"{i}.png", color=(i, i, i)) for i in range(3)]
    output_dir = tmp_path / "out"

    result = evaluate_unpaired_collections(generated, reference, output_dir, save_graphs=True)

    assert (result.generated_count, result.reference_count) == (2, 3)
    with result.image_statistics_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["collection", "path", *FEATURE_NAMES]
    assert [(row["collection"], row["path"]) for row in rows] == [
        ("generated", str(generated[0])),
        ("generated", str(generated[1])),
        *(("reference", str(path)) for path in reference),
    ]
    assert float(rows[0]["mean_r"]) == pytest.approx(10 / 255)
    with result.feature_comparison_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        comparison = list(reader)
    assert reader.fieldnames == list(COMPARISON_FIELDS)
    assert [row["feature"] for row in comparison] == list(FEATURE_NAMES)
    assert result.graph_path is not None and result.graph_path.is_file()

    first = result.image_statistics_csv.read_bytes()
    evaluate_unpaired_collections(generated, reference, output_dir, save_graphs=False)
    assert result.image_statistics_csv.read_bytes() == first


def test_evaluate_unpaired_collections_rejects_empty(tmp_path: Path) -> None:
    image = write_rgb_image(tmp_path / "x.png")
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_unpaired_collections([], [image], tmp_path, save_graphs=False)
