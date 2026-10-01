"""Paired evaluator contract: coverage modes, valid-region support and result metadata.

Tiny CPU fixtures only; these check software behaviour, not image-quality claims.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.image_helpers import write_rgb_image
from virtual_staining.evaluation.evaluator import (
    EvaluationCoverageError,
    EvaluationInputError,
    EvaluationSample,
    evaluate_pair,
    evaluate_samples,
)
from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    MetricDefinition,
    MetricResult,
    resolve_metrics,
)

_ERRORS = resolve_metrics(
    [{"name": name} for name in ("mae", "mse", "rmse", "psnr")], BUILTIN_METRIC_DEFINITIONS
)


def _pair(tmp_path: Path, name: str, target: Any = (10, 20, 30), generated: Any = (12, 20, 30)):
    return (
        write_rgb_image(tmp_path / f"{name}_target.png", color=target),
        write_rgb_image(tmp_path / f"{name}_generated.png", color=generated),
    )


def _sample(tmp_path: Path, name: str, **kwargs: Any) -> EvaluationSample:
    target, generated = _pair(tmp_path, name)
    return EvaluationSample(name, "HE", "S1", target, generated, **kwargs)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _strict_json(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _mask(path: Path, array: np.ndarray, mode: str = "L") -> Path:
    Image.fromarray(array.astype(np.uint8), mode="L").convert(mode).save(path)
    return path


# --- input failures and coverage ---------------------------------------------------------


def _failure_samples(tmp_path: Path) -> list[EvaluationSample]:
    good = _sample(tmp_path, "good")
    unreadable = tmp_path / "bad_generated.png"
    unreadable.write_bytes(b"not an image")
    small = write_rgb_image(tmp_path / "small_generated.png", size=(8, 8))
    gray = tmp_path / "gray_generated.png"
    Image.new("L", (16, 16)).save(gray)
    return [
        good,
        EvaluationSample("missing", "HE", "S1", good.target_path, tmp_path / "absent.png"),
        EvaluationSample("unreadable", "HE", "S1", good.target_path, unreadable),
        EvaluationSample("shape", "HE", "S1", good.target_path, small),
        EvaluationSample("gray", "HE", "S1", good.target_path, gray),
    ]


_FAILURE_REASONS = [
    "",
    "missing_generated",
    "unreadable_generated",
    "shape_mismatch",
    "unsupported_generated_mode",
]


def test_strict_mode_records_known_input_failures_and_fails(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"

    with pytest.raises(EvaluationCoverageError, match="4 of 5 samples had input failures"):
        evaluate_samples(_failure_samples(tmp_path), output)

    coverage = _read_csv(output / "coverage.csv")
    assert [row["reason"] for row in coverage] == _FAILURE_REASONS
    assert [row["status"] for row in coverage] == ["evaluated"] + ["failed"] * 4
    assert not (output / "per_image_metrics.csv").exists()
    assert not (output / "evaluation_result.json").exists()


def test_permissive_mode_excludes_known_input_failures(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"

    result = evaluate_samples(_failure_samples(tmp_path), output, input_failures="permissive")

    assert (result.num_requested, result.num_evaluated, result.num_excluded) == (5, 1, 4)
    coverage = _read_csv(result.coverage_csv)
    assert [row["reason"] for row in coverage] == _FAILURE_REASONS
    assert [row["status"] for row in coverage] == ["evaluated"] + ["excluded"] * 4
    assert [row["sample_id"] for row in _read_csv(result.metrics_csv)] == ["good"]
    counts = _strict_json(result.result_json)["counts"]
    assert counts == {"requested": 5, "evaluated": 1, "excluded": 4, "failed": 0}
    assert counts["requested"] == counts["evaluated"] + counts["excluded"] + counts["failed"]


def test_nothing_evaluated_is_never_a_successful_result(tmp_path: Path) -> None:
    target, _ = _pair(tmp_path, "a")
    missing = [EvaluationSample("a", "HE", "S1", target, tmp_path / "absent.png")]

    with pytest.raises(EvaluationCoverageError, match="No sample could be evaluated"):
        evaluate_samples(missing, tmp_path / "out", input_failures="permissive")
    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_samples([], tmp_path / "out")
    with pytest.raises(ValueError, match="at least one requested metric"):
        evaluate_samples([_sample(tmp_path, "b")], tmp_path / "out", metrics=())


@pytest.mark.parametrize("mode", ["strict", "permissive"])
def test_unexpected_evaluator_errors_propagate_in_every_mode(tmp_path: Path, mode: Any) -> None:
    def broken(*_: Any) -> dict[str, MetricResult]:
        raise ZeroDivisionError("metric bug")

    definition = MetricDefinition("broken", "1", "tests", broken, higher_is_better=True)
    metrics = resolve_metrics([{"name": "broken"}], {"broken": definition})

    with pytest.raises(ZeroDivisionError, match="metric bug"):
        evaluate_samples(
            [_sample(tmp_path, "a")], tmp_path / "out", metrics=metrics, input_failures=mode
        )


# --- dynamic reports and result metadata ---------------------------------------------------


def _scaled_error_definition() -> MetricDefinition:
    def parse(raw: Mapping[str, Any], field: str) -> dict[str, Any]:
        unknown = set(raw) - {"scale"}
        if unknown:
            raise ValueError(f"{field} has unknown keys {sorted(unknown)}")
        scale = raw.get("scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise TypeError(f"{field}.scale must be a number")
        return {"scale": float(scale)}

    def evaluate(
        target: np.ndarray,
        generated: np.ndarray,
        support: np.ndarray | None,
        requested: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, MetricResult]:
        scale = requested["scaled_error"]["scale"]
        return {"scaled_error": MetricResult.of(scale * float(np.abs(target - generated).max()))}

    return MetricDefinition(
        name="scaled_error",
        version="2",
        source="tests.external",
        evaluator=evaluate,
        higher_is_better=False,
        parse_options=parse,
    )


def test_external_metric_appears_in_reports_and_result_metadata(tmp_path: Path) -> None:
    definitions = {
        "scaled_error": _scaled_error_definition(),
        "ssim": BUILTIN_METRIC_DEFINITIONS["ssim"],
    }
    metrics = resolve_metrics(
        [{"name": "scaled_error", "options": {"scale": 10}}, {"name": "ssim"}], definitions
    )

    result = evaluate_samples([_sample(tmp_path, "a")], tmp_path / "out", metrics=metrics)

    row = _read_csv(result.metrics_csv)[0]
    assert list(row)[8:] == [
        "scaled_error",
        "scaled_error_status",
        "scaled_error_reason",
        "ssim",
        "ssim_status",
        "ssim_reason",
    ]
    assert float(row["scaled_error"]) == pytest.approx(10 * 2 / 255)
    assert row["scaled_error_status"] == "finite"
    assert [r["metric"] for r in _read_csv(result.summary_csv)] == ["scaled_error", "ssim"]
    payload = _strict_json(result.result_json)
    assert payload["schema_version"] == 1
    assert payload["statuses"] == ["finite", "positive_infinity", "undefined", "unavailable"]
    custom = payload["metrics"][0]
    assert custom["name"] == "scaled_error"
    assert (custom["version"], custom["source"]) == ("2", "tests.external")
    assert custom["options"] == {"scale": 10.0}
    assert custom["higher_is_better"] is False


def test_non_finite_results_are_empty_cells_with_status_and_strict_json(tmp_path: Path) -> None:
    target, generated = _pair(tmp_path, "same", generated=(10, 20, 30))
    samples = [EvaluationSample("same", "HE", "S1", target, generated)]

    result = evaluate_samples(samples, tmp_path / "out")

    row = _read_csv(result.metrics_csv)[0]
    assert (row["psnr"], row["psnr_status"]) == ("inf", "positive_infinity")
    assert (row["pcc_gray"], row["pcc_gray_status"]) == ("", "undefined")
    assert "constant" in row["pcc_gray_reason"]
    assert math.isinf(float(row["psnr"]))
    _strict_json(result.result_json)


# --- valid-region support -------------------------------------------------------------------


def test_binary_support_restricts_pointwise_errors(tmp_path: Path) -> None:
    target = np.zeros((4, 8, 3), np.uint8)
    generated = target.copy()
    generated[:, 4:] = 51  # error only in the right half
    Image.fromarray(target).save(tmp_path / "t.png")
    Image.fromarray(generated).save(tmp_path / "g.png")
    left = np.zeros((4, 8), np.uint8)
    left[:, :2] = 255
    right = np.zeros((4, 8), np.uint8)
    right[:, 4:] = 255
    left_mask = _mask(tmp_path / "left.png", left)
    right_mask = _mask(tmp_path / "right_1.png", right, mode="1")

    clean, _ = evaluate_pair(
        tmp_path / "t.png", tmp_path / "g.png", metrics=_ERRORS, support_path=left_mask
    )
    errors, _ = evaluate_pair(
        tmp_path / "t.png", tmp_path / "g.png", metrics=_ERRORS, support_path=right_mask
    )

    assert clean["mae"] == MetricResult("finite", 0.0, support_count=8, support_fraction=0.25)
    assert clean["psnr"].status == "positive_infinity"
    assert errors["mae"].value == pytest.approx(0.2)
    assert errors["rmse"].value == pytest.approx(0.2)
    assert (errors["mse"].support_count, errors["mse"].support_fraction) == (16, 0.5)


def test_support_columns_are_reported_per_metric(tmp_path: Path) -> None:
    support = _mask(tmp_path / "support.png", np.full((16, 16), 255))
    sample = _sample(tmp_path, "a", support_path=support)

    result = evaluate_samples([sample], tmp_path / "out", metrics=_ERRORS[:1])

    row = _read_csv(result.metrics_csv)[0]
    assert row["support_path"] == str(support)
    assert (row["mae_support_count"], row["mae_support_fraction"]) == ("256", "1.0")
    assert _strict_json(result.result_json)["valid_region_support"] is True


def test_empty_support_is_undefined_not_perfect(tmp_path: Path) -> None:
    support = _mask(tmp_path / "empty.png", np.zeros((16, 16)))
    target, generated = _pair(tmp_path, "a")

    results, _ = evaluate_pair(target, generated, metrics=_ERRORS, support_path=support)

    for result in results.values():
        assert (result.status, result.value) == ("undefined", None)
        assert (result.support_count, result.support_fraction) == (0, 0.0)


@pytest.mark.parametrize(
    ("array", "mode", "reason"),
    [
        (np.full((16, 16), 128), "L", "malformed_support"),
        (np.full((16, 16), 255), "RGB", "malformed_support"),
        (np.full((16, 8), 255), "L", "support_shape_mismatch"),
    ],
    ids=["soft", "rgb", "shape"],
)
def test_malformed_or_mismatched_support_is_rejected(
    tmp_path: Path, array: np.ndarray, mode: str, reason: str
) -> None:
    support = _mask(tmp_path / "support.png", array, mode=mode)
    target, generated = _pair(tmp_path, "a")

    with pytest.raises(EvaluationInputError) as caught:
        evaluate_pair(target, generated, metrics=_ERRORS, support_path=support)
    assert caught.value.reason == reason
    with pytest.raises(EvaluationCoverageError):
        evaluate_samples(
            [EvaluationSample("a", "HE", "S1", target, generated, support)],
            tmp_path / "out",
            metrics=_ERRORS,
        )


def test_missing_support_file_is_a_known_input_failure(tmp_path: Path) -> None:
    sample = _sample(tmp_path, "a", support_path=tmp_path / "absent.png")

    with pytest.raises(EvaluationCoverageError):
        evaluate_samples([sample], tmp_path / "out", metrics=_ERRORS)
    assert _read_csv(tmp_path / "out" / "coverage.csv")[0]["reason"] == "missing_support"


@pytest.mark.parametrize("name", ["ssim", "pcc_gray", "pcc_rgb_mean"])
def test_support_with_ssim_or_pcc_fails_before_reading_inputs(tmp_path: Path, name: str) -> None:
    metrics = resolve_metrics([{"name": "mae"}, {"name": name}], BUILTIN_METRIC_DEFINITIONS)
    sample = EvaluationSample(
        "a",
        "HE",
        "S1",
        tmp_path / "absent_t.png",
        tmp_path / "absent_g.png",
        tmp_path / "absent_m.png",
    )

    with pytest.raises(ValueError, match="do not support it"):
        evaluate_samples([sample], tmp_path / "out", metrics=metrics)
    assert not (tmp_path / "out").exists()


def test_support_must_be_supplied_for_all_samples_or_none(tmp_path: Path) -> None:
    support = _mask(tmp_path / "support.png", np.full((16, 16), 255))
    samples = [_sample(tmp_path, "a", support_path=support), _sample(tmp_path, "b")]

    with pytest.raises(ValueError, match="every sample or for none"):
        evaluate_samples(samples, tmp_path / "out", metrics=_ERRORS)
