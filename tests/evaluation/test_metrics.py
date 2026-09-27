from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from virtual_staining import metrics as metrics_module
from virtual_staining.definitions import Definitions
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.metrics import (
    BUILTIN_METRIC_DEFINITIONS,
    DEFAULT_METRIC_NAMES,
    MetricDefinition,
    MetricEvaluatorError,
    MetricResult,
    compute_metrics,
    compute_pcc,
    compute_ssim,
    default_metrics,
    resolve_metrics,
)


def _rgb(value: float, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _random_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    target = rng.integers(0, 256, (12, 10, 3)).astype(np.float32) / 255.0
    generated = rng.integers(0, 256, (12, 10, 3)).astype(np.float32) / 255.0
    return target, generated


def _request(*names: str) -> tuple[Any, ...]:
    return resolve_metrics([{"name": name} for name in names], BUILTIN_METRIC_DEFINITIONS)


# --- numeric contract ----------------------------------------------------------------

# Values of the previous full-image implementation on the fixed random pair.
_REFERENCE_VALUES = {
    "ssim": 0.08502276986837387,
    "psnr": 7.827645756962429,
    "mae": 0.3287254869937897,
    "rmse": 0.4060857147208553,
    "mse": 0.1649056077003479,
    "pcc_rgb_mean": 0.046097655400981506,
    "pcc_gray": 0.0451877061164742,
    "pcc_r": 0.10128831873782929,
    "pcc_g": -0.013612820698442602,
    "pcc_b": 0.05061746816355783,
}


def test_builtin_finite_values_are_unchanged() -> None:
    results = compute_metrics(_request(*_REFERENCE_VALUES), *_random_pair())

    for name, expected in _REFERENCE_VALUES.items():
        assert results[name].status == "finite"
        assert results[name].value == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_ssim_call_pins_every_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def recording(*args: Any, **kwargs: Any) -> float:
        calls.append(kwargs)
        return 0.5

    monkeypatch.setattr(metrics_module, "structural_similarity", recording)

    assert compute_ssim(_rgb(0.2), _rgb(0.3)) == 0.5
    assert calls == [
        {
            "data_range": 1.0,
            "channel_axis": 2,
            "win_size": 7,
            "gaussian_weights": False,
            "use_sample_covariance": True,
            "K1": 0.01,
            "K2": 0.03,
        }
    ]


def test_identical_images_have_positive_infinite_psnr() -> None:
    result = compute_metrics(_request("psnr", "mae"), _rgb(0.5), _rgb(0.5))

    assert result["psnr"] == MetricResult("positive_infinity", math.inf)
    assert result["mae"] == MetricResult("finite", 0.0)


def test_constant_data_pcc_is_undefined() -> None:
    assert math.isnan(compute_pcc(np.ones((8, 8)), np.zeros((8, 8))))
    # A constant that float rounding would give a tiny non-zero standard deviation.
    assert math.isnan(compute_pcc(np.full((16, 16), 20 / 255, np.float32), np.arange(256.0)))
    results = compute_metrics(
        _request("pcc_gray", "pcc_r", "pcc_rgb_mean"), _rgb(1.0), _random_pair()[1][:8, :8]
    )

    for name in ("pcc_gray", "pcc_r", "pcc_rgb_mean"):
        assert results[name].status == "undefined"
        assert results[name].value is None
        assert "constant" in (results[name].reason or "")


def test_ssim_too_small_for_its_window_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: Any, **__: Any) -> float:
        raise AssertionError("SSIM must not be reparameterized for small images")

    monkeypatch.setattr(metrics_module, "structural_similarity", forbidden)

    result = compute_metrics(_request("ssim"), _rgb(0.2, 6, 20), _rgb(0.3, 6, 20))["ssim"]

    assert result.status == "unavailable"
    assert "7x7 SSIM window" in (result.reason or "")


def test_inputs_are_never_rescaled_or_reshaped() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_metrics(_request("mae"), _rgb(0.5) * 255, _rgb(0.5))
    with pytest.raises(ValueError, match="same shape"):
        compute_metrics(_request("mae"), _rgb(0.5, 8, 8), _rgb(0.5, 8, 9))
    with pytest.raises(ValueError, match="H x W x 3"):
        compute_metrics(_request("mae"), np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32))


# --- request resolution -----------------------------------------------------------------


def test_default_request_is_the_builtin_default_set_in_report_order() -> None:
    assert [metric.name for metric in default_metrics()] == list(DEFAULT_METRIC_NAMES)
    assert DEFAULT_METRIC_NAMES == (
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "pcc_gray",
        "pcc_rgb_mean",
    )


@pytest.mark.parametrize(
    ("request_value", "error", "message"),
    [
        ([{"name": "fid"}], ValueError, "not a registered metric definition"),
        ([{"name": "mae"}, {"name": "mae"}], ValueError, "requested more than once"),
        ([{"name": "mae", "options": {"window": 3}}], ValueError, "unknown keys"),
        ([{"name": "mae", "options": "window=3"}], TypeError, "must be a mapping"),
        ([{"name": "mae", "weight": 1}], ValueError, "unknown keys"),
        ([{"name": 3}], TypeError, "name must be a string"),
        (["mae"], TypeError, "must be a mapping"),
        ("mae", TypeError, "list of mappings"),
        ([], ValueError, "at least one metric"),
    ],
)
def test_malformed_requests_are_rejected(
    request_value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        resolve_metrics(request_value, BUILTIN_METRIC_DEFINITIONS)


def _custom(
    name: str = "custom",
    *,
    evaluator: Any = None,
    parse_options: Any = None,
    **kwargs: Any,
) -> MetricDefinition:
    def evaluate(
        target: np.ndarray,
        generated: np.ndarray,
        support: np.ndarray | None,
        requested: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, MetricResult]:
        return {name: MetricResult.of(1.0)}

    return MetricDefinition(
        name=name,
        version="1",
        source="tests",
        evaluator=evaluator or evaluate,
        higher_is_better=True,
        **({"parse_options": parse_options} if parse_options else {}),
        **kwargs,
    )


def test_options_must_resolve_to_json_compatible_values() -> None:
    nan_options = _custom(parse_options=lambda raw, field: {"scale": math.nan})
    with pytest.raises(ValueError, match="JSON-compatible"):
        resolve_metrics([{"name": "custom"}], {"custom": nan_options})
    not_a_dict = _custom(parse_options=lambda raw, field: [1])
    with pytest.raises(TypeError, match="must return a dict"):
        resolve_metrics([{"name": "custom"}], {"custom": not_a_dict})


def test_duplicate_metric_definition_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate metric definition 'mae'"):
        builtin_definitions().extend(metrics=[_custom("mae")])
    assert "custom" in Definitions().extend(metrics=[_custom()]).metrics


# --- evaluator groups and result validity -------------------------------------------


def test_only_requested_evaluator_groups_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: Any, **__: Any) -> float:
        raise AssertionError("unrequested evaluator group executed")

    monkeypatch.setattr(metrics_module, "structural_similarity", forbidden)
    monkeypatch.setattr(metrics_module, "compute_pcc", forbidden)

    results = compute_metrics(_request("rmse", "mae"), *_random_pair())

    assert list(results) == ["rmse", "mae"]


def test_shared_group_runs_once_and_returns_only_requested_outputs() -> None:
    calls: list[set[str]] = []

    def group(
        target: np.ndarray,
        generated: np.ndarray,
        support: np.ndarray | None,
        requested: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, MetricResult]:
        calls.append(set(requested))
        return {"a": MetricResult.of(1.0), "b": MetricResult.of(2.0), "c": MetricResult.of(3.0)}

    definitions = {name: _custom(name, evaluator=group) for name in ("a", "b", "c")}
    request = resolve_metrics([{"name": "b"}, {"name": "a"}], definitions)

    results = compute_metrics(request, _rgb(0.1), _rgb(0.2))

    assert calls == [{"a", "b"}]
    assert list(results) == ["b", "a"]


@pytest.mark.parametrize(
    "evaluator",
    [
        lambda *_: {"custom": MetricResult.of(-math.inf)},
        lambda *_: {"custom": 1.0},
        lambda *_: {},
        lambda *_: None,
    ],
    ids=["negative_infinity", "bare_float", "missing_output", "not_a_mapping"],
)
def test_malformed_evaluator_output_is_an_error(evaluator: Any) -> None:
    request = resolve_metrics([{"name": "custom"}], {"custom": _custom(evaluator=evaluator)})

    with pytest.raises(MetricEvaluatorError):
        compute_metrics(request, _rgb(0.1), _rgb(0.2))


@pytest.mark.parametrize(
    ("status", "value", "reason"),
    [
        ("finite", math.nan, None),
        ("finite", math.inf, None),
        ("positive_infinity", 3.0, None),
        ("undefined", 1.0, "why"),
        ("undefined", None, None),
        ("unavailable", None, ""),
        ("negative_infinity", -math.inf, None),
    ],
)
def test_result_states_outside_the_contract_are_rejected(
    status: Any, value: float | None, reason: str | None
) -> None:
    with pytest.raises(MetricEvaluatorError):
        MetricResult(status, value, reason)


def test_identity_records_definition_options_direction_and_presentation() -> None:
    custom = _custom(
        parse_options=lambda raw, field: {"scale": float(raw.get("scale", 1.0))},
        thresholds=(0.5,),
    ).resolve({"scale": 2}, "custom")

    identity = custom.identity()

    assert json.loads(json.dumps(identity, allow_nan=False)) == identity
    assert identity["name"] == "custom"
    assert (identity["version"], identity["source"]) == ("1", "tests")
    assert identity["options"] == {"scale": 2.0}
    assert identity["higher_is_better"] is True
    assert identity["input"]["range"] == [0.0, 1.0]
    assert identity["presentation"] == {"thresholds": [0.5], "plot_range": None}
    assert custom.request() == {"name": "custom", "options": {"scale": 2.0}}
    assert _request("mae")[0].request() == {"name": "mae"}
