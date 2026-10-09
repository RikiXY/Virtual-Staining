"""Evaluation image metrics: the definition contract and the built-in definitions.

A :class:`MetricDefinition` is an immutable, explicitly supplied description of one scalar
metric of a target/generated RGB image pair. Callers make external metrics available by
passing definitions in :class:`virtual_staining.definitions.Definitions`; configuration then
requests them by name. Nothing is discovered or imported from configuration.

Every metric receives two float ``H x W x 3`` arrays holding RGB in ``[0, 1]`` on the same
pixel grid; inputs are never resized, cropped, clipped, rescaled or channel-converted.
Each requested metric yields one :class:`MetricResult` with an explicit status.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, cast, get_args

import numpy as np
from skimage.metrics import structural_similarity

MetricStatus = Literal["finite", "positive_infinity", "undefined", "unavailable"]
METRIC_STATUSES: tuple[str, ...] = get_args(MetricStatus)
BUILTIN_METRIC_SOURCE = "virtual_staining"


class MetricEvaluatorError(RuntimeError):
    """A metric evaluator returned something outside the result contract (a defect)."""


@dataclass(frozen=True)
class MetricResult:
    """One requested metric value with its validity status.

    ``finite`` carries a finite float, ``positive_infinity`` carries ``inf`` (e.g. PSNR of
    identical images); ``undefined`` (the formula has no value, e.g. PCC of constant data)
    and ``unavailable`` (the metric cannot be computed for this input, e.g. an image
    smaller than the SSIM window) carry no value and a reason. ``support_count`` and
    ``support_fraction`` are set when the metric was restricted to a valid region.
    """

    status: MetricStatus
    value: float | None = None
    reason: str | None = None
    support_count: int | None = None
    support_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.status == "finite":
            ok = isinstance(self.value, float) and math.isfinite(self.value)
        elif self.status == "positive_infinity":
            ok = self.value == math.inf
        elif self.status in ("undefined", "unavailable"):
            ok = self.value is None and bool(self.reason)
        else:
            ok = False
        if not ok:
            raise MetricEvaluatorError(
                f"Invalid metric result: status={self.status!r} value={self.value!r} "
                f"reason={self.reason!r}"
            )

    @classmethod
    def of(cls, value: float) -> MetricResult:
        """Classify a computed number; NaN and negative infinity are evaluator defects."""
        number = float(value)
        if math.isfinite(number):
            return cls("finite", number)
        if number == math.inf:
            return cls("positive_infinity", number)
        raise MetricEvaluatorError(f"Metric evaluator produced {number!r}, not a valid result")

    @classmethod
    def undefined(cls, reason: str) -> MetricResult:
        return cls("undefined", reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> MetricResult:
        return cls("unavailable", reason=reason)


#: ``evaluator(target, generated, support, requested)`` computes the requested outputs of
#: one evaluator group. ``requested`` maps each requested metric name of the group to its
#: validated options; ``support`` is None or a boolean ``H x W`` valid-region mask. Metrics
#: sharing one evaluator callable form a group that runs once per image pair.
MetricEvaluator = Callable[
    [np.ndarray, np.ndarray, "np.ndarray | None", Mapping[str, Mapping[str, Any]]],
    Mapping[str, MetricResult],
]


def no_options(raw: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Option parser of a metric that accepts no options."""
    if raw:
        raise ValueError(f"{field} has unknown keys {sorted(raw)}; this metric takes no options")
    return {}


@dataclass(frozen=True)
class MetricDefinition:
    """One registered evaluation metric; its ``name`` is also its output identity.

    ``parse_options(raw, field)`` validates raw options strictly and returns them
    normalized as a JSON-compatible dict. Bump ``version`` whenever the same options would
    compute a different number. ``higher_is_better`` is None when ranking is meaningless.
    ``supports_valid_region`` declares that the evaluator honours a valid-region support
    mask. ``thresholds`` and ``plot_range`` are presentation heuristics only (CLI colouring,
    comparison threshold shares, plot axes); they are not quality or acceptance criteria.
    """

    name: str
    version: str
    source: str
    evaluator: MetricEvaluator
    higher_is_better: bool | None
    parse_options: Callable[[Mapping[str, Any], str], dict[str, Any]] = no_options
    supports_valid_region: bool = False
    thresholds: tuple[float, ...] = ()
    plot_range: tuple[float, float] | None = None

    def resolve(self, raw: object, field: str) -> ResolvedMetric:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{field} must be a mapping")
        options = self.parse_options(raw, field)
        if not isinstance(options, dict):
            raise TypeError(f"{field}: the options parser of {self.name!r} must return a dict")
        try:
            encoded = json.dumps(options, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} did not resolve to JSON-compatible options: {exc}") from exc
        return ResolvedMetric(self, MappingProxyType(json.loads(encoded)))


@dataclass(frozen=True)
class ResolvedMetric:
    """A metric definition bound to its validated options: one requested output."""

    definition: MetricDefinition
    options: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.definition.name

    def request(self) -> dict[str, Any]:
        """The canonical request spelling: ``name`` plus ``options`` when non-empty."""
        return {"name": self.name, **({"options": dict(self.options)} if self.options else {})}

    def identity(self) -> dict[str, Any]:
        definition = self.definition
        return {
            "name": definition.name,
            "version": definition.version,
            "source": definition.source,
            "options": dict(self.options),
            # The input contract every metric shares (see the module docstring).
            "applicability": "paired_image",
            "input": {"layout": "HWC", "channels": 3, "color": "RGB", "range": [0.0, 1.0]},
            "supports_valid_region": definition.supports_valid_region,
            "higher_is_better": definition.higher_is_better,
            "presentation": {
                "thresholds": list(definition.thresholds),
                "plot_range": list(definition.plot_range) if definition.plot_range else None,
            },
        }


def resolve_metrics(
    request: object,
    definitions: Mapping[str, MetricDefinition],
    *,
    field: str = "evaluation.metrics",
) -> tuple[ResolvedMetric, ...]:
    """Resolve an ordered request ``[{name, options?}, ...]``; None is the default set."""
    if request is None:
        request = [{"name": name} for name in DEFAULT_METRIC_NAMES]
    if isinstance(request, str | bytes | Mapping) or not isinstance(request, Sequence):
        raise TypeError(f"{field} must be a list of mappings with 'name' and optional 'options'")
    if not request:
        raise ValueError(f"{field} must request at least one metric")
    resolved: list[ResolvedMetric] = []
    for index, entry in enumerate(request):
        entry_field = f"{field}[{index}]"
        if not isinstance(entry, Mapping):
            raise TypeError(f"{entry_field} must be a mapping with 'name' and optional 'options'")
        unknown = sorted(set(entry) - {"name", "options"})
        if unknown:
            raise ValueError(f"{entry_field} has unknown keys {unknown}")
        name = entry.get("name")
        if not isinstance(name, str):
            raise TypeError(f"{entry_field}.name must be a string")
        if name not in definitions:
            raise ValueError(
                f"{entry_field}.name={name!r} is not a registered metric definition; "
                f"registered: {sorted(definitions)}. Definitions are supplied explicitly in "
                "Python."
            )
        if any(metric.name == name for metric in resolved):
            raise ValueError(f"{entry_field}.name={name!r} is requested more than once")
        options = entry.get("options", {})
        resolved.append(definitions[name].resolve(options, f"{entry_field}.options"))
    return tuple(resolved)


def check_valid_region_support(metrics: Iterable[ResolvedMetric]) -> None:
    """Reject a request that would restrict a metric without valid-region support."""
    unsupported = [m.name for m in metrics if not m.definition.supports_valid_region]
    if unsupported:
        raise ValueError(
            f"Valid-region support was supplied but metrics {unsupported} do not support it; "
            "only pointwise error metrics (e.g. mae, mse, rmse, psnr) can be restricted."
        )


def compute_metrics(
    metrics: Sequence[ResolvedMetric],
    target: np.ndarray,
    generated: np.ndarray,
    support: np.ndarray | None = None,
) -> dict[str, MetricResult]:
    """Run each requested evaluator group once and return exactly the requested results."""
    if target.shape != generated.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError(
            "Metric inputs must be H x W x 3 arrays of the same shape; "
            f"got {target.shape} and {generated.shape}"
        )
    for image in (target, generated):
        if not np.issubdtype(image.dtype, np.floating) or not (
            image.min() >= 0.0 and image.max() <= 1.0
        ):
            raise ValueError("Metric inputs must be floating-point RGB in [0, 1]")
    support_count = support_fraction = None
    if support is not None:
        check_valid_region_support(metrics)
        if support.dtype != np.bool_ or support.shape != target.shape[:2]:
            raise ValueError("Valid-region support must be a boolean H x W mask on the image grid")
        support_count = int(support.sum())
        support_fraction = support_count / support.size
        if support_count == 0:
            empty = MetricResult.undefined("empty valid-region support")
            return {m.name: replace(empty, support_count=0, support_fraction=0.0) for m in metrics}

    groups: dict[MetricEvaluator, dict[str, Mapping[str, Any]]] = {}
    for metric in metrics:
        groups.setdefault(metric.definition.evaluator, {})[metric.name] = metric.options
    computed: dict[str, MetricResult] = {}
    for evaluator, requested in groups.items():
        output = evaluator(target, generated, support, requested)
        for name in requested:
            result = output.get(name) if isinstance(output, Mapping) else None
            if not isinstance(result, MetricResult):
                raise MetricEvaluatorError(
                    f"Metric evaluator for {name!r} returned {result!r}, not a MetricResult"
                )
            computed[name] = result
    if support is None:
        return {m.name: computed[m.name] for m in metrics}
    return {
        m.name: replace(
            computed[m.name], support_count=support_count, support_fraction=support_fraction
        )
        for m in metrics
    }


# --- built-in numerical definitions ------------------------------------------------------


def compute_mae(target: np.ndarray, generated: np.ndarray) -> float:
    return float(np.mean(np.abs(target - generated)))


def compute_mse(target: np.ndarray, generated: np.ndarray) -> float:
    return float(np.mean((target - generated) ** 2))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    return float(np.sqrt(compute_mse(target, generated)))


def compute_psnr(target: np.ndarray, generated: np.ndarray) -> float:
    return _psnr_from_mse(compute_mse(target, generated))


def _psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


SSIM_WIN_SIZE = 7


def compute_ssim(target: np.ndarray, generated: np.ndarray) -> float:
    """Mean SSIM with every parameter pinned so a scikit-image default change cannot move it."""
    result = structural_similarity(
        target,
        generated,
        data_range=1.0,
        channel_axis=2,
        win_size=SSIM_WIN_SIZE,
        gaussian_weights=False,
        use_sample_covariance=True,
        K1=0.01,
        K2=0.03,
    )
    return float(cast(float, result))


def compute_pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when either input is constant (undefined)."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    # Exact constancy test: np.std of a constant array can round to a tiny non-zero value.
    if np.ptp(a_flat) == 0.0 or np.ptp(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _rgb_to_gray_float(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def compute_pcc_gray(target: np.ndarray, generated: np.ndarray) -> float:
    return compute_pcc(_rgb_to_gray_float(target), _rgb_to_gray_float(generated))


def compute_pcc_rgb(target: np.ndarray, generated: np.ndarray) -> tuple[float, float, float, float]:
    """Per-channel PCC and their mean over the channels where PCC is defined."""
    pcc_r = compute_pcc(target[..., 0], generated[..., 0])
    pcc_g = compute_pcc(target[..., 1], generated[..., 1])
    pcc_b = compute_pcc(target[..., 2], generated[..., 2])
    pcc_values = np.array([pcc_r, pcc_g, pcc_b], dtype=np.float64)
    pcc_rgb_mean = float("nan") if np.isnan(pcc_values).all() else float(np.nanmean(pcc_values))
    return pcc_r, pcc_g, pcc_b, pcc_rgb_mean


def _pcc_result(value: float) -> MetricResult:
    if math.isnan(value):
        return MetricResult.undefined("constant input: Pearson correlation is undefined")
    return MetricResult.of(value)


def _evaluate_ssim(
    target: np.ndarray,
    generated: np.ndarray,
    support: np.ndarray | None,
    requested: Mapping[str, Mapping[str, Any]],
) -> dict[str, MetricResult]:
    del support, requested
    if min(target.shape[:2]) < SSIM_WIN_SIZE:
        return {
            "ssim": MetricResult.unavailable(
                f"image {target.shape[1]}x{target.shape[0]} is smaller than the fixed "
                f"{SSIM_WIN_SIZE}x{SSIM_WIN_SIZE} SSIM window"
            )
        }
    return {"ssim": MetricResult.of(compute_ssim(target, generated))}


def _evaluate_error(
    target: np.ndarray,
    generated: np.ndarray,
    support: np.ndarray | None,
    requested: Mapping[str, Mapping[str, Any]],
) -> dict[str, MetricResult]:
    if support is not None:
        # Valid pixels, all RGB channels: (N, 3).
        target, generated = target[support], generated[support]
    results: dict[str, MetricResult] = {}
    if "mae" in requested:
        results["mae"] = MetricResult.of(compute_mae(target, generated))
    if requested.keys() & {"mse", "rmse", "psnr"}:
        mse = compute_mse(target, generated)
        if "mse" in requested:
            results["mse"] = MetricResult.of(mse)
        if "rmse" in requested:
            results["rmse"] = MetricResult.of(float(np.sqrt(mse)))
        if "psnr" in requested:
            results["psnr"] = MetricResult.of(_psnr_from_mse(mse))
    return results


def _evaluate_pcc(
    target: np.ndarray,
    generated: np.ndarray,
    support: np.ndarray | None,
    requested: Mapping[str, Mapping[str, Any]],
) -> dict[str, MetricResult]:
    del support
    results: dict[str, MetricResult] = {}
    if "pcc_gray" in requested:
        results["pcc_gray"] = _pcc_result(compute_pcc_gray(target, generated))
    if requested.keys() & {"pcc_r", "pcc_g", "pcc_b", "pcc_rgb_mean"}:
        values = compute_pcc_rgb(target, generated)
        for name, value in zip(("pcc_r", "pcc_g", "pcc_b", "pcc_rgb_mean"), values, strict=True):
            results[name] = _pcc_result(value)
    return results


def _builtin(
    name: str,
    evaluator: MetricEvaluator,
    higher_is_better: bool,
    thresholds: tuple[float, ...],
    plot_range: tuple[float, float],
    *,
    support: bool = False,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        version="1",
        source=BUILTIN_METRIC_SOURCE,
        evaluator=evaluator,
        higher_is_better=higher_is_better,
        supports_valid_region=support,
        thresholds=thresholds,
        plot_range=plot_range,
    )


_PCC_THRESHOLDS = (0.95, 0.90, 0.80)
_UNIT = (0.0, 1.0)
BUILTIN_METRICS: tuple[MetricDefinition, ...] = (
    # Pointwise errors share one group and may be restricted to a valid region.
    _builtin("mae", _evaluate_error, False, (0.06, 0.10, 0.16), _UNIT, support=True),
    _builtin("mse", _evaluate_error, False, (0.0036, 0.0100, 0.0256), _UNIT, support=True),
    _builtin("rmse", _evaluate_error, False, (0.08, 0.12, 0.20), _UNIT, support=True),
    _builtin("psnr", _evaluate_error, True, (25.0, 20.0, 15.0), (0.0, 60.0), support=True),
    _builtin("ssim", _evaluate_ssim, True, (0.85, 0.75, 0.65), _UNIT),
    _builtin("pcc_gray", _evaluate_pcc, True, _PCC_THRESHOLDS, (-1.0, 1.0)),
    _builtin("pcc_r", _evaluate_pcc, True, _PCC_THRESHOLDS, (-1.0, 1.0)),
    _builtin("pcc_g", _evaluate_pcc, True, _PCC_THRESHOLDS, (-1.0, 1.0)),
    _builtin("pcc_b", _evaluate_pcc, True, _PCC_THRESHOLDS, (-1.0, 1.0)),
    _builtin("pcc_rgb_mean", _evaluate_pcc, True, _PCC_THRESHOLDS, (-1.0, 1.0)),
)
BUILTIN_METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(
    {definition.name: definition for definition in BUILTIN_METRICS}
)
#: The request used when none is configured, in report order.
DEFAULT_METRIC_NAMES = ("mae", "mse", "rmse", "psnr", "ssim", "pcc_gray", "pcc_rgb_mean")


def default_metrics() -> tuple[ResolvedMetric, ...]:
    """The built-in default request resolved against the built-in definitions."""
    return resolve_metrics(None, BUILTIN_METRIC_DEFINITIONS)
