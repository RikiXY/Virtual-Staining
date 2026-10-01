"""Standalone evaluation metric; deliberately separate from training L1."""

import math

import numpy as np

from virtual_staining.config import reject_unknown_keys
from virtual_staining.metrics import MetricDefinition, MetricResult


def parse_options(raw, field):
    reject_unknown_keys(raw, frozenset({"scale"}), field)
    scale = raw.get("scale", 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{field}.scale must be finite and positive")
    return {"scale": float(scale)}


def scaled_max_error(target, generated, support, requested):
    return {
        "scaled_max_error": MetricResult.of(
            float(np.abs(target - generated).max()) * requested["scaled_max_error"]["scale"]
        )
    }


SCALED_MAX_ERROR = MetricDefinition(
    "scaled_max_error", "1", "mexina_external_example", scaled_max_error, False, parse_options
)
