from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from virtual_staining.config.losses import LossTermConfig
from virtual_staining.loss_definitions import LossDefinition, LossParams


@dataclass
class StepLosses:
    loss_G: float
    loss_D: float
    raw: dict[str, float] | None = None
    weighted: dict[str, float] | None = None
    current_weight: dict[str, float] | None = None


@dataclass(frozen=True)
class LossTermResult:
    name: str
    raw: torch.Tensor
    weighted: torch.Tensor
    current_weight: float
    stage: Literal["generator", "discriminator"] = "generator"

    @property
    def component_key(self) -> str:
        return f"{self.stage}_{self.name}"


@dataclass(frozen=True)
class LossEvaluationContext:
    epoch: int = 0
    global_step: int | None = None
    masks: dict[str, torch.Tensor] | None = None


@dataclass
class LossAggregate:
    total: torch.Tensor
    raw: dict[str, float]
    weighted: dict[str, float]
    current_weight: dict[str, float]


class ConfiguredLossEvaluator:
    """Evaluates configured Pix2Pix loss terms for training and validation.

    Terms are resolved against their canonical ``LossDefinition`` once, at construction.
    """

    def __init__(
        self,
        *,
        generator_terms: tuple[LossTermConfig, ...] = (),
        discriminator_terms: tuple[LossTermConfig, ...] = (),
    ) -> None:
        self.generator_terms = generator_terms
        self.discriminator_terms = discriminator_terms
        self._generator = _resolve(generator_terms)
        self._discriminator = _resolve(discriminator_terms)

    @property
    def needs_discriminator_logits(self) -> bool:
        return any(
            definition.context == "adversarial"
            for _, definition, _ in (*self._generator, *self._discriminator)
        )

    def generator_total(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        discriminator_fake: torch.Tensor | None = None,
        context: LossEvaluationContext,
    ) -> LossAggregate:
        total = prediction.sum() * 0.0
        results: list[LossTermResult] = []
        for term, definition, params in self._generator:
            if definition.context == "adversarial":
                if discriminator_fake is None:
                    raise ValueError(
                        f"generator {term.name} loss requires discriminator_fake logits"
                    )
                raw = definition.adversarial_loss(discriminator_fake, target_is_real=True)
            else:
                raw = _ensure_scalar(
                    definition.reconstruction_loss(prediction, target, params, masks=context.masks)
                )
            result = _term_result(term, raw, context, "generator")
            total = total + result.weighted
            results.append(result)
        return _aggregate_loss_results(total, results)

    def discriminator_total(
        self,
        *,
        discriminator_real: torch.Tensor,
        discriminator_fake: torch.Tensor,
        context: LossEvaluationContext,
    ) -> LossAggregate:
        total = discriminator_real.sum() * 0.0
        results: list[LossTermResult] = []
        for term, definition, _ in self._discriminator:
            raw = definition.adversarial_loss(
                discriminator_real, target_is_real=True
            ) + definition.adversarial_loss(discriminator_fake, target_is_real=False)
            result = _term_result(term, raw, context, "discriminator")
            total = total + result.weighted
            results.append(result)
        return _aggregate_loss_results(total, results)


def _resolve(
    terms: tuple[LossTermConfig, ...],
) -> tuple[tuple[LossTermConfig, LossDefinition, LossParams], ...]:
    return tuple((term, term.definition, term.resolved_params()) for term in terms)


def _term_result(
    term: LossTermConfig,
    raw: torch.Tensor,
    context: LossEvaluationContext,
    stage: Literal["generator", "discriminator"],
) -> LossTermResult:
    current_weight = term.current_weight(epoch=context.epoch, global_step=context.global_step)
    return LossTermResult(
        name=term.name,
        raw=raw,
        weighted=raw * current_weight,
        current_weight=current_weight,
        stage=stage,
    )


def _ensure_scalar(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value
    return value.mean()


def _aggregate_loss_results(
    total: torch.Tensor,
    results: list[LossTermResult],
) -> LossAggregate:
    raw: dict[str, float] = {}
    weighted: dict[str, float] = {}
    current_weight: dict[str, float] = {}
    for result in results:
        raw[result.component_key] = float(result.raw.detach().item())
        weighted[result.component_key] = float(result.weighted.detach().item())
        current_weight[result.component_key] = result.current_weight
    return LossAggregate(
        total=total,
        raw=raw,
        weighted=weighted,
        current_weight=current_weight,
    )
