from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from virtual_staining.config.losses import LossTermConfig, output_component_key
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
    """``masks`` maps a mask source to its per-output masks, e.g. ``foreground_mask.HE``."""

    epoch: int = 0
    global_step: int | None = None
    masks: Mapping[str, Mapping[str, torch.Tensor]] | None = None


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
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        discriminator_fake: torch.Tensor | None = None,
        context: LossEvaluationContext,
    ) -> LossAggregate:
        """One joint adversarial term plus each reconstruction term's mean over outputs.

        A reconstruction term contributes ``weight * mean(term(prediction[o], target[o]))``
        over the ordered outputs, so one output keeps its exact scale. Per-output
        components are reported as ``<key>__<output>``: ``raw`` is that output's term,
        ``weighted`` its contribution to the total (``weight * raw / M``) and
        ``current_weight`` the scheduled term weight.
        """
        names = tuple(predictions)
        if not names or tuple(targets) != names:
            raise ValueError(
                f"generator loss targets {tuple(targets)} must match predictions {names}"
            )
        total = predictions[names[0]].sum() * 0.0
        raw: dict[str, float] = {}
        weighted: dict[str, float] = {}
        current_weight: dict[str, float] = {}
        for term, definition, params in self._generator:
            weight = term.current_weight(epoch=context.epoch, global_step=context.global_step)
            key = f"generator_{term.name}"
            if definition.context == "adversarial":
                if discriminator_fake is None:
                    raise ValueError(
                        f"generator {term.name} loss requires discriminator_fake logits"
                    )
                value = definition.adversarial_loss(discriminator_fake, target_is_real=True)
                total = total + value * weight
                raw[key] = float(value.detach().item())
                weighted[key] = float((value * weight).detach().item())
                current_weight[key] = weight
                continue
            per_output = [
                _ensure_scalar(
                    definition.reconstruction_loss(
                        predictions[name],
                        targets[name],
                        params,
                        masks=_output_masks(term, params, context.masks, name),
                    )
                )
                for name in names
            ]
            term_raw = torch.stack(per_output).mean()
            total = total + term_raw * weight
            for name, value in zip(names, per_output, strict=True):
                output_key = output_component_key(key, name)
                raw[output_key] = float(value.detach().item())
                weighted[output_key] = float((value * weight / len(names)).detach().item())
                current_weight[output_key] = weight
        return LossAggregate(total=total, raw=raw, weighted=weighted, current_weight=current_weight)

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


def _output_masks(
    term: LossTermConfig,
    params: LossParams,
    masks: Mapping[str, Mapping[str, torch.Tensor]] | None,
    output: str,
) -> dict[str, torch.Tensor] | None:
    """Project per-output masks onto the primitive's ``{source: mask}`` for one output.

    A mask is never shared between outputs; a required mask missing for ``output`` fails.
    """
    source = params.mask.source
    if not params.mask.enabled:
        return None
    by_output = (masks or {}).get(source, {})
    if output not in by_output:
        raise ValueError(
            f"loss {term.name!r} requires batch mask {source!r} for output {output!r}, "
            "but the batch provides none"
        )
    return {source: by_output[output]}


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
