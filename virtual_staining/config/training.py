from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from virtual_staining.checkpoint_selection import CheckpointMode
from virtual_staining.config.validation import parse_bool_strict, parse_choice, reject_unknown_keys

if TYPE_CHECKING:
    from virtual_staining.definitions import MethodDefinition

AugmentationIntensity = Literal["light", "medium", "strong"]

TRAINING_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "epochs",
        "seed",
        "num_workers",
        "validate_rate",
        "checkpoint_rate",
        "checkpoint_top_k",
        "log_rate",
        "resume",
        "early_stopping",
        "augmentation",
    }
)
_EARLY_STOPPING_KEYS: frozenset[str] = frozenset({"monitor", "mode", "patience", "min_delta"})
_AUGMENTATION_KEYS: frozenset[str] = frozenset(
    {"enabled", "expansion_factor", "intensity", "photometric_inputs"}
)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Stop when a method-reported validation monitor stops improving."""

    monitor: str
    mode: CheckpointMode
    patience: int = 15
    min_delta: float = 0.0

    def validate(self) -> None:
        if not self.monitor.strip():
            raise ValueError("training.early_stopping.monitor must not be blank")
        if self.mode not in set(get_args(CheckpointMode)):
            raise ValueError("training.early_stopping.mode must be one of ['max', 'min']")
        if self.patience < 0:
            raise ValueError("training.early_stopping.patience must be >= 0")
        if self.min_delta < 0:
            raise ValueError("training.early_stopping.min_delta must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


@dataclass(frozen=True)
class AugmentationConfig:
    """Paired augmentation; ``photometric_inputs`` None means "resolve the default".

    ``resolve`` fills the effective list: empty for ``light`` (no photometric
    transforms), otherwise the preparation reference input when it is selected, else the
    first selected model input. Targets and masks never receive photometric transforms.
    """

    enabled: bool = False
    expansion_factor: int = 1
    intensity: AugmentationIntensity = "light"
    photometric_inputs: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.expansion_factor < 1:
            raise ValueError("augmentation.expansion_factor must be greater than or equal to 1")
        if self.intensity not in set(get_args(AugmentationIntensity)):
            raise ValueError(
                f"augmentation.intensity must be one of {sorted(get_args(AugmentationIntensity))}"
            )
        names = self.photometric_inputs
        if names is None:
            return
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"augmentation.photometric_inputs has duplicate names {duplicates}")
        if names and self.intensity == "light":
            raise ValueError(
                "augmentation.photometric_inputs must be empty for intensity='light', which "
                "has no photometric transforms"
            )

    def resolve(self, model_inputs: tuple[str, ...], reference: str | None) -> AugmentationConfig:
        """Return this config with the effective ``photometric_inputs`` for ``model_inputs``."""
        names = self.photometric_inputs
        if names is None:
            if self.intensity == "light":
                names = ()
            else:
                names = (reference if reference in model_inputs else model_inputs[0],)
        unknown = [name for name in names if name not in model_inputs]
        if unknown:
            raise ValueError(
                f"augmentation.photometric_inputs {unknown} are not selected model.inputs "
                f"{list(model_inputs)}; targets and masks never receive photometric transforms"
            )
        return replace(self, photometric_inputs=tuple(names))

    @property
    def effective_expansion_factor(self) -> int:
        return self.expansion_factor if self.enabled else 1

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": self.enabled,
            "expansion_factor": self.expansion_factor,
            "intensity": self.intensity,
        }
        if self.photometric_inputs is not None:
            data["photometric_inputs"] = list(self.photometric_inputs)
        return data


def _parse_augmentation_config(raw: Any) -> AugmentationConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("augmentation must be a YAML mapping")
    reject_unknown_keys(raw, _AUGMENTATION_KEYS, "augmentation")
    expansion_factor = raw.get("expansion_factor", 1)
    if isinstance(expansion_factor, bool) or not isinstance(expansion_factor, int):
        raise TypeError("augmentation.expansion_factor must be an integer")
    photometric = raw.get("photometric_inputs")
    if photometric is not None:
        if isinstance(photometric, str) or not isinstance(photometric, list | tuple):
            raise TypeError("augmentation.photometric_inputs must be a list of input names")
        if not all(isinstance(name, str) for name in photometric):
            raise TypeError("augmentation.photometric_inputs must contain names")
        photometric = tuple(photometric)
    config = AugmentationConfig(
        photometric_inputs=photometric,
        enabled=parse_bool_strict(raw.get("enabled", False), "augmentation.enabled"),
        expansion_factor=expansion_factor,
        intensity=cast(
            AugmentationIntensity,
            parse_choice(
                raw.get("intensity", "light"),
                "augmentation.intensity",
                set(get_args(AugmentationIntensity)),
            ),
        ),
    )
    config.validate()
    return config


def _parse_early_stopping_config(
    raw: Any, method: MethodDefinition, outputs: tuple[str, ...]
) -> EarlyStoppingConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("training.early_stopping must be a YAML mapping")
    reject_unknown_keys(raw, _EARLY_STOPPING_KEYS, "training.early_stopping")
    monitor = raw.get("monitor", method.resolve_default_monitor(outputs))
    if monitor is None:
        raise ValueError(
            f"training.early_stopping.monitor is required for method.name={method.name!r} "
            f"with model.outputs {list(outputs)}"
        )
    if not isinstance(monitor, str):
        raise TypeError("training.early_stopping.monitor must be a string")
    # The selected method decides which validation columns exist and their direction.
    default_mode = method.monitor_mode(monitor, "training.early_stopping.monitor")
    config = EarlyStoppingConfig(
        monitor=monitor,
        mode=cast(
            CheckpointMode,
            parse_choice(
                raw.get("mode", default_mode),
                "training.early_stopping.mode",
                set(get_args(CheckpointMode)),
            ),
        ),
        patience=int(raw.get("patience", 15)),
        min_delta=float(raw.get("min_delta", 0.0)),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class TrainingConfig:
    """Generic training lifecycle owned by ``Trainer``; optimization belongs to the method."""

    batch_size: int
    epochs: int
    seed: int | None
    num_workers: int
    validate_rate: int
    checkpoint_rate: int
    checkpoint_top_k: int = 3
    log_rate: int = 15
    resume: str | None = None
    early_stopping: EarlyStoppingConfig | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any], *, method: MethodDefinition, outputs: tuple[str, ...]
    ) -> TrainingConfig:
        reject_unknown_keys(data, TRAINING_KEYS, "training")
        if "epochs" not in data:
            raise ValueError("training.epochs is required")
        return cls(
            batch_size=int(data.get("batch_size", 8)),
            epochs=int(data["epochs"]),
            seed=data.get("seed"),
            num_workers=int(data.get("num_workers", min(4, os.cpu_count() or 1))),
            validate_rate=int(data.get("validate_rate", 10)),
            checkpoint_rate=int(data.get("checkpoint_rate", 10)),
            checkpoint_top_k=int(data.get("checkpoint_top_k", 3)),
            log_rate=int(data.get("log_rate", 15)),
            resume=data.get("resume"),
            early_stopping=_parse_early_stopping_config(
                data.get("early_stopping"), method, outputs
            ),
            augmentation=_parse_augmentation_config(data.get("augmentation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "validate_rate": self.validate_rate,
            "checkpoint_rate": self.checkpoint_rate,
            "checkpoint_top_k": self.checkpoint_top_k,
            "log_rate": self.log_rate,
            "resume": self.resume,
            "augmentation": self.augmentation.to_dict(),
        }
        if self.early_stopping is not None:
            data["early_stopping"] = self.early_stopping.to_dict()
        return {key: value for key, value in data.items() if value is not None}

    def validate(self) -> None:
        for field_name, value in (
            ("batch_size", self.batch_size),
            ("epochs", self.epochs),
            ("validate_rate", self.validate_rate),
            ("checkpoint_rate", self.checkpoint_rate),
            ("checkpoint_top_k", self.checkpoint_top_k),
            ("log_rate", self.log_rate),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.early_stopping is not None:
            self.early_stopping.validate()
        self.augmentation.validate()
