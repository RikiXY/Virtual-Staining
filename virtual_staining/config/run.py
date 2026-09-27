from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.evaluation import EvaluationConfig
from virtual_staining.config.experiment_data import DataConfig
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.method import DEFAULT_METHOD_NAME, METHOD_KEYS, MethodConfig
from virtual_staining.config.model import MODEL_KEYS, ModelConfig
from virtual_staining.config.project import PROJECT_KEYS, ProjectConfig
from virtual_staining.config.training import TRAINING_KEYS, TrainingConfig
from virtual_staining.config.validation import reject_unknown_keys
from virtual_staining.definitions import Definitions, MethodDefinition, ResolutionContext

_TOP_LEVEL_KEYS = PROJECT_KEYS | frozenset(
    {"preprocessing", "training", "inference", "evaluation", "method", "model", "data"}
)
# Sections whose keys are split between the framework and the selected method definition.
_SHARED_SECTION_KEYS: Mapping[str, frozenset[str]] = {
    "method": METHOD_KEYS,
    "model": MODEL_KEYS,
    "training": TRAINING_KEYS,
}


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return value


def _default_definitions() -> Definitions:
    from virtual_staining.methods.builtin import builtin_definitions

    return builtin_definitions()


def _split_sections(
    raw: Mapping[str, Any], definition: MethodDefinition
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split shared sections into framework keys and keys owned by ``definition``."""
    unsupported = sorted(set(definition.owned_keys) - set(_SHARED_SECTION_KEYS))
    if unsupported:
        raise ValueError(
            f"method definition {definition.name!r} may own keys only in "
            f"{sorted(_SHARED_SECTION_KEYS)}; got {unsupported}"
        )
    common: dict[str, dict[str, Any]] = {}
    owned: dict[str, dict[str, Any]] = {}
    for name, framework_keys in _SHARED_SECTION_KEYS.items():
        method_keys = definition.owned_keys.get(name, frozenset())
        if method_keys & framework_keys:
            raise ValueError(
                f"method definition {definition.name!r} cannot own framework keys "
                f"{sorted(method_keys & framework_keys)} in {name}"
            )
        section = _section(raw, name)
        reject_unknown_keys(section, framework_keys | method_keys, name)
        common[name] = {key: value for key, value in section.items() if key in framework_keys}
        owned[name] = {key: value for key, value in section.items() if key in method_keys}
    return common, owned


@dataclass(frozen=True)
class RunConfig:
    """A run configuration resolved against explicitly supplied method definitions."""

    project: ProjectConfig
    method: MethodConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
    evaluation: EvaluationConfig | None
    data: DataConfig = field(default_factory=DataConfig)
    # The definition set this config was resolved with; a live reference, never serialized.
    definitions: Definitions = field(
        default_factory=_default_definitions, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        definition = self.method.definition
        if self.data.pairing != definition.pairing:
            raise ValueError(
                f"method.name={definition.name!r} requires data.pairing={definition.pairing!r}"
            )
        self._validate_inference(definition)
        if (
            self.evaluation is not None
            and self.evaluation.protocol == "unpaired"
            and self.data.pairing != "unpaired"
        ):
            raise ValueError(
                "evaluation.protocol='unpaired' requires data.pairing='unpaired' "
                "(independent data.domains collections)"
            )
        definition.validate(self)
        if self.preprocessing is None:
            return
        configured = set(self.preprocessing.inputs.modalities)
        requested = set(self.model.inputs)
        if not requested.issubset(configured):
            raise ValueError("model.inputs must be a subset of preprocessing.inputs.modalities")
        if self.model.target != self.preprocessing.inputs.target_modality:
            raise ValueError("model.target must equal preprocessing.inputs.target_modality")

    def _validate_inference(self, definition: MethodDefinition) -> None:
        if self.inference is None:
            return
        direction = self.inference.direction
        directions = list(definition.prediction_directions)
        if direction is not None and len(directions) < 2:
            raise ValueError(
                f"inference.direction is not supported by method.name={definition.name!r}, "
                f"which has the single prediction direction {directions}"
            )
        if direction is not None and direction not in directions:
            raise ValueError(
                f"inference.direction must be one of {directions} for "
                f"method.name={definition.name!r}. Got {direction!r}."
            )
        if self.inference.checkpoint_metric is not None:
            definition.checkpoint_metric_mode(
                self.inference.checkpoint_metric, "inference.checkpoint_metric"
            )

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], definitions: Definitions | None = None
    ) -> RunConfig:
        """Resolve a raw run mapping; YAML loading and Python callers share this path.

        ``definitions`` defaults to the built-in set. ``method.name`` is resolved before
        any method-owned key is validated, so the selected definition owns its options.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("A run configuration must be a mapping")
        reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")
        definitions = definitions if definitions is not None else _default_definitions()
        project = ProjectConfig.from_mapping(dict(raw))
        name = _section(raw, "method").get("name", DEFAULT_METHOD_NAME)
        if not isinstance(name, str):
            raise TypeError("method.name must be a string")
        definition = definitions.method(name)
        common, owned = _split_sections(raw, definition)
        model = ModelConfig.from_mapping(common["model"])
        training = (
            TrainingConfig.from_mapping(common["training"], method=definition)
            if "training" in raw
            else None
        )
        options = definition.parse_options(
            owned,
            ResolutionContext(
                definitions=definitions,
                image_size=project.image_size,
                inputs=model.inputs,
                target=model.target,
                training=training,
            ),
        )
        return cls(
            project=project,
            method=MethodConfig(definition=definition, options=options),
            data=DataConfig.from_mapping(_section(raw, "data")),
            model=model,
            preprocessing=(
                PreprocessingConfig.from_mapping(
                    _section(raw, "preprocessing"),
                    dataset_root=project.dataset_root,
                    default_image_size=project.image_size,
                )
                if "preprocessing" in raw
                else None
            ),
            training=training,
            inference=(
                InferenceConfig.from_mapping(_section(raw, "inference"))
                if "inference" in raw
                else None
            ),
            evaluation=(
                EvaluationConfig.from_mapping(_section(raw, "evaluation"))
                if "evaluation" in raw
                else None
            ),
            definitions=definitions,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, definitions: Definitions | None = None) -> RunConfig:
        return cls.from_mapping(load_yaml_mapping(path), definitions)

    def to_dict(self) -> dict[str, Any]:
        data = self.project.to_dict()
        owned = self.method.sections()
        data["method"] = self.method.to_dict()
        data["data"] = self.data.to_dict()
        data["model"] = {**self.model.to_dict(), **owned.get("model", {})}
        if self.preprocessing is not None:
            data["preprocessing"] = self.preprocessing.to_dict()
        if self.training is not None:
            data["training"] = {**self.training.to_dict(), **owned.get("training", {})}
        if self.inference is not None:
            data["inference"] = self.inference.to_dict()
        if self.evaluation is not None:
            data["evaluation"] = self.evaluation.to_dict()
        return data
