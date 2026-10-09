from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.evaluation import EvaluationConfig
from virtual_staining.config.experiment_data import DataConfig
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.loader import dump_yaml_mapping, load_yaml_mapping
from virtual_staining.config.method import DEFAULT_METHOD_NAME, METHOD_KEYS, MethodConfig
from virtual_staining.config.model import MODEL_KEYS, SUPERSEDED_MODEL_KEYS, ModelConfig
from virtual_staining.config.project import PROJECT_KEYS, ProjectConfig
from virtual_staining.config.stages import VALID_STAGES, StageName
from virtual_staining.config.training import TRAINING_KEYS, TrainingConfig
from virtual_staining.config.validation import reject_superseded_keys, reject_unknown_keys
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


def _validate_preparation_support(data: DataConfig, stages: Sequence[str]) -> None:
    if "prepare" in stages and data.pairing == "unpaired":
        raise ValueError("prepare with data.pairing='unpaired' is unsupported")


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
    method: MethodConfig | None
    model: ModelConfig | None
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
    evaluation: EvaluationConfig | None
    data: DataConfig = field(default_factory=DataConfig)
    stages: tuple[StageName, ...] = ()
    # The definition set this config was resolved with; a live reference, never serialized.
    definitions: Definitions = field(
        default_factory=_default_definitions, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        self.validate_stages(self.stages)
        definition = self.method.definition if self.method is not None else None
        if definition is not None and self.data.pairing != definition.pairing:
            raise ValueError(
                f"method.name={definition.name!r} requires data.pairing={definition.pairing!r}"
            )
        # The protocol defaults to the training pairing but may be overridden independently.
        protocol = (self.evaluation.protocol if self.evaluation else None) or self.data.pairing
        if self.evaluation is not None and protocol == "paired":
            if self.evaluation.reference_collection is not None:
                raise ValueError(
                    "evaluation.reference_collection applies to the unpaired protocol only"
                )
        elif self.evaluation is not None:
            if self.evaluation.metrics is not None:
                raise ValueError(
                    "evaluation.metrics applies to the paired protocol only; the unpaired "
                    "protocol reports fixed appearance-distribution diagnostics"
                )
            if self.evaluation.input_failures != "strict":
                raise ValueError("evaluation.input_failures applies to the paired protocol only")
        if definition is None:
            return
        assert self.model is not None
        self._validate_inference(definition)
        self._resolve_augmentation()
        definition.validate(self)
        if self.evaluation is not None and protocol == "unpaired" and len(self.model.outputs) > 1:
            raise ValueError(
                "evaluation.protocol='unpaired' compares one generated collection with one "
                f"independent reference collection; model.outputs {list(self.model.outputs)} "
                "has several simultaneous outputs and no unambiguous per-output reference "
                "contract exists. Use the paired protocol or a one-output model."
            )
        if self.preprocessing is None:
            return
        inputs = self.preprocessing.inputs
        # Subsets in any order: model order is authoritative and datasets select by name.
        unknown = sorted(set(self.model.inputs) - set(inputs.modalities))
        if unknown:
            raise ValueError(
                f"model.inputs {unknown} are not in preprocessing.inputs.modalities "
                f"{list(inputs.modalities)}"
            )
        unknown = sorted(set(self.model.outputs) - set(inputs.target_modalities))
        if unknown:
            raise ValueError(
                f"model.outputs {unknown} are not in preprocessing.inputs.target_modalities "
                f"{list(inputs.target_modalities)}"
            )

    def _resolve_augmentation(self) -> None:
        """Fill the effective ``photometric_inputs`` and check paired-geometry rules."""
        if self.training is None:
            return
        assert self.model is not None
        augmentation = self.training.augmentation.resolve(
            self.model.inputs,
            self.preprocessing.inputs.reference if self.preprocessing is not None else None,
        )
        object.__setattr__(self, "training", replace(self.training, augmentation=augmentation))
        width, height = self.project.image_size
        if augmentation.enabled and width != height:
            raise ValueError(
                "training.augmentation.enabled=true requires a square image_size: every "
                f"preset includes RandomRotate90, got image_size [{width}, {height}]"
            )

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
        cls,
        raw: Mapping[str, Any],
        definitions: Definitions | None = None,
        *,
        stages: Sequence[str] = (),
    ) -> RunConfig:
        """Resolve a raw run mapping; YAML loading and Python callers share this path.

        ``stages`` selects the union of operation requirements; omission is inspection
        without execution requirements. Supplied sections are always validated.
        ``definitions`` defaults to the built-in set; the selected method owns its options.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("A run configuration must be a mapping")
        reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")
        for section in _TOP_LEVEL_KEYS - PROJECT_KEYS:
            if section in raw:
                _section(raw, section)
        definitions = definitions if definitions is not None else _default_definitions()
        project = ProjectConfig.from_mapping(dict(raw))
        selected = cls.check_stages(stages)
        data = DataConfig.from_mapping(_section(raw, "data"))
        _validate_preparation_support(data, selected)
        inference = (
            InferenceConfig.from_mapping(_section(raw, "inference")) if "inference" in raw else None
        )
        # Supplied method-owned sections are validated even for inactive operations.
        resolve_method = bool(set(selected) - {"prepare"}) or any(
            key in raw for key in ("method", "model", "training", "inference")
        )
        model = None
        training = None
        method = None
        if resolve_method:
            name = _section(raw, "method").get("name", DEFAULT_METHOD_NAME)
            if not isinstance(name, str):
                raise TypeError("method.name must be a string")
            definition = definitions.method(name)
            reject_superseded_keys(_section(raw, "model"), SUPERSEDED_MODEL_KEYS, "model")
            common, owned = _split_sections(raw, definition)
            model = ModelConfig.from_mapping(common["model"])
            training = (
                TrainingConfig.from_mapping(
                    common["training"], method=definition, outputs=model.outputs
                )
                if "training" in raw
                else None
            )
            options = definition.parse_options(
                owned,
                ResolutionContext(
                    definitions=definitions,
                    image_size=project.image_size,
                    inputs=model.inputs,
                    outputs=model.outputs,
                    training=training,
                    stages=selected,
                ),
            )
            method = MethodConfig(definition=definition, options=options)
        return cls(
            project=project,
            method=method,
            data=data,
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
            inference=inference,
            evaluation=(
                EvaluationConfig.from_mapping(_section(raw, "evaluation"), definitions.metrics)
                if "evaluation" in raw
                else None
            ),
            definitions=definitions,
            stages=selected,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        definitions: Definitions | None = None,
        *,
        stages: Sequence[str] = (),
    ) -> RunConfig:
        return cls.from_mapping(load_yaml_mapping(path), definitions, stages=stages)

    @staticmethod
    def check_stages(stages: Sequence[str]) -> tuple[StageName, ...]:
        unknown = [stage for stage in stages if stage not in VALID_STAGES]
        if unknown:
            raise ValueError(
                f"Unknown stage(s): {', '.join(unknown)}. Allowed stages: {', '.join(VALID_STAGES)}"
            )
        return cast(tuple[StageName, ...], tuple(stages))

    def validate_stages(self, stages: Sequence[str]) -> None:
        """Check operation requirements without reading artifacts or creating directories."""
        selected = self.check_stages(stages)
        _validate_preparation_support(self.data, selected)
        for stage in selected:
            if stage == "prepare":
                if self.preprocessing is None:
                    raise ValueError("preprocessing is required for prepare")
                continue
            for name in ("results_path", "run_name"):
                if getattr(self.project, name) is None:
                    raise ValueError(f"{name} is required for {stage}")
            if self.method is None or self.model is None:
                raise ValueError(f"method and model are required for {stage}")
            if stage == "train" and self.training is None:
                raise ValueError("training is required for train")
            if stage == "infer":
                if self.inference is None:
                    raise ValueError("inference is required for infer")
                if (
                    self.inference.checkpoint_path is None
                    and self.inference.checkpoint_policy is None
                ):
                    raise ValueError(
                        "inference.checkpoint_path or inference.checkpoint_policy "
                        "is required for infer"
                    )
            self.method.definition.validate_stage(self, stage)
            if stage == "train" and self.data.pairing == "unpaired":
                missing = (
                    set(self.model.inputs) | set(self.model.outputs)
                ) - self.data.domains.keys()
                if missing:
                    raise ValueError(f"data.domains requires training domains {sorted(missing)}")
            if stage == "evaluate":
                protocol = (
                    self.evaluation.protocol if self.evaluation else None
                ) or self.data.pairing
                if protocol == "paired" and (
                    self.evaluation is None or self.evaluation.metrics is None
                ):
                    from virtual_staining.metrics import resolve_metrics

                    resolve_metrics(None, self.definitions.metrics)
                if protocol == "unpaired":
                    directions = self.method.definition.prediction_directions
                    direction = (
                        self.inference.direction if self.inference else None
                    ) or directions[0]
                    outputs = self.method.definition.prediction_outputs(self, direction)
                    if len(outputs) != 1:
                        raise ValueError(
                            "evaluation.protocol='unpaired' requires one predicted output"
                        )
                    if (
                        not (self.evaluation and self.evaluation.reference_collection)
                        and outputs[0] not in self.data.domains
                    ):
                        raise ValueError(
                            "evaluation.reference_collection or "
                            f"data.domains.{outputs[0]} is required"
                        )

    def resolved_yaml(self) -> str:
        """Canonical snapshot bytes, with execution context outside the authored schema."""
        context = f"# Resolve with stages: {', '.join(self.stages)}\n" if self.stages else ""
        return context + dump_yaml_mapping(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        data = self.project.to_dict()
        owned = self.method.sections() if self.method is not None else {}
        if self.method is not None:
            data["method"] = self.method.to_dict()
        data["data"] = self.data.to_dict()
        if self.model is not None:
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
