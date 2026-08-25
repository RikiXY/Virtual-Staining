from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.evaluation import EvaluationConfig
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.config.validation import _TOP_LEVEL_KEYS, reject_unknown_keys


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return value


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
    evaluation: EvaluationConfig | None

    def __post_init__(self) -> None:
        if self.preprocessing is None:
            return
        configured = set(self.preprocessing.inputs.modalities)
        requested = set(self.model.inputs)
        if not requested.issubset(configured):
            raise ValueError("model.inputs must be a subset of preprocessing.inputs.modalities")
        if self.model.target != self.preprocessing.inputs.target_modality:
            raise ValueError("model.target must equal preprocessing.inputs.target_modality")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = load_yaml_mapping(path)
        reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")
        project = ProjectConfig.from_mapping(raw)
        config = cls(
            project=project,
            model=ModelConfig.from_mapping(_section(raw, "model")),
            preprocessing=(
                PreprocessingConfig.from_mapping(
                    _section(raw, "preprocessing"),
                    dataset_root=project.dataset_root,
                    default_image_size=project.image_size,
                )
                if "preprocessing" in raw
                else None
            ),
            training=(
                TrainingConfig.from_mapping(_section(raw, "training"))
                if "training" in raw
                else None
            ),
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
        )
        return config

    def to_dict(self) -> dict[str, Any]:
        data = self.project.to_dict()
        data["model"] = self.model.to_dict()
        if self.preprocessing is not None:
            data["preprocessing"] = self.preprocessing.to_dict()
        if self.training is not None:
            data["training"] = self.training.to_dict()
        if self.inference is not None:
            data["inference"] = self.inference.to_dict()
        if self.evaluation is not None:
            data["evaluation"] = self.evaluation.to_dict()
        return data
