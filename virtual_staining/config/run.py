from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.evaluation import EvaluationConfig
from virtual_staining.config.experiment_data import ExperimentDataConfig
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.method import MethodConfig
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
    method: MethodConfig
    data: ExperimentDataConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
    evaluation: EvaluationConfig | None

    def __post_init__(self) -> None:
        expected_pairing = "unpaired" if self.method.name == "cyclegan" else "paired"
        if self.method.name != "custom" and self.data.pairing != expected_pairing:
            raise ValueError(
                f"method '{self.method.name}' requires data.pairing='{expected_pairing}'"
            )
        if self.method.name == "cyclegan":
            if self.model.generator.architecture == "concat_unet":
                raise ValueError("CycleGAN requires a 'resnet' or custom tensor generator")
            if len(self.model.inputs) != 1:
                raise ValueError("CycleGAN requires exactly one model input domain")
            required_domains = {self.model.inputs[0], self.model.target}
            if set(self.data.domains) != required_domains:
                raise ValueError(
                    "CycleGAN data.domains must contain exactly the model source and target "
                    f"domains: {sorted(required_domains)}"
                )
            if self.training is not None:
                if self.training.augmentation.enabled:
                    raise ValueError(
                        "CycleGAN currently requires training.augmentation.enabled=false"
                    )
                generator_losses = {term.name for term in self.training.losses.active_generator}
                discriminator_losses = {
                    term.name for term in self.training.losses.active_discriminator
                }
                required_generator = {"adversarial_lsgan", "cycle_l1"}
                if not required_generator.issubset(generator_losses):
                    raise ValueError(
                        "CycleGAN training.losses.generator requires adversarial_lsgan and cycle_l1"
                    )
                if "adversarial_lsgan" not in discriminator_losses:
                    raise ValueError(
                        "CycleGAN training.losses.discriminator requires adversarial_lsgan"
                    )
            if self.inference is not None and self.inference.direction is None:
                object.__setattr__(self, "inference", replace(self.inference, direction="A_to_B"))
        elif self.inference is not None and self.inference.direction == "B_to_A":
            raise ValueError("inference.direction='B_to_A' is only supported by CycleGAN")
        elif self.method.name == "pix2pix" and self.model.generator.architecture == "resnet":
            raise ValueError("Pix2Pix requires a 'concat_unet' or custom mapping generator")
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
            method=MethodConfig.from_mapping(_section(raw, "method")),
            data=ExperimentDataConfig.from_mapping(_section(raw, "data")),
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
        data["method"] = self.method.to_dict()
        data["data"] = self.data.to_dict()
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
