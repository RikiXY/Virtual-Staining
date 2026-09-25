from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from virtual_staining.config.data import PreprocessingConfig
from virtual_staining.config.evaluation import EvaluationConfig
from virtual_staining.config.experiment_data import DataConfig
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.method import MethodConfig
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import PROJECT_KEYS, ProjectConfig
from virtual_staining.config.training import TrainingConfig
from virtual_staining.config.validation import reject_unknown_keys

_TOP_LEVEL_KEYS = PROJECT_KEYS | frozenset(
    {"preprocessing", "training", "inference", "evaluation", "method", "model", "data"}
)
_METHOD_LOSSES: dict[str, frozenset[str]] = {
    "pix2pix": frozenset({"adversarial_bce", "l1", "ssim"}),
    "cyclegan": frozenset({"adversarial_lsgan", "cycle_l1", "identity_l1"}),
}
# Two stride-2 stages must round-trip exactly; residual reflection padding needs >= 2 px.
_RESNET_SIZE_MULTIPLE = 4
_RESNET_MIN_SIZE = 8


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a YAML mapping")
    return value


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    method: MethodConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
    evaluation: EvaluationConfig | None
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        self._validate_method_contract()
        if self.preprocessing is None:
            return
        configured = set(self.preprocessing.inputs.modalities)
        requested = set(self.model.inputs)
        if not requested.issubset(configured):
            raise ValueError("model.inputs must be a subset of preprocessing.inputs.modalities")
        if self.model.target != self.preprocessing.inputs.target_modality:
            raise ValueError("model.target must equal preprocessing.inputs.target_modality")

    def _validate_method_contract(self) -> None:
        method = self.method.name
        if method == "pix2pix":
            if self.data.pairing != "paired":
                raise ValueError("method.name='pix2pix' requires data.pairing='paired'")
            if self.model.generator.architecture != "concat_unet":
                raise ValueError(
                    "method.name='pix2pix' requires model.generator.architecture='concat_unet'"
                )
            if self.inference is not None and self.inference.direction is not None:
                raise ValueError("inference.direction is supported only for method.name='cyclegan'")
            if self.evaluation is not None and self.evaluation.protocol == "unpaired":
                raise ValueError(
                    "evaluation.protocol='unpaired' requires method.name='cyclegan' "
                    "(pix2pix has no independent data.domains collections)"
                )
        else:
            self._validate_cyclegan()
        if self.training is not None:
            losses = self.training.losses
            unsupported = sorted(
                {term.name for term in (*losses.generator, *losses.discriminator)}
                - _METHOD_LOSSES[method]
            )
            if unsupported:
                raise ValueError(
                    f"training.losses {unsupported} are not supported by method.name={method!r}; "
                    f"supported: {sorted(_METHOD_LOSSES[method])}"
                )

    def _validate_cyclegan(self) -> None:
        if self.data.pairing != "unpaired":
            raise ValueError("method.name='cyclegan' requires data.pairing='unpaired'")
        if len(self.model.inputs) != 1:
            raise ValueError(
                "method.name='cyclegan' requires exactly one model.inputs entry (domain A); "
                f"got {list(self.model.inputs)}"
            )
        domain_a, domain_b = self.model.inputs[0], self.model.target
        if domain_a == domain_b:
            raise ValueError(
                "cyclegan model.inputs[0] and model.target must name different domains"
            )
        missing = sorted({domain_a, domain_b} - set(self.data.domains))
        extra = sorted(set(self.data.domains) - {domain_a, domain_b})
        if missing or extra:
            raise ValueError(
                f"data.domains keys must be exactly [{domain_a!r}, {domain_b!r}] "
                f"(model.inputs[0], model.target); missing={missing}, extra={extra}"
            )
        if self.model.generator.architecture != "resnet":
            raise ValueError(
                "method.name='cyclegan' requires model.generator.architecture='resnet'"
            )
        width, height = self.project.image_size
        if (
            width % _RESNET_SIZE_MULTIPLE
            or height % _RESNET_SIZE_MULTIPLE
            or min(width, height) < _RESNET_MIN_SIZE
        ):
            raise ValueError(
                f"image_size {[width, height]} is invalid for the resnet generator: both "
                f"dimensions must be multiples of {_RESNET_SIZE_MULTIPLE} and at least "
                f"{_RESNET_MIN_SIZE}"
            )
        monitors: list[tuple[str, str]] = []
        if self.training is not None:
            training = self.training
            if training.augmentation.enabled:
                raise ValueError(
                    "method.name='cyclegan' requires training.augmentation.enabled=false"
                )
            generator = {term.name for term in training.losses.active_generator}
            discriminator = {term.name for term in training.losses.active_discriminator}
            for role, names, required in (
                ("generator", generator, "adversarial_lsgan"),
                ("generator", generator, "cycle_l1"),
                ("discriminator", discriminator, "adversarial_lsgan"),
            ):
                if required not in names:
                    raise ValueError(
                        f"method.name='cyclegan' requires an active training.losses.{role} "
                        f"term {required!r}"
                    )
            if training.scheduler.name == "reduce_on_plateau":
                monitors.append(("training.scheduler.monitor", training.scheduler.monitor))
            if training.early_stopping is not None:
                monitors.append(
                    ("training.early_stopping.monitor", training.early_stopping.monitor)
                )
        if self.inference is not None and self.inference.checkpoint_metric is not None:
            monitors.append(("inference.checkpoint_metric", self.inference.checkpoint_metric))
        for field_name, monitor in monitors:
            if monitor.startswith("val_"):
                raise ValueError(
                    f"{field_name}={monitor!r} is a paired image-fidelity metric that "
                    "method.name='cyclegan' does not report; use loss_G_val or a loss_val_* column"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = load_yaml_mapping(path)
        reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")
        project = ProjectConfig.from_mapping(raw)
        config = cls(
            project=project,
            method=MethodConfig.from_mapping(_section(raw, "method")),
            data=DataConfig.from_mapping(_section(raw, "data")),
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
