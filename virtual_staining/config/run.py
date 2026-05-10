from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from virtual_staining.common.dimensions import parse_wh_size_from_aliases
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.sections import section_with_shared_fields
from virtual_staining.config.validation import _TOP_LEVEL_KEYS, reject_unknown_keys
from virtual_staining.evaluation.config import EvaluationConfig
from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig, ModelConfig

if TYPE_CHECKING:
    from virtual_staining.training.config import InferenceConfig, TrainingConfig


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: Any | None
    evaluation: EvaluationConfig | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = load_yaml_mapping(path)
        if any(
            key in raw for key in ("training", "inference", "model", "evaluation", "preprocessing")
        ):
            reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")

        project = _parse_project(raw)
        model = _parse_model(raw.get("model", {}))
        training = _parse_training(raw, project) if "training" in raw or "epochs" in raw else None
        inference = _parse_inference(raw, project) if "inference" in raw else None
        preprocessing = None
        evaluation = _parse_evaluation(raw.get("evaluation", {})) if "evaluation" in raw else None

        run_config = cls(
            project=project,
            model=model,
            training=training,
            inference=inference,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )
        return run_config


def _parse_project(raw: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        dataset_root=Path(raw["dataset_root"]),
        results_path=Path(raw["results_path"]),
        run_name=raw["run_name"],
        image_size=parse_wh_size_from_aliases(raw, ("model_image_size", "image_size"), (256, 256)),
    )


def _parse_model(model_raw: dict[str, Any]) -> ModelConfig:
    gen_raw = model_raw.get("generator", {})
    disc_raw = model_raw.get("discriminator", {})
    return ModelConfig(
        generator=GeneratorConfig(
            name=gen_raw.get("name", "unet"),
            in_channels=int(gen_raw.get("in_channels", 3)),
            out_channels=int(gen_raw.get("out_channels", 3)),
            base_channels=int(gen_raw.get("base_channels", 64)),
            bilinear=bool(gen_raw.get("bilinear", False)),
        ),
        discriminator=DiscriminatorConfig(
            name=disc_raw.get("name", "patchgan"),
            in_channels=int(disc_raw.get("in_channels", 6)),
            ndf=int(disc_raw.get("ndf", 64)),
            use_sigmoid=bool(disc_raw.get("use_sigmoid", False)),
        ),
    )


def _parse_training(raw: dict[str, Any], project: ProjectConfig) -> TrainingConfig:
    from virtual_staining.training.config import _TRAINING_KEYS, TrainingConfig

    data = section_with_shared_fields(
        raw, "training", {"dataset_root", "results_path", "run_name", "image_size"}
    )
    reject_unknown_keys(data, _TRAINING_KEYS, "training")
    section_project = _parse_project(data)

    config = TrainingConfig(
        batch_size=int(data.get("batch_size", 8)),
        epochs=int(data["epochs"]),
        lr_g=float(data.get("lr_g", 2e-4)),
        lr_d=float(data.get("lr_d", 2e-4)),
        beta1=float(data.get("beta1", 0.5)),
        beta2=float(data.get("beta2", 0.999)),
        l1_weight=float(data.get("l1_weight", 25.0)),
        seed=data.get("seed"),
        num_workers=int(data.get("num_workers", min(4, os.cpu_count() or 1))),
        validate_rate=int(data.get("validate_rate", 10)),
        checkpoint_rate=int(data.get("checkpoint_rate", 10)),
        log_rate=int(data.get("log_rate", 15)),
        resume=data.get("resume"),
        train_dir=Path(data["train_dir"]) if data.get("train_dir") else None,
        val_dir=Path(data["val_dir"]) if data.get("val_dir") else None,
        project=section_project,
    )
    config.validate()
    return config


def _parse_inference(raw: dict[str, Any], project: ProjectConfig) -> InferenceConfig:
    from virtual_staining.training.config import _INFERENCE_KEYS, InferenceConfig

    data = section_with_shared_fields(
        raw, "inference", {"dataset_root", "results_path", "run_name", "image_size"}
    )
    reject_unknown_keys(data, _INFERENCE_KEYS, "inference")
    section_project = _parse_project(data)

    config = InferenceConfig(
        checkpoint_policy=data.get("checkpoint_policy"),
        checkpoint_path=Path(data["checkpoint"]) if data.get("checkpoint") else None,
        test_dir=Path(data["test_dir"]) if data.get("test_dir") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        project=section_project,
    )
    config.validate()
    return config


def _parse_evaluation(eval_raw: dict[str, Any]) -> EvaluationConfig:
    from virtual_staining.config.validation import parse_bool_strict

    return EvaluationConfig(
        save_graphs=parse_bool_strict(eval_raw.get("save_graphs", False), "save_graphs"),
        target_dir=Path(eval_raw["target_dir"]) if eval_raw.get("target_dir") else None,
        generated_dir=Path(eval_raw["generated_dir"]) if eval_raw.get("generated_dir") else None,
        output_dir=Path(eval_raw["output_dir"]) if eval_raw.get("output_dir") else None,
    )
