from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from virtual_staining.common.dimensions import parse_wh_size_from_aliases
from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.sections import section_with_shared_fields
from virtual_staining.config.validation import (
    _TOP_LEVEL_KEYS,
    parse_bool_strict,
    reject_unknown_keys,
)
from virtual_staining.evaluation.config import _EVALUATION_KEYS, EvaluationConfig
from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig, ModelConfig

_FLAT_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "save_graphs",
        "target_dir",
        "generated_dir",
        "output_dir",
    }
)

if TYPE_CHECKING:
    from virtual_staining.data.config import PreprocessingConfig
    from virtual_staining.inference.config import InferenceConfig
    from virtual_staining.training.config import TrainingConfig


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig | None
    inference: InferenceConfig | None
    preprocessing: PreprocessingConfig | None
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
        training = _parse_training(raw) if "training" in raw or "epochs" in raw else None
        inference = _parse_inference(raw) if "inference" in raw else None
        preprocessing = (
            _parse_preprocessing(raw)
            if "preprocessing" in raw or {"source_name", "target_name"} <= set(raw)
            else None
        )
        evaluation = (
            _parse_evaluation(raw)
            if "evaluation" in raw or any(key in raw for key in _FLAT_EVALUATION_KEYS)
            else None
        )

        return cls(
            project=project,
            model=model,
            training=training,
            inference=inference,
            preprocessing=preprocessing,
            evaluation=evaluation,
        )


def _parse_project(raw: dict[str, Any]) -> ProjectConfig:
    project = ProjectConfig(
        dataset_root=Path(raw["dataset_root"]),
        results_path=Path(raw["results_path"]),
        run_name=raw["run_name"],
        image_size=parse_wh_size_from_aliases(raw, ("model_image_size", "image_size"), (256, 256)),
    )
    project.validate()
    return project


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


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    from virtual_staining.training.config import _TRAINING_KEYS, TrainingConfig

    data = section_with_shared_fields(
        raw, "training", {"dataset_root", "results_path", "run_name", "image_size"}
    )
    reject_unknown_keys(data, _TRAINING_KEYS, "training")

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
    )
    config.validate()
    return config


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    from virtual_staining.inference.config import _INFERENCE_KEYS, InferenceConfig

    data = section_with_shared_fields(
        raw, "inference", {"dataset_root", "results_path", "run_name", "image_size"}
    )
    reject_unknown_keys(data, _INFERENCE_KEYS, "inference")

    config = InferenceConfig(
        checkpoint_policy=data.get("checkpoint_policy"),
        checkpoint_path=Path(data["checkpoint"]) if data.get("checkpoint") else None,
        test_dir=Path(data["test_dir"]) if data.get("test_dir") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
    )
    config.validate()
    return config


def _parse_preprocessing(raw: dict[str, Any]) -> PreprocessingConfig:
    from virtual_staining.data.config import _PREPROCESSING_KEYS, PreprocessingConfig, _pair

    data = section_with_shared_fields(raw, "preprocessing", {"dataset_root", "image_size"})
    reject_unknown_keys(data, _PREPROCESSING_KEYS, "preprocessing")

    config = PreprocessingConfig(
        dataset_root=Path(data["dataset_root"]),
        source_name=data["source_name"],
        target_name=data["target_name"],
        image_size=parse_wh_size_from_aliases(data, ("patch_size", "image_size"), (256, 256)),
        grid_movement=_pair(data.get("grid_movement"), (256, 256)),
        margin=int(data.get("margin", 200)),
        seed=data.get("seed"),
        save_masks=parse_bool_strict(data.get("save_masks", False), "save_masks"),
        train_ratio=float(data.get("train_ratio", 0.8)),
        val_ratio=float(data.get("val_ratio", 0.05)),
        test_ratio=float(data.get("test_ratio", 0.15)),
        min_foreground_ratio=float(data.get("min_foreground_ratio", 0.25)),
        max_white_ratio=float(data.get("max_white_ratio", 0.7)),
        white_threshold=int(data.get("white_threshold", 250)),
        max_largest_white_component_ratio=float(
            data.get("max_largest_white_component_ratio", 0.20)
        ),
    )
    config.validate()
    return config


def _parse_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    data = section_with_shared_fields(
        raw,
        "evaluation",
        {"dataset_root", "results_path", "run_name"},
    )
    reject_unknown_keys(data, _EVALUATION_KEYS, "evaluation")

    return EvaluationConfig(
        save_graphs=parse_bool_strict(data.get("save_graphs", False), "save_graphs"),
        target_dir=Path(data["target_dir"]) if data.get("target_dir") else None,
        generated_dir=Path(data["generated_dir"]) if data.get("generated_dir") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
    )
