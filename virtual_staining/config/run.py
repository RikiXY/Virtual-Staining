from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.sections import section_with_shared_fields
from virtual_staining.config.utilities import (
    _COMPARE_KEYS,
    _COMPARE_PANELS_KEYS,
    _ORGANIZE_KEYS,
    CompareConfig,
    ComparePanelsConfig,
    OrganizeConfig,
)
from virtual_staining.config.validation import (
    _TOP_LEVEL_KEYS,
    parse_bool_strict,
    reject_unknown_keys,
)
from virtual_staining.evaluation.config import _EVALUATION_KEYS, EvaluationConfig
from virtual_staining.models.config import DiscriminatorConfig, GeneratorConfig, ModelConfig
from virtual_staining.utils.dimensions import parse_wh_size_from_aliases

_MODEL_KEYS: frozenset[str] = frozenset({"generator", "discriminator"})

_GENERATOR_KEYS: frozenset[str] = frozenset(
    {"name", "in_channels", "out_channels", "base_channels", "bilinear"}
)

_DISCRIMINATOR_KEYS: frozenset[str] = frozenset({"name", "in_channels", "ndf", "use_sigmoid"})

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
    compare: CompareConfig | None
    compare_panels: ComparePanelsConfig | None
    organize: OrganizeConfig | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = load_yaml_mapping(path)
        if any(
            key in raw
            for key in (
                "training",
                "inference",
                "model",
                "evaluation",
                "preprocessing",
                "compare",
                "compare_panels",
                "organize",
            )
        ):
            reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")

        project = _parse_project(raw)
        model = _parse_model(raw.get("model", {}))
        training = _parse_training(raw) if "training" in raw else None
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
        compare = _parse_compare(raw) if "compare" in raw else None
        compare_panels = _parse_compare_panels(raw) if "compare_panels" in raw else None
        organize = _parse_organize(raw) if "organize" in raw else None

        return cls(
            project=project,
            model=model,
            training=training,
            inference=inference,
            preprocessing=preprocessing,
            evaluation=evaluation,
            compare=compare,
            compare_panels=compare_panels,
            organize=organize,
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of the resolved config."""
        data: dict[str, Any] = {
            "dataset_root": str(self.project.dataset_root),
            "results_path": str(self.project.results_path),
            "run_name": self.project.run_name,
            "image_size": list(self.project.image_size),
            "model": {
                "generator": {
                    "name": self.model.generator.name,
                    "in_channels": self.model.generator.in_channels,
                    "out_channels": self.model.generator.out_channels,
                    "base_channels": self.model.generator.base_channels,
                    "bilinear": self.model.generator.bilinear,
                },
                "discriminator": {
                    "name": self.model.discriminator.name,
                    "in_channels": self.model.discriminator.in_channels,
                    "ndf": self.model.discriminator.ndf,
                    "use_sigmoid": self.model.discriminator.use_sigmoid,
                },
            },
        }

        if self.training is not None:
            data["training"] = {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "lr_g": self.training.lr_g,
                "lr_d": self.training.lr_d,
                "beta1": self.training.beta1,
                "beta2": self.training.beta2,
                "l1_weight": self.training.l1_weight,
                "seed": self.training.seed,
                "num_workers": self.training.num_workers,
                "validate_rate": self.training.validate_rate,
                "checkpoint_rate": self.training.checkpoint_rate,
                "log_rate": self.training.log_rate,
                "resume": self.training.resume,
                "train_dir": str(self.training.train_dir) if self.training.train_dir else None,
                "val_dir": str(self.training.val_dir) if self.training.val_dir else None,
            }

        if self.inference is not None:
            data["inference"] = {
                "checkpoint_policy": self.inference.checkpoint_policy,
                "checkpoint_path": (
                    str(self.inference.checkpoint_path) if self.inference.checkpoint_path else None
                ),
                "test_dir": str(self.inference.test_dir) if self.inference.test_dir else None,
                "output_dir": (
                    str(self.inference.output_dir) if self.inference.output_dir else None
                ),
            }

        if self.preprocessing is not None:
            data["preprocessing"] = {
                "dataset_root": str(self.preprocessing.dataset_root),
                "source_name": self.preprocessing.source_name,
                "target_name": self.preprocessing.target_name,
                "image_size": list(self.preprocessing.image_size),
                "grid_movement": list(self.preprocessing.grid_movement),
                "margin": self.preprocessing.margin,
                "seed": self.preprocessing.seed,
                "save_masks": self.preprocessing.save_masks,
                "train_ratio": self.preprocessing.train_ratio,
                "val_ratio": self.preprocessing.val_ratio,
                "test_ratio": self.preprocessing.test_ratio,
                "min_foreground_ratio": self.preprocessing.min_foreground_ratio,
                "max_white_ratio": self.preprocessing.max_white_ratio,
                "white_threshold": self.preprocessing.white_threshold,
                "max_largest_white_component_ratio": (
                    self.preprocessing.max_largest_white_component_ratio
                ),
            }

        if self.evaluation is not None:
            data["evaluation"] = {
                "save_graphs": self.evaluation.save_graphs,
                "target_dir": (
                    str(self.evaluation.target_dir) if self.evaluation.target_dir else None
                ),
                "generated_dir": (
                    str(self.evaluation.generated_dir) if self.evaluation.generated_dir else None
                ),
                "output_dir": (
                    str(self.evaluation.output_dir) if self.evaluation.output_dir else None
                ),
            }

        if self.compare is not None:
            data["compare"] = {
                "mode": self.compare.mode,
                "run_a": str(self.compare.run_a) if self.compare.run_a else None,
                "run_b": str(self.compare.run_b) if self.compare.run_b else None,
                "csv_a": str(self.compare.csv_a) if self.compare.csv_a else None,
                "csv_b": str(self.compare.csv_b) if self.compare.csv_b else None,
                "label_a": self.compare.label_a,
                "label_b": self.compare.label_b,
                "column": self.compare.column,
                "output_dir": str(self.compare.output_dir) if self.compare.output_dir else None,
                "higher_is_better": self.compare.higher_is_better,
                "lower_is_better": self.compare.lower_is_better,
                "bins": self.compare.bins,
                "min_value": self.compare.min_value,
                "max_value": self.compare.max_value,
                "thresholds": list(self.compare.thresholds) if self.compare.thresholds else None,
                "tolerance": self.compare.tolerance,
                "sample_id_column": self.compare.sample_id_column,
            }

        if self.compare_panels is not None:
            data["compare_panels"] = {
                "mode": self.compare_panels.mode,
                "run_path": (
                    str(self.compare_panels.run_path) if self.compare_panels.run_path else None
                ),
                "hide_graphs_path": self.compare_panels.hide_graphs_path,
                "source_image": (
                    str(self.compare_panels.source_image)
                    if self.compare_panels.source_image
                    else None
                ),
                "generated_image": (
                    str(self.compare_panels.generated_image)
                    if self.compare_panels.generated_image
                    else None
                ),
                "target_image": (
                    str(self.compare_panels.target_image)
                    if self.compare_panels.target_image
                    else None
                ),
                "save_path": (
                    str(self.compare_panels.save_path) if self.compare_panels.save_path else None
                ),
                "with_diagnostics": self.compare_panels.with_diagnostics,
            }

        if self.organize is not None:
            data["organize"] = {
                "run_path": str(self.organize.run_path) if self.organize.run_path else None,
                "metrics_csv": (
                    str(self.organize.metrics_csv) if self.organize.metrics_csv else None
                ),
                "output_dir": str(self.organize.output_dir) if self.organize.output_dir else None,
                "metrics": list(self.organize.metrics) if self.organize.metrics else None,
                "top_k": self.organize.top_k,
                "mode": self.organize.mode,
                "include_all_ranked": self.organize.include_all_ranked,
                "overwrite": self.organize.overwrite,
            }

        return _drop_nones(data)


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_nones(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nones(item) for item in value]
    return value


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
    reject_unknown_keys(model_raw, _MODEL_KEYS, "model")
    gen_raw = model_raw.get("generator", {})
    reject_unknown_keys(gen_raw, _GENERATOR_KEYS, "model.generator")
    disc_raw = model_raw.get("discriminator", {})
    reject_unknown_keys(disc_raw, _DISCRIMINATOR_KEYS, "model.discriminator")
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

    data = section_with_shared_fields(raw, "training", set())
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

    data = section_with_shared_fields(raw, "inference", set())
    reject_unknown_keys(data, _INFERENCE_KEYS, "inference")

    config = InferenceConfig(
        checkpoint_policy=data.get("checkpoint_policy"),
        checkpoint_path=Path(data["checkpoint_path"]) if data.get("checkpoint_path") else None,
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


def _parse_compare(raw: dict[str, Any]) -> CompareConfig:
    data = section_with_shared_fields(raw, "compare", set())
    reject_unknown_keys(data, _COMPARE_KEYS, "compare")
    raw_mode = str(data.get("mode", "paired"))
    if raw_mode not in {"paired", "unpaired"}:
        raise ValueError("compare.mode must be 'paired' or 'unpaired'")
    mode = cast(Literal["paired", "unpaired"], raw_mode)
    thresholds = data.get("thresholds")
    return CompareConfig(
        mode=mode,
        run_a=Path(data["run_a"]) if data.get("run_a") else None,
        run_b=Path(data["run_b"]) if data.get("run_b") else None,
        csv_a=Path(data["csv_a"]) if data.get("csv_a") else None,
        csv_b=Path(data["csv_b"]) if data.get("csv_b") else None,
        label_a=data.get("label_a"),
        label_b=data.get("label_b"),
        column=str(data.get("column", "ssim")),
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        higher_is_better=(
            parse_bool_strict(data["higher_is_better"], "higher_is_better")
            if "higher_is_better" in data
            else None
        ),
        lower_is_better=(
            parse_bool_strict(data["lower_is_better"], "lower_is_better")
            if "lower_is_better" in data
            else None
        ),
        bins=int(data.get("bins", 30)),
        min_value=float(data["min_value"]) if data.get("min_value") is not None else None,
        max_value=float(data["max_value"]) if data.get("max_value") is not None else None,
        thresholds=tuple(float(value) for value in thresholds) if thresholds is not None else None,
        tolerance=float(data.get("tolerance", 0.0)),
        sample_id_column=str(data.get("sample_id_column", "sample_id")),
    )


def _parse_compare_panels(raw: dict[str, Any]) -> ComparePanelsConfig:
    data = section_with_shared_fields(raw, "compare_panels", set())
    reject_unknown_keys(data, _COMPARE_PANELS_KEYS, "compare_panels")
    raw_mode = str(data.get("mode", "from_metrics"))
    if raw_mode not in {"single", "from_metrics"}:
        raise ValueError("compare_panels.mode must be 'single' or 'from_metrics'")
    mode = cast(Literal["single", "from_metrics"], raw_mode)
    return ComparePanelsConfig(
        mode=mode,
        run_path=Path(data["run_path"]) if data.get("run_path") else None,
        hide_graphs_path=(
            parse_bool_strict(data["hide_graphs_path"], "hide_graphs_path")
            if "hide_graphs_path" in data
            else False
        ),
        source_image=Path(data["source_image"]) if data.get("source_image") else None,
        generated_image=Path(data["generated_image"]) if data.get("generated_image") else None,
        target_image=Path(data["target_image"]) if data.get("target_image") else None,
        save_path=Path(data["save_path"]) if data.get("save_path") else None,
        with_diagnostics=(
            parse_bool_strict(data["with_diagnostics"], "with_diagnostics")
            if "with_diagnostics" in data
            else False
        ),
    )


def _parse_organize(raw: dict[str, Any]) -> OrganizeConfig:
    data = section_with_shared_fields(raw, "organize", set())
    reject_unknown_keys(data, _ORGANIZE_KEYS, "organize")
    raw_mode = str(data.get("mode", "hardlink"))
    if raw_mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("organize.mode must be one of: hardlink, symlink, copy")
    mode = cast(Literal["hardlink", "symlink", "copy"], raw_mode)
    metrics = data.get("metrics")
    return OrganizeConfig(
        run_path=Path(data["run_path"]) if data.get("run_path") else None,
        metrics_csv=Path(data["metrics_csv"]) if data.get("metrics_csv") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        metrics=tuple(str(metric) for metric in metrics) if metrics is not None else None,
        top_k=int(data.get("top_k", 20)),
        mode=mode,
        include_all_ranked=(
            parse_bool_strict(data["include_all_ranked"], "include_all_ranked")
            if "include_all_ranked" in data
            else False
        ),
        overwrite=(
            parse_bool_strict(data["overwrite"], "overwrite") if "overwrite" in data else False
        ),
    )
