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
from virtual_staining.models.config import (
    DiscriminatorConfig,
    GeneratorConfig,
    ModelConfig,
    ModelName,
    NormName,
)
from virtual_staining.utils.dimensions import parse_wh_size_from_aliases

_MODEL_KEYS: frozenset[str] = frozenset({"name", "generator", "discriminator"})

_GENERATOR_KEYS: frozenset[str] = frozenset(
    {"name", "in_channels", "out_channels", "base_channels", "norm", "dropout", "bilinear"}
)

_DISCRIMINATOR_KEYS: frozenset[str] = frozenset(
    {"name", "in_channels", "ndf", "norm", "use_sigmoid"}
)

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
    from virtual_staining.training.config import (
        AugmentationConfig,
        LossConfig,
        TrainingConfig,
    )


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig | None
    augmentation: AugmentationConfig
    losses: LossConfig | None
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
                "augmentation",
                "losses",
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
        augmentation = _parse_augmentation(raw.get("augmentation", {}))
        losses = _parse_losses(raw["losses"]) if "losses" in raw else None
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
            augmentation=augmentation,
            losses=losses,
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
                "name": self.model.name,
                "generator": {
                    "name": self.model.generator.name,
                    "in_channels": self.model.generator.in_channels,
                    "out_channels": self.model.generator.out_channels,
                    "base_channels": self.model.generator.base_channels,
                    "norm": self.model.generator.norm,
                    "dropout": self.model.generator.dropout,
                    "bilinear": self.model.generator.bilinear,
                },
                "discriminator": {
                    "name": self.model.discriminator.name,
                    "in_channels": self.model.discriminator.in_channels,
                    "ndf": self.model.discriminator.ndf,
                    "norm": self.model.discriminator.norm,
                    "use_sigmoid": self.model.discriminator.use_sigmoid,
                },
            },
        }

        if self.project.manifest_path_override is not None:
            data["manifest_path"] = str(self.project.manifest_path_override)

        if self.training is not None:
            training_data: dict[str, Any] = {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "lr_g": self.training.lr_g,
                "lr_d": self.training.lr_d,
                "beta1": self.training.beta1,
                "beta2": self.training.beta2,
                "seed": self.training.seed,
                "num_workers": self.training.num_workers,
                "validate_rate": self.training.validate_rate,
                "checkpoint_rate": self.training.checkpoint_rate,
                "checkpoint_top_k": self.training.checkpoint_top_k,
                "log_rate": self.training.log_rate,
                "resume": self.training.resume,
            }
            if self.training.scheduler.name != "none":
                training_data["scheduler"] = self.training.scheduler.to_yaml_dict()
            data["training"] = training_data

        data["augmentation"] = self.augmentation.to_yaml_dict()

        if self.losses is not None:
            data["losses"] = self.losses.to_yaml_dict()

        if self.inference is not None:
            data["inference"] = {
                "checkpoint_policy": self.inference.checkpoint_policy,
                "checkpoint_path": (
                    str(self.inference.checkpoint_path) if self.inference.checkpoint_path else None
                ),
                "checkpoint_metric": self.inference.checkpoint_metric,
                "checkpoint_rank": self.inference.checkpoint_rank,
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
                "save_discarded_patches": self.preprocessing.save_discarded_patches,
                "mask_strategy": self.preprocessing.mask_strategy,
                "source_mask_strategy": self.preprocessing.source_mask_strategy,
                "target_mask_strategy": self.preprocessing.target_mask_strategy,
                "mask_scale": self.preprocessing.mask_scale,
                "lowres_mask_filtering": self.preprocessing.lowres_mask_filtering,
                "tiled_io": self.preprocessing.tiled_io,
                "max_memory_gb": self.preprocessing.max_memory_gb,
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
    manifest_path_raw = raw.get("manifest_path")
    project = ProjectConfig(
        dataset_root=Path(raw["dataset_root"]),
        results_path=Path(raw["results_path"]),
        run_name=raw["run_name"],
        image_size=parse_wh_size_from_aliases(raw, ("model_image_size", "image_size"), (256, 256)),
        manifest_path_override=Path(manifest_path_raw) if manifest_path_raw else None,
    )
    project.validate()
    return project


def _parse_model(model_raw: dict[str, Any]) -> ModelConfig:
    reject_unknown_keys(model_raw, _MODEL_KEYS, "model")
    gen_raw = model_raw.get("generator", {})
    reject_unknown_keys(gen_raw, _GENERATOR_KEYS, "model.generator")
    disc_raw = model_raw.get("discriminator", {})
    reject_unknown_keys(disc_raw, _DISCRIMINATOR_KEYS, "model.discriminator")
    model_name = cast(
        ModelName,
        _parse_supported_choice(model_raw.get("name", "pix2pix"), "model.name", {"pix2pix"}),
    )
    generator_name = cast(
        Literal["unet"],
        _parse_supported_choice(gen_raw.get("name", "unet"), "model.generator.name", {"unet"}),
    )
    generator_norm = cast(
        NormName,
        _parse_supported_choice(
            gen_raw.get("norm", "batch"),
            "model.generator.norm",
            {"batch", "instance"},
        ),
    )
    generator_dropout = parse_bool_strict(gen_raw.get("dropout", False), "model.generator.dropout")
    bilinear = parse_bool_strict(gen_raw.get("bilinear", False), "model.generator.bilinear")
    if bilinear:
        raise ValueError(
            "model.generator.bilinear=True is not supported. The bilinear "
            "upsampling path has a channel mismatch bug. Use 'bilinear: false' "
            "(the default)."
        )
    discriminator_name = cast(
        Literal["patchgan"],
        _parse_supported_choice(
            disc_raw.get("name", "patchgan"),
            "model.discriminator.name",
            {"patchgan"},
        ),
    )
    discriminator_norm = cast(
        NormName,
        _parse_supported_choice(
            disc_raw.get("norm", "instance"),
            "model.discriminator.norm",
            {"batch", "instance"},
        ),
    )
    use_sigmoid = parse_bool_strict(
        disc_raw.get("use_sigmoid", False), "model.discriminator.use_sigmoid"
    )
    if use_sigmoid:
        raise ValueError(
            "model.discriminator.use_sigmoid=True cannot be used with "
            "BCEWithLogitsLoss. BCEWithLogitsLoss applies sigmoid internally; "
            "adding a second sigmoid in the discriminator saturates gradients "
            "and breaks training. Set 'use_sigmoid: false' (the default) or "
            "implement explicit loss switching."
        )
    return ModelConfig(
        name=model_name,
        generator=GeneratorConfig(
            name=generator_name,
            in_channels=int(gen_raw.get("in_channels", 3)),
            out_channels=int(gen_raw.get("out_channels", 3)),
            base_channels=int(gen_raw.get("base_channels", 64)),
            norm=generator_norm,
            dropout=generator_dropout,
            bilinear=bilinear,
        ),
        discriminator=DiscriminatorConfig(
            name=discriminator_name,
            in_channels=int(disc_raw.get("in_channels", 6)),
            ndf=int(disc_raw.get("ndf", 64)),
            norm=discriminator_norm,
            use_sigmoid=use_sigmoid,
        ),
    )


def _parse_supported_choice(raw: Any, field: str, choices: set[str]) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{field} must be a string. Supported values: {sorted(choices)}.")
    if raw not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}. Got {raw!r}.")
    return raw


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    from virtual_staining.training.config import (
        _TRAINING_KEYS,
        TrainingConfig,
        parse_learning_rate_scheduler_config,
    )

    data = section_with_shared_fields(raw, "training", set())
    reject_unknown_keys(data, _TRAINING_KEYS, "training")

    epochs = int(data["epochs"])
    scheduler = parse_learning_rate_scheduler_config(
        data.get("scheduler", {}),
        epochs=epochs,
        legacy_lr_schedule=data.get("lr_schedule"),
        legacy_decay_start_epoch=data.get("decay_start_epoch"),
    )
    config = TrainingConfig(
        batch_size=int(data.get("batch_size", 8)),
        epochs=epochs,
        lr_g=float(data.get("lr_g", 2e-4)),
        lr_d=float(data.get("lr_d", 2e-4)),
        beta1=float(data.get("beta1", 0.5)),
        beta2=float(data.get("beta2", 0.999)),
        seed=data.get("seed"),
        num_workers=int(data.get("num_workers", min(4, os.cpu_count() or 1))),
        validate_rate=int(data.get("validate_rate", 10)),
        checkpoint_rate=int(data.get("checkpoint_rate", 10)),
        checkpoint_top_k=int(data.get("checkpoint_top_k", 3)),
        log_rate=int(data.get("log_rate", 15)),
        resume=data.get("resume"),
        scheduler=scheduler,
    )
    config.validate()
    return config


def _parse_augmentation(raw: Any) -> AugmentationConfig:
    from virtual_staining.training.config import parse_augmentation_config

    return parse_augmentation_config(raw)


def _parse_losses(raw: Any) -> LossConfig:
    from virtual_staining.training.config import parse_loss_config

    return parse_loss_config(raw)


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    from virtual_staining.inference.config import _INFERENCE_KEYS, InferenceConfig

    data = section_with_shared_fields(raw, "inference", set())
    reject_unknown_keys(data, _INFERENCE_KEYS, "inference")

    config = InferenceConfig(
        checkpoint_policy=data.get("checkpoint_policy"),
        checkpoint_path=Path(data["checkpoint_path"]) if data.get("checkpoint_path") else None,
        checkpoint_metric=data.get("checkpoint_metric"),
        checkpoint_rank=(
            int(data["checkpoint_rank"])
            if "checkpoint_rank" in data and data["checkpoint_rank"] is not None
            else None
        ),
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
    )
    config.validate()
    return config


def _parse_preprocessing(raw: dict[str, Any]) -> PreprocessingConfig:
    from virtual_staining.data.config import (
        _PREPROCESSING_KEYS,
        PreprocessingConfig,
        _optional_strategy,
        _pair,
    )

    data = section_with_shared_fields(raw, "preprocessing", {"dataset_root", "image_size"})
    reject_unknown_keys(data, _PREPROCESSING_KEYS, "preprocessing")

    max_memory_gb_raw = data.get("max_memory_gb")

    config = PreprocessingConfig(
        dataset_root=Path(data["dataset_root"]),
        source_name=data["source_name"],
        target_name=data["target_name"],
        image_size=parse_wh_size_from_aliases(data, ("patch_size", "image_size"), (256, 256)),
        grid_movement=_pair(data.get("grid_movement"), (256, 256)),
        margin=int(data.get("margin", 200)),
        seed=data.get("seed"),
        save_masks=parse_bool_strict(data.get("save_masks", False), "save_masks"),
        save_discarded_patches=parse_bool_strict(
            data.get("save_discarded_patches", False),
            "save_discarded_patches",
        ),
        mask_strategy=_optional_strategy(
            data.get("mask_strategy", "connected_components"),
            "mask_strategy",
        )
        or "connected_components",
        source_mask_strategy=_optional_strategy(
            data.get("source_mask_strategy"),
            "source_mask_strategy",
        ),
        target_mask_strategy=_optional_strategy(
            data.get("target_mask_strategy"),
            "target_mask_strategy",
        ),
        mask_scale=float(data.get("mask_scale", 1.0)),
        lowres_mask_filtering=parse_bool_strict(
            data.get("lowres_mask_filtering", False),
            "lowres_mask_filtering",
        ),
        tiled_io=parse_bool_strict(data.get("tiled_io", False), "tiled_io"),
        max_memory_gb=None if max_memory_gb_raw is None else float(max_memory_gb_raw),
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
