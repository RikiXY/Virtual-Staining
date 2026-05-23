from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.config_helpers import write_yaml
from virtual_staining.config import RunConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.models.config import ModelConfig
from virtual_staining.training.config import (
    AugmentationConfig,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    LossConfig,
    LossScheduleConfig,
    TrainingConfig,
)


def _make_project(**overrides: object) -> ProjectConfig:
    defaults: dict[str, object] = {
        "dataset_root": Path("data/root"),
        "results_path": Path("results"),
        "run_name": "my_run",
        "image_size": (256, 256),
    }
    defaults.update(overrides)
    return ProjectConfig(
        dataset_root=defaults["dataset_root"],  # type: ignore[arg-type]
        results_path=defaults["results_path"],  # type: ignore[arg-type]
        run_name=defaults["run_name"],  # type: ignore[arg-type]
        image_size=defaults["image_size"],  # type: ignore[arg-type]
    )


def _make_training_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "batch_size": 8,
        "epochs": 50,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "seed": 42,
        "num_workers": 4,
        "validate_rate": 10,
        "checkpoint_rate": 10,
        "checkpoint_top_k": 3,
        "log_rate": 15,
        "resume": None,
        "scheduler": LearningRateSchedulerConfig(),
        "early_stopping": None,
    }
    defaults.update(overrides)
    return TrainingConfig(
        batch_size=defaults["batch_size"],  # type: ignore[arg-type]
        epochs=defaults["epochs"],  # type: ignore[arg-type]
        lr_g=defaults["lr_g"],  # type: ignore[arg-type]
        lr_d=defaults["lr_d"],  # type: ignore[arg-type]
        beta1=defaults["beta1"],  # type: ignore[arg-type]
        beta2=defaults["beta2"],  # type: ignore[arg-type]
        seed=defaults["seed"],  # type: ignore[arg-type]
        num_workers=defaults["num_workers"],  # type: ignore[arg-type]
        validate_rate=defaults["validate_rate"],  # type: ignore[arg-type]
        checkpoint_rate=defaults["checkpoint_rate"],  # type: ignore[arg-type]
        checkpoint_top_k=defaults["checkpoint_top_k"],  # type: ignore[arg-type]
        log_rate=defaults["log_rate"],  # type: ignore[arg-type]
        resume=defaults["resume"],  # type: ignore[arg-type]
        scheduler=defaults["scheduler"],  # type: ignore[arg-type]
        early_stopping=defaults["early_stopping"],  # type: ignore[arg-type]
    )


def test_frozen() -> None:
    config = _make_training_config()
    with pytest.raises((AttributeError, TypeError)):
        config.epochs = 999  # type: ignore[misc]


def test_run_config_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        training:
          epochs: 20
          batch_size: 4
          seed: 7
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.dataset_root == Path("/data")
    assert run_config.project.run_root == Path("/results") / "yaml_run"
    assert run_config.project.image_size == (128, 128)
    assert run_config.training is not None
    assert run_config.training.epochs == 20
    assert run_config.training.batch_size == 4
    assert run_config.training.seed == 7
    assert run_config.training.checkpoint_top_k == 3
    assert run_config.model == ModelConfig()


def test_training_checkpoint_top_k_parse(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          checkpoint_top_k: 5
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    assert run_config.training.checkpoint_top_k == 5


def test_training_scheduler_parses_linear_decay(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          scheduler:
            name: linear_decay
            decay_start_epoch: 10
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    assert run_config.training.scheduler.name == "linear_decay"
    assert run_config.training.scheduler.decay_start_epoch == 10
    assert run_config.to_yaml_dict()["training"]["scheduler"] == {
        "name": "linear_decay",
        "decay_start_epoch": 10,
    }


def test_training_scheduler_parses_legacy_linear_decay_keys(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          lr_schedule: linear_decay
          decay_start_epoch: 10
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    assert run_config.training.scheduler.name == "linear_decay"
    assert run_config.training.scheduler.decay_start_epoch == 10


def test_training_scheduler_parses_reduce_on_plateau(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          scheduler:
            name: reduce_on_plateau
            monitor: val_ssim
            mode: max
            factor: 0.5
            patience: 5
            min_lr: 0.00002
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    scheduler = run_config.training.scheduler
    assert scheduler.name == "reduce_on_plateau"
    assert scheduler.monitor == "val_ssim"
    assert scheduler.mode == "max"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 5
    assert scheduler.min_lr == pytest.approx(0.00002)


def test_training_early_stopping_parses_valid_config(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          early_stopping:
            monitor: val_ssim
            mode: max
            patience: 5
            min_delta: 0.001
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    early_stopping = run_config.training.early_stopping
    assert early_stopping == EarlyStoppingConfig(
        monitor="val_ssim",
        mode="max",
        patience=5,
        min_delta=0.001,
    )
    assert run_config.to_yaml_dict()["training"]["early_stopping"] == {
        "monitor": "val_ssim",
        "mode": "max",
        "patience": 5,
        "min_delta": 0.001,
    }


@pytest.mark.parametrize(
    ("early_stopping_yaml", "match"),
    [
        ("monitor: train_loss", "training.early_stopping.monitor"),
        ("monitor: val_ssim\n  mode: median", "training.early_stopping.mode"),
        ("monitor: val_ssim\n  patience: -1", "patience"),
        ("monitor: val_ssim\n  min_delta: -0.1", "min_delta"),
    ],
)
def test_training_early_stopping_rejects_invalid_values(
    tmp_path: Path,
    early_stopping_yaml: str,
    match: str,
) -> None:
    yaml_file = tmp_path / "bad_early_stopping.yaml"
    early_stopping_body = textwrap.indent(
        "\n".join(line.strip() for line in early_stopping_yaml.splitlines()),
        "            ",
    )
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          early_stopping:
{early_stopping_body}
    """,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        RunConfig.from_yaml(yaml_file)


def test_training_scheduler_rejects_decay_start_at_or_after_epochs() -> None:
    config = _make_training_config(
        epochs=10,
        scheduler=LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=10),
    )

    with pytest.raises(ValueError, match="decay_start_epoch"):
        config.validate()


@pytest.mark.parametrize(
    ("scheduler_yaml", "match"),
    [
        ("name: cosine", "training.scheduler.name"),
        ("name: reduce_on_plateau\n  monitor: val_loss", "training.scheduler.monitor"),
        ("name: reduce_on_plateau\n  mode: median", "training.scheduler.mode"),
        ("name: reduce_on_plateau\n  factor: 1.0", "factor"),
        ("name: reduce_on_plateau\n  factor: 0.0", "factor"),
        ("name: reduce_on_plateau\n  patience: -1", "patience"),
        ("name: reduce_on_plateau\n  min_lr: -0.1", "min_lr"),
    ],
)
def test_training_scheduler_rejects_invalid_values(
    tmp_path: Path,
    scheduler_yaml: str,
    match: str,
) -> None:
    yaml_file = tmp_path / "bad_scheduler.yaml"
    scheduler_body = textwrap.indent(
        "\n".join(line.strip() for line in scheduler_yaml.splitlines()),
        "            ",
    )
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        training:
          epochs: 20
          scheduler:
{scheduler_body}
    """,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        RunConfig.from_yaml(yaml_file)


def test_run_config_model_explicit_contract_parses(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        model:
          name: pix2pix
          generator:
            name: unet
            in_channels: 1
            out_channels: 3
            base_channels: 32
            norm: instance
            dropout: true
            bilinear: false
          discriminator:
            name: patchgan
            in_channels: 4
            ndf: 32
            norm: batch
            use_sigmoid: false
        training:
          epochs: 1
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.model.name == "pix2pix"
    assert run_config.model.generator.norm == "instance"
    assert run_config.model.generator.dropout is True
    assert run_config.model.discriminator.norm == "batch"


def test_run_config_rejects_unsupported_bilinear_generator(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        model:
          generator:
            bilinear: true
        training:
          epochs: 1
    """,
    )

    with pytest.raises(ValueError, match=r"model\.generator\.bilinear.*bilinear: false"):
        RunConfig.from_yaml(yaml_file)


def test_run_config_rejects_non_bool_bilinear_generator(tmp_path: Path) -> None:
    yaml_file = tmp_path / "train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: yaml_run
        image_size: [128, 128]
        model:
          generator:
            bilinear: "false"
        training:
          epochs: 1
    """,
    )

    with pytest.raises(TypeError, match=r"model\.generator\.bilinear"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_run_yaml_defaults(tmp_path: Path) -> None:
    yaml_file = tmp_path / "minimal.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: minimal_run
        training:
          epochs: 10
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.training is not None
    config = run_config.training
    assert config.batch_size == 8
    assert config.lr_g == pytest.approx(2e-4)
    assert config.log_rate == 15
    assert config.resume is None
    assert run_config.losses is None


def test_augmentation_config_defaults_to_disabled() -> None:
    config = AugmentationConfig()

    assert config.enabled is False
    assert config.expansion_factor == 1
    assert config.effective_expansion_factor == 1
    assert config.intensity == "light"
    assert config.to_yaml_dict() == {
        "enabled": False,
        "expansion_factor": 1,
        "intensity": "light",
    }


@pytest.mark.parametrize("intensity", ["light", "medium", "strong"])
def test_run_config_parses_augmentation_section(tmp_path: Path, intensity: str) -> None:
    yaml_file = tmp_path / "augmentation.yaml"
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: /data
        results_path: /results
        run_name: augmented_run
        training:
          epochs: 1
        augmentation:
          enabled: true
          expansion_factor: 3
          intensity: {intensity}
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.augmentation.enabled is True
    assert run_config.augmentation.expansion_factor == 3
    assert run_config.augmentation.effective_expansion_factor == 3
    assert run_config.augmentation.intensity == intensity
    assert run_config.to_yaml_dict()["augmentation"] == {
        "enabled": True,
        "expansion_factor": 3,
        "intensity": intensity,
    }


@pytest.mark.parametrize(
    ("augmentation_yaml", "match"),
    [
        ("enabled: 'true'", "augmentation.enabled"),
        ("expansion_factor: 0", "expansion_factor"),
        ("expansion_factor: 1.5", "expansion_factor"),
        ("intensity: extreme", "augmentation.intensity"),
        ("unknown: value", "unknown"),
    ],
)
def test_run_config_rejects_invalid_augmentation_section(
    tmp_path: Path,
    augmentation_yaml: str,
    match: str,
) -> None:
    yaml_file = tmp_path / "bad_augmentation.yaml"
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: /data
        results_path: /results
        run_name: bad_augmented_run
        training:
          epochs: 1
        augmentation:
{textwrap.indent(augmentation_yaml, "          ")}
    """,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        RunConfig.from_yaml(yaml_file)


def test_loss_config_defaults_to_empty_lists() -> None:
    config = LossConfig()

    assert config.generator == ()
    assert config.discriminator == ()
    assert config.active_generator == ()
    assert config.active_discriminator == ()
    assert config.to_yaml_dict() == {"generator": [], "discriminator": []}


def test_run_config_from_yaml_parses_explicit_ssim_loss(tmp_path: Path) -> None:
    yaml_file = tmp_path / "losses.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ssim_run
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
              weight: 1.0
              enabled: true
              params:
                data_range: 1.0
                window_size: 11
                sigma: 1.5
                channel_mode: rgb
                reduction: mean
              schedule:
                type: constant
          discriminator: []
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.losses is not None
    assert len(run_config.losses.generator) == 1
    term = run_config.losses.generator[0]
    assert term.name == "ssim"
    assert term.weight == pytest.approx(1.0)
    assert term.enabled is True
    assert term.is_active is True
    assert term.params["window_size"] == 11
    assert term.params["sigma"] == pytest.approx(1.5)
    assert term.schedule.type == "constant"
    assert run_config.losses.active_generator == (term,)
    assert run_config.losses.discriminator == ()


def test_run_config_from_yaml_parses_scheduled_masked_ssim_loss(tmp_path: Path) -> None:
    yaml_file = tmp_path / "losses.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ssim_run
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
              weight: 1.0
              params:
                mask:
                  enabled: true
                  source: foreground_mask
                  foreground_weight: 1.0
                  background_weight: 0.25
                  ignore_empty_mask: true
              schedule:
                type: linear_warmup
                start_epoch: 0
                end_epoch: 4
          discriminator: []
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.losses is not None
    term = run_config.losses.generator[0]
    assert term.requires_mask is True
    assert term.mask.background_weight == pytest.approx(0.25)
    assert term.current_weight(epoch=0) == pytest.approx(0.0)
    assert term.current_weight(epoch=2) == pytest.approx(0.5)
    assert term.current_weight(epoch=4) == pytest.approx(1.0)


def test_lr_scheduler_config_does_not_consume_loss_term_schedule(tmp_path: Path) -> None:
    yaml_file = tmp_path / "schedules.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: schedules_run
        training:
          epochs: 10
          scheduler:
            name: linear_decay
            decay_start_epoch: 5
        losses:
          generator:
            - name: ssim
              weight: 1.0
              schedule:
                type: linear_warmup
                start_epoch: 0
                end_epoch: 4
          discriminator: []
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.training is not None
    assert run_config.training.scheduler.name == "linear_decay"
    assert run_config.losses is not None
    assert run_config.losses.generator[0].schedule.type == "linear_warmup"


def test_run_config_from_yaml_requires_explicit_loss_weight(tmp_path: Path) -> None:
    yaml_file = tmp_path / "losses_minimal.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ssim_run
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
    """,
    )

    with pytest.raises(ValueError, match="weight is required"):
        RunConfig.from_yaml(yaml_file)


def test_zero_weight_and_disabled_losses_are_inactive(tmp_path: Path) -> None:
    yaml_file = tmp_path / "inactive_losses.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: inactive_losses
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
              weight: 0.0
              enabled: true
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)

    assert run_config.losses is not None
    term = run_config.losses.generator[0]
    assert term.enabled is True
    assert term.weight == pytest.approx(0.0)
    assert term.is_active is False
    assert run_config.losses.active_generator == ()


def test_loss_config_to_yaml_dict_preserves_explicit_losses(tmp_path: Path) -> None:
    yaml_file = tmp_path / "losses_round_trip.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ssim_run
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
              weight: 1.0
              enabled: false
              params:
                reduction: mean
          discriminator: []
    """,
    )

    data = RunConfig.from_yaml(yaml_file).to_yaml_dict()

    assert data["losses"] == {
        "generator": [
            {
                "name": "ssim",
                "weight": 1.0,
                "enabled": False,
                "params": {"reduction": "mean"},
                "schedule": {"type": "constant"},
            }
        ],
        "discriminator": [],
    }


@pytest.mark.parametrize(
    ("loss_yaml", "match"),
    [
        ("name: mse\n  weight: 1.0", "losses.generator\\[0\\].name"),
        ("name: ssim\n  weight: -1.0", "weight"),
        ("weight: 1.0", "name is required"),
        ("name: ssim", "weight is required"),
        ("name: ssim\n  weight: 1.0\n  target: image", "target"),
        ("name: ssim\n  weight: 1.0\n  typo: true", "typo"),
        ("name: ssim\n  weight: 1.0\n  schedule: {type: linear}", "schedule"),
        ("name: ssim\n  weight: 1.0\n  schedule: {type: linear_warmup}", "end_epoch"),
        ("name: ssim\n  weight: 1.0\n  schedule: {type: step}", "epoch"),
        ("name: ssim\n  weight: 1.0\n  enabled: 'true'", "enabled"),
        ("name: ssim\n  weight: 1.0\n  params: {window_size: 4}", "window_size"),
        ("name: ssim\n  weight: 1.0\n  params: {data_range: 0.0}", "data_range"),
        ("name: ssim\n  weight: 1.0\n  params: {sigma: 0.0}", "sigma"),
        ("name: ssim\n  weight: 1.0\n  params: {channel_mode: lab}", "channel_mode"),
        ("name: ssim\n  weight: 1.0\n  params: {reduction: median}", "reduction"),
        ("name: ssim\n  weight: 1.0\n  params: {mask: {source: target_mask}}", "source"),
        (
            "name: ssim\n  weight: 1.0\n  params: {mask: {foreground_weight: -1.0}}",
            "foreground_weight",
        ),
        ("name: ms_ssim\n  weight: 1.0", "losses.generator\\[0\\].name"),
    ],
)
def test_loss_config_rejects_invalid_generator_terms(
    tmp_path: Path, loss_yaml: str, match: str
) -> None:
    yaml_file = tmp_path / "bad_losses.yaml"
    loss_body = textwrap.dedent(loss_yaml).strip()
    loss_lines = [line.strip() for line in loss_body.splitlines()]
    list_item = "- " + "\n  ".join(loss_lines)
    indented_loss_yaml = textwrap.indent(list_item, "            ")
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: /data
        results_path: /results
        run_name: bad_losses
        training:
          epochs: 1
        losses:
          generator:
{indented_loss_yaml}
    """,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        RunConfig.from_yaml(yaml_file)


def test_loss_config_rejects_ssim_discriminator_term(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad_discriminator_loss.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: bad_losses
        training:
          epochs: 1
        losses:
          discriminator:
            - name: ssim
              weight: 1.0
    """,
    )

    with pytest.raises(ValueError, match="losses.generator"):
        RunConfig.from_yaml(yaml_file)


def test_loss_config_rejects_duplicate_loss_names(tmp_path: Path) -> None:
    yaml_file = tmp_path / "duplicate_losses.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: duplicate_losses
        training:
          epochs: 1
        losses:
          generator:
            - name: ssim
              weight: 1.0
            - name: ssim
              weight: 0.5
    """,
    )

    with pytest.raises(ValueError, match="Duplicate loss"):
        RunConfig.from_yaml(yaml_file)


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (LossScheduleConfig(type="constant"), {0: 1.0, 5: 1.0}),
        (
            LossScheduleConfig(type="linear_warmup", start_epoch=0, end_epoch=4),
            {0: 0.0, 2: 0.5, 4: 1.0},
        ),
        (
            LossScheduleConfig(type="linear_decay", start_epoch=0, end_epoch=4),
            {0: 1.0, 2: 0.5, 4: 0.0},
        ),
        (LossScheduleConfig(type="step", epoch=3, factor=0.25), {2: 1.0, 3: 0.25}),
        (LossScheduleConfig(type="turn_on_after_epoch", epoch=3), {2: 0.0, 3: 1.0}),
        (LossScheduleConfig(type="turn_off_after_epoch", epoch=3), {2: 1.0, 3: 0.0}),
    ],
)
def test_loss_schedule_multiplier_values(
    schedule: LossScheduleConfig, expected: dict[int, float]
) -> None:
    schedule.validate()
    for epoch, value in expected.items():
        assert schedule.multiplier(epoch=epoch) == pytest.approx(value)


def test_cosine_loss_schedule_boundaries() -> None:
    schedule = LossScheduleConfig(type="cosine", start_epoch=0, end_epoch=4)
    assert schedule.multiplier(epoch=0) == pytest.approx(1.0)
    assert schedule.multiplier(epoch=4) == pytest.approx(0.0)


def test_training_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
          batch_size: 2
          log_rate: 3
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.dataset_root == Path("/data")
    assert run_config.project.run_root == Path("/results") / "section_run"
    assert run_config.project.image_size == (512, 512)
    assert run_config.training is not None
    assert run_config.training.epochs == 30
    assert run_config.training.batch_size == 2
    assert run_config.training.log_rate == 3


def test_inference_from_run_yaml_section(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_path: checkpoints/ep030.pth
          output_dir: /results/section_run/custom_output
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    config = run_config.inference
    assert config.output_dir == Path("/results/section_run/custom_output")
    assert config.checkpoint_path == Path("checkpoints/ep030.pth")


def test_training_yaml_rejects_train_dir_and_val_dir(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
          train_dir: /custom/train
          val_dir: /custom/val
    """,
    )

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in training: train_dir, val_dir"):
        RunConfig.from_yaml(yaml_file)


def test_inference_yaml_rejects_test_dir(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_path: checkpoints/ep030.pth
          test_dir: /custom/test
    """,
    )

    with pytest.raises(ValueError, match="Unknown key\\(s\\) in inference: test_dir"):
        RunConfig.from_yaml(yaml_file)


def test_inference_latest_checkpoint_policy(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_policy: latest
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    config = run_config.inference
    assert config.checkpoint_policy == "latest"
    assert config.checkpoint_path is None
    assert config.output_dir is None


def test_inference_generic_best_checkpoint_policy_with_metric(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_policy: best
          checkpoint_metric: val_ssim
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    assert run_config.inference.checkpoint_policy == "best"
    assert run_config.inference.checkpoint_metric == "val_ssim"


def test_inference_top_k_checkpoint_policy_with_rank(tmp_path: Path) -> None:
    yaml_file = tmp_path / "run.yaml"
    write_yaml(
        yaml_file,
        f"""\
        dataset_root: {tmp_path / "data"}
        results_path: {tmp_path / "results"}
        run_name: section_run
        image_size: [512, 512]
        training:
          epochs: 30
        inference:
          checkpoint_policy: top_k
          checkpoint_metric: val_mae
          checkpoint_rank: 2
    """,
    )

    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.inference is not None
    assert run_config.inference.checkpoint_policy == "top_k"
    assert run_config.inference.checkpoint_metric == "val_mae"
    assert run_config.inference.checkpoint_rank == 2


def test_inference_unknown_checkpoint_policy_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad_policy.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        inference:
          checkpoint_policy: best_ssim
    """,
    )

    with pytest.raises(ValueError, match="Supported values"):
        RunConfig.from_yaml(yaml_file)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"epochs": 0}, "epochs"),
        ({"lr_g": 0.0}, "lr_g"),
        ({"lr_d": -0.1}, "lr_d"),
        ({"beta1": -0.1}, "beta1"),
        ({"beta2": 1.0}, "beta2"),
        ({"num_workers": -1}, "num_workers"),
        ({"validate_rate": 0}, "validate_rate"),
        ({"checkpoint_rate": 0}, "checkpoint_rate"),
        ({"checkpoint_top_k": 0}, "checkpoint_top_k"),
        ({"checkpoint_top_k": -1}, "checkpoint_top_k"),
        ({"log_rate": 0}, "log_rate"),
    ],
)
def test_training_invalid_values_raise_value_error(
    overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _make_training_config(**overrides).validate()


def test_project_config_rejects_empty_run_name(tmp_path: Path) -> None:
    yaml_content = "dataset_root: /data\nresults_path: /results\nrun_name: ''\n"
    yaml_file = tmp_path / "bad.yaml"
    write_yaml(yaml_file, yaml_content)
    with pytest.raises(ValueError, match="run_name"):
        RunConfig.from_yaml(yaml_file)


def test_project_config_rejects_invalid_image_size(tmp_path: Path) -> None:
    yaml_content = (
        "dataset_root: /data\nresults_path: /results\nrun_name: run\nimage_size: [0, 256]\n"
    )
    yaml_file = tmp_path / "bad.yaml"
    write_yaml(yaml_file, yaml_content)
    with pytest.raises(ValueError, match="image_size"):
        RunConfig.from_yaml(yaml_file)


def test_non_square_image_size_training_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "ns_train.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ns_run
        image_size: [320, 256]
        training:
          epochs: 5
    """,
    )
    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.image_size == (320, 256)


def test_non_square_image_size_inference_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "ns_infer.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: ns_run
        image_size: [320, 256]
        inference:
          checkpoint_path: checkpoints/ep010.pth
    """,
    )
    run_config = RunConfig.from_yaml(yaml_file)
    assert run_config.project.image_size == (320, 256)


def test_training_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "typo.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        training:
          epochs: 10
          bathc_size: 4
    """,
    )
    with pytest.raises(ValueError, match="bathc_size"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_yaml_unknown_top_level_key_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "typo_top.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        typo_field: oops
        training:
          epochs: 10
    """,
    )
    with pytest.raises(ValueError, match="typo_field"):
        RunConfig.from_yaml(yaml_file)


def test_inference_from_yaml_unknown_section_key_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "typo_inf.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: my_run
        inference:
          checkpoint_path: checkpoints/ep010.pth
          checkpont_policy: latest
    """,
    )
    with pytest.raises(ValueError, match="checkpont_policy"):
        RunConfig.from_yaml(yaml_file)


def test_training_from_yaml_invalid_image_size_raises_value_error(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad_training.yaml"
    write_yaml(
        yaml_file,
        """\
        dataset_root: /data
        results_path: /results
        run_name: bad_training
        image_size: [256, -1]
    """,
    )
    with pytest.raises(ValueError, match="image_size"):
        RunConfig.from_yaml(yaml_file)


@pytest.mark.parametrize(
    ("inference_yaml", "field"),
    [
        ("checkpoint_path: ''", "checkpoint_path"),
        ("checkpoint_path: '   '", "checkpoint_path"),
        ("checkpoint_policy: best", "checkpoint_metric"),
        ("checkpoint_policy: top_k\ncheckpoint_rank: 0", "checkpoint_rank"),
        ("checkpoint_policy: latest\ncheckpoint_rank: 1", "checkpoint_rank"),
    ],
)
def test_inference_from_yaml_invalid_values_raise_value_error(
    tmp_path: Path, inference_yaml: str, field: str
) -> None:
    yaml_content = (
        "dataset_root: /data\n"
        "results_path: /results\n"
        "run_name: inference_run\n"
        "image_size: [256, 256]\n"
        "inference:\n"
        f"{textwrap.indent(inference_yaml, '  ')}\n"
    )
    yaml_file = tmp_path / "bad_inference.yaml"
    write_yaml(yaml_file, yaml_content)
    with pytest.raises(ValueError, match=field):
        RunConfig.from_yaml(yaml_file)
