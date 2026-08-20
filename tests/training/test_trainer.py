from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tests.image_helpers import write_rgb_pair
from tests.manifest_helpers import make_manifest_record
from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest, Split
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.checkpoint_selection import update_checkpoint_selection
from virtual_staining.training.config import (
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    TrainingConfig,
)
from virtual_staining.training.loss_config import LossConfig, LossTermConfig
from virtual_staining.training.results import EpochMetrics
from virtual_staining.training.trainer import Trainer
from virtual_staining.training.validator import validate_epoch


def _make_project(dataset_root: Path, results_path: Path, run_name: str) -> ProjectConfig:
    return ProjectConfig(
        dataset_root=dataset_root,
        results_path=results_path,
        run_name=run_name,
        image_size=(32, 32),
    )


def _pix2pix_losses() -> LossConfig:
    return LossConfig(
        generator=(
            LossTermConfig(name="adversarial_bce", weight=1.0),
            LossTermConfig(name="l1", weight=25.0),
        ),
        discriminator=(LossTermConfig(name="adversarial_bce", weight=1.0),),
    )


def _validate(trainer: Trainer, epoch: int = 0) -> EpochMetrics:
    return validate_epoch(
        epoch=epoch,
        generator=trainer.generator,
        discriminator=trainer.discriminator,
        val_loader=trainer.val_loader,
        loss_evaluator=trainer._loss_evaluator,
        losses=trainer.losses,
        device=trainer.device,
        amp_enabled=trainer._amp_enabled,
        output_dir=trainer._output_val_dir,
    )


class _FailingDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise AssertionError("validation image metrics must not require discriminator outputs")


def _make_manifest_dataset(
    dataset_root: Path,
    split: Split,
    prefixes: list[str],
    transform: transforms.Compose,
) -> PairedManifestDataset:
    split_dir = dataset_root / "splits" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for prefix in prefixes:
        write_rgb_pair(split_dir, prefix, size=(32, 32))
        records.append(make_manifest_record(prefix, split, ext=".png", width=32, height=32))
    manifest = DatasetManifest(records=tuple(records), dataset_root=dataset_root)
    return PairedManifestDataset(manifest.filter_split(split), transform=transform)


def _make_train_val_loaders(
    dataset_root: Path,
    project: ProjectConfig,
    *,
    train_prefixes: list[str],
    val_prefixes: list[str],
    batch_size: int = 1,
    shuffle: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(project.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_loader = DataLoader(
        _make_manifest_dataset(dataset_root, "train", train_prefixes, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    val_loader = DataLoader(
        _make_manifest_dataset(dataset_root, "val", val_prefixes, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return train_loader, val_loader


def _make_resume_trainer(
    config: TrainingConfig,
    run_paths: RunPaths,
    project: ProjectConfig,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
) -> Trainer:
    device = torch.device("cpu")
    train_loader, val_loader = _make_train_val_loaders(
        project.dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )
    return Trainer(
        config=config,
        run_paths=run_paths,
        generator=generator.to(device),
        discriminator=discriminator.to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        image_size=project.image_size,
        train_dir=project.dataset_root / "splits" / "train",
        val_dir=project.dataset_root / "splits" / "val",
        losses=_pix2pix_losses(),
    )


def _make_trainer(
    tmp_path: Path,
    checkpoint_rate: int,
    scheduler: LearningRateSchedulerConfig | None = None,
) -> tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig]:
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "smoke_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=42,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=checkpoint_rate,
        log_rate=1,
        scheduler=scheduler or LearningRateSchedulerConfig(),
    )
    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )

    device = torch.device("cpu")
    return (
        Trainer(
            config=config,
            run_paths=run_paths,
            generator=UNetGenerator().to(device),
            discriminator=PatchGANDiscriminator().to(device),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            image_size=project.image_size,
            train_dir=dataset_root / "splits" / "train",
            val_dir=dataset_root / "splits" / "val",
            losses=_pix2pix_losses(),
        ),
        config,
        run_paths,
        project,
    )


@pytest.fixture()
def smoke_trainer(tmp_path: Path) -> tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig]:
    """Trainer that never saves a checkpoint (checkpoint_rate=2 > epochs=1)."""
    return _make_trainer(tmp_path, checkpoint_rate=2)


@pytest.fixture()
def checkpointing_trainer(
    tmp_path: Path,
) -> tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig]:
    """Trainer that saves a checkpoint every epoch, for round-trip tests."""
    return _make_trainer(tmp_path, checkpoint_rate=1)


def test_trainer_smoke_run_creates_expected_files(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, run_paths, _ = smoke_trainer

    trainer.train(seed=42)

    run_root = run_paths.root
    assert (run_paths.metrics_dir / "train.csv").exists()
    assert (run_paths.metrics_dir / "validation.csv").exists()
    assert (run_paths.metrics_dir / "all.csv").exists()
    assert (run_paths.logs_dir / "training.log").exists()
    assert not (run_root / "run_metadata.json").exists()


def test_trainer_metrics_csv_structure(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, _ = smoke_trainer

    trainer.train(seed=42)

    train_metrics_path = run_paths.metrics_dir / "train.csv"
    validation_metrics_path = run_paths.metrics_dir / "validation.csv"
    all_metrics_path = run_paths.metrics_dir / "all.csv"
    with train_metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        train_rows = list(csv.DictReader(metrics_file))
    with validation_metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        validation_rows = list(csv.DictReader(metrics_file))
    with all_metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        all_rows = list(csv.DictReader(metrics_file))

    assert len(train_rows) == config.epochs
    assert len(validation_rows) == config.epochs
    assert len(all_rows) == config.epochs
    assert {
        "epoch",
        "loss_G_train",
        "loss_D_train",
    } <= set(train_rows[0].keys())
    assert {
        "epoch",
        "loss_G_val",
        "loss_D_val",
        "val_ssim",
        "val_mae",
        "val_rmse",
        "val_psnr",
        "val_pcc_gray",
        "val_pcc_rgb_mean",
    } <= set(validation_rows[0].keys())
    assert {
        "epoch",
        "loss_G_train",
        "loss_D_train",
        "loss_G_val",
        "loss_D_val",
        "val_ssim",
        "val_mae",
        "val_rmse",
        "val_psnr",
        "val_pcc_gray",
        "val_pcc_rgb_mean",
    } <= set(all_rows[0].keys())
    for row in validation_rows:
        assert row["loss_G_val"] != ""
        assert row["loss_D_val"] != ""
        assert row["val_ssim"] != ""
        assert row["val_mae"] != ""
        assert row["val_rmse"] != ""
    for row in all_rows:
        assert row["loss_G_train"] != ""
        assert row["loss_D_train"] != ""
        assert row["loss_G_val"] != ""
        assert row["loss_D_val"] != ""
        assert row["val_ssim"] != ""
        assert row["val_mae"] != ""
        assert row["val_rmse"] != ""
    for row in train_rows:
        assert float(row["loss_G_train"]) > 0
        assert float(row["loss_D_train"]) > 0


def test_trainer_train_losses_are_epoch_averages(tmp_path: Path) -> None:
    """Train losses in metrics/train.csv must be averages over all batches."""
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "avg_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=0,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=2,
        log_rate=1,
    )

    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000", "00001_00001"],
        val_prefixes=["00256_00000"],
    )

    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    trainer = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=PatchGANDiscriminator().to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=dataset_root / "splits" / "train",
        val_dir=dataset_root / "splits" / "val",
        losses=_pix2pix_losses(),
    )
    trainer.train(seed=0)

    metrics_path = run_paths.metrics_dir / "train.csv"
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert float(rows[0]["loss_G_train"]) > 0
    assert float(rows[0]["loss_D_train"]) > 0


def test_trainer_metrics_csv_includes_configured_loss_components(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "loss_component_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=0,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=2,
        log_rate=1,
    )
    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )
    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    losses = LossConfig(
        generator=(
            LossTermConfig(name="adversarial_bce", weight=1.0),
            LossTermConfig(name="l1", weight=25.0),
            LossTermConfig(name="ssim", weight=1.0, params={"window_size": 3}),
        ),
        discriminator=(LossTermConfig(name="adversarial_bce", weight=1.0),),
    )

    trainer = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=PatchGANDiscriminator().to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=dataset_root / "splits" / "train",
        val_dir=dataset_root / "splits" / "val",
        losses=losses,
    )
    trainer.train(seed=0)

    with (run_paths.metrics_dir / "train.csv").open(newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with (run_paths.metrics_dir / "validation.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (run_paths.metrics_dir / "all.csv").open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    row = rows[0]
    train_row = train_rows[0]
    all_row = all_rows[0]
    assert "loss_train_raw_generator_ssim" in train_row
    assert "loss_train_weighted_generator_ssim" in train_row
    assert "loss_train_current_weight_generator_ssim" in train_row
    assert "loss_val_raw_generator_ssim" in row
    assert "loss_val_weighted_generator_ssim" in row
    assert "loss_val_current_weight_generator_ssim" in row
    assert "loss_train_raw_discriminator_adversarial_bce" in train_row
    assert "loss_train_total_generator" in train_row
    assert "loss_val_total_generator" in row
    assert float(train_row["loss_train_raw_generator_ssim"]) >= 0.0
    assert float(train_row["loss_train_weighted_generator_ssim"]) >= 0.0
    assert float(train_row["loss_train_current_weight_generator_ssim"]) == pytest.approx(1.0)
    assert train_row["loss_G_train"] == train_row["loss_train_total_generator"]
    assert row["loss_G_val"] == row["loss_val_total_generator"]
    assert "loss_train_raw_generator_ssim" in all_row
    assert "loss_val_raw_generator_ssim" in all_row
    assert all_row["loss_G_train"] == all_row["loss_train_total_generator"]
    assert all_row["loss_G_val"] == all_row["loss_val_total_generator"]


def test_validate_restores_models_that_started_in_train_mode(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, _run_paths, _project = smoke_trainer
    trainer.generator.train()
    trainer.discriminator.train()

    _validate(trainer)

    assert trainer.generator.training is True
    assert trainer.discriminator.training is True


def test_validate_preserves_models_that_started_in_eval_mode(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, _run_paths, _project = smoke_trainer
    trainer.generator.eval()
    trainer.discriminator.eval()

    _validate(trainer)

    assert trainer.generator.training is False
    assert trainer.discriminator.training is False


def test_validate_image_metrics_do_not_require_discriminator_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "reconstruction_val_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=1,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=0,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=2,
        log_rate=1,
    )
    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )
    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    losses = LossConfig(generator=(LossTermConfig(name="l1", weight=1.0),), discriminator=())
    trainer = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=_FailingDiscriminator().to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=dataset_root / "splits" / "train",
        val_dir=dataset_root / "splits" / "val",
        losses=losses,
    )

    metrics = _validate(trainer)

    assert metrics.loss_D == pytest.approx(0.0)
    assert math.isfinite(metrics.image["val_ssim"])
    assert metrics.image["val_mae"] >= 0.0


def test_trainer_checkpoint_round_trip(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer

    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    train_loader, val_loader = _make_train_val_loaders(
        project.dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )

    trainer_2 = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=PatchGANDiscriminator().to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=project.dataset_root / "splits" / "train",
        val_dir=project.dataset_root / "splits" / "val",
        losses=_pix2pix_losses(),
    )

    start_epoch = trainer_2.resume(checkpoint_path)
    assert start_epoch == 1


def test_resume_accepts_latest_and_relative_checkpoint_names(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer
    trainer.train(seed=42)

    relative_trainer = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )
    latest_trainer = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )

    assert relative_trainer.resume("ep000.pth") == 1
    assert latest_trainer.resume("latest") == 1


def test_resume_rejects_invalid_checkpoint_paths(tmp_path: Path) -> None:
    trainer, _config, _run_paths, _project = _make_trainer(tmp_path, checkpoint_rate=1)

    with pytest.raises(ValueError, match=r"must end with '.pth'"):
        trainer.resume("checkpoint.txt")
    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        trainer.resume("missing.pth")
    with pytest.raises(FileNotFoundError, match="resume='latest'.*no checkpoints found"):
        trainer.resume("latest")


def test_resume_restores_scheduler_state(tmp_path: Path) -> None:
    trainer, config, run_paths, project = _make_trainer(
        tmp_path,
        checkpoint_rate=1,
        scheduler=LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=0),
    )
    assert trainer._scheduler_G is not None
    trainer._opt_G.step()
    trainer._opt_D.step()
    trainer._step_lr_schedulers(epoch=0, val_metrics=None)
    checkpoint_path = trainer._checkpoint_manager.save(0)

    resumed = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )
    assert resumed._scheduler_G is not None

    assert resumed.resume(checkpoint_path) == 1
    assert resumed._scheduler_G.state_dict()["last_epoch"] == 1


def test_checkpoint_architecture_metadata_present(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, run_paths, _ = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "architecture" in ck
    gen = ck["architecture"]["generator"]
    assert gen["class"] == "UNetGenerator"
    assert gen["in_channels"] == 3
    assert gen["out_channels"] == 3
    assert gen["base_channels"] == 64
    assert gen["norm"] == "batch"
    assert gen["dropout"] is False
    assert gen["bilinear"] is False
    assert gen["output_activation"] == "tanh"
    assert "name" not in ck["architecture"]
    assert ck["format_version"] == 2
    assert ck["normalization_contract"] == {
        "input_range": "[-1, 1]",
        "output_range": "[-1, 1]",
    }
    disc = ck["architecture"]["discriminator"]
    assert disc["class"] == "PatchGANDiscriminator"
    assert disc["in_channels"] == 6
    assert disc["ndf"] == 64
    assert disc["norm"] == "instance"
    assert disc["use_sigmoid"] is False


def test_training_writes_best_checkpoint_record(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, run_paths, _ = checkpointing_trainer
    result = trainer.train(seed=42)

    best_record = json.loads((run_paths.checkpoints_dir / "best.json").read_text(encoding="utf-8"))

    loss_selection = best_record["metrics"]["loss_G_val"]
    assert "default_metric" not in best_record
    assert loss_selection["mode"] == "min"
    assert loss_selection["best"]["epoch"] == 0
    assert loss_selection["best"]["checkpoint_path"] == "ep000.pth"
    assert isinstance(loss_selection["best"]["metric_value"], float)
    assert result.best_checkpoint_path == run_paths.checkpoints_dir / "ep000.pth"


def _make_policy_selection_trainer(
    tmp_path: Path,
    *,
    epochs: int = 2,
    validate_rate: int = 1,
    checkpoint_top_k: int = 3,
    losses: LossConfig | None = None,
    discriminator: nn.Module | None = None,
    scheduler: LearningRateSchedulerConfig | None = None,
    early_stopping: EarlyStoppingConfig | None = None,
) -> tuple[Trainer, RunPaths]:
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "policy_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=epochs,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=0,
        num_workers=0,
        validate_rate=validate_rate,
        checkpoint_rate=1,
        checkpoint_top_k=checkpoint_top_k,
        log_rate=1,
        scheduler=scheduler or LearningRateSchedulerConfig(),
        early_stopping=early_stopping,
    )
    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )
    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    trainer = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=(discriminator or PatchGANDiscriminator()).to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=dataset_root / "splits" / "train",
        val_dir=dataset_root / "splits" / "val",
        losses=losses or _pix2pix_losses(),
    )
    return trainer, run_paths


def _stub_epoch_metrics() -> EpochMetrics:
    return EpochMetrics(loss_G=1.0, loss_D=1.0)


def _stub_optimizer_epoch(trainer: Trainer) -> EpochMetrics:
    trainer._opt_D.step()
    trainer._opt_G.step()
    return _stub_epoch_metrics()


def test_training_best_checkpoint_policy_uses_higher_is_better_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(
        tmp_path,
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=float(10 - epoch),
            loss_D=1.0,
            image={"val_ssim": [0.2, 0.8][epoch], "val_mae": [0.4, 0.1][epoch]},
        ),
    )

    result = trainer.train(seed=0)

    best_record = json.loads((run_paths.checkpoints_dir / "best.json").read_text(encoding="utf-8"))
    ssim_selection = best_record["metrics"]["val_ssim"]
    mae_selection = best_record["metrics"]["val_mae"]
    assert "default_metric" not in best_record
    assert ssim_selection["mode"] == "max"
    assert ssim_selection["best"]["epoch"] == 1
    assert ssim_selection["best"]["checkpoint_path"] == "ep001.pth"
    assert ssim_selection["best"]["metric_value"] == pytest.approx(0.8)
    assert mae_selection["mode"] == "min"
    assert mae_selection["best"]["epoch"] == 1
    ranked_values = [
        (record["rank"], record["epoch"], record["metric_value"])
        for record in ssim_selection["records"]
    ]
    assert ranked_values == [(1, 1, 0.8), (2, 0, 0.2)]
    assert (
        ssim_selection["records"][0]["checkpoint_path"] == ssim_selection["best"]["checkpoint_path"]
    )
    assert result.best_checkpoint_path == run_paths.checkpoints_dir / "ep001.pth"


def test_training_best_checkpoint_policy_uses_lower_is_better_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(
        tmp_path,
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=float(epoch + 1),
            loss_D=1.0,
            image={"val_mae": [0.4, 0.1][epoch]},
        ),
    )

    trainer.train(seed=0)

    best_record = json.loads((run_paths.checkpoints_dir / "best.json").read_text(encoding="utf-8"))
    mae_selection = best_record["metrics"]["val_mae"]
    assert "default_metric" not in best_record
    assert mae_selection["mode"] == "min"
    assert mae_selection["best"]["epoch"] == 1
    assert mae_selection["best"]["metric_value"] == pytest.approx(0.1)
    ranked_values = [
        (record["rank"], record["epoch"], record["metric_value"])
        for record in mae_selection["records"]
    ]
    assert ranked_values == [(1, 1, 0.1), (2, 0, 0.4)]


def test_training_reports_ranked_loss_checkpoint_in_progress_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(tmp_path)
    progress_updates: list[dict[str, object]] = []

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(loss_G=float(epoch + 1), loss_D=1.0),
    )
    monkeypatch.setattr(
        "virtual_staining.training.trainer.emit_progress_update",
        lambda **kwargs: progress_updates.append(kwargs),
    )

    result = trainer.train(seed=0)

    assert result.best_checkpoint_path == run_paths.checkpoints_dir / "ep000.pth"
    assert progress_updates[-1]["best_checkpoint_name"] == "ep000.pth"
    assert progress_updates[-1]["best_checkpoint_loss_G_val"] == pytest.approx(1.0)


def test_resumed_training_restores_ranked_loss_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(tmp_path)
    checkpoint_path = trainer._checkpoint_manager.save(0)
    update_checkpoint_selection(
        run_paths.checkpoints_dir,
        metrics={"loss_G_val": 0.5},
        modes={"loss_G_val": "min"},
        top_k=3,
        epoch=0,
        checkpoint_path=checkpoint_path,
    )
    progress_updates: list[dict[str, object]] = []

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(loss_G=float("nan"), loss_D=1.0),
    )
    monkeypatch.setattr(
        "virtual_staining.training.trainer.emit_progress_update",
        lambda **kwargs: progress_updates.append(kwargs),
    )

    result = trainer.train(seed=0, start_epoch=1)

    assert result.best_checkpoint_path == checkpoint_path
    assert progress_updates[-1]["best_checkpoint_name"] == "ep000.pth"
    assert progress_updates[-1]["best_checkpoint_loss_G_val"] == pytest.approx(0.5)


def test_linear_decay_scheduler_steps_active_optimizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(
        tmp_path,
        scheduler=LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=1),
    )
    initial_lr_g = trainer._opt_G.param_groups[0]["lr"]
    initial_lr_d = trainer._opt_D.param_groups[0]["lr"]

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(loss_G=1.0, loss_D=1.0, image={"val_ssim": 0.5}),
    )

    trainer.train(seed=0)

    assert trainer._opt_G.param_groups[0]["lr"] < initial_lr_g
    assert trainer._opt_D.param_groups[0]["lr"] < initial_lr_d


def test_reduce_on_plateau_scheduler_steps_from_validation_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(
        tmp_path,
        scheduler=LearningRateSchedulerConfig(
            name="reduce_on_plateau",
            monitor="val_ssim",
            mode="max",
            factor=0.5,
            patience=0,
        ),
    )
    initial_lr_g = trainer._opt_G.param_groups[0]["lr"]
    initial_lr_d = trainer._opt_D.param_groups[0]["lr"]

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=1.0,
            loss_D=1.0,
            image={"val_ssim": [0.8, 0.7][epoch]},
        ),
    )

    trainer.train(seed=0)

    assert trainer._opt_G.param_groups[0]["lr"] == pytest.approx(initial_lr_g * 0.5)
    assert trainer._opt_D.param_groups[0]["lr"] == pytest.approx(initial_lr_d * 0.5)


def test_scheduler_skips_discriminator_when_discriminator_loss_inactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(
        tmp_path,
        losses=LossConfig(generator=(LossTermConfig(name="l1", weight=1.0),), discriminator=()),
        scheduler=LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=1),
    )
    initial_lr_d = trainer._opt_D.param_groups[0]["lr"]

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(loss_G=1.0, loss_D=0.0, image={"val_ssim": 0.5}),
    )

    trainer.train(seed=0)

    assert trainer._opt_G.param_groups[0]["lr"] < 2e-4
    assert trainer._opt_D.param_groups[0]["lr"] == pytest.approx(initial_lr_d)


def test_metric_checkpoint_selection_does_not_require_discriminator_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(
        tmp_path,
        losses=LossConfig(generator=(LossTermConfig(name="l1", weight=1.0),), discriminator=()),
        discriminator=_FailingDiscriminator(),
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_epoch_metrics())
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=1.0,
            loss_D=0.0,
            image={"val_ssim": [0.7, 0.6][epoch]},
        ),
    )

    trainer.train(seed=0)

    best_record = json.loads((run_paths.checkpoints_dir / "best.json").read_text(encoding="utf-8"))
    ssim_selection = best_record["metrics"]["val_ssim"]
    assert ssim_selection["best"]["epoch"] == 0
    assert ssim_selection["best"]["metric_value"] == pytest.approx(0.7)
    assert ssim_selection["records"][0]["epoch"] == 0
    assert ssim_selection["records"][0]["metric_value"] == pytest.approx(0.7)


def test_early_stopping_max_mode_stops_after_patience_validation_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, run_paths = _make_policy_selection_trainer(
        tmp_path,
        epochs=5,
        validate_rate=2,
        early_stopping=EarlyStoppingConfig(
            monitor="val_ssim",
            mode="max",
            patience=1,
            min_delta=0.0,
        ),
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=1.0,
            loss_D=1.0,
            image={"val_ssim": {1: 0.8, 3: 0.7}[epoch]},
        ),
    )

    result = trainer.train(seed=0)

    assert result.stopped_early is True
    assert result.final_epoch == 3
    assert result.stop_epoch == 3
    assert result.early_stopping_monitor == "val_ssim"
    assert result.early_stopping_mode == "max"
    assert result.early_stopping_best_epoch == 1
    assert result.early_stopping_best_value == pytest.approx(0.8)
    with (run_paths.metrics_dir / "validation.csv").open(newline="", encoding="utf-8") as f:
        validation_rows = list(csv.DictReader(f))
    with (run_paths.metrics_dir / "all.csv").open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    assert [row["epoch"] for row in validation_rows] == ["1", "3"]
    assert [row["epoch"] for row in all_rows] == ["0", "1", "2", "3"]
    assert (run_paths.checkpoints_dir / "ep003.pth").exists()


def test_early_stopping_min_mode_can_monitor_validation_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(
        tmp_path,
        epochs=4,
        early_stopping=EarlyStoppingConfig(
            monitor="loss_val_total_generator",
            mode="min",
            patience=1,
            min_delta=0.0,
        ),
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=[0.5, 0.6, 0.7, 0.8][epoch],
            loss_D=1.0,
            raw={"generator_l1": [0.5, 0.6, 0.7, 0.8][epoch]},
            weighted={"generator_l1": [0.5, 0.6, 0.7, 0.8][epoch]},
            current_weight={"generator_l1": 1.0},
            image={"val_mae": [0.3, 0.4, 0.5, 0.6][epoch]},
        ),
    )

    result = trainer.train(seed=0)

    assert result.stopped_early is True
    assert result.final_epoch == 1
    assert result.stop_epoch == 1
    assert result.early_stopping_best_epoch == 0
    assert result.early_stopping_best_value == pytest.approx(0.5)


def test_omitted_early_stopping_preserves_full_epoch_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(tmp_path, epochs=3)

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_optimizer_epoch(trainer))
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=1.0,
            loss_D=1.0,
            image={"val_ssim": [0.8, 0.7, 0.6][epoch]},
        ),
    )

    result = trainer.train(seed=0)

    assert result.stopped_early is False
    assert result.final_epoch == 2
    assert result.stop_epoch is None


def test_early_stopping_on_validation_image_metric_without_discriminator_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _run_paths = _make_policy_selection_trainer(
        tmp_path,
        epochs=3,
        losses=LossConfig(generator=(LossTermConfig(name="l1", weight=1.0),), discriminator=()),
        discriminator=_FailingDiscriminator(),
        early_stopping=EarlyStoppingConfig(monitor="val_mae", mode="min", patience=1),
    )

    monkeypatch.setattr(trainer, "_train_epoch", lambda *args: _stub_epoch_metrics())
    monkeypatch.setattr(
        trainer,
        "_validate",
        lambda epoch: EpochMetrics(
            loss_G=1.0,
            loss_D=0.0,
            image={"val_mae": [0.1, 0.2, 0.3][epoch]},
        ),
    )

    result = trainer.train(seed=0)

    assert result.stopped_early is True
    assert result.final_epoch == 1
    assert result.early_stopping_monitor == "val_mae"


def test_load_checkpoint_validates_matching_architecture(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    trainer_2 = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )
    start_epoch = trainer_2.resume(checkpoint_path)
    assert start_epoch == 1


def test_load_checkpoint_raises_on_architecture_mismatch(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    trainer_mismatch = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(base_channels=32), PatchGANDiscriminator()
    )
    with pytest.raises(ValueError, match="base_channels"):
        trainer_mismatch.resume(checkpoint_path)


def test_short_run_writes_final_checkpoint(tmp_path: Path) -> None:
    trainer, _config, run_paths, _ = _make_trainer(tmp_path, checkpoint_rate=10)
    trainer.train(seed=42)

    checkpoints = sorted(run_paths.checkpoints_dir.glob("ep*.pth"))
    assert len(checkpoints) == 1
    assert checkpoints[0].name == "ep000.pth"


def test_no_duplicate_final_checkpoint_when_already_checkpointed(tmp_path: Path) -> None:
    trainer, _config, run_paths, _ = _make_trainer(tmp_path, checkpoint_rate=1)
    trainer.train(seed=42)

    checkpoints = sorted(run_paths.checkpoints_dir.glob("ep*.pth"))
    assert len(checkpoints) == 1
    assert checkpoints[0].name == "ep000.pth"


def test_checkpoint_rate_creates_multiple_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_project(dataset_root, tmp_path / "results", "multi_epoch_run")
    config = TrainingConfig(
        batch_size=1,
        epochs=2,
        lr_g=2e-4,
        lr_d=2e-4,
        beta1=0.5,
        beta2=0.999,
        seed=42,
        num_workers=0,
        validate_rate=1,
        checkpoint_rate=1,
        log_rate=1,
    )
    run_paths = RunPaths(project.run_root)
    run_paths.create_directories()
    train_loader, val_loader = _make_train_val_loaders(
        dataset_root,
        project,
        train_prefixes=["00000_00000"],
        val_prefixes=["00256_00000"],
    )
    trainer = Trainer(
        config=config,
        run_paths=run_paths,
        generator=UNetGenerator().to("cpu"),
        discriminator=PatchGANDiscriminator().to("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        image_size=project.image_size,
        train_dir=dataset_root / "splits" / "train",
        val_dir=dataset_root / "splits" / "val",
        losses=_pix2pix_losses(),
    )
    trainer.train(seed=42)

    checkpoints = sorted(run_paths.checkpoints_dir.glob("ep*.pth"))
    assert len(checkpoints) == 2
    assert checkpoints[0].name == "ep000.pth"
    assert checkpoints[1].name == "ep001.pth"


def test_load_checkpoint_raises_on_missing_architecture(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
    tmp_path: Path,
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ck.pop("architecture")
    no_arch_path = tmp_path / "no_arch.pth"
    torch.save(ck, no_arch_path)

    trainer_2 = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )
    with pytest.raises(ValueError, match="architecture metadata"):
        trainer_2.resume(no_arch_path)
