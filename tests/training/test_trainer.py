from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tests.image_helpers import write_rgb_pair
from tests.manifest_helpers import make_manifest_record
from virtual_staining.config.project import ProjectConfig
from virtual_staining.data.dataset import PairedManifestDataset
from virtual_staining.data.manifest import DatasetManifest, Split
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.models.config import ModelConfig
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.config import LossConfig, LossTermConfig, TrainingConfig
from virtual_staining.training.trainer import Trainer


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
        model_config=ModelConfig(),
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
            model_config=ModelConfig(),
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
    assert (run_paths.metrics_dir / "metrics.csv").exists()
    assert (run_paths.logs_dir / "training.log").exists()
    assert not (run_root / "run_metadata.json").exists()


def test_trainer_metrics_csv_structure(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, _ = smoke_trainer

    trainer.train(seed=42)

    metrics_path = run_paths.metrics_dir / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert len(rows) == config.epochs
    assert {
        "epoch",
        "loss_G_train",
        "loss_D_train",
        "loss_G_val",
        "loss_D_val",
    } <= set(rows[0].keys())
    for row in rows:
        assert row["loss_G_val"] != ""
        assert row["loss_D_val"] != ""
        assert float(row["loss_G_train"]) > 0
        assert float(row["loss_D_train"]) > 0


def test_trainer_train_losses_are_epoch_averages(tmp_path: Path) -> None:
    """Train losses in metrics/metrics.csv must be averages over all batches."""
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
        model_config=ModelConfig(),
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

    metrics_path = run_paths.metrics_dir / "metrics.csv"
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
        model_config=ModelConfig(),
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

    with (run_paths.metrics_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    row = rows[0]
    assert "loss_train_raw_generator_ssim" in row
    assert "loss_train_weighted_generator_ssim" in row
    assert "loss_train_current_weight_generator_ssim" in row
    assert "loss_val_raw_generator_ssim" in row
    assert "loss_val_weighted_generator_ssim" in row
    assert "loss_val_current_weight_generator_ssim" in row
    assert "loss_train_raw_discriminator_adversarial_bce" in row
    assert "loss_train_total_generator" in row
    assert "loss_val_total_generator" in row
    assert float(row["loss_train_raw_generator_ssim"]) >= 0.0
    assert float(row["loss_train_weighted_generator_ssim"]) >= 0.0
    assert float(row["loss_train_current_weight_generator_ssim"]) == pytest.approx(1.0)
    assert row["loss_G_train"] == row["loss_train_total_generator"]
    assert row["loss_G_val"] == row["loss_val_total_generator"]


def test_validate_restores_models_that_started_in_train_mode(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, run_paths, _project = smoke_trainer
    trainer.generator.train()
    trainer.discriminator.train()

    trainer._validate(epoch=0, log_file=run_paths.logs_dir / "training.log")

    assert trainer.generator.training is True
    assert trainer.discriminator.training is True


def test_validate_preserves_models_that_started_in_eval_mode(
    smoke_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, _config, run_paths, _project = smoke_trainer
    trainer.generator.eval()
    trainer.discriminator.eval()

    trainer._validate(epoch=0, log_file=run_paths.logs_dir / "training.log")

    assert trainer.generator.training is False
    assert trainer.discriminator.training is False


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
        model_config=ModelConfig(),
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

    start_epoch = trainer_2._checkpoint_manager.load(checkpoint_path)
    assert start_epoch == 1


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
    assert ck["architecture"]["name"] == "pix2pix"
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

    assert best_record["policy"] == "best_val_loss"
    assert best_record["metric"] == "loss_G_val"
    assert best_record["epoch"] == 0
    assert best_record["checkpoint_path"] == "ep000.pth"
    assert isinstance(best_record["metric_value"], float)
    assert result.best_checkpoint_path == run_paths.checkpoints_dir / "ep000.pth"


def test_load_checkpoint_validates_matching_architecture(
    checkpointing_trainer: tuple[Trainer, TrainingConfig, RunPaths, ProjectConfig],
) -> None:
    trainer, config, run_paths, project = checkpointing_trainer
    trainer.train(seed=42)

    checkpoint_path = next(run_paths.checkpoints_dir.glob("*.pth"))
    trainer_2 = _make_resume_trainer(
        config, run_paths, project, UNetGenerator(), PatchGANDiscriminator()
    )
    start_epoch = trainer_2._checkpoint_manager.load(checkpoint_path)
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
        trainer_mismatch._checkpoint_manager.load(checkpoint_path)


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
        model_config=ModelConfig(),
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
        trainer_2._checkpoint_manager.load(no_arch_path)
