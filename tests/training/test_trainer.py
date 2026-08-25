from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch
from torch.utils.data import DataLoader

from virtual_staining.config.project import ProjectConfig
from virtual_staining.experiment.run_paths import RunPaths
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import ConcatUNetGenerator
from virtual_staining.training.config import TrainingConfig
from virtual_staining.training.helpers import unpack_batch
from virtual_staining.training.trainer import Trainer


def _session() -> ExperimentSession:
    return cast(ExperimentSession, SimpleNamespace(log_metrics=lambda *_args, **_kwargs: None))


def test_unpack_batch_preserves_named_inputs_and_validates_shapes() -> None:
    batch = {
        "inputs": {"LF": torch.zeros(2, 3, 8, 8), "AF": torch.ones(2, 3, 8, 8)},
        "target": torch.zeros(2, 3, 8, 8),
        "masks": {"foreground_mask": torch.ones(2, 1, 8, 8)},
    }
    inputs, target, masks = unpack_batch(batch, torch.device("cpu"), ("LF", "AF"))
    assert tuple(inputs) == ("LF", "AF")
    assert target.shape == (2, 3, 8, 8)
    assert masks["foreground_mask"].shape == (2, 1, 8, 8)


def test_trainer_requires_named_generator_and_keeps_validation_dir(tmp_path: Path) -> None:
    project = ProjectConfig(
        dataset_root=tmp_path / "dataset",
        results_path=tmp_path / "results",
        run_name="run",
        image_size=(8, 8),
    )
    paths = RunPaths(project.run_root)
    paths.create_directories()
    generator = ConcatUNetGenerator(("LF", "AF"), base_channels=4)
    discriminator = PatchGANDiscriminator(in_channels=9, ndf=4)
    sample = {
        "inputs": {"LF": torch.zeros(1, 3, 8, 8), "AF": torch.zeros(1, 3, 8, 8)},
        "target": torch.zeros(1, 3, 8, 8),
        "masks": {},
    }
    loader = DataLoader([sample], batch_size=1)  # pyright: ignore[reportArgumentType]
    trainer = Trainer(
        TrainingConfig(
            batch_size=1,
            epochs=1,
            lr_g=2e-4,
            lr_d=2e-4,
            beta1=0.5,
            beta2=0.999,
            seed=0,
            num_workers=0,
            validate_rate=1,
            checkpoint_rate=1,
        ),
        paths,
        generator,
        discriminator,
        loader,
        loader,
        torch.device("cpu"),
        experiment_session=_session(),
        config_hash="sha256:test",
        image_size=(8, 8),
        train_dir=tmp_path / "train",
        val_dir=tmp_path / "val",
    )
    assert trainer._input_names == ("LF", "AF")
    assert trainer._val_dir == tmp_path / "val"
