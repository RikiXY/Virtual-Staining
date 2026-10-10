from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tests.checkpoint_helpers import assert_nested_equal
from tests.config_helpers import cyclegan_config_data, pix2pix_config_data, write_config_data
from virtual_staining.checkpoint_contract import read_checkpoint
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.experiment.session import ExperimentSession
from virtual_staining.methods.cyclegan import CycleGANMethod
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.training.benchmarking import TrainingBenchmarkRecorder
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.training.helpers import unpack_batch
from virtual_staining.training.trainer import Trainer


def test_unpack_batch_preserves_named_inputs_and_validates_shapes() -> None:
    batch = {
        "inputs": {"LF": torch.zeros(2, 3, 8, 8), "AF": torch.ones(2, 3, 8, 8)},
        "targets": {"stained": torch.zeros(2, 3, 8, 8)},
        "masks": {"foreground_mask": {"stained": torch.ones(2, 1, 8, 8)}},
    }
    inputs, targets, masks = unpack_batch(batch, torch.device("cpu"), ("LF", "AF"), ("stained",))
    assert tuple(inputs) == ("LF", "AF")
    assert targets["stained"].shape == (2, 3, 8, 8)
    assert masks["foreground_mask"]["stained"].shape == (2, 1, 8, 8)


def _pix2pix_config(tmp_path: Path, inputs: tuple[str, ...]) -> RunConfig:
    data = pix2pix_config_data(tmp_path, inputs=inputs, image_size=(8, 8))
    data["training"]["epochs"] = 1
    return RunConfig.from_mapping(data)


def test_trainer_requires_named_generator(tmp_path: Path) -> None:
    run_config = _pix2pix_config(tmp_path, ("LF", "AF"))
    assert run_config.training is not None
    paths = RunLayout.from_project(run_config.project)
    ensure_run_directories(paths)
    method = Pix2PixMethod(run_config, torch.device("cpu"))
    sample = {
        "inputs": {"LF": torch.zeros(1, 3, 8, 8), "AF": torch.zeros(1, 3, 8, 8)},
        "targets": {"stained": torch.zeros(1, 3, 8, 8)},
        "masks": {},
    }
    loader = DataLoader([sample], batch_size=1)  # pyright: ignore[reportArgumentType]
    trainer = Trainer(
        run_config.training,
        paths,
        method,
        loader,
        loader,
        torch.device("cpu"),
        config_hash="sha256:test",
    )
    assert trainer.method.name == "pix2pix"
    assert method.generator.input_names == ("LF", "AF")


def test_trainer_resumes_from_v4_checkpoint_at_next_epoch(tmp_path: Path) -> None:
    run_config = _pix2pix_config(tmp_path, ("LF",))
    training = run_config.training
    assert training is not None
    paths = RunLayout.from_project(run_config.project)
    ensure_run_directories(paths)
    loader = DataLoader([], batch_size=1)  # pyright: ignore[reportArgumentType]

    def build() -> Trainer:
        return Trainer(
            training,
            paths,
            Pix2PixMethod(run_config, torch.device("cpu")),
            loader,
            loader,
            torch.device("cpu"),
            config_hash="sha256:test",
        )

    build()._checkpoints.save(2)
    assert build().resume("latest") == 3
    assert build().resume("ep002.pth") == 3


class _EpochRecordingDataset(Dataset):
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"domain_a": torch.zeros(3, 32, 32), "domain_b": torch.zeros(3, 32, 32)}


def test_trainer_calls_optional_dataset_epoch_hook(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(
        write_config_data(tmp_path / "run.yaml", cyclegan_config_data(tmp_path))
    )
    assert config.training is not None
    paths = RunLayout.from_project(config.project)
    ensure_run_directories(paths)
    dataset = _EpochRecordingDataset()
    loader = DataLoader(dataset, batch_size=1)
    trainer = Trainer(
        config.training,
        paths,
        CycleGANMethod(config, torch.device("cpu"), seed=0),
        loader,
        loader,
        torch.device("cpu"),
        config_hash="sha256:test",
    )

    trainer.train(seed=0)

    assert dataset.epochs == [0, 1]


@pytest.mark.parametrize("method_name", ["pix2pix", "cyclegan"])
@pytest.mark.parametrize("scheduler_name", ["linear_decay", "reduce_on_plateau"])
@pytest.mark.parametrize("tracked", [False, True])
def test_reused_real_method_checkpoint_matches_final_state(
    tmp_path: Path, method_name: str, scheduler_name: str, tracked: bool
) -> None:
    torch.manual_seed(7)
    data = (
        pix2pix_config_data(tmp_path, image_size=(64, 64))
        if method_name == "pix2pix"
        else cyclegan_config_data(tmp_path)
    )
    data["training"].update(
        epochs=3,
        checkpoint_rate=2,
        scheduler={
            "name": scheduler_name,
            "decay_start_epoch": 0,
            "monitor": "loss_G_val",
            "patience": 0,
            "factor": 0.5,
        },
    )
    config = RunConfig.from_mapping(data)
    assert config.training is not None
    paths = RunLayout.from_project(config.project)
    ensure_run_directories(paths)
    device = torch.device("cpu")

    def build_method() -> Pix2PixMethod | CycleGANMethod:
        return (
            Pix2PixMethod(config, device)
            if method_name == "pix2pix"
            else CycleGANMethod(config, device, seed=7)
        )

    method = build_method()
    sample = (
        {
            "inputs": {name: torch.rand(3, 64, 64) * 2 - 1 for name in ("LF", "AF")},
            "targets": {"stained": torch.rand(3, 64, 64) * 2 - 1},
            "masks": {},
        }
        if method_name == "pix2pix"
        else {"domain_a": torch.rand(3, 32, 32) * 2 - 1, "domain_b": torch.rand(3, 32, 32) * 2 - 1}
    )
    loader = DataLoader([sample, sample], batch_size=2)  # pyright: ignore[reportArgumentType]
    experiment = Mock(spec=ExperimentSession) if tracked else None
    recorder = TrainingBenchmarkRecorder(device, warmup_batches=0) if tracked else None
    if recorder is not None:
        recorder.start_run()
    trainer = Trainer(
        config.training,
        paths,
        method,
        loader,
        loader,
        device,
        config_hash="sha256:test",
        experiment_session=experiment,
        benchmark_recorder=recorder,
    )
    with patch.object(trainer._checkpoints, "save", wraps=trainer._checkpoints.save) as save:
        trainer.train(seed=7)
    assert [call.args[0] for call in save.call_args_list] == [0, 1, 2]
    if recorder is not None:
        recorder.finish_run()
        assert recorder.report(metadata={})["summary"]["phases"]["checkpoint"]["count"] == 3
    retained = read_checkpoint(paths.checkpoints_dir / "ep002.pth")
    # The old finalization would save exactly this live state again.
    comparison = MethodCheckpointManager(method, tmp_path / "comparison", config_hash="sha256:test")
    assert_nested_equal(retained, read_checkpoint(comparison.save(2)))
    assert isinstance(retained, dict)
    assert_nested_equal(retained["state"], method.state_dict())
    restored = build_method()
    assert (
        MethodCheckpointManager(restored, paths.checkpoints_dir).load(
            paths.checkpoints_dir / "ep002.pth"
        )
        == 3
    )
    assert_nested_equal(restored.state_dict(), method.state_dict())
    assert retained["method"] == method.checkpoint_identity().method_metadata()
    assert retained["format_version"] == 4
    assert retained["config_hash"] == "sha256:test"
    assert all(state["last_epoch"] == 3 for state in retained["state"]["schedulers"].values())
    if experiment is not None:
        assert [call.kwargs["step"] for call in experiment.log_metrics.call_args_list] == [0, 1, 2]
