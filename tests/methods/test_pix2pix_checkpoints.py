from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.checkpoint_helpers import assert_nested_equal
from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.config.losses import parse_loss_config
from virtual_staining.config.method import MethodConfig
from virtual_staining.config.model import ModelConfig
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.run import RunConfig
from virtual_staining.config.training import LearningRateSchedulerConfig, TrainingConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.inference.runner import load_inference_generator, predict_batch
from virtual_staining.methods.pix2pix import Pix2PixMethod
from virtual_staining.training.checkpoints import MethodCheckpointManager

_CPU = torch.device("cpu")
_LOSSES = parse_loss_config(
    {
        "generator": [{"name": "l1", "weight": 1.0}, {"name": "adversarial_bce", "weight": 1.0}],
        "discriminator": [{"name": "adversarial_bce", "weight": 1.0}],
    }
)
_LINEAR = {
    "epochs": 4,
    "scheduler": LearningRateSchedulerConfig(name="linear_decay", decay_start_epoch=1),
}
_PLATEAU = {
    "scheduler": LearningRateSchedulerConfig(
        name="reduce_on_plateau", monitor="loss_G_val", mode="min", factor=0.5, patience=2
    )
}


def _config(
    tmp_path: Path,
    *,
    image_size: tuple[int, int] = (64, 64),
    training: dict[str, Any] | None = None,
    **model: Any,
) -> RunConfig:
    model_mapping: dict[str, Any] = {
        "inputs": ["LF", "AF"],
        "target": "stained",
        "generator": {"base_channels": 4},
        "discriminator": {"ndf": 4},
    }
    model_mapping.update(model)
    return RunConfig(
        project=ProjectConfig(
            dataset_root=tmp_path / "dataset",
            results_path=tmp_path / "results",
            run_name="run",
            image_size=image_size,
        ),
        method=MethodConfig(),
        model=ModelConfig.from_mapping(model_mapping),
        training=TrainingConfig(
            **{
                "batch_size": 1,
                "epochs": 2,
                "lr_g": 2e-4,
                "lr_d": 2e-4,
                "beta1": 0.5,
                "beta2": 0.999,
                "seed": 0,
                "num_workers": 0,
                "validate_rate": 1,
                "checkpoint_rate": 1,
                "losses": _LOSSES,
                **(training or {}),
            }
        ),
        inference=InferenceConfig(checkpoint_policy="latest"),
        preprocessing=None,
        evaluation=None,
    )


def _manager(config: RunConfig, method: Pix2PixMethod) -> MethodCheckpointManager:
    paths = RunLayout.from_project(config.project)
    ensure_run_directories(paths)
    return MethodCheckpointManager(
        method,
        paths.checkpoints_dir,
        image_size=config.project.image_size,
        config_hash="sha256:test",
    )


def _batch() -> dict[str, Any]:
    torch.manual_seed(1)
    return {
        "inputs": {"LF": torch.rand(2, 3, 64, 64) * 2 - 1, "AF": torch.rand(2, 3, 64, 64) * 2 - 1},
        "target": torch.rand(2, 3, 64, 64) * 2 - 1,
        "masks": {},
    }


def _trained_checkpoint(
    tmp_path: Path, epoch: int = 4, training: dict[str, Any] | None = None
) -> tuple[RunConfig, Pix2PixMethod, Path]:
    config = _config(tmp_path, training=training)
    torch.manual_seed(0)
    method = Pix2PixMethod(config, _CPU)
    method.train_mode()
    method.step(_batch(), epoch=0, global_step=0)
    return config, method, _manager(config, method).save(epoch)


def _assert_modules_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    right_state = right.state_dict()
    for name, value in left.state_dict().items():
        assert torch.equal(value, right_state[name]), name


def test_pix2pix_v4_payload_holds_method_owned_state_only(tmp_path: Path) -> None:
    _config_, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {
        "format_version",
        "epoch",
        "method",
        "image_size",
        "normalization",
        "config_hash",
        "state",
    }
    assert payload["format_version"] == 4
    assert payload["method"]["name"] == "pix2pix"
    assert payload["method"]["pairing"] == "paired"
    assert payload["method"]["inputs"] == ["LF", "AF"]
    assert payload["method"]["outputs"] == ["stained"]
    assert payload["method"]["prediction_directions"] == ["forward"]
    assert payload["method"]["components"]["generator"]["class"] == "ConcatUNetGenerator"
    assert payload["method"]["components"]["generator"]["base_channels"] == 4
    assert payload["method"]["components"]["discriminator"]["class"] == "PatchGANDiscriminator"
    assert set(payload["state"]) == {
        "models",
        "optimization",
        "optimizers",
        "scalers",
        "schedulers",
    }
    assert payload["state"]["optimization"]["generator"] == {
        "optimizer": {
            "class": "Adam",
            "lr": 2e-4,
            "betas": [0.5, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
        },
        "scheduler": None,
    }


def test_pix2pix_v4_save_resume_round_trip(tmp_path: Path) -> None:
    config, source, path = _trained_checkpoint(tmp_path, epoch=4)
    torch.manual_seed(99)
    resumed = Pix2PixMethod(config, _CPU)

    assert _manager(config, resumed).load(path) == 5

    _assert_modules_equal(source.generator, resumed.generator)
    _assert_modules_equal(source.discriminator, resumed.discriminator)
    source_state, resumed_state = source.state_dict(), resumed.state_dict()
    for name in ("generator", "discriminator"):
        source_opt = source_state["optimizers"][name]
        resumed_opt = resumed_state["optimizers"][name]
        assert source_opt["param_groups"] == resumed_opt["param_groups"]
        for key, values in source_opt["state"].items():
            for field, value in values.items():
                assert torch.equal(value, resumed_opt["state"][key][field])
    assert source.learning_rates() == resumed.learning_rates()


def test_pix2pix_v4_inference_round_trip(tmp_path: Path) -> None:
    config, source, path = _trained_checkpoint(tmp_path)
    paths = RunLayout.from_project(config.project)

    generator, resolved_path = load_inference_generator(config, paths, _CPU)

    assert resolved_path == path
    assert not generator.training
    _assert_modules_equal(source.generator, generator)
    source.generator.eval()
    inputs = _batch()["inputs"]
    assert torch.equal(
        predict_batch(generator, inputs, _CPU), predict_batch(source.generator, inputs, _CPU)
    )


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"inputs": ["AF", "LF"]}, r"method.inputs\[0\]"),
        ({"target": "other"}, r"method.outputs\[0\]"),
        ({"generator": {"base_channels": 8}}, "method.components.generator.base_channels"),
        ({"generator": {"base_channels": 4, "norm": "instance"}}, "generator.norm"),
        ({"generator": {"base_channels": 4, "dropout": True}}, "generator.dropout"),
        ({"discriminator": {"ndf": 8}}, "method.components.discriminator.ndf"),
    ],
)
def test_pix2pix_config_mismatch_is_rejected_for_training_and_inference(
    tmp_path: Path, changes: dict[str, Any], field: str
) -> None:
    _trained_checkpoint(tmp_path)
    config = _config(tmp_path, **changes)
    method = Pix2PixMethod(config, _CPU)
    before = {name: value.clone() for name, value in method.generator.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match=field):
        _manager(config, method).load(
            RunLayout.from_project(config.project).checkpoints_dir / "ep004.pth"
        )
    with pytest.raises(CheckpointCompatibilityError, match=field):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)
    for name, value in method.generator.state_dict().items():
        assert torch.equal(value, before[name])


def test_pix2pix_image_size_mismatch_is_rejected(tmp_path: Path) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    resized = dataclasses.replace(
        config, project=dataclasses.replace(config.project, image_size=(32, 32))
    )
    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        _manager(resized, Pix2PixMethod(resized, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        load_inference_generator(resized, RunLayout.from_project(resized.project), _CPU)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["method"]["components"]["generator"].update(
                {"class": "ResNetGenerator"}
            ),
            "generator.class",
        ),
        (lambda payload: payload["method"].update({"name": "cyclegan"}), "method.name"),
        (lambda payload: payload["method"].update({"pairing": "unpaired"}), "method.pairing"),
        (
            lambda payload: payload["method"].update({"prediction_directions": ["backward"]}),
            "method.prediction_directions",
        ),
        (
            lambda payload: payload["normalization"].update({"output_range": "[0, 1]"}),
            "normalization.output_range",
        ),
        (lambda payload: payload.update({"format_version": 3}), "format version 3"),
        (lambda payload: payload.pop("format_version"), "unversioned"),
        (lambda payload: payload.pop("state"), "no method state"),
    ],
)
def test_pix2pix_stored_identity_mismatch_is_rejected(tmp_path: Path, mutate, message) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        _manager(config, Pix2PixMethod(config, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)


def test_pix2pix_rejects_legacy_v3_layout(tmp_path: Path) -> None:
    config, method, path = _trained_checkpoint(tmp_path)
    torch.save(
        {
            "format_version": 3,
            "epoch": 4,
            "architecture": {"generator": {}, "discriminator": {}},
            "generator_state_dict": method.generator.state_dict(),
            "discriminator_state_dict": method.discriminator.state_dict(),
            "image_size": list(config.project.image_size),
        },
        path,
    )
    with pytest.raises(CheckpointCompatibilityError, match="unsupported format version 3"):
        _manager(config, Pix2PixMethod(config, _CPU)).load(path)
    with pytest.raises(CheckpointCompatibilityError, match="unsupported format version 3"):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)


def _plateau(**changes: Any) -> dict[str, Any]:
    return {"scheduler": dataclasses.replace(_PLATEAU["scheduler"], **changes)}


def _reject(config: RunConfig, state: dict[str, Any], match: str) -> None:
    """Loading ``state`` must fail deliberately and leave the target runtime untouched."""
    torch.manual_seed(5)
    target = Pix2PixMethod(config, _CPU)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(CheckpointCompatibilityError, match=match):
        target.load_state_dict(state)
    assert_nested_equal(target.state_dict(), before)


def _drop_first_key(mapping: dict[str, Any]) -> None:
    del mapping[next(iter(mapping))]


def _first_tensor(state: dict[str, Any]) -> str:
    return next(iter(state["models"]["generator"]))


_MALFORMED: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "missing_group": (
        lambda s: s.pop("scalers"),
        r"state has mismatched keys: missing \['scalers'\]",
    ),
    "extra_group": (lambda s: s.update(replay_pools={}), r"unexpected \['replay_pools'\]"),
    "models_not_mapping": (lambda s: s.update(models=[]), "state.models must be a mapping"),
    "missing_model_role": (
        lambda s: s["models"].pop("discriminator"),
        r"state.models has mismatched keys: missing \['discriminator'\]",
    ),
    "wrong_model_keys": (
        lambda s: _drop_first_key(s["models"]["generator"]),
        "state.models.generator has mismatched keys",
    ),
    "non_tensor_weight": (
        lambda s: s["models"]["generator"].update({_first_tensor(s): [1.0]}),
        "must be a tensor",
    ),
    "wrong_shape": (
        lambda s: s["models"]["generator"].update({_first_tensor(s): torch.zeros(1)}),
        "state.models.generator.* has shape",
    ),
    "wrong_dtype": (
        lambda s: s["models"]["generator"].update(
            {_first_tensor(s): s["models"]["generator"][_first_tensor(s)].double()}
        ),
        "has dtype torch.float64; expected torch.float32",
    ),
    "optimizer_not_mapping": (
        lambda s: s["optimizers"].update(generator=[]),
        "state.optimizers.generator must be a mapping",
    ),
    "optimizer_missing_state": (
        lambda s: s["optimizers"]["discriminator"].pop("state"),
        "state.optimizers.discriminator has mismatched keys",
    ),
    "optimizer_group_count": (
        lambda s: s["optimizers"]["generator"]["param_groups"].append(
            copy.deepcopy(s["optimizers"]["generator"]["param_groups"][0])
        ),
        "param_groups must be a list of 1 groups",
    ),
    "optimizer_group_params": (
        lambda s: s["optimizers"]["generator"]["param_groups"][0]["params"].pop(),
        r"param_groups\[0\].params does not match",
    ),
    "optimizer_group_keys": (
        lambda s: s["optimizers"]["generator"]["param_groups"][0].pop("betas"),
        r"param_groups\[0\] has mismatched keys",
    ),
    "optimizer_state_index": (
        lambda s: s["optimizers"]["generator"]["state"].update({10_000: {}}),
        "unknown parameter index 10000",
    ),
    "optimizer_state_shape": (
        lambda s: s["optimizers"]["generator"]["state"][0].update(exp_avg=torch.zeros(1, 1, 1)),
        r"state.optimizers.generator.state\[0\].exp_avg has shape",
    ),
    "scaler_not_mapping": (
        lambda s: s["scalers"].update(generator=1.0),
        "state.scalers.generator must be a mapping",
    ),
    "scaler_bad_keys": (
        lambda s: s["scalers"].update(generator={"scale": 1.0}),
        "state.scalers.generator has mismatched keys",
    ),
    "scaler_bad_value": (
        lambda s: s["scalers"].update(
            generator={
                "scale": "big",
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 0,
            }
        ),
        "state.scalers.generator.scale must be a number",
    ),
    "optimization_missing_role": (
        lambda s: s["optimization"].pop("generator"),
        "state.optimization has mismatched keys",
    ),
    "optimizer_class": (
        lambda s: s["optimization"]["generator"]["optimizer"].update({"class": "SGD"}),
        "optimization.generator.optimizer.class is 'SGD' in the checkpoint but 'Adam'",
    ),
    "optimizer_betas": (
        lambda s: s["optimization"]["discriminator"]["optimizer"].update(betas=[0.9, 0.999]),
        r"optimization.discriminator.optimizer.betas\[0\] is 0.9",
    ),
    "optimizer_initial_lr": (
        lambda s: s["optimization"]["generator"]["optimizer"].update(lr=1e-3),
        "optimization.generator.optimizer.lr is 0.001",
    ),
    "scheduler_state_without_scheduler": (
        lambda s: s["schedulers"].update(generator={"last_epoch": 0}),
        "state.schedulers.generator is present but the current run has no scheduler",
    ),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_pix2pix_malformed_state_is_rejected_before_mutation(tmp_path: Path, case: str) -> None:
    config, source, _path = _trained_checkpoint(tmp_path)
    state = copy.deepcopy(source.state_dict())
    mutate, match = _MALFORMED[case]
    mutate(state)
    _reject(config, state, match)


_POLICY_CHANGES: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
    "scheduler_present_to_absent": (_LINEAR, {}, "scheduler is {.*} in the checkpoint but None"),
    "scheduler_absent_to_present": ({}, _LINEAR, "scheduler is None in the checkpoint"),
    "scheduler_type": (_LINEAR, {**_PLATEAU, "epochs": 4}, "scheduler.name is 'linear_decay'"),
    "linear_decay_start": (
        _LINEAR,
        {"epochs": 4, "scheduler": LearningRateSchedulerConfig("linear_decay", 2)},
        "scheduler.decay_start_epoch is 1 in the checkpoint but 2",
    ),
    "linear_decay_horizon": (
        _LINEAR,
        {**_LINEAR, "epochs": 6},
        "scheduler.epochs is 4 in the checkpoint but 6",
    ),
    "plateau_factor": (_PLATEAU, _plateau(factor=0.1), "scheduler.factor is 0.5"),
    "plateau_patience": (_PLATEAU, _plateau(patience=5), "scheduler.patience is 2"),
    "plateau_min_lr": (_PLATEAU, _plateau(min_lr=1e-6), "scheduler.min_lr is 0.0"),
    "plateau_mode": (_PLATEAU, _plateau(mode="max"), "scheduler.mode is 'min'"),
    "plateau_monitor": (
        _PLATEAU,
        _plateau(monitor="val_ssim"),
        "scheduler.monitor is 'loss_G_val'",
    ),
    "optimizer_beta": ({}, {"beta1": 0.9}, r"optimizer.betas\[0\] is 0.5 .* but 0.9"),
    "optimizer_lr": ({}, {"lr_d": 1e-3}, "discriminator.optimizer.lr is 0.0002"),
}


@pytest.mark.parametrize("case", sorted(_POLICY_CHANGES))
def test_pix2pix_resume_rejects_changed_optimization_policy(tmp_path: Path, case: str) -> None:
    saved, current, match = _POLICY_CHANGES[case]
    _config_, _source, path = _trained_checkpoint(tmp_path, training=saved)
    config = _config(tmp_path, training=current)
    torch.manual_seed(5)
    target = Pix2PixMethod(config, _CPU)
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match=f"{path}.*{match}"):
        _manager(config, target).load(path)
    assert_nested_equal(target.state_dict(), before)


def test_decayed_learning_rate_is_not_mistaken_for_a_policy_change(tmp_path: Path) -> None:
    config = _config(tmp_path, training=_LINEAR)
    source = Pix2PixMethod(config, _CPU)
    source.train_mode()
    for epoch in range(3):
        source.step(_batch(), epoch=epoch, global_step=epoch)
        source.step_schedulers(epoch=epoch, validation_metrics=None)
    assert source.learning_rates()["lr_g"] < config.training.lr_g  # type: ignore[union-attr]
    path = _manager(config, source).save(2)

    resumed = Pix2PixMethod(config, _CPU)
    assert _manager(config, resumed).load(path) == 3
    assert resumed.learning_rates() == source.learning_rates()
    assert_nested_equal(resumed.state_dict(), source.state_dict())


def test_pix2pix_resume_round_trips_optimizer_scaler_and_scheduler_state(
    tmp_path: Path,
) -> None:
    config, source, _path = _trained_checkpoint(tmp_path, training=_PLATEAU)
    for value in (1.0, 2.0, 3.0, 4.0):
        source._scheduler_G.step(value)  # type: ignore[union-attr]
        source._scheduler_D.step(value)  # type: ignore[union-attr]
    path = _manager(config, source).save(4)
    assert source.learning_rates()["lr_g"] < 2e-4

    torch.manual_seed(42)
    resumed = Pix2PixMethod(config, _CPU)
    assert _manager(config, resumed).load(path) == 5

    assert_nested_equal(resumed.state_dict(), source.state_dict())
    assert resumed.learning_rates() == source.learning_rates()


def _epoch(method: Pix2PixMethod, epoch: int) -> None:
    # Global RNG is not checkpointed; both runs re-seed it identically per epoch.
    torch.manual_seed(1_000 + epoch)
    method.train_mode()
    for step in range(2):
        method.step(_batch(), epoch=epoch, global_step=epoch * 2 + step)
    method.step_schedulers(epoch=epoch, validation_metrics=None)


def test_resumed_training_matches_continuous_training_given_identical_rng_seeding(
    tmp_path: Path,
) -> None:
    """Resume restores model/optimizer/scaler/scheduler state at a completed epoch.

    Global, DataLoader-worker, and augmentation RNG state are not persisted, so this equality
    holds only because both runs re-seed the global RNG identically before every epoch.
    """
    config = _config(tmp_path, training=_LINEAR, generator={"base_channels": 4, "dropout": True})
    torch.manual_seed(0)
    continuous = Pix2PixMethod(config, _CPU)
    torch.manual_seed(0)
    interrupted = Pix2PixMethod(config, _CPU)
    for epoch in range(2):
        _epoch(continuous, epoch)
        _epoch(interrupted, epoch)
    path = _manager(config, interrupted).save(1)

    torch.manual_seed(77)
    resumed = Pix2PixMethod(config, _CPU)
    start = _manager(config, resumed).load(path)
    for epoch in range(start, 4):
        _epoch(continuous, epoch)
        _epoch(resumed, epoch)

    assert_nested_equal(resumed.state_dict(), continuous.state_dict())
    assert resumed.learning_rates() == continuous.learning_rates()


def test_restore_failure_after_preflight_voids_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, path = _trained_checkpoint(tmp_path)
    target = Pix2PixMethod(config, _CPU)

    def fail(_state: object) -> None:
        raise RuntimeError("device went away")

    monkeypatch.setattr(target._opt_D, "load_state_dict", fail)
    with pytest.raises(RuntimeError, match="partially restored and must be discarded"):
        _manager(config, target).load(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.pop("models"), r"state.models has no 'generator' entry"),
        (lambda s: s["models"].pop("generator"), r"state.models has no 'generator' entry"),
        (
            lambda s: _drop_first_key(s["models"]["generator"]),
            "state.models.generator has mismatched keys",
        ),
        (
            lambda s: s["models"]["generator"].update({_first_tensor(s): torch.zeros(2)}),
            "state.models.generator.* has shape",
        ),
    ],
)
def test_pix2pix_inference_rejects_malformed_generator_state(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object], match: str
) -> None:
    config, _method, path = _trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload["state"])
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match=f"{path}.*{match}"):
        load_inference_generator(config, RunLayout.from_project(config.project), _CPU)


def test_checkpoints_are_read_on_cpu_with_restricted_unpickling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, path = _trained_checkpoint(tmp_path)
    calls: list[dict[str, Any]] = []
    real_load = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        payload = real_load(*args, **kwargs)
        weights = payload["state"]["models"]["generator"].values()
        assert all(tensor.device.type == "cpu" for tensor in weights)
        return payload

    monkeypatch.setattr(torch, "load", spy)
    _manager(config, Pix2PixMethod(config, _CPU)).load(path)
    load_inference_generator(config, RunLayout.from_project(config.project), _CPU)

    assert calls == [{"map_location": "cpu", "weights_only": True}] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_deserialized_checkpoint_is_restored_onto_the_execution_device(
    tmp_path: Path,
) -> None:
    cuda = torch.device("cuda")
    config = _config(tmp_path)
    source = Pix2PixMethod(config, cuda)
    source.train_mode()
    source.step(_batch(), epoch=0, global_step=0)
    path = _manager(config, source).save(0)

    resumed = Pix2PixMethod(config, cuda)
    _manager(config, resumed).load(path)

    assert all(p.device.type == "cuda" for p in resumed.generator.parameters())
    optimizer_state = resumed.state_dict()["optimizers"]["generator"]["state"]
    assert all(
        v.device.type == "cuda" for s in optimizer_state.values() for v in s.values() if v.ndim
    )
    assert_nested_equal(resumed.state_dict(), source.state_dict())
    generator = load_inference_generator(config, RunLayout.from_project(config.project), cuda)[0]
    assert next(generator.parameters()).device.type == "cuda"
