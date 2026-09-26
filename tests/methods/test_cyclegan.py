from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.checkpoint_helpers import assert_nested_equal
from tests.config_helpers import cyclegan_config_data, write_config_data
from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.run import RunConfig
from virtual_staining.experiment.run_layout import RunLayout, ensure_run_directories
from virtual_staining.methods.cyclegan import CycleGANMethod, ReplayPool
from virtual_staining.methods.registry import resolve_training_method
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.training.preview import ValidationPreviewWriter

_CPU = torch.device("cpu")


def _config(tmp_path: Path, mutate: Callable[[dict[str, Any]], None] | None = None) -> RunConfig:
    data = cyclegan_config_data(tmp_path)
    if mutate is not None:
        mutate(data)
    return RunConfig.from_yaml(write_config_data(tmp_path / "run.yaml", data))


def _method(config: RunConfig, *, seed: int = 7) -> CycleGANMethod:
    torch.manual_seed(0)
    method = CycleGANMethod(config, _CPU, seed=seed)
    method.train_mode()
    return method


def _batch(seed: int = 1) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "domain_a": torch.rand(2, 3, 32, 32, generator=generator) * 2 - 1,
        "domain_b": torch.rand(2, 3, 32, 32, generator=generator) * 2 - 1,
        "path_a": ["a0", "a1"],
        "path_b": ["b0", "b1"],
    }


def _params(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(b, a) for b, a in zip(before, module.parameters(), strict=True))


def _manager(config: RunConfig, method: CycleGANMethod) -> MethodCheckpointManager:
    paths = RunLayout.from_project(config.project)
    ensure_run_directories(paths)
    return MethodCheckpointManager(
        method, paths.checkpoints_dir, image_size=config.project.image_size
    )


def test_resolver_builds_cyclegan_with_method_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)

    method = resolve_training_method(config, _CPU, seed=7)

    assert isinstance(method, CycleGANMethod)
    assert (method.name, method.pairing) == ("cyclegan", "unpaired")
    assert method.input_names == ("label_free",)
    assert method.output_names == ("stained",)
    assert method.prediction_directions == ("A_to_B", "B_to_A")
    assert set(method.component_metadata()) == {"G_A_to_B", "G_B_to_A", "D_A", "D_B"}
    with pytest.raises(ValueError, match="resolved training seed"):
        resolve_training_method(config, _CPU)


def test_component_metadata_describes_all_four_components(tmp_path: Path) -> None:
    metadata = _method(_config(tmp_path)).component_metadata()

    assert metadata["G_A_to_B"] == {
        "class": "ResnetGenerator",
        "output_activation": "tanh",
        "in_channels": 3,
        "out_channels": 3,
        "architecture": "resnet",
        "base_channels": 4,
        "norm": "instance",
        "blocks": 1,
    }
    assert metadata["G_B_to_A"] == metadata["G_A_to_B"]
    assert metadata["D_A"] == {
        "class": "PatchGANDiscriminator",
        "architecture": "patchgan",
        "conditional": False,
        "in_channels": 3,
        "ndf": 4,
        "norm": "instance",
        "use_sigmoid": False,
    }
    assert metadata["D_B"] == metadata["D_A"]


def test_full_step_is_finite_and_reports_configured_components(tmp_path: Path) -> None:
    method = _method(_config(tmp_path))

    metrics = method.step(_batch(), epoch=0, global_step=0)

    assert set(metrics.losses) == {"loss_G", "loss_D"}
    assert all(math.isfinite(value) for value in metrics.losses.values())
    assert set(metrics.raw) == {
        "generator_adversarial_lsgan",
        "generator_cycle_l1",
        "generator_identity_l1",
        "discriminator_adversarial_lsgan",
    }
    assert metrics.current_weight["generator_cycle_l1"] == 10.0
    assert metrics.weighted["generator_cycle_l1"] == pytest.approx(
        10.0 * metrics.raw["generator_cycle_l1"]
    )
    assert metrics.component_totals == {
        "generator": metrics.losses["loss_G"],
        "discriminator": metrics.losses["loss_D"],
    }


def test_generator_phase_updates_generators_with_frozen_discriminators(tmp_path: Path) -> None:
    method = _method(_config(tmp_path))
    method.step(_batch(0), epoch=0, global_step=0)
    discriminator_grads = {
        name: [p.grad.clone() for p in module.parameters() if p.grad is not None]
        for name, module in (("D_A", method.D_A), ("D_B", method.D_B))
    }
    assert all(discriminator_grads.values())
    frozen_during_call: list[bool] = []
    hooks = [
        module.register_forward_pre_hook(
            lambda mod, _args: frozen_during_call.append(
                not any(p.requires_grad for p in mod.parameters())
            )
        )
        for module in (method.D_A, method.D_B)
    ]
    before = {name: _params(m) for name, m in method._models().items()}
    real_a, real_b = method._unpack_batch(_batch())

    method._generator_phase(real_a, real_b, epoch=0, global_step=1)

    for hook in hooks:
        hook.remove()
    assert frozen_during_call and all(frozen_during_call)
    assert _changed(before["G_A_to_B"], method.G_A_to_B)
    assert _changed(before["G_B_to_A"], method.G_B_to_A)
    assert not _changed(before["D_A"], method.D_A)
    assert not _changed(before["D_B"], method.D_B)
    for name, module in (("D_A", method.D_A), ("D_B", method.D_B)):
        assert all(p.requires_grad for p in module.parameters())
        after = [p.grad for p in module.parameters() if p.grad is not None]
        assert all(torch.equal(b, a) for b, a in zip(discriminator_grads[name], after, strict=True))


def test_discriminator_phase_uses_detached_fakes_and_updates_discriminators(
    tmp_path: Path,
) -> None:
    method = _method(_config(tmp_path))
    real_a, real_b = method._unpack_batch(_batch())
    _objective, fake_a, fake_b = method._generator_phase(real_a, real_b, epoch=0, global_step=0)
    assert fake_a.requires_grad and fake_b.requires_grad
    seen_inputs: list[torch.Tensor] = []
    hooks = [
        module.register_forward_pre_hook(lambda _mod, args: seen_inputs.append(args[0]))
        for module in (method.D_A, method.D_B)
    ]
    before = {name: _params(m) for name, m in method._models().items()}

    method._discriminator_phase(real_a, real_b, fake_a, fake_b, epoch=0, global_step=0)

    for hook in hooks:
        hook.remove()
    assert len(seen_inputs) == 4
    assert not any(tensor.requires_grad for tensor in seen_inputs)
    assert _changed(before["D_A"], method.D_A)
    assert _changed(before["D_B"], method.D_B)
    assert not _changed(before["G_A_to_B"], method.G_A_to_B)
    assert not _changed(before["G_B_to_A"], method.G_B_to_A)
    assert all(not image.requires_grad for image in method._pool_A.images + method._pool_B.images)


def test_identity_loss_can_be_disabled(tmp_path: Path) -> None:
    def disable_identity(data: dict[str, Any]) -> None:
        data["training"]["losses"]["generator"][2]["enabled"] = False

    method = _method(_config(tmp_path, disable_identity))
    calls: list[int] = []
    method.G_B_to_A.register_forward_hook(lambda *_args: calls.append(1))

    metrics = method.step(_batch(), epoch=0, global_step=0)

    assert "generator_identity_l1" not in metrics.raw
    assert len(calls) == 2  # rec_a and fake_a only; no identity pass
    assert math.isfinite(metrics.losses["loss_G"])


def test_replay_pool_fills_then_bounds_capacity() -> None:
    pool = ReplayPool(3, seed=0)
    images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).requires_grad_()

    first = pool.query(images[:3] * 1)
    assert torch.equal(first, images[:3].detach())
    for _ in range(20):
        out = pool.query(torch.full((2, 1, 1, 1), 9.0, requires_grad=True) * 1)
        assert out.shape == (2, 1, 1, 1)
        assert not out.requires_grad
    assert len(pool.images) == 3
    assert all(not image.requires_grad for image in pool.images)


def test_replay_pool_capacity_zero_passes_through() -> None:
    pool = ReplayPool(0, seed=0)
    images = torch.randn(2, 3, 4, 4, requires_grad=True)

    out = pool.query(images)

    assert torch.equal(out, images.detach()) and not out.requires_grad
    assert pool.images == []


def _drive(pool: ReplayPool, steps: int, offset: int = 0) -> list[float]:
    return [
        float(pool.query(torch.full((1, 1, 1, 1), float(offset + step))).item())
        for step in range(steps)
    ]


def test_replay_pool_is_deterministic_and_mixes_history() -> None:
    first, second = ReplayPool(4, seed=3), ReplayPool(4, seed=3)

    outputs = _drive(first, 60)

    assert outputs == _drive(second, 60)
    assert outputs != [float(step) for step in range(60)]
    assert outputs != _drive(ReplayPool(4, seed=4), 60)


def test_replay_pool_state_round_trip_continues_identically() -> None:
    pool = ReplayPool(4, seed=3)
    _drive(pool, 10)
    restored = ReplayPool(4, seed=99)

    restored.load_state_dict(pool.state_dict(), _CPU)

    assert _drive(restored, 30, offset=100) == _drive(pool, 30, offset=100)


def test_replay_pool_rejects_malformed_state() -> None:
    pool = ReplayPool(2, seed=0)
    state = pool.state_dict()
    with pytest.raises(CheckpointCompatibilityError, match="capacity"):
        ReplayPool(3, seed=0).load_state_dict(state, _CPU)
    with pytest.raises(CheckpointCompatibilityError, match="rng_state"):
        pool.load_state_dict({**state, "rng_state": [1, 2]}, _CPU)
    with pytest.raises(CheckpointCompatibilityError, match="rng_state"):
        pool.load_state_dict({**state, "rng_state": torch.zeros(3, dtype=torch.uint8)}, _CPU)
    with pytest.raises(CheckpointCompatibilityError, match="images"):
        pool.load_state_dict({**state, "images": [torch.zeros(1)] * 3}, _CPU)
    with pytest.raises(CheckpointCompatibilityError, match=r"images\[0\]"):
        pool.load_state_dict({**state, "images": [torch.zeros(1, 3, 4, 4, dtype=torch.long)]}, _CPU)
    with pytest.raises(CheckpointCompatibilityError, match=r"images\[1\]"):
        pool.load_state_dict(
            {**state, "images": [torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 8, 8)]}, _CPU
        )


def test_method_pools_use_independent_seed_derived_rngs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first, second = _method(config, seed=7), _method(config, seed=7)
    other = _method(config, seed=8)

    state = first._pool_A.state_dict()["rng_state"]
    assert not torch.equal(state, first._pool_B.state_dict()["rng_state"])
    assert torch.equal(state, second._pool_A.state_dict()["rng_state"])
    assert not torch.equal(state, other._pool_A.state_dict()["rng_state"])


def test_validation_is_deterministic_restores_modes_and_writes_previews(tmp_path: Path) -> None:
    method = _method(_config(tmp_path))
    method.step(_batch(), epoch=0, global_step=0)
    method.D_B.eval()
    loader = [_batch(3), _batch(4)]

    writer = ValidationPreviewWriter(tmp_path / "val")
    first = method.validate(loader, epoch=2, preview_sink=writer)  # type: ignore[arg-type]
    second = method.validate(loader, epoch=2, preview_sink=writer)  # type: ignore[arg-type]

    assert first == second
    assert set(first.losses) == {"loss_G", "loss_D"}
    assert first.image == {}
    assert "discriminator_adversarial_lsgan" in first.raw
    assert (method.G_A_to_B.training, method.G_B_to_A.training) == (True, True)
    assert (method.D_A.training, method.D_B.training) == (True, False)
    assert sorted(path.name for path in (tmp_path / "val").iterdir()) == [
        "epoch2_batch0_preview.tif",
        "epoch2_batch1_preview.tif",
    ]
    assert method.checkpoint_selection_metrics(first) == {"loss_G_val": first.losses["loss_G"]}
    assert (
        method.validation_metric(first, "loss_val_raw_generator_cycle_l1")
        == first.raw["generator_cycle_l1"]
    )
    assert method.validation_metric(first, "val_ssim") is None


def _with_linear_decay(data: dict[str, Any]) -> None:
    data["training"]["epochs"] = 4
    data["training"]["scheduler"] = {"name": "linear_decay", "decay_start_epoch": 0}


def test_cyclegan_v4_checkpoint_round_trip_restores_all_state(tmp_path: Path) -> None:
    config = _config(tmp_path, _with_linear_decay)
    source = _method(config)
    for step in range(3):
        source.step(_batch(step), epoch=0, global_step=step)
    source.step_schedulers(epoch=0, validation_metrics=None)
    path = _manager(config, source).save(0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["method"]["pairing"] == "unpaired"
    assert payload["method"]["prediction_directions"] == ["A_to_B", "B_to_A"]
    assert set(payload["state"]) == {
        "models",
        "optimization",
        "optimizers",
        "scalers",
        "schedulers",
        "replay_pools",
    }

    torch.manual_seed(123)
    resumed = CycleGANMethod(config, _CPU, seed=999)
    assert _manager(config, resumed).load(path) == 1

    for name, model in source._models().items():
        resumed_state = resumed._models()[name].state_dict()
        assert all(torch.equal(v, resumed_state[k]) for k, v in model.state_dict().items())
    source_state, resumed_state = source.state_dict(), resumed.state_dict()
    for role in ("generators", "discriminators"):
        assert (
            source_state["optimizers"][role]["param_groups"]
            == resumed_state["optimizers"][role]["param_groups"]
        )
        assert source_state["scalers"][role] == resumed_state["scalers"][role]
        assert source_state["schedulers"][role] == resumed_state["schedulers"][role]
    assert source.learning_rates() == resumed.learning_rates()
    for name in ("fake_A", "fake_B"):
        stored, restored = source_state["replay_pools"][name], resumed_state["replay_pools"][name]
        assert len(stored["images"]) == len(restored["images"]) == 6
        assert all(
            torch.equal(a, b) for a, b in zip(stored["images"], restored["images"], strict=True)
        )
        assert torch.equal(stored["rng_state"], restored["rng_state"])

    # Continuation: identical subsequent steps produce identical parameters and pools.
    for method in (source, resumed):
        method.step(_batch(10), epoch=1, global_step=3)
    for name, model in source._models().items():
        resumed_state = resumed._models()[name].state_dict()
        assert all(torch.equal(v, resumed_state[k]) for k, v in model.state_dict().items())
    assert all(
        torch.equal(a, b)
        for a, b in zip(source._pool_B.images, resumed._pool_B.images, strict=True)
    )


def test_semantic_domain_mismatch_is_rejected_before_state_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    path = _manager(config, _method(config)).save(0)

    def rename_domain(data: dict[str, Any]) -> None:
        data["model"]["inputs"] = ["autofluorescence"]
        data["data"]["domains"] = {"autofluorescence": "a", "stained": "b"}

    renamed = _config(tmp_path, rename_domain)
    method = _method(renamed)
    monkeypatch.setattr(method, "load_state_dict", lambda _state: pytest.fail("state interpreted"))

    with pytest.raises(CheckpointCompatibilityError, match=r"method.inputs\[0\]"):
        _manager(renamed, method).load(path)


def _first_key(mapping: dict[str, Any]) -> str:
    return next(iter(mapping))


_MALFORMED: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "missing_pools": (
        lambda state: state.pop("replay_pools"),
        r"state has mismatched keys: missing \['replay_pools'\]",
    ),
    "missing_optimization": (
        lambda state: state.pop("optimization"),
        r"state has mismatched keys: missing \['optimization'\]",
    ),
    "missing_model_role": (
        lambda state: state["models"].pop("D_B"),
        r"state.models has mismatched keys: missing \['D_B'\]",
    ),
    "wrong_model_keys": (
        lambda state: state["models"]["G_A_to_B"].popitem(),
        "state.models.G_A_to_B has mismatched keys",
    ),
    "wrong_shape": (
        lambda state: state["models"]["D_A"].update(
            {_first_key(state["models"]["D_A"]): torch.zeros(3)}
        ),
        "state.models.D_A.* has shape",
    ),
    "wrong_dtype": (
        lambda state: state["models"]["G_B_to_A"].update(
            {key: value.half() for key, value in list(state["models"]["G_B_to_A"].items())[:1]}
        ),
        "state.models.G_B_to_A.* has dtype torch.float16",
    ),
    "scheduler_without_config": (
        lambda state: state["schedulers"].update({"generators": {"x": 1}}),
        "state.schedulers.generators is present but the current run has no scheduler",
    ),
    "optimizer_not_mapping": (
        lambda state: state["optimizers"].update({"generators": []}),
        "state.optimizers.generators must be a mapping",
    ),
    "optimizer_params": (
        lambda state: state["optimizers"]["discriminators"]["param_groups"][0]["params"].pop(),
        r"state.optimizers.discriminators.param_groups\[0\].params does not match",
    ),
    "optimizer_policy": (
        lambda state: state["optimization"]["generators"]["optimizer"].update(eps=1e-6),
        "state.optimization.generators.optimizer.eps is 1e-06",
    ),
    "scaler_not_mapping": (
        lambda state: state["scalers"].update({"discriminators": None}),
        "state.scalers.discriminators must be a mapping",
    ),
    "pool_missing_rng": (
        lambda state: state["replay_pools"]["fake_A"].pop("rng_state"),
        r"state.replay_pools.fake_A has mismatched keys: missing \['rng_state'\]",
    ),
    "pool_missing_role": (
        lambda state: state["replay_pools"].pop("fake_B"),
        "state.replay_pools has mismatched keys",
    ),
    "pool_capacity": (
        lambda state: state["replay_pools"]["fake_B"].update(capacity=1),
        "state.replay_pools.fake_B.capacity is 1",
    ),
    "pool_image_type": (
        lambda state: state["replay_pools"]["fake_A"]["images"].append([0.0]),
        r"state.replay_pools.fake_A.images\[2\] must be a floating",
    ),
    "pool_image_shape": (
        lambda state: state["replay_pools"]["fake_B"]["images"].__setitem__(
            0, torch.zeros(2, 3, 32, 32)
        ),
        r"state.replay_pools.fake_B.images\[0\] must be a floating",
    ),
    "pool_rng_dtype": (
        lambda state: state["replay_pools"]["fake_B"].update(rng_state=torch.zeros(5)),
        "state.replay_pools.fake_B.rng_state must be a uint8 tensor",
    ),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_state_is_rejected_before_mutation(tmp_path: Path, case: str) -> None:
    config = _config(tmp_path)
    source = _method(config)
    source.step(_batch(), epoch=0, global_step=0)
    state = copy.deepcopy(source.state_dict())
    mutate, match = _MALFORMED[case]
    mutate(state)
    target = _method(config, seed=3)
    target.step(_batch(5), epoch=0, global_step=0)
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match=match):
        target.load_state_dict(state)

    assert_nested_equal(target.state_dict(), before)


def _with_plateau(data: dict[str, Any]) -> None:
    data["training"]["scheduler"] = {"name": "reduce_on_plateau", "factor": 0.5, "patience": 1}


def _with_longer_linear_decay(data: dict[str, Any]) -> None:
    _with_linear_decay(data)
    data["training"]["epochs"] = 8


@pytest.mark.parametrize(
    ("saved", "current", "match"),
    [
        (_with_linear_decay, None, "generators.scheduler is {.*} in the checkpoint but None"),
        (None, _with_linear_decay, "generators.scheduler is None in the checkpoint"),
        (_with_linear_decay, _with_plateau, "scheduler.name is 'linear_decay'"),
        (_with_linear_decay, _with_longer_linear_decay, "scheduler.epochs is 4 .* but 8"),
        (
            _with_plateau,
            lambda data: (_with_plateau(data), data["training"]["scheduler"].update(patience=3)),
            "scheduler.patience is 1 .* but 3",
        ),
    ],
)
def test_cyclegan_resume_rejects_changed_scheduler_policy(
    tmp_path: Path,
    saved: Callable[[dict[str, Any]], None] | None,
    current: Callable[[dict[str, Any]], None] | None,
    match: str,
) -> None:
    config = _config(tmp_path, saved)
    source = _method(config)
    source.step(_batch(), epoch=0, global_step=0)
    path = _manager(config, source).save(0)
    changed = _config(tmp_path, current)
    target = _method(changed, seed=3)
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match=match):
        _manager(changed, target).load(path)
    assert_nested_equal(target.state_dict(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cyclegan_replay_pools_are_restored_onto_the_execution_device(tmp_path: Path) -> None:
    cuda = torch.device("cuda")
    config = _config(tmp_path)
    source = CycleGANMethod(config, cuda, seed=7)
    source.train_mode()
    source.step(_batch(), epoch=0, global_step=0)
    path = _manager(config, source).save(0)

    resumed = CycleGANMethod(config, cuda, seed=9)
    _manager(config, resumed).load(path)

    assert all(image.device.type == "cuda" for image in resumed._pool_A.images)
    assert_nested_equal(resumed.state_dict(), source.state_dict())
    resumed.step(_batch(2), epoch=1, global_step=1)
