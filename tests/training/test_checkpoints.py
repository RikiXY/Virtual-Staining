from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast, get_args

import pytest
import torch

import virtual_staining.checkpoint_contract as checkpoint_contract
import virtual_staining.training.checkpoints as training_checkpoints
from virtual_staining.checkpoint_contract import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCompatibilityError,
)
from virtual_staining.checkpoint_selection import (
    RANKED_CHECKPOINT_POLICIES,
    SUPPORTED_CHECKPOINT_POLICIES,
    CheckpointMode,
    resolve_checkpoint_path,
    update_checkpoint_selection,
)
from virtual_staining.config.inference import InferenceConfig
from virtual_staining.training.checkpoints import MethodCheckpointManager
from virtual_staining.training.runtime import TrainingMethodRuntime


@pytest.mark.parametrize("policy", sorted(SUPPORTED_CHECKPOINT_POLICIES))
@pytest.mark.parametrize("mode", get_args(CheckpointMode))
def test_checkpoint_policy_and_mode_contracts(tmp_path: Path, policy: str, mode: str) -> None:
    config = InferenceConfig.from_mapping(
        {"checkpoint_policy": policy, "checkpoint_metric": "val_ssim"}
    )
    first, latest = tmp_path / "ep001.pth", tmp_path / "ep002.pth"
    for epoch, path, value in ((1, first, 0.1), (2, latest, 0.9)):
        path.touch()
        update_checkpoint_selection(
            tmp_path,
            metrics={"val_ssim": value},
            modes={"val_ssim": mode},
            top_k=2,
            epoch=epoch,
            checkpoint_path=path,
        )
    expected = latest if policy == "latest" or mode == "max" else first
    assert config.checkpoint_policy == policy
    assert resolve_checkpoint_path(tmp_path, policy=policy, metric="val_ssim") == expected


@pytest.mark.parametrize("policy", sorted(RANKED_CHECKPOINT_POLICIES))
def test_ranked_checkpoint_policies_require_metric_and_accept_rank(policy: str) -> None:
    with pytest.raises(ValueError, match="checkpoint_metric is required"):
        InferenceConfig.from_mapping({"checkpoint_policy": policy})
    config = InferenceConfig.from_mapping(
        {"checkpoint_policy": policy, "checkpoint_metric": "val_ssim", "checkpoint_rank": 2}
    )
    assert config.checkpoint_rank == 2


def test_checkpoint_policy_validation_remains_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown checkpoint_policy"):
        InferenceConfig.from_mapping({"checkpoint_policy": "unknown"})
    with pytest.raises(ValueError, match="Unsupported checkpoint policy"):
        resolve_checkpoint_path(tmp_path, policy="unknown")
    with pytest.raises(ValueError, match="checkpoint_rank is supported only"):
        InferenceConfig.from_mapping({"checkpoint_policy": "latest", "checkpoint_rank": 1})


def test_checkpoint_selection_rejects_unknown_mode_without_writing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ep001.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match="mode must be one of"):
        update_checkpoint_selection(
            tmp_path,
            metrics={"val_ssim": 0.5},
            modes={"val_ssim": "unknown"},
            top_k=2,
            epoch=1,
            checkpoint_path=checkpoint,
        )
    assert not (tmp_path / "best.json").exists()


class _FakeMethod:
    """Tiny method whose state deliberately does not resemble a GAN."""

    def __init__(self, **overrides: Any) -> None:
        self.name = overrides.get("name", "fake")
        self.pairing = overrides.get("pairing", "unpaired")
        self.input_names = overrides.get("input_names", ("left", "right"))
        self.output_names = overrides.get("output_names", ("middle",))
        self.prediction_directions = overrides.get("prediction_directions", ("sideways",))
        self.components = overrides.get(
            "components",
            {"alpha": {"class": "Alpha", "width": 3}, "beta": {"class": "Beta", "depth": 2}},
        )
        self.alpha = torch.nn.Linear(2, 3)
        self.beta = torch.nn.Conv2d(1, 1, 1)
        self.joint = torch.optim.SGD(
            [*self.alpha.parameters(), *self.beta.parameters()], lr=0.1, momentum=0.9
        )
        self.counter = overrides.get("counter", 123)
        self.loaded_states: list[Mapping[str, Any]] = []

    def component_metadata(self) -> Mapping[str, object]:
        return self.components

    def state_dict(self) -> dict[str, Any]:
        return {
            "models": {"alpha": self.alpha.state_dict(), "beta": self.beta.state_dict()},
            "optimizers": {"joint": self.joint.state_dict()},
            "custom_runtime_state": {"counter": self.counter},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.loaded_states.append(state)
        self.alpha.load_state_dict(state["models"]["alpha"])
        self.beta.load_state_dict(state["models"]["beta"])
        self.joint.load_state_dict(state["optimizers"]["joint"])
        self.counter = state["custom_runtime_state"]["counter"]


def _fake_manager(
    root: Path, method: _FakeMethod, image_size: tuple[int, int] = (16, 16)
) -> MethodCheckpointManager:
    return MethodCheckpointManager(
        cast(TrainingMethodRuntime, method),
        root,
        image_size=image_size,
        device=torch.device("cpu"),
        config_hash="sha256:abc",
    )


def _read(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_generic_checkpoint_round_trips_arbitrary_method_state(tmp_path: Path) -> None:
    source = _FakeMethod(counter=123)
    source.joint.zero_grad()
    source.alpha(torch.ones(1, 2)).sum().backward()
    source.joint.step()
    path = _fake_manager(tmp_path, source).save(7)

    payload = _read(path)
    assert path.name == "ep007.pth"
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION == 4
    assert payload["epoch"] == 7
    assert payload["method"] == {
        "name": "fake",
        "pairing": "unpaired",
        "inputs": ["left", "right"],
        "outputs": ["middle"],
        "prediction_directions": ["sideways"],
        "components": {
            "alpha": {"class": "Alpha", "width": 3},
            "beta": {"class": "Beta", "depth": 2},
        },
    }
    assert payload["image_size"] == [16, 16]
    assert payload["config_hash"] == "sha256:abc"
    assert set(payload["state"]) == {"models", "optimizers", "custom_runtime_state"}

    target = _FakeMethod(counter=0)
    manager = _fake_manager(tmp_path, target)
    assert manager.latest() == path
    assert manager.load(path) == 8
    assert len(target.loaded_states) == 1
    assert target.counter == 123
    for name, value in source.alpha.state_dict().items():
        assert torch.equal(value, target.alpha.state_dict()[name])
    assert target.joint.state_dict()["state"].keys() == source.joint.state_dict()["state"].keys()


_IDENTITY_MISMATCHES: dict[str, tuple[dict[str, Any], str]] = {
    "method": ({"name": "other"}, "method.name"),
    "pairing": ({"pairing": "paired"}, "method.pairing"),
    "input_order": ({"input_names": ("right", "left")}, r"method.inputs\[0\]"),
    "input_set": ({"input_names": ("left",)}, "method.inputs"),
    "outputs": ({"output_names": ("other",)}, r"method.outputs\[0\]"),
    "directions": ({"prediction_directions": ("forward",)}, "method.prediction_directions"),
    "component_class": (
        {
            "components": {
                "alpha": {"class": "Gamma", "width": 3},
                "beta": {"class": "Beta", "depth": 2},
            }
        },
        "method.components.alpha.class",
    ),
    "component_parameter": (
        {
            "components": {
                "alpha": {"class": "Alpha", "width": 4},
                "beta": {"class": "Beta", "depth": 2},
            }
        },
        "method.components.alpha.width",
    ),
    "component_missing": (
        {"components": {"alpha": {"class": "Alpha", "width": 3}}},
        "method.components.beta",
    ),
}


@pytest.mark.parametrize("case", sorted(_IDENTITY_MISMATCHES))
def test_semantic_mismatch_is_rejected_before_state_loading(tmp_path: Path, case: str) -> None:
    path = _fake_manager(tmp_path, _FakeMethod()).save(0)
    overrides, field_pattern = _IDENTITY_MISMATCHES[case]
    target = _FakeMethod(**overrides)
    with pytest.raises(CheckpointCompatibilityError, match=field_pattern):
        _fake_manager(tmp_path, target).load(path)
    assert target.loaded_states == []


def test_image_size_mismatch_is_rejected_before_state_loading(tmp_path: Path) -> None:
    path = _fake_manager(tmp_path, _FakeMethod()).save(0)
    target = _FakeMethod()
    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        _fake_manager(tmp_path, target, image_size=(32, 16)).load(path)
    assert target.loaded_states == []


def _legacy_v3(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 3,
        "epoch": 0,
        "architecture": {"generator": {}, "discriminator": {}},
        "normalization_contract": payload["normalization"],
        "generator_state_dict": {},
        "discriminator_state_dict": {},
        "optimizerG_state_dict": {},
        "optimizerD_state_dict": {},
        "image_size": payload["image_size"],
    }


_PAYLOAD_MUTATIONS: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "not_mapping": (lambda payload: [payload], "not a mapping"),
    "unversioned": (
        lambda payload: {k: v for k, v in payload.items() if k != "format_version"},
        "unversioned",
    ),
    "v3": (_legacy_v3, "unsupported format version 3"),
    "v2": (lambda payload: {**payload, "format_version": 2}, "unsupported format version 2"),
    "v5": (lambda payload: {**payload, "format_version": 5}, "unsupported format version 5"),
    "string_version": (
        lambda payload: {**payload, "format_version": "4"},
        "unsupported format version '4'",
    ),
    "bad_epoch": (lambda payload: {**payload, "epoch": -1}, "malformed epoch"),
    "missing_method": (
        lambda payload: {k: v for k, v in payload.items() if k != "method"},
        "'method' metadata",
    ),
    "missing_pairing": (
        lambda payload: {
            **payload,
            "method": {k: v for k, v in payload["method"].items() if k != "pairing"},
        },
        "method.pairing",
    ),
    "extra_method_key": (
        lambda payload: {**payload, "method": {**payload["method"], "topology": "gan"}},
        "unexpected method metadata keys",
    ),
    "normalization": (
        lambda payload: {
            **payload,
            "normalization": {**payload["normalization"], "input_range": "[0, 1]"},
        },
        "normalization.input_range",
    ),
    "missing_state": (
        lambda payload: {k: v for k, v in payload.items() if k != "state"},
        "no method state",
    ),
    "state_not_mapping": (lambda payload: {**payload, "state": [1, 2]}, "malformed method state"),
    "state_empty": (lambda payload: {**payload, "state": {}}, "malformed method state"),
    "state_bad_keys": (
        lambda payload: {**payload, "state": {1: "x"}},
        "malformed method state",
    ),
}


@pytest.mark.parametrize("case", sorted(_PAYLOAD_MUTATIONS))
def test_malformed_or_unsupported_payload_is_rejected_before_state_loading(
    tmp_path: Path, case: str
) -> None:
    path = _fake_manager(tmp_path, _FakeMethod()).save(0)
    mutate, message = _PAYLOAD_MUTATIONS[case]
    torch.save(mutate(_read(path)), path)
    target = _FakeMethod()
    with pytest.raises(CheckpointCompatibilityError, match=message):
        _fake_manager(tmp_path, target).load(path)
    assert target.loaded_states == []


def test_generic_checkpoint_layer_has_no_topology_or_legacy_paths() -> None:
    forbidden = ("generator", "discriminator", "optimizerg", "optimizerd", "scalerg", "loss_g")
    for module in (checkpoint_contract, training_checkpoints):
        source = Path(cast(str, module.__file__)).read_text(encoding="utf-8")
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        strings = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        tokens = " ".join((*names, *strings)).lower()
        assert not [word for word in forbidden if word in tokens], module.__name__
        assert "legacy" not in source.lower()
        assert "migrat" not in source.lower()
    for obsolete in (
        "make_arch_metadata",
        "validate_checkpoint_metadata",
        "check_generator_arch",
        "check_discriminator_arch",
        "load_legacy_v3",
    ):
        assert not hasattr(checkpoint_contract, obsolete)
    assert not hasattr(training_checkpoints, "CheckpointManager")
