"""Guards for the explicit method/component definition seam.

Generic layers must not know the built-in methods, depend on GAN loss configuration, or
gain any code-discovery mechanism; the built-ins must use the public definition path.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from virtual_staining.definitions import ComponentDefinition, MethodDefinition
from virtual_staining.methods.builtin import (
    PIX2PIX_IMAGE_METRICS,
    CycleGANDefinition,
    Pix2PixDefinition,
    builtin_definitions,
    builtin_method_definitions,
)
from virtual_staining.metrics import BUILTIN_METRICS

_PACKAGE = Path(__file__).resolve().parents[2] / "virtual_staining"
_RUNTIMES = ("virtual_staining.methods.pix2pix", "virtual_staining.methods.cyclegan")
# Modules allowed to name the built-in methods: their definitions and runtimes, the
# canonical built-in loss registry, and the documented default ``method.name``.
_BUILTIN_NAME_OWNERS = {
    "methods/builtin.py",
    "methods/pix2pix.py",
    "methods/cyclegan.py",
    "loss_definitions.py",
    "config/method.py",
}


def _loaded_after(code: str) -> set[str]:
    script = textwrap.dedent(code) + "\nimport sys\nprint('\\n'.join(sorted(sys.modules)))\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PACKAGE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return set(completed.stdout.split())


def test_config_resolution_imports_no_method_runtime_or_torch() -> None:
    loaded = _loaded_after(
        """
        from pathlib import Path
        from virtual_staining.config.run import RunConfig
        for path in sorted(Path("config/runs").glob("*.yaml")):
            RunConfig.from_yaml(path).to_dict()
        """
    )
    assert not loaded & {*_RUNTIMES, "torch"}


def test_generic_trainer_does_not_import_gan_loss_configuration() -> None:
    loaded = _loaded_after(
        "import virtual_staining.training.trainer, virtual_staining.training.runtime"
    )
    assert "virtual_staining.config.losses" not in loaded
    assert "virtual_staining.loss_definitions" not in loaded
    assert not loaded & {*_RUNTIMES, "virtual_staining.methods.builtin"}


def test_generic_inference_runner_imports_no_concrete_loader() -> None:
    loaded = _loaded_after("import virtual_staining.inference.runner")
    assert not loaded & {*_RUNTIMES, "virtual_staining.methods.builtin"}


def _string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_only_definition_owners_name_the_builtin_methods() -> None:
    offenders = sorted(
        str(path.relative_to(_PACKAGE))
        for path in _PACKAGE.rglob("*.py")
        if str(path.relative_to(_PACKAGE)) not in _BUILTIN_NAME_OWNERS
        and _string_constants(path) & {"pix2pix", "cyclegan"}
    )
    assert offenders == []


# Layers that resolve configuration, definitions, checkpoints and runtimes. (The CLI's
# lazy subcommand import and version probing live elsewhere and never read user input.)
_SEAM = (
    "definitions.py",
    "methods",
    "models",
    "training",
    "inference",
    "checkpoint_contract.py",
    "checkpoint_selection.py",
    "config/run.py",
    "config/method.py",
    "config/model.py",
    "config/training.py",
    "config/scheduler.py",
    "config/evaluation.py",
    "metrics.py",
    "evaluation",
)


def test_no_code_discovery_or_import_strings_exist() -> None:
    forbidden_calls = {
        "import_module",
        "__import__",
        "entry_points",
        "iter_modules",
        "walk_packages",
    }
    forbidden_modules = {"importlib", "pkgutil", "importlib.metadata", "pkg_resources"}
    violations: list[str] = []
    paths = [
        path
        for entry in _SEAM
        for path in (
            (_PACKAGE / entry).rglob("*.py") if (_PACKAGE / entry).is_dir() else [_PACKAGE / entry]
        )
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                violations += [f"{path}: import {n}" for n in names if n in forbidden_modules]
            elif isinstance(node, ast.Attribute | ast.Name):
                name = node.attr if isinstance(node, ast.Attribute) else node.id
                if name in forbidden_calls:
                    violations.append(f"{path}: {name}")
        if {"class_path", "entry_point"} & {c.lower() for c in _string_constants(path)}:
            violations.append(f"{path}: plugin key")
    assert violations == []


def test_default_definitions_register_each_builtin_once_through_the_public_api() -> None:
    definitions = builtin_definitions()
    names = [definition.name for definition in builtin_method_definitions()]

    assert sorted(names) == ["cyclegan", "pix2pix"] and len(set(names)) == len(names)
    assert set(definitions.methods) == {"pix2pix", "cyclegan"}
    assert all(isinstance(d, MethodDefinition) for d in definitions.methods.values())
    assert set(definitions.components) == {"concat_unet", "resnet", "patchgan"}
    assert all(isinstance(d, ComponentDefinition) for d in definitions.components.values())
    assert tuple(definitions.metrics.values()) == BUILTIN_METRICS
    assert builtin_definitions() is definitions


def test_pix2pix_explicitly_owns_its_reused_validation_image_metrics() -> None:
    assert {name: metric.name for name, metric in PIX2PIX_IMAGE_METRICS.items()} == {
        "ssim": "ssim",
        "psnr": "psnr",
        "mae": "mae",
        "rmse": "rmse",
        "pcc_rgb_mean": "pcc_rgb_mean",
        "pcc_gray": "pcc_gray",
    }
    # Validation metrics are per output; loss_G_val stays the joint default metric.
    assert list(Pix2PixDefinition().checkpoint_modes(("PAS", "HE")).items()) == [
        ("loss_G_val", "min"),
        ("val_ssim__PAS", "max"),
        ("val_ssim__HE", "max"),
        ("val_psnr__PAS", "max"),
        ("val_psnr__HE", "max"),
        ("val_mae__PAS", "min"),
        ("val_mae__HE", "min"),
        ("val_rmse__PAS", "min"),
        ("val_rmse__HE", "min"),
        ("val_pcc_rgb_mean__PAS", "max"),
        ("val_pcc_rgb_mean__HE", "max"),
        ("val_pcc_gray__PAS", "max"),
        ("val_pcc_gray__HE", "max"),
    ]
    assert dict(CycleGANDefinition.checkpoint_metrics) == {"loss_G_val": "min"}


def test_pix2pix_monitor_names_accept_safe_output_identifiers() -> None:
    definition = Pix2PixDefinition()

    assert definition.monitor_mode("val_ssim__H-E_2", "monitor") == "max"
    assert definition.monitor_mode("val_mae__PAS", "monitor") == "min"
    assert definition.monitor_mode("loss_val_raw_generator_l1__H-E_2", "monitor") == "min"
    assert definition.resolve_default_monitor(("HE",)) == "val_ssim__HE"
    assert definition.resolve_default_monitor(("HE", "PAS")) is None
    with pytest.raises(ValueError, match="val_<metric>__<output>"):
        definition.monitor_mode("val_ssim", "monitor")


@pytest.mark.parametrize(
    "module",
    [
        "definitions.py",
        "config/run.py",
        "config/training.py",
        "training/trainer.py",
        "training/runtime.py",
        "inference/runner.py",
    ],
)
def test_generic_modules_do_not_branch_on_method_names(module: str) -> None:
    tree = ast.parse((_PACKAGE / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            assert not any(
                isinstance(o, ast.Constant) and o.value in {"pix2pix", "cyclegan"} for o in operands
            ), f"{module}:{node.lineno}"
