from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, content: str) -> Path:
    """Write a dedented YAML snippet and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def yaml_section(name: str, content: str) -> str:
    """Return a YAML section with consistently indented dedented content."""
    body = textwrap.dedent(content).strip()
    return f"{name}:\n{textwrap.indent(body, '  ')}"


def write_run_config(
    tmp_path: Path,
    section_yaml: str = "",
    *,
    filename: str = "run.yaml",
    dataset_root: Path | None = None,
    results_path: Path | None = None,
    run_name: str = "current_run",
) -> Path:
    """Write the minimal project config shared by run-level tests."""
    dataset_root = tmp_path / "data" if dataset_root is None else dataset_root
    results_path = tmp_path / "results" if results_path is None else results_path
    section = textwrap.dedent(section_yaml).strip()
    content = f"dataset_root: {dataset_root}\nresults_path: {results_path}\nrun_name: {run_name}\n"
    if section:
        content += f"{section}\n"
    data = yaml.safe_load(content)
    if "model" not in data:
        data["model"] = {"inputs": ["label_free"], "target": "stained"}
    content = yaml.safe_dump(data, sort_keys=False)
    return write_yaml(tmp_path / filename, content)


def write_queue_config(
    tmp_path: Path,
    jobs_yaml: str,
    *,
    continue_on_failure: bool = False,
    name: str = "nightly",
) -> Path:
    """Write a local queue YAML file with dedented job entries."""
    jobs = textwrap.dedent(jobs_yaml).rstrip()
    content = (
        f"name: {name}\n"
        f"continue_on_failure: {'true' if continue_on_failure else 'false'}\n"
        "jobs:\n"
        f"{jobs}\n"
    )
    return write_yaml(tmp_path / "config" / "queues" / f"{name}.yaml", content)


def cyclegan_config_data(tmp_path: Path) -> dict[str, Any]:
    """Return a canonical tiny CycleGAN run configuration mapping for tests to adjust."""
    return {
        "dataset_root": str(tmp_path / "dataset"),
        "results_path": str(tmp_path / "results"),
        "run_name": "cyclegan_run",
        "image_size": [32, 32],
        "method": {"name": "cyclegan"},
        "data": {
            "pairing": "unpaired",
            "domains": {"label_free": "domains/label_free", "stained": "domains/stained"},
        },
        "model": {
            "inputs": ["label_free"],
            "target": "stained",
            "generator": {"architecture": "resnet", "base_channels": 4, "blocks": 1},
            "discriminator": {"ndf": 4},
        },
        "training": {
            "batch_size": 2,
            "epochs": 2,
            "seed": 7,
            "num_workers": 0,
            "validate_rate": 1,
            "checkpoint_rate": 1,
            "log_rate": 1,
            "losses": {
                "generator": [
                    {"name": "adversarial_lsgan", "weight": 1.0},
                    {"name": "cycle_l1", "weight": 10.0},
                    {"name": "identity_l1", "weight": 5.0},
                ],
                "discriminator": [{"name": "adversarial_lsgan", "weight": 1.0}],
            },
        },
        "inference": {"checkpoint_policy": "latest"},
    }


def write_config_data(path: Path, data: dict[str, Any]) -> Path:
    """Write a run configuration mapping as YAML and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
