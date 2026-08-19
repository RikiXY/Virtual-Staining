from __future__ import annotations

import textwrap
from pathlib import Path

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
    training = data.get("training")
    if isinstance(training, dict):
        training["augmentation"] = data.pop("augmentation", training.get("augmentation", {}))
        training["losses"] = data.pop("losses", training.get("losses", {}))
    preprocessing = data.get("preprocessing")
    if isinstance(preprocessing, dict) and "image_size" in preprocessing:
        preprocessing["patch_size"] = preprocessing.pop("image_size")
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
