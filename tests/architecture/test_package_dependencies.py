from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

PROJECT = "virtual_staining"
COMPONENTS = {
    "cli",
    "applications",
    "config",
    "checkpoint_contract",
    "checkpoint_selection",
    "metrics",
    "data",
    "models",
    "experiment",
    "training",
    "inference",
    "evaluation",
    "utils",
}
ALLOWED_EDGES = {
    "cli": {"applications", "cli", "metrics"},
    "applications": {
        "checkpoint_contract",
        "checkpoint_selection",
        "config",
        "data",
        "evaluation",
        "experiment",
        "inference",
        "metrics",
        "models",
        "training",
        "utils",
    },
    "config": {"config", "checkpoint_selection", "metrics", "utils"},
    "checkpoint_selection": {"metrics"},
    "checkpoint_contract": set(),
    "metrics": set(),
    "data": {"config", "data", "utils"},
    "models": {"models"},
    "experiment": {"config", "experiment"},
    "training": {
        "checkpoint_contract",
        "checkpoint_selection",
        "config",
        "experiment",
        "metrics",
        "models",
        "training",
        "utils",
    },
    "inference": {
        "checkpoint_contract",
        "checkpoint_selection",
        "config",
        "data",
        "experiment",
        "inference",
        "models",
        "utils",
    },
    "evaluation": {"config", "evaluation", "metrics", "utils"},
    "utils": {"utils"},
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(Path("virtual_staining")).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PROJECT, *parts))


def _component(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != PROJECT:
        return None
    return parts[1] if parts[1] in COMPONENTS else None


def _imports(path: Path) -> set[str]:
    module_name = _module_name(path)
    package = module_name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolve_name("." * node.level + (node.module or ""), package)
            else:
                base = node.module or ""
            imports.update(
                base if not alias.name else f"{base}.{alias.name}" for alias in node.names
            )
    return imports


def _edges() -> list[tuple[str, str, Path, str]]:
    edges: list[tuple[str, str, Path, str]] = []
    for path in sorted(Path(PROJECT).glob("**/*.py")):
        source = _component(_module_name(path))
        if source is None:
            continue
        for imported in sorted(_imports(path)):
            target = _component(imported)
            if target is not None:
                edges.append((source, target, path, imported))
    return edges


def test_package_dependencies_match_allowlist_and_topologically_sort() -> None:
    assert set(ALLOWED_EDGES) == COMPONENTS
    assert all(
        source in COMPONENTS and targets <= COMPONENTS for source, targets in ALLOWED_EDGES.items()
    )

    violations = [
        f"{path}: {source} -> {imported} ({target})"
        for source, target, path, imported in _edges()
        if source != target and target not in ALLOWED_EDGES[source]
    ]
    assert not violations, "Dependency allowlist violations:\n" + "\n".join(violations)

    remaining = {
        component: set(targets) - {component} for component, targets in ALLOWED_EDGES.items()
    }
    order: list[str] = []
    while remaining:
        ready = sorted(component for component, targets in remaining.items() if not targets)
        assert ready, f"Dependency allowlist contains a cycle: {remaining}"
        order.extend(ready)
        for component in ready:
            remaining.pop(component)
        for targets in remaining.values():
            targets.difference_update(ready)
    assert set(order) == COMPONENTS


def test_cli_commands_use_application_or_cli_surfaces() -> None:
    violations: list[str] = []
    for path in sorted(Path("virtual_staining/cli").glob("*.py")):
        if path.name in {"__init__.py", "_output.py", "_progress.py"}:
            continue
        for imported in _imports(path):
            if imported.startswith("virtual_staining.") and not imported.startswith(
                ("virtual_staining.applications", "virtual_staining.cli")
            ):
                violations.append(f"{path}: {imported}")
    assert not violations, "CLI command boundary violations:\n" + "\n".join(violations)
