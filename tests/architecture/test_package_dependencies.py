from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

PROJECT = "virtual_staining"
APPLICATION_SURFACES = {"cli", "applications"}
FOUNDATIONAL_COMPONENTS = {"utils", "metrics", "split_contract"}


def _module_name(path: Path) -> str:
    relative = path.relative_to(Path(PROJECT)).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PROJECT, *parts))


def _component(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != PROJECT:
        return None
    return parts[1]


def _imports(path: Path) -> set[str]:
    module_name = _module_name(path)
    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module or ""
            )
            imports.update(
                base if not alias.name else f"{base}.{alias.name}" for alias in node.names
            )
    return imports


def _internal_edges() -> list[tuple[str, str, Path, str]]:
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


def test_lower_layers_do_not_depend_on_application_surfaces() -> None:
    violations = [
        f"{path}: {source} -> {imported}"
        for source, target, path, imported in _internal_edges()
        if source not in APPLICATION_SURFACES and target in APPLICATION_SURFACES
    ]
    assert not violations, "Lower-layer imports of application surfaces:\n" + "\n".join(violations)


def test_foundational_components_remain_leaf_dependencies() -> None:
    violations = [
        f"{path}: {source} -> {imported}"
        for source, target, path, imported in _internal_edges()
        if source in FOUNDATIONAL_COMPONENTS and target != source
    ]
    assert not violations, "Foundational dependency violations:\n" + "\n".join(violations)


def test_config_does_not_depend_on_runtime_domains() -> None:
    forbidden = {
        "data",
        "experiment",
        "models",
        "training",
        "inference",
        "evaluation",
        *APPLICATION_SURFACES,
    }
    violations = [
        f"{path}: config -> {imported}"
        for source, target, path, imported in _internal_edges()
        if source == "config" and target in forbidden
    ]
    assert not violations, "Config dependency violations:\n" + "\n".join(violations)


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


def test_alignment_dependency_boundary() -> None:
    alignment = "virtual_staining.data.alignment"
    allowed = {
        "models.py": (f"{alignment}.models.",),
        "warping.py": (f"{alignment}.models.",),
        "registration.py": (
            "virtual_staining.config.",
            f"{alignment}.models.",
            f"{alignment}.warping.",
        ),
        "__init__.py": (f"{alignment}.",),
    }
    violations = []
    for path in Path("virtual_staining/data/alignment").rglob("*.py"):
        for imported in _imports(path):
            if imported.startswith("virtual_staining.") and not imported.startswith(
                allowed[path.name]
            ):
                violations.append(f"{path}: {imported}")
    for module in ("preprocessing", "slide_set_processor"):
        path = Path(f"virtual_staining/data/{module}.py")
        for imported in _imports(path):
            if imported.startswith(f"{alignment}.") and (
                module == "preprocessing" or imported.removeprefix(f"{alignment}.").count(".")
            ):
                violations.append(f"{path}: use only the public alignment API: {imported}")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in {
                "SIFT_create",
                "BFMatcher",
                "estimateAffinePartial2D",
                "warpAffine",
                "invertAffineTransform",
            }:
                violations.append(f"{path}: alignment implementation: {node.attr}")
    assert not violations, "Alignment boundary violations:\n" + "\n".join(violations)
