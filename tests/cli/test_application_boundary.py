from __future__ import annotations

import ast
from pathlib import Path


def _project_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return {name for name in imports if name.startswith("virtual_staining.")}


def test_cli_commands_depend_on_applications_and_cli_presentation_only() -> None:
    command_files = (
        path
        for path in Path("virtual_staining/cli").glob("*.py")
        if path.name not in {"__init__.py", "_output.py"}
    )
    violations = {
        str(path): sorted(
            name
            for name in _project_imports(path)
            if not name.startswith(("virtual_staining.applications", "virtual_staining.cli"))
        )
        for path in command_files
    }
    assert not {path: names for path, names in violations.items() if names}


def test_lower_layers_do_not_depend_on_applications_or_cli() -> None:
    lower_layers = ("config", "data", "evaluation", "experiment", "inference", "models", "training")
    violations: dict[str, list[str]] = {}
    for package in lower_layers:
        for path in Path("virtual_staining", package).glob("**/*.py"):
            forbidden = sorted(
                name
                for name in _project_imports(path)
                if name.startswith(("virtual_staining.applications", "virtual_staining.cli"))
            )
            if forbidden:
                violations[str(path)] = forbidden
    assert not violations
