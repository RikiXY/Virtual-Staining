from typing import TYPE_CHECKING, Any

from virtual_staining.config.loader import load_yaml_mapping
from virtual_staining.config.project import ProjectConfig
from virtual_staining.config.validation import (
    parse_bool_strict,
    reject_unknown_keys,
)

if TYPE_CHECKING:
    from virtual_staining.config.run import RunConfig

__all__ = [
    "load_yaml_mapping",
    "parse_bool_strict",
    "ProjectConfig",
    "RunConfig",
    "reject_unknown_keys",
]


def __getattr__(name: str) -> Any:
    if name == "RunConfig":
        from virtual_staining.config.run import RunConfig

        return RunConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
