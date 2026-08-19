from typing import TYPE_CHECKING, Any

from virtual_staining.inference.config import InferenceConfig

if TYPE_CHECKING:
    from virtual_staining.inference.runner import InferenceResult

__all__ = [
    "InferenceConfig",
    "InferenceResult",
]


def __getattr__(name: str) -> Any:
    if name == "InferenceResult":
        from virtual_staining.inference.runner import InferenceResult

        return InferenceResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
