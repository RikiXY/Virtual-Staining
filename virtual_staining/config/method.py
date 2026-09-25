from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_choice, reject_unknown_keys

MethodName = Literal["pix2pix", "cyclegan"]
_METHOD_KEYS = frozenset({"name", "replay_buffer_size"})
_BUILTIN_METHODS = {"pix2pix", "cyclegan"}
DEFAULT_CYCLEGAN_REPLAY_BUFFER_SIZE = 50


@dataclass(frozen=True)
class MethodConfig:
    """Select the built-in image-translation method for a run."""

    name: MethodName = "pix2pix"
    replay_buffer_size: int | None = None

    def __post_init__(self) -> None:
        if self.name == "cyclegan":
            if self.replay_buffer_size is None:
                raise ValueError("method.replay_buffer_size must be resolved for cyclegan")
            if self.replay_buffer_size < 0:
                raise ValueError("method.replay_buffer_size must be >= 0")
        elif self.replay_buffer_size is not None:
            raise ValueError(
                "method.replay_buffer_size is supported only for method.name='cyclegan'"
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MethodConfig:
        reject_unknown_keys(data, _METHOD_KEYS, "method")
        name = cast(
            MethodName,
            parse_choice(data.get("name", "pix2pix"), "method.name", _BUILTIN_METHODS),
        )
        replay_buffer_size = data.get("replay_buffer_size")
        if replay_buffer_size is not None and (
            isinstance(replay_buffer_size, bool) or not isinstance(replay_buffer_size, int)
        ):
            raise TypeError("method.replay_buffer_size must be an integer")
        if name == "cyclegan" and replay_buffer_size is None:
            replay_buffer_size = DEFAULT_CYCLEGAN_REPLAY_BUFFER_SIZE
        return cls(name=name, replay_buffer_size=replay_buffer_size)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name}
        if self.replay_buffer_size is not None:
            data["replay_buffer_size"] = self.replay_buffer_size
        return data
