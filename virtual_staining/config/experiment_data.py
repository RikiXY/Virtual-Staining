from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from virtual_staining.config.validation import reject_unknown_keys

Pairing = Literal["paired", "unpaired"]


@dataclass(frozen=True)
class ExperimentDataConfig:
    """Training data semantics; paired manifest data remains the default."""

    pairing: Pairing = "paired"
    domains: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pairing not in {"paired", "unpaired"}:
            raise ValueError("data.pairing must be 'paired' or 'unpaired'")
        if self.pairing == "paired" and self.domains:
            raise ValueError("data.domains is only used with data.pairing='unpaired'")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ExperimentDataConfig:
        reject_unknown_keys(data, frozenset({"pairing", "domains"}), "data")
        pairing = str(data.get("pairing", "paired"))
        raw_domains = data.get("domains", {})
        if not isinstance(raw_domains, dict):
            raise TypeError("data.domains must be a YAML mapping of domain name to directory")
        domains: dict[str, Path] = {}
        for name, path in raw_domains.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("data.domains keys must be non-empty strings")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"data.domains.{name} must be a non-empty path string")
            domains[name] = Path(path)
        return cls(pairing=cast(Pairing, pairing), domains=domains)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"pairing": self.pairing}
        if self.domains:
            result["domains"] = {name: str(path) for name, path in self.domains.items()}
        return result
