from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_choice, reject_unknown_keys

DataPairing = Literal["paired", "unpaired"]
_DATA_KEYS = frozenset({"pairing", "domains"})


@dataclass(frozen=True)
class DataConfig:
    """Experiment data pairing and, for unpaired data, per-domain image sources."""

    pairing: DataPairing = "paired"
    domains: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pairing == "paired" and self.domains:
            raise ValueError("data.domains is supported only with data.pairing='unpaired'")
        if self.pairing == "unpaired" and not self.domains:
            raise ValueError("data.pairing='unpaired' requires data.domains")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DataConfig:
        reject_unknown_keys(data, _DATA_KEYS, "data")
        pairing = parse_choice(
            data.get("pairing", "paired"), "data.pairing", {"paired", "unpaired"}
        )
        raw_domains = data.get("domains", {})
        if not isinstance(raw_domains, dict):
            raise TypeError("data.domains must be a YAML mapping of domain name to path")
        domains: dict[str, str] = {}
        for name, spec in raw_domains.items():
            if not isinstance(spec, str) or not spec.strip():
                raise TypeError(f"data.domains.{name} must be a non-empty path or pattern string")
            domains[str(name)] = spec
        return cls(pairing=cast(DataPairing, pairing), domains=domains)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"pairing": self.pairing}
        if self.domains:
            data["domains"] = dict(self.domains)
        return data
