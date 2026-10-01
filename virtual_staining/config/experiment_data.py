from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from virtual_staining.config.validation import parse_choice, reject_unknown_keys

DataPairing = Literal["paired", "unpaired"]
# content: SHA-256 of every consumed file, verified stable while read (publication freeze).
# membership: locators, sizes, and semantic metadata only; explicitly unverified.
HashPolicy = Literal["content", "membership"]
# auto: strongest biological unit (patient > specimen > set) with complete metadata.
GroupValidation = Literal["auto", "patient", "specimen", "set", "unavailable"]
_DATA_KEYS = frozenset({"pairing", "domains", "hash_policy", "group_validation", "group_metadata"})


@dataclass(frozen=True)
class DataConfig:
    """Experiment data pairing, unpaired domain sources, and consumed-data provenance policy."""

    pairing: DataPairing = "paired"
    domains: dict[str, str] = field(default_factory=dict)
    hash_policy: HashPolicy = "content"
    group_validation: GroupValidation = "auto"
    # Unpaired only: CSV sidecar of path,domain,split,set_id,specimen_id,patient_id.
    group_metadata: Path | None = None

    def __post_init__(self) -> None:
        if self.pairing == "paired" and self.domains:
            raise ValueError("data.domains is supported only with data.pairing='unpaired'")
        if self.pairing == "unpaired" and not self.domains:
            raise ValueError("data.pairing='unpaired' requires data.domains")
        if self.pairing == "paired" and self.group_metadata is not None:
            raise ValueError(
                "data.group_metadata is supported only with data.pairing='unpaired'; paired "
                "group identities come from the prepared slide-set metadata"
            )

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
        hash_policy = parse_choice(
            data.get("hash_policy", "content"), "data.hash_policy", {"content", "membership"}
        )
        group_validation = parse_choice(
            data.get("group_validation", "auto"),
            "data.group_validation",
            {"auto", "patient", "specimen", "set", "unavailable"},
        )
        group_metadata = data.get("group_metadata")
        if group_metadata is not None and (
            not isinstance(group_metadata, str) or not group_metadata.strip()
        ):
            raise TypeError("data.group_metadata must be a non-empty path string")
        return cls(
            pairing=cast(DataPairing, pairing),
            domains=domains,
            hash_policy=cast(HashPolicy, hash_policy),
            group_validation=cast(GroupValidation, group_validation),
            group_metadata=Path(group_metadata) if group_metadata else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"pairing": self.pairing}
        if self.domains:
            data["domains"] = dict(self.domains)
        data["hash_policy"] = self.hash_policy
        data["group_validation"] = self.group_validation
        if self.group_metadata is not None:
            data["group_metadata"] = str(self.group_metadata)
        return data
