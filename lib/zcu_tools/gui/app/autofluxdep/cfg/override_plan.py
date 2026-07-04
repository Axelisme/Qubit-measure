"""Autofluxdep run-time cfg override contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

OverrideMode: TypeAlias = Literal["after_first_point", "all_points"]
_VALID_MODES: frozenset[str] = frozenset({"after_first_point", "all_points"})


@dataclass(frozen=True)
class OverridePath:
    """One Default cfg path that generation may patch during a run."""

    path: str
    mode: OverrideMode
    source: str
    reason: str

    def __post_init__(self) -> None:
        _validate_path(self.path)
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unsupported override mode {self.mode!r}")
        if not self.source.strip():
            raise ValueError("override source must be non-empty")
        if not self.reason.strip():
            raise ValueError("override reason must be non-empty")

    def to_wire(self) -> dict[str, str]:
        return {
            "path": self.path,
            "mode": self.mode,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OverridePlan:
    """Declared set of Default cfg paths a node may override at run time."""

    paths: tuple[OverridePath, ...] = ()

    def __post_init__(self) -> None:
        paths = tuple(self.paths)
        object.__setattr__(self, "paths", paths)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for entry in paths:
            if entry.path in seen:
                duplicates.add(entry.path)
            seen.add(entry.path)
        if duplicates:
            raise ValueError(
                "duplicate override paths: " + ", ".join(sorted(duplicates))
            )

    def to_wire(self) -> list[dict[str, str]]:
        return [entry.to_wire() for entry in self.paths]


def override_plan_to_wire(plan: OverridePlan) -> list[dict[str, str]]:
    """Return the JSON-safe wire/artifact representation of an override plan."""
    return plan.to_wire()


def validate_override_plan_base_cfg(
    plan: OverridePlan,
    base_cfg: Mapping[str, object],
    *,
    node_name: str,
) -> None:
    """Fast-fail a run whose declared override path is absent from base cfg."""
    for entry in plan.paths:
        if not _path_exists(base_cfg, entry.path):
            raise ValueError(
                f"override path {entry.path!r} for node {node_name!r} "
                "is absent from run-start base_cfg"
            )


def _validate_path(path: str) -> None:
    if path != path.strip():
        raise ValueError(
            f"override path must not have surrounding whitespace: {path!r}"
        )
    if not path:
        raise ValueError("override path must be non-empty")
    parts = path.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"override path has an empty segment: {path!r}")
    if parts[0] == "generation":
        raise ValueError("override path must target Default cfg, not generation")


def _path_exists(tree: Mapping[str, object], path: str) -> bool:
    node: object = tree
    for part in path.split("."):
        if not isinstance(node, Mapping):
            return False
        if part not in node:
            return False
        node = node[part]
    if isinstance(node, Mapping):
        return False
    return True


__all__ = [
    "OverrideMode",
    "OverridePath",
    "OverridePlan",
    "override_plan_to_wire",
    "validate_override_plan_base_cfg",
]
