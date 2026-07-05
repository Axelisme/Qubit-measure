"""Autofluxdep run-time cfg override contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

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
    _by_path: Mapping[str, OverridePath] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        paths = tuple(self.paths)
        object.__setattr__(self, "paths", paths)
        by_path: dict[str, OverridePath] = {}
        duplicates: set[str] = set()
        for entry in paths:
            if entry.path in by_path:
                duplicates.add(entry.path)
            by_path[entry.path] = entry
        if duplicates:
            raise ValueError(
                "duplicate override paths: " + ", ".join(sorted(duplicates))
            )
        object.__setattr__(self, "_by_path", MappingProxyType(by_path))

    def to_wire(self) -> list[dict[str, str]]:
        return [entry.to_wire() for entry in self.paths]

    @property
    def by_path(self) -> Mapping[str, OverridePath]:
        return self._by_path


@dataclass(frozen=True)
class RunCfgSnapshot:
    """Run-start cfg truth for one enabled node."""

    base_cfg: Mapping[str, object]
    override_plan: OverridePlan
    knobs: Mapping[str, Any]

    def to_wire(self) -> dict[str, object]:
        return {
            "base_cfg": deepcopy(dict(self.base_cfg)),
            "override_plan": override_plan_to_wire(self.override_plan),
        }


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
        if _is_module_root_path(entry.path):
            raise ValueError(
                f"override path {entry.path!r} for node {node_name!r} targets a "
                "module root; whole-module replacement is not allowed"
            )
        if not _path_exists(base_cfg, entry.path):
            raise ValueError(
                f"override path {entry.path!r} for node {node_name!r} "
                "is absent from run-start base_cfg"
            )


def apply_override_patches(
    base_cfg: Mapping[str, object],
    plan: OverridePlan,
    patches: Mapping[str, object],
    *,
    flux_idx: int,
    node_name: str,
) -> dict[str, object]:
    """Return this point's cfg by applying declared generation patches.

    The input base cfg is never mutated. Every patch path must be declared by the
    builder's run-start OverridePlan, must exist in the run-start base cfg, and
    must be legal for this flux index. Whole-module replacement is rejected, but
    a declared nested object such as ``modules.readout.pulse_cfg.waveform`` may
    be replaced atomically.
    """

    by_path = plan.by_path
    unknown = set(patches) - set(by_path)
    if unknown:
        raise ValueError(
            f"node {node_name!r} generated undeclared override path(s): "
            + ", ".join(sorted(unknown))
        )
    required = {
        entry.path
        for entry in plan.paths
        if entry.mode == "all_points"
        or (entry.mode == "after_first_point" and flux_idx > 0)
    }
    missing = required - set(patches)
    if missing:
        raise ValueError(
            f"node {node_name!r} missed generated override path(s): "
            + ", ".join(sorted(missing))
        )

    point_cfg = deepcopy(dict(base_cfg))
    for path, value in patches.items():
        entry = by_path[path]
        if entry.mode == "after_first_point" and flux_idx == 0:
            raise ValueError(
                f"node {node_name!r} generated initial-only path {path!r} "
                "at flux index 0"
            )
        _set_leaf(point_cfg, path, deepcopy(value), node_name=node_name)
    return point_cfg


def module_override_paths(
    *,
    prefix: str,
    leaf_paths: tuple[str, ...],
    source: str,
    reason: str,
    mode: OverrideMode = "all_points",
) -> tuple[OverridePath, ...]:
    """Declare one module dependency as leaf-level override paths."""
    return tuple(
        OverridePath(f"{prefix}.{leaf_path}", mode, source, reason)
        for leaf_path in leaf_paths
    )


def module_leaf_patches(
    *,
    prefix: str,
    module: object,
    leaf_paths: tuple[str, ...],
) -> dict[str, object]:
    """Extract declared patches from a module-shaped object.

    Missing paths are skipped so raw dict fixtures and pydantic module objects can
    share one path list; ``apply_override_patches`` later enforces that declared
    required paths were actually produced for this point.
    """

    patches: dict[str, object] = {}
    for leaf_path in leaf_paths:
        found, value = _try_get_path(module, leaf_path)
        if found:
            patches[f"{prefix}.{leaf_path}"] = value
    return patches


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
    if isinstance(node, Mapping) and _is_module_root_path(path):
        return False
    return True


def _set_leaf(
    tree: dict[str, object],
    path: str,
    value: object,
    *,
    node_name: str,
) -> None:
    parts = path.split(".")
    node: object = tree
    for part in parts[:-1]:
        if not isinstance(node, dict):
            raise ValueError(
                f"override path {path!r} for node {node_name!r} cannot descend "
                f"through {type(node).__name__}"
            )
        if part not in node:
            raise ValueError(
                f"override path {path!r} for node {node_name!r} is absent "
                "from run-start base_cfg"
            )
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(
            f"override path {path!r} for node {node_name!r} cannot assign "
            f"inside {type(node).__name__}"
        )
    leaf = parts[-1]
    if leaf not in node:
        raise ValueError(
            f"override path {path!r} for node {node_name!r} is absent "
            "from run-start base_cfg"
        )
    if isinstance(node[leaf], Mapping) and _is_module_root_path(path):
        raise ValueError(
            f"override path {path!r} for node {node_name!r} targets a mapping; "
            "whole-module replacement is not allowed"
        )
    node[leaf] = value


def _try_get_path(tree: object, path: str) -> tuple[bool, object]:
    node: object = tree
    for part in path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return False, None
            node = node[part]
            continue
        if not hasattr(node, part):
            return False, None
        node = getattr(node, part)
    return True, node


def _is_module_root_path(path: str) -> bool:
    parts = path.split(".")
    return path == "modules" or (len(parts) == 2 and parts[0] == "modules")


__all__ = [
    "OverrideMode",
    "OverridePath",
    "OverridePlan",
    "RunCfgSnapshot",
    "apply_override_patches",
    "module_leaf_patches",
    "module_override_paths",
    "override_plan_to_wire",
    "validate_override_plan_base_cfg",
]
