"""Autofluxdep run-time cfg override contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

OverrideMode: TypeAlias = Literal["after_first_point", "all_points", "fallback"]
_VALID_MODES: frozenset[str] = frozenset(
    {"after_first_point", "all_points", "fallback"}
)


@dataclass(frozen=True)
class OverridePath:
    """One Default cfg path that runtime may patch during a run."""

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
    """Return this point's cfg by applying declared runtime patches.

    The input base cfg is never mutated. Every patch path must be declared by the
    builder's run-start OverridePlan, must exist in the run-start base cfg, and
    must be legal for this flux index. Whole-module replacement is rejected, but
    a declared nested object may be replaced atomically. ``fallback`` paths are
    optional: when no patch is supplied, the run-start base cfg remains in effect.
    """

    by_path = plan.by_path
    unknown = set(patches) - set(by_path)
    if unknown:
        raise ValueError(
            f"node {node_name!r} produced undeclared override path(s): "
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
    found, node = _try_get_path(tree, path)
    if not found:
        return False
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
    if _is_module_root_path(path):
        raise ValueError(
            f"override path {path!r} for node {node_name!r} targets a module root; "
            "whole-module replacement is not allowed"
        )
    parts = path.split(".")
    node: object = tree
    parent: object | None = None
    parent_part = ""
    for part in parts[:-1]:
        parent = node
        parent_part = part
        node = _get_child(node, part, path=path, node_name=node_name)
    leaf = parts[-1]

    if isinstance(node, dict):
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
        return

    if isinstance(node, Mapping):
        raise ValueError(
            f"override path {path!r} for node {node_name!r} cannot assign "
            f"inside immutable mapping {type(node).__name__}"
        )
    found, _value = _try_get_object_child(node, leaf)
    if not found:
        raise ValueError(
            f"override path {path!r} for node {node_name!r} is absent "
            "from run-start base_cfg"
        )

    replacement = _copy_object_with_updated_leaf(
        node, leaf, value, path=path, node_name=node_name
    )
    if parent is None:
        raise ValueError(
            f"override path {path!r} for node {node_name!r} cannot replace "
            "the cfg root object"
        )
    _assign_child(
        parent,
        parent_part,
        replacement,
        path=path,
        node_name=node_name,
    )


def _get_child(
    node: object,
    part: str,
    *,
    path: str,
    node_name: str,
) -> object:
    if isinstance(node, Mapping):
        if part not in node:
            raise ValueError(
                f"override path {path!r} for node {node_name!r} is absent "
                "from run-start base_cfg"
            )
        return node[part]
    found, value = _try_get_object_child(node, part)
    if found:
        return value
    raise ValueError(
        f"override path {path!r} for node {node_name!r} cannot descend "
        f"through {type(node).__name__}"
    )


def _assign_child(
    parent: object,
    part: str,
    value: object,
    *,
    path: str,
    node_name: str,
) -> None:
    if isinstance(parent, dict):
        parent[part] = value
        return
    if hasattr(parent, part):
        try:
            setattr(parent, part, value)
        except Exception as exc:
            raise ValueError(
                f"override path {path!r} for node {node_name!r} cannot replace "
                f"{part!r} on {type(parent).__name__}"
            ) from exc
        return
    raise ValueError(
        f"override path {path!r} for node {node_name!r} cannot replace "
        f"{part!r} on {type(parent).__name__}"
    )


def _copy_object_with_updated_leaf(
    obj: object,
    leaf: str,
    value: object,
    *,
    path: str,
    node_name: str,
) -> object:
    if _is_sweep_cfg_like(obj) and leaf in {"start", "stop", "expts", "step"}:
        return _copy_sweep_cfg_with_updated_leaf(obj, leaf, value)

    clone = deepcopy(obj)
    try:
        setattr(clone, leaf, value)
    except Exception as exc:
        raise ValueError(
            f"override path {path!r} for node {node_name!r} cannot assign "
            f"{leaf!r} inside {type(obj).__name__}"
        ) from exc
    return clone


def _is_sweep_cfg_like(obj: object) -> bool:
    return all(
        hasattr(obj, field_name) for field_name in ("start", "stop", "expts", "step")
    )


def _copy_sweep_cfg_with_updated_leaf(
    sweep: object,
    leaf: str,
    value: object,
) -> object:
    start = float(getattr(sweep, "start"))
    stop = float(getattr(sweep, "stop"))
    expts = int(getattr(sweep, "expts"))
    step = float(getattr(sweep, "step"))

    if leaf == "start":
        start = float(cast(Any, value))
    elif leaf == "stop":
        stop = float(cast(Any, value))
    elif leaf == "expts":
        expts = _coerce_sweep_expts(value)
    elif leaf == "step":
        step = float(cast(Any, value))
        stop = start + step * (expts - 1)

    if leaf != "step":
        step = 0.0 if expts == 1 else (stop - start) / (expts - 1)

    sweep_type = cast(Any, type(sweep))
    return sweep_type(start=start, stop=stop, expts=expts, step=step)


def _coerce_sweep_expts(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("SweepCfg expts patch must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("SweepCfg expts patch must be an integer")


def _try_get_path(tree: object, path: str) -> tuple[bool, object]:
    node: object = tree
    for part in path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return False, None
            node = node[part]
            continue
        found, value = _try_get_object_child(node, part)
        if not found:
            return False, None
        node = value
    return True, node


def _try_get_object_child(node: object, part: str) -> tuple[bool, object]:
    model_fields = _model_field_names(node)
    if model_fields is not None:
        if part not in model_fields:
            return False, None
        return True, getattr(node, part)
    if not hasattr(node, part):
        return False, None
    return True, getattr(node, part)


def _model_field_names(node: object) -> frozenset[str] | None:
    fields = getattr(type(node), "model_fields", None)
    if not isinstance(fields, Mapping):
        return None
    return frozenset(str(key) for key in fields)


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
