"""Wire projection and prefix queries for canonical cfg binding targets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from zcu_tools.gui.cfg import DirectValue, EvalValue
from zcu_tools.gui.cfg.binding import CfgDraft, SettableTarget, SettableTargetKind


def project_target_entries(draft: CfgDraft) -> list[dict[str, object]]:
    """Project live binding targets into the existing flat wire entry shape."""
    return project_targets(draft.iter_settable_targets())


def project_targets(targets: Iterable[SettableTarget]) -> list[dict[str, object]]:
    """Project an already-selected target snapshot without walking a draft."""
    return [_target_entry(target) for target in targets]


def _target_entry(target: SettableTarget) -> dict[str, object]:
    kind = (
        "moduleref_key"
        if target.kind is SettableTargetKind.REFERENCE_KEY
        else target.kind.value
    )
    entry: dict[str, object] = {
        "path": target.path,
        "kind": kind,
        "value": _wire_value(target.get_value()),
        "type": _wire_type(target.value_type),
    }
    choices = target.choices()
    if choices is not None:
        entry["choices"] = list(choices)
    return entry


def _wire_value(value: object) -> object:
    if isinstance(value, EvalValue):
        return value.expr
    if isinstance(value, DirectValue):
        return value.value
    return value


def _wire_type(value_type: type) -> str:
    if value_type is int:
        return "integer"
    if value_type is float:
        return "number"
    if value_type is str:
        return "string"
    if value_type is bool:
        return "bool"
    return value_type.__name__


def build_settable_tree(
    draft: CfgDraft, prefix: str | None = None
) -> dict[str, object]:
    """Build the existing nested wire view from nominal binding targets.

    ``prefix`` is a read projection policy only. Unknown prefixes return an
    empty object; they never become setter aliases.
    """
    targets = tuple(draft.iter_settable_targets())
    tree = _build_tree(targets)
    if not prefix:
        return tree

    target = next((item for item in targets if item.path == prefix), None)
    if target is not None:
        if target.kind is SettableTargetKind.REFERENCE_KEY:
            return _dict_at(tree, prefix.rsplit(".", 1)[0])
        if target.kind is SettableTargetKind.SWEEP_EDGE:
            return _dict_at(tree, prefix.rsplit(".", 1)[0])
        node = _node_at(tree, prefix)
        if node is _MISSING:
            return {}
        return {prefix.rsplit(".", 1)[-1]: node}

    if not any(item.path.startswith(prefix + ".") for item in targets):
        return {}
    return _dict_at(tree, prefix)


def _build_tree(targets: Iterable[SettableTarget]) -> dict[str, object]:
    tree: dict[str, object] = {}
    for target in targets:
        if target.kind is SettableTargetKind.REFERENCE_KEY:
            ref_path = target.path.rsplit(".", 1)[0]
            node = _ensure_dict(tree, ref_path.split("."))
            node["$ref"] = {
                "current": _wire_value(target.get_value()),
                "options": list(target.choices() or ()),
            }
            continue
        parts = target.path.split(".")
        parent = _ensure_dict(tree, parts[:-1])
        value = _wire_value(target.get_value())
        choices = target.choices()
        parent[parts[-1]] = (
            {"$value": value, "$choices": list(choices)}
            if choices is not None
            else value
        )
    return tree


def _ensure_dict(root: dict[str, object], parts: list[str]) -> dict[str, object]:
    node = root
    for part in parts:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise RuntimeError(f"wire projection collision at {part!r}")
        node = cast("dict[str, object]", child)
    return node


_MISSING = object()


def _node_at(root: dict[str, object], path: str) -> object:
    node: object = root
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _dict_at(root: dict[str, object], path: str) -> dict[str, object]:
    node = _node_at(root, path)
    if not isinstance(node, dict):
        return {}
    return cast("dict[str, object]", node)


__all__ = ["build_settable_tree", "project_target_entries", "project_targets"]
