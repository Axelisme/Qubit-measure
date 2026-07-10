"""Path operations for the shared configuration Spec/Value trees."""

from __future__ import annotations

from .model import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    ReferenceSpec,
    ReferenceValue,
)


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise RuntimeError("Cfg tree path must not be empty")
    return parts


def _resolve_spec_matches(
    root: CfgSectionSpec,
    parts: tuple[str, ...],
    path: str,
) -> list[CfgNodeSpec]:
    node: CfgNodeSpec = root
    for index, part in enumerate(parts):
        if isinstance(node, CfgSectionSpec):
            child = node.fields.get(part)
            if child is None:
                raise KeyError(
                    f"Cfg spec path {path!r} segment {part!r} not found; "
                    f"available: {', '.join(node.fields)}"
                )
            node = child
            continue
        if isinstance(node, ReferenceSpec):
            remaining = parts[index:]
            matches: list[CfgNodeSpec] = []
            for allowed in node.allowed:
                try:
                    matches.extend(_resolve_spec_matches(allowed, remaining, path))
                except (KeyError, RuntimeError):
                    continue
            if not matches:
                allowed_labels = ", ".join(shape.label for shape in node.allowed)
                raise KeyError(
                    f"Cfg ref path {'.'.join(remaining)!r} not found in any "
                    f"allowed shape (allowed: {allowed_labels})"
                )
            return matches
        raise RuntimeError(
            f"Cfg spec path {path!r} cannot descend into {type(node).__name__} "
            f"at {part!r}"
        )
    return [node]


def resolve_spec_path(root: CfgSectionSpec, path: str) -> CfgNodeSpec:
    """Resolve a dotted path through sections and reference allowed shapes."""
    matches = _resolve_spec_matches(root, _split_path(path), path)
    first = matches[0]
    first_type = _spec_type_signature(first)
    if not all(_spec_type_signature(match) == first_type for match in matches):
        raise TypeError(
            f"Cfg ref path {path!r} resolves to inconsistent spec types: "
            + ", ".join(_format_spec_type(match) for match in matches)
        )
    return first


def _spec_type_signature(spec: CfgNodeSpec) -> tuple[type, type | str | None]:
    if isinstance(spec, ReferenceSpec):
        return (type(spec), spec.kind)
    type_ = getattr(spec, "type", None)
    return (type(spec), type_ if isinstance(type_, type) else None)


def _format_spec_type(spec: CfgNodeSpec) -> str:
    signature = _spec_type_signature(spec)[1]
    if isinstance(signature, type):
        return f"{type(spec).__name__}[{signature.__name__}]"
    if isinstance(signature, str):
        return f"{type(spec).__name__}[{signature}]"
    return type(spec).__name__


def _resolve_value_parent(
    root: CfgSectionValue,
    path: str,
) -> tuple[CfgSectionValue, str]:
    parts = _split_path(path)
    section = root
    for part in parts[:-1]:
        child = section.fields.get(part)
        if isinstance(child, ReferenceValue):
            section = child.value
        elif isinstance(child, CfgSectionValue):
            section = child
        else:
            raise RuntimeError(
                f"Cfg value path {path!r} cannot descend into "
                f"{type(child).__name__} at {part!r}"
            )
    return section, parts[-1]


def read_value_path(
    root: CfgSectionValue,
    path: str,
) -> CfgNodeValue | None:
    """Read an existing leaf from a dotted Value-tree path."""
    section, leaf = _resolve_value_parent(root, path)
    if leaf not in section.fields:
        raise KeyError(f"Cfg value path {path!r} leaf {leaf!r} not found")
    return section.fields[leaf]


def replace_value_path(
    root: CfgSectionValue,
    path: str,
    value: CfgNodeValue | None,
) -> None:
    """Replace an existing leaf at a dotted Value-tree path in place."""
    section, leaf = _resolve_value_parent(root, path)
    if leaf not in section.fields:
        raise KeyError(f"Cfg value path {path!r} leaf {leaf!r} not found")
    section.fields[leaf] = value


__all__ = ["read_value_path", "replace_value_path", "resolve_spec_path"]
