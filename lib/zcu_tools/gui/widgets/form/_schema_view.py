"""Project the public partial-input schema into typed, active view rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import FormPreview, FormValue, FormWritePath

type _Kind = Literal["text", "integer", "number", "complex", "boolean", "json", "model"]


@dataclass(frozen=True)
class FieldView:
    """One active schema row; no validation or writable model mirror.

    path is the canonical model-field path. kind selects a Qt representation.
    label, description, unit and placeholder are display text; collapsed is
    the initial model-group fold hint. nullable permits null; active identifies
    an enabled nullable model. expression permits numeric Expr input. options
    is current choice data, or None for free input (empty means no choices).
    raw is current JSON or a schema static default when present is False.
    value is the ordinary preview leaf when has_value is True; otherwise None
    is only a placeholder. children are current model rows, never container
    keys/indexes. Only kind=model can have children.
    """

    path: FormWritePath
    kind: _Kind
    label: str
    description: str
    unit: str
    placeholder: str
    collapsed: bool
    nullable: bool
    active: bool
    expression: bool
    options: tuple[JsonValue, ...] | None
    raw: JsonValue
    present: bool
    value: FormValue
    has_value: bool
    children: tuple[FieldView, ...] = ()


def _objects(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    return (
        tuple(item for item in value if isinstance(item, dict))
        if isinstance(value, list)
        else ()
    )


def _properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _text(schema: dict[str, JsonValue], key: str) -> str:
    value = schema.get(key)
    return value if isinstance(value, str) else ""


def _variants(schema: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return _objects(schema.get("anyOf")) or (schema,)


def _kind(variants: tuple[dict[str, JsonValue], ...]) -> _Kind:
    if any("__complex__" in _properties(node) for node in variants):
        return "complex"
    for node in variants:
        if "expr" in _properties(node) or node.get("type") == "null":
            continue
        if "properties" in node or "discriminator" in node:
            return "model"
        kind: _Kind | None = None
        match node.get("type"):
            case "integer":
                kind = "integer"
            case "number":
                kind = "number"
            case "boolean":
                kind = "boolean"
            case "array" | "object":
                kind = "json"
            case _:
                pass
        if kind is not None:
            return kind
    return "text"


def _branch(
    schema: dict[str, JsonValue], raw: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], str, tuple[JsonValue, ...]]:
    discriminator = schema.get("discriminator")
    key = (
        _text(discriminator, "propertyName") if isinstance(discriminator, dict) else ""
    )
    options: list[JsonValue] = []
    selected: dict[str, JsonValue] = {}
    for branch in _objects(schema.get("oneOf")):
        selector = _properties(branch).get(key)
        if isinstance(selector, dict) and "const" in selector:
            tag = selector["const"]
            options.append(tag)
            if raw.get(key) == tag:
                selected = branch
    return selected, key, tuple(options)


def _field(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    data: dict[str, JsonValue],
    preview: FormPreview,
) -> FieldView:
    variants = _variants(schema)
    direct = next(
        (
            node
            for node in variants
            if "expr" not in _properties(node) and node.get("type") != "null"
        ),
        schema,
    )
    metadata = {**direct, **schema}
    raw = data.get(path[-1])
    kind = _kind(variants)
    options = direct.get("enum", schema.get("enum"))
    if options is None and "const" in direct:
        options = [direct["const"]]
    nullable = any(node.get("type") == "null" for node in variants)
    present = path[-1] in data
    field = FieldView(
        path=path,
        kind=kind,
        label=_text(metadata, "title") or path[-1],
        description=_text(metadata, "description"),
        unit=_text(metadata, "unit"),
        placeholder=_text(metadata, "placeholder"),
        collapsed=metadata.get("collapsed") is True,
        nullable=nullable,
        active=not nullable or (present and raw is not None),
        expression=any("expr" in _properties(node) for node in variants),
        options=tuple(options) if isinstance(options, list) else None,
        raw=raw if present else direct.get("default"),
        present=present,
        value=preview.values.get(path),
        has_value=path in preview.values,
    )
    if kind != "model" or not field.active:
        return field
    nested = raw if isinstance(raw, dict) else {}
    selector = ""
    choices: tuple[JsonValue, ...] = ()
    if "discriminator" in direct:
        branch, selector, choices = _branch(direct, nested)
        properties = _properties(branch)
        # An unknown/missing tag exposes only its selector; do not guess a branch.
        if not properties:
            selector_schema: dict[str, JsonValue] = {"type": "string"}
            properties = {selector: selector_schema}
    else:
        properties = _properties(direct)
    children = tuple(
        replace(child, options=choices) if child.path[-1] == selector else child
        for name, node in properties.items()
        if isinstance(node, dict)
        for child in (_field(node, (*path, name), nested, preview),)
    )
    return replace(field, children=children)


def project_fields(
    schema: dict[str, JsonValue], data: dict[str, JsonValue], preview: FormPreview
) -> tuple[FieldView, ...]:
    """Project a public input schema and detached data/preview into active rows.

    Preserve declaration order and metadata. Inspect the discriminator only
    to choose its visible schema branch; unknown tags show a selector without
    guessing children. Do not evaluate, validate, write or generate defaults.
    """
    return tuple(
        _field(node, (name,), data, preview)
        for name, node in _properties(schema).items()
        if isinstance(node, dict)
    )


def flatten_fields(fields: tuple[FieldView, ...]) -> tuple[FieldView, ...]:
    """Return fields and all active children in declaration-order preorder."""
    return tuple(
        item for field in fields for item in (field, *flatten_fields(field.children))
    )
