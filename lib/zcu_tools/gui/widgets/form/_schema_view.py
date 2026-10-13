"""Project the public partial-input schema into typed, active view rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import FormPreview, FormValue, FormWritePath

type _Kind = Literal[
    "text", "integer", "number", "complex", "boolean", "json", "model", "key"
]


@dataclass(frozen=True)
class FieldView:
    """One active schema row; no validation or writable model mirror.

    path is the canonical model-field path. kind selects a Qt representation.
    label, description, unit and placeholder are display text; collapsed is
    the initial model-group fold hint. nullable permits null; active identifies
    an enabled nullable model. expression permits scalar Expr input. options
    is current choice data, or None for free input (empty means no choices);
    for kind=key it holds suggested library keys, never a restriction, or
    None when the captured library offers none. decimals is the schema's
    display digit count for a number, or None to show the full value. fixed
    marks a one-value Literal (schema const, not nullable): its row shows the
    value as read-only text, so options is None and expression is False.
    raw is current JSON or a schema static default when present is False.
    value is the ordinary preview leaf when has_value is True; otherwise None
    is only a placeholder. key_or_inline marks a field the schema marks
    x-form-key-or-inline (an untagged str-or-model field): its row offers Key
    and Inline modes, kind is key for a string or unfilled value and model
    for an object. group is the schema's x-form-group name: the editor draws
    the row in that foldable group of its model level, or ungrouped for
    None. children are current model rows,
    never container keys/indexes. Only kind=model can have children.
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
    key_or_inline: bool
    decimals: int | None
    fixed: bool
    group: str | None
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
    # Nullable scalar fields wrap another scalar/Expr anyOf. Flatten only
    # alternatives, not model properties/container items; retain outer metadata.
    branches = _objects(schema.get("anyOf"))
    if not branches:
        return (schema,)
    metadata = {key: value for key, value in schema.items() if key != "anyOf"}
    return tuple(
        variant for branch in branches for variant in _variants({**metadata, **branch})
    )


def _direct(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    # Value constraints are inside the ordinary Expr alternative.
    return next(
        (
            node
            for node in _variants(schema)
            if "expr" not in _properties(node) and node.get("type") != "null"
        ),
        schema,
    )


def _options(schema: dict[str, JsonValue]) -> tuple[JsonValue, ...] | None:
    if schema.get("not") == {}:
        return ()
    values = schema.get("enum")
    if isinstance(values, list):
        return tuple(values)
    return (schema["const"],) if "const" in schema else None


def _key_and_inline(
    schema: dict[str, JsonValue], variants: tuple[dict[str, JsonValue], ...]
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]] | None:
    """Split a field the schema marks x-form-key-or-inline into key and inline.

    Core decides which fields are str-or-model; unmarked fields return None.
    variants are the field's flattened alternatives: the string node is the
    library key, the other non-null node the inline model. A marked field
    without exactly one of each raises ValueError.
    """
    if schema.get("x-form-key-or-inline") is not True:
        return None
    keys = [node for node in variants if node.get("type") == "string"]
    inline = [node for node in variants if node.get("type") not in ("string", "null")]
    if len(keys) != 1 or len(inline) != 1:
        raise ValueError(
            "x-form-key-or-inline schema requires one key and one inline alternative"
        )
    return keys[0], inline[0]


def _shown(node: dict[str, JsonValue]) -> bool:
    """Whether a property schema is drawn, i.e. not marked hidden.

    The schema marks Display(hidden=True) fields and one-value structural
    tags outside a union; their input stays in state and is still validated.
    """
    return node.get("hidden") is not True


def _decimals(metadata: dict[str, JsonValue]) -> int | None:
    value = metadata.get("decimals")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _selector_title(schema: dict[str, JsonValue], key: str) -> str:
    """Title of the tag field key in the first branch declaring one, else key.

    A placeholder selector (no branch selected yet) is then labelled like
    the selector of a selected branch.
    """
    for branch in _objects(schema.get("oneOf")):
        node = _properties(branch).get(key)
        if isinstance(node, dict) and (title := _text(node, "title")):
            return title
    return key


def _library_keys(key: dict[str, JsonValue]) -> tuple[JsonValue, ...] | None:
    keys = key.get("x-form-library-keys")
    return tuple(keys) if isinstance(keys, list) and keys else None


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
        tags = branch.get("x-form-discriminator-values")
        if not isinstance(tags, list):
            raise ValueError(
                "Tagged branch schema requires x-form-discriminator-values"
            )
        if raw.get(key) in tags:
            selected = branch
        selector = _properties(branch).get(key)
        if isinstance(selector, dict):
            allowed = _options(_direct(selector))
            options.extend(tag for tag in tags if allowed is None or tag in allowed)
    return selected, key, tuple(options)


def _field(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    data: dict[str, JsonValue],
    preview: FormPreview,
) -> FieldView:
    variants = _variants(schema)
    raw = data.get(path[-1])
    present = path[-1] in data
    split = _key_and_inline(schema, variants)
    if split is None:
        direct = _direct(schema)
        kind: _Kind = _kind(variants)
        options = _options(direct)
    else:
        key, inline = split
        # The JSON type is the structural selector: an object is inline.
        inline_mode = isinstance(raw if present else key.get("default"), dict)
        direct = inline if inline_mode else key
        kind = "model" if inline_mode else "key"
        options = _library_keys(key)
    metadata = {**direct, **schema}
    nullable = any(node.get("type") == "null" for node in variants)
    # One legal value leaves nothing to choose or compute.
    fixed = split is None and "const" in direct and not nullable
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
        expression=not fixed and any("expr" in _properties(node) for node in variants),
        options=None if fixed else options,
        raw=raw if present else direct.get("default"),
        present=present,
        value=preview.values.get(path),
        has_value=path in preview.values,
        key_or_inline=split is not None,
        decimals=_decimals(metadata),
        fixed=fixed,
        group=_text(schema, "x-form-group") or None,
    )
    if kind != "model" or not field.active:
        return field
    # Recursion through _children: each nested model level projects its rows.
    return replace(field, children=_children(direct, path, raw, preview))


def _children(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    raw: JsonValue,
    preview: FormPreview,
) -> tuple[FieldView, ...]:
    """Project the drawn property rows of the active model value raw at path.

    schema is the model's node. A tagged model shows its selected branch, its
    selector a plain tag dropdown of the branch choices; an unknown or missing
    tag shows only a placeholder selector, titled like the branch selectors.
    Properties the schema marks hidden have no rows.
    """
    nested = raw if isinstance(raw, dict) else {}
    selector = ""
    choices: tuple[JsonValue, ...] = ()
    if "discriminator" in schema:
        branch, selector, choices = _branch(schema, nested)
        properties = _properties(branch)
        # An unknown/missing tag exposes only its selector; do not guess a branch.
        if not properties:
            selector_schema: dict[str, JsonValue] = {
                "type": "string",
                "title": _selector_title(schema, selector),
            }
            properties = {selector: selector_schema}
    else:
        properties = _properties(schema)
    return tuple(
        replace(child, options=choices, expression=False, fixed=False)
        if child.path[-1] == selector
        else child
        for name, node in properties.items()
        if isinstance(node, dict) and _shown(node)
        for child in (_field(node, (*path, name), nested, preview),)
    )


def project_fields(
    schema: dict[str, JsonValue], data: dict[str, JsonValue], preview: FormPreview
) -> tuple[FieldView, ...]:
    """Project a public input schema and detached data/preview into active rows.

    Preserve declaration order and metadata. Inspect the discriminator only
    to choose its visible schema branch; unknown tags show a selector without
    guessing children. Fields the schema marks hidden have no rows; their
    input and errors stay with the state. Do not evaluate, validate, write
    or generate defaults.
    """
    return tuple(
        _field(node, (name,), data, preview)
        for name, node in _properties(schema).items()
        if isinstance(node, dict) and _shown(node)
    )


def flatten_fields(fields: tuple[FieldView, ...]) -> tuple[FieldView, ...]:
    """Return fields and all active children in declaration-order preorder."""
    return tuple(
        item for field in fields for item in (field, *flatten_fields(field.children))
    )
