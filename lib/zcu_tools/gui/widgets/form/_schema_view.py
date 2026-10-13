"""Project the public partial-input schema into typed, active view rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import FormPreview, FormValue, FormWritePath, Option

type _Kind = Literal[
    "text", "integer", "number", "complex", "boolean", "json", "model", "choice"
]


@dataclass(frozen=True)
class CommandView:
    """One explicit action an option lists in the schema.

    name identifies it for FormState.run_command(path, name, env); label is
    its menu text. Both are nonempty strings.
    """

    name: str
    label: str


@dataclass(frozen=True)
class OptionView:
    """One entry of a field's schema x-form-options list, in schema order.

    label is the display text. value is the JSON partial input written at
    the field when this option is chosen. fields is the embedded partial
    object schema of the option's editable children (without the tags its
    value fixes), or None for an option without children. commands are the
    explicit actions available while this option is selected, possibly none.
    """

    label: str
    value: JsonValue
    fields: dict[str, JsonValue] | None
    commands: tuple[CommandView, ...]

    def matches(self, current: JsonValue) -> bool:
        """Whether current, the field's JSON input, selects this option.

        The core's Option rule decides: every leaf of value is in current.
        """
        return Option(self.label, self.value).matches(current)


@dataclass(frozen=True)
class FieldView:
    """One active schema row; no validation or writable model mirror.

    path is the canonical model-field path. kind selects a Qt representation:
    choice is an option dropdown without children, model a row with child
    rows (a nested model, or an option dropdown whose selected option has
    fields). label, description, unit and placeholder are display text;
    collapsed is the initial model-group fold hint. nullable permits null;
    active identifies an enabled nullable model. expression permits scalar
    Expr input: the schema offers an Expr alternative, which the core does
    only for text fields; an option row never has it. options is the
    schema's x-form-options list, or None for a row without one (free
    input, a bool, an ordinary model). selected is the index in options of
    the first option matching the input, or None when none matches (the
    input is unfilled, or it is shown as missing when present). decimals is
    the schema's display digit count for a number, or None to show the full
    value. fixed marks a one-value Literal (schema const, not nullable): its
    row shows the value as read-only text, so options is None and
    expression is False. raw is current JSON or a schema static default
    when present is False. value is the ordinary preview leaf when
    has_value is True; otherwise None is only a placeholder. group is the
    schema's x-form-group name: the editor draws the row in that foldable
    group of its model level, or ungrouped for None. children are current
    model rows, never container keys/indexes; an option row's children come
    from its selected option's fields. Only kind=model can have children.
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
    options: tuple[OptionView, ...] | None
    selected: int | None
    raw: JsonValue
    present: bool
    value: FormValue
    has_value: bool
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


def _string(node: dict[str, JsonValue], key: str) -> str:
    value = node.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"x-form-options entry needs a nonempty {key!r} string")
    return value


def _option_views(schema: dict[str, JsonValue]) -> tuple[OptionView, ...] | None:
    """Read the field's x-form-options list, or None when the schema has none.

    An entry without a label, value, object fields or {name, label} commands
    raises ValueError: the core owns this list and the view does not repair it.
    """
    entries = schema.get("x-form-options")
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("x-form-options must be a list")
    options: list[OptionView] = []
    for entry in entries:
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError("x-form-options entry needs a label and a value")
        fields = entry.get("fields")
        if fields is not None and not isinstance(fields, dict):
            raise ValueError("x-form-options fields must be an object schema")
        listed = entry.get("commands", [])
        if not isinstance(listed, list):
            raise ValueError("x-form-options commands must be a list")
        commands: list[CommandView] = []
        for item in listed:
            if not isinstance(item, dict):
                raise ValueError("x-form-options command must be an object")
            commands.append(CommandView(_string(item, "name"), _string(item, "label")))
        options.append(
            OptionView(_string(entry, "label"), entry["value"], fields, tuple(commands))
        )
    return tuple(options)


def _shown(node: dict[str, JsonValue]) -> bool:
    """Whether a property schema is drawn, i.e. not marked hidden.

    The schema marks Display(hidden=True) fields and one-value structural
    tags outside a union; their input stays in state and is still validated.
    """
    return node.get("hidden") is not True


def _decimals(metadata: dict[str, JsonValue]) -> int | None:
    value = metadata.get("decimals")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _kind(variants: tuple[dict[str, JsonValue], ...]) -> _Kind:
    if any("__complex__" in _properties(node) for node in variants):
        return "complex"
    for node in variants:
        if "expr" in _properties(node) or node.get("type") == "null":
            continue
        if "properties" in node:
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


def _field(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    data: dict[str, JsonValue],
    preview: FormPreview,
) -> FieldView:
    variants = _variants(schema)
    raw = data.get(path[-1])
    present = path[-1] in data
    direct = _direct(schema)
    nullable = any(node.get("type") == "null" for node in variants)
    default = direct.get("default")
    current = raw if present else default
    # One legal value leaves nothing to choose or compute.
    fixed = "const" in direct and not nullable
    options = None if fixed else _option_views(schema)
    selected: int | None = None
    children_schema = direct
    if options is None:
        if "oneOf" in direct:
            # The core lists every tagged union's options; never guess branches.
            raise ValueError(f"Tagged union field {path!r} has no x-form-options")
        kind = _kind(variants)
        expression = any("expr" in _properties(node) for node in variants)
    else:
        selected = next(
            (index for index, item in enumerate(options) if item.matches(current)),
            None,
        )
        fields = None if selected is None else options[selected].fields
        kind = "choice" if fields is None else "model"
        children_schema = fields or {}
        expression = False
    metadata = {**direct, **schema}
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
        expression=expression,
        options=options,
        selected=selected,
        raw=current,
        present=present,
        value=preview.values.get(path),
        has_value=path in preview.values,
        decimals=_decimals(metadata),
        fixed=fixed,
        group=_text(schema, "x-form-group") or None,
    )
    if kind != "model" or not field.active:
        return field
    # Recursion through _children: each nested model level projects its rows.
    return replace(field, children=_children(children_schema, path, raw, preview))


def _children(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    raw: JsonValue,
    preview: FormPreview,
) -> tuple[FieldView, ...]:
    """Project the drawn property rows of the active model value raw at path.

    schema is the model's node, or the selected option's fields schema.
    Properties the schema marks hidden have no rows.
    """
    nested = raw if isinstance(raw, dict) else {}
    return tuple(
        _field(node, (*path, name), nested, preview)
        for name, node in _properties(schema).items()
        if isinstance(node, dict) and _shown(node)
    )


def project_fields(
    schema: dict[str, JsonValue], data: dict[str, JsonValue], preview: FormPreview
) -> tuple[FieldView, ...]:
    """Project a public input schema and detached data/preview into active rows.

    Preserve declaration order and metadata. A field with x-form-options
    shows the children of its selected option's fields only; a tagged union
    without that list raises ValueError instead of guessing a branch. Fields
    the schema marks hidden have no rows; their input and errors stay with
    the state. Do not evaluate, validate, write or generate defaults.
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
