"""Public measure wire types for one current, single-layer Form observation."""

from typing import Literal, TypedDict

from pydantic import JsonValue

from zcu_tools.gui.form import FormDescription, FormFieldDescription

from ..services.tab_form import TabFormResource
from .form_observation import FormError
from .handlers._wire_values import context_wire_value


class FormConstraintsWire(TypedDict):
    """Leaf hints: ge/gt and le/lt are inclusive/exclusive numeric bounds.

    min_length/max_length are string or container bounds; pattern is a string
    regular expression. None means undeclared. Hints do not replace validation.
    """

    ge: int | float | None
    gt: int | float | None
    le: int | float | None
    lt: int | float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None


class FormCommandWire(TypedDict):
    """Available command: name is the run_command identifier; label is UI text."""

    name: str
    label: str


class FormOptionWire(TypedDict):
    """Ordered option: label is UI text; value is detached writable input JSON.

    commands lists actions available when this option is selected. No child
    descriptions or captured sources are included.
    """

    label: str
    value: JsonValue
    commands: list[FormCommandWire]


class FormFieldWire(TypedDict):
    """Single field metadata, without recursive children or validation branches.

    path is the full model-field path; name is its final segment. kind names
    the control category. title is display text; description/unit/placeholder
    are optional text. decimals is display precision, not rounding input.
    hidden omits a UI row, group names a same-level UI group, collapsed is an
    initial model fold hint. can_expand reports structural expandability, not
    guaranteed nonempty children, without running child hooks.
    nullable permits null, expression permits Expr.
    choices is a legal-value list or None when undeclared. options is the
    ordered selector list or None for a non-selector. selected is the first
    matching index or None for missing/no match. selection_fixed means the
    annotation permits one non-null selector branch at this level, without
    freezing inner choices; nullable None is an enable/disable capability, not
    a branch. It carries no value and ignores dynamic menu size. fixed
    constrains the complete value, fixed_value is meaningful only when fixed.
    Both flags may be true. constraints are hints.
    """

    path: list[str]
    name: str
    kind: Literal[
        "model",
        "integer",
        "number",
        "complex",
        "string",
        "boolean",
        "sequence",
        "mapping",
        "literal",
        "enum",
        "union",
        "unknown",
    ]
    title: str
    description: str | None
    unit: str | None
    placeholder: str | None
    decimals: int | None
    hidden: bool
    group: str | None
    collapsed: bool
    can_expand: bool
    nullable: bool
    expression: bool
    choices: list[JsonValue] | None
    options: list[FormOptionWire] | None
    selected: int | None
    selection_fixed: bool
    fixed: bool
    fixed_value: JsonValue
    constraints: FormConstraintsWire


class FormDescriptionWire(TypedDict):
    """path is the requested model-field path, [] for root.

    target is field metadata, None only at root. children contains only direct
    metadata in declaration order, with canonical paths through flattened
    wrappers. Leaves, refs, disabled models and unmatched selectors have none.
    """

    path: list[str]
    target: FormFieldWire | None
    children: list[FormFieldWire]


class FormFieldObservation(TypedDict):
    """Atomic current field read from the tab's fixed capture.

    description is single-layer metadata. input_present distinguishes missing
    input from explicit null; input retains authored expression/complex tags.
    value is an ordinary leaf or subtree, None for null/unavailable, never proof
    of validity. errors contains absolute root and target/descendant errors.
    All JSON containers are detached. No read grants mutation authority.
    """

    description: FormDescriptionWire
    input_present: bool
    input: JsonValue
    value: JsonValue
    errors: list[FormError]


def description_wire(description: FormDescription) -> FormDescriptionWire:
    """Encode core metadata without resolving annotations or evaluating hooks.

    description is a core single-layer result; the returned JSON containers are
    detached by that owner. No schema generation, source reads or mutation run.
    """
    return {
        "path": list(description.path),
        "target": _field_wire(description.target)
        if description.target is not None
        else None,
        "children": [_field_wire(child) for child in description.children],
    }


def _field_wire(field: FormFieldDescription) -> FormFieldWire:
    hints = field.constraints
    return {
        "path": list(field.path),
        "name": field.name,
        "kind": field.kind,
        "title": field.title,
        "description": field.description,
        "unit": field.unit,
        "placeholder": field.placeholder,
        "decimals": field.decimals,
        "hidden": field.hidden,
        "group": field.group,
        "collapsed": field.collapsed,
        "can_expand": field.can_expand,
        "nullable": field.nullable,
        "expression": field.expression,
        "choices": list(field.choices) if field.choices is not None else None,
        "options": [
            {
                "label": option.label,
                "value": option.value,
                "commands": [
                    {"name": command.name, "label": command.label}
                    for command in option.commands
                ],
            }
            for option in field.options
        ]
        if field.options is not None
        else None,
        "selected": field.selected,
        "selection_fixed": field.selection_fixed,
        "fixed": field.fixed,
        "fixed_value": field.fixed_value,
        "constraints": {
            "ge": hints.ge,
            "gt": hints.gt,
            "le": hints.le,
            "lt": hints.lt,
            "min_length": hints.min_length,
            "max_length": hints.max_length,
            "pattern": hints.pattern,
        },
    }


def describe_form(
    resource: TabFormResource, path: tuple[str, ...]
) -> FormFieldObservation:
    """Read description and local values in this owner turn using fixed sources.

    resource must be live; path is canonical model-field names, () for root.
    Unknown/inactive/whole-leaf descendants raise FormPathError from core;
    other declaration failures retain their exception types. Revoked resources
    raise RESOURCE_GONE. Busy/notification/observer reads are allowed.
    Only Form preview root/descendant errors are returned; use get_cfg or Run
    admission for lowered cfg errors. No defaults, capture, library
    materialization or publication occurs.
    """
    description = resource.describe(path)
    preview = resource.preview()
    raw: JsonValue = resource.input.data
    present = True
    for part in path:
        present = isinstance(raw, dict) and part in raw
        raw = raw.get(part) if isinstance(raw, dict) else None
    value: JsonValue = None
    subtree: dict[str, JsonValue] = {}
    for leaf_path, leaf_value in preview.values.items():
        segments = list(leaf_path)
        if tuple(segments) == path:
            value = context_wire_value(leaf_value)
            break
        if tuple(segments[: len(path)]) != path:
            continue
        parent = subtree
        for name in segments[len(path) : -1]:
            child = parent.setdefault(name, {})
            if not isinstance(child, dict):
                raise RuntimeError("Conflicting Form preview paths")
            parent = child
        parent[segments[-1]] = context_wire_value(leaf_value)
    else:
        value = subtree or None
    return {
        "description": description_wire(description),
        "input_present": present,
        "input": raw,
        "value": value,
        "errors": [
            {"path": list(error_path), "message": message}
            for error_path, message in (
                preview.errors.messages.items() if preview.errors is not None else ()
            )
            if not error_path or error_path[: len(path)] == path
        ],
    }
