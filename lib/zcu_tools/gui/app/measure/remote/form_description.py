"""Public measure wire types for one current, single-layer Form observation."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict

from pydantic import JsonValue

from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.form import (
    FormDescription,
    FormFieldDescription,
    FormValue,
    FormWritePath,
)
from zcu_tools.gui.form_adapter import LogicalFieldObservation

from ..services.tab_form import TabFormResource
from .form_errors import FormError
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


class FormInputCommandWire(FormCommandWire):
    """target is the replaced model path; inputs are participating field paths."""

    target: list[str | int]
    inputs: list[list[str | int]]


class FormOptionWire(TypedDict):
    """Ordered option: label is UI text; value is detached writable input JSON.

    commands lists actions available when this option is selected. No child
    descriptions or captured sources are included.
    """

    label: str
    value: JsonValue
    commands: list[FormCommandWire]


class FormElementWire(TypedDict):
    """One fixed element: original index, captured name and canonical typed path."""

    index: int
    name: str
    path: list[str | int]


class FormFieldWire(TypedDict):
    """Single field metadata, without recursive children or validation branches.

    path is the full typed field/index path. name is the field name, or a
    FixedList element's captured name. elements lists original indexes, names
    and paths for kind="elements", otherwise None. kind names
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
    a branch, even while disabled. It carries no value and ignores dynamic
    menu size. fixed
    constrains the complete value, fixed_value is meaningful only when fixed.
    Both flags may be true. constraints are hints. commands lists field actions
    with target and participating inputs. selector=False hides the dropdown,
    retaining options and active structure.
    """

    path: list[str | int]
    name: str
    kind: Literal[
        "model",
        "elements",
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
    readonly: bool
    fixed_value: JsonValue
    constraints: FormConstraintsWire
    commands: list[FormInputCommandWire]
    selector: bool
    elements: list[FormElementWire] | None


class FormDescriptionWire(TypedDict):
    """path is the requested model-field path, [] for root.

    target is field metadata, None only at root. children contains only direct
    metadata in declaration order, with canonical paths through flattened
    wrappers. Leaves, refs, disabled models and unmatched selectors have none.
    """

    path: list[str | int]
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
        "elements": None
        if field.elements is None
        else [
            {"index": item.index, "name": item.name, "path": list(item.path)}
            for item in field.elements
        ],
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
        "selector": field.selector,
        "selection_fixed": field.selection_fixed,
        "fixed": field.fixed,
        "readonly": field.readonly,
        "fixed_value": field.fixed_value,
        "commands": [
            {
                "name": item.name,
                "label": item.label,
                "target": list(item.target),
                "inputs": [list(path) for path in item.inputs],
            }
            for item in field.commands
        ],
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
    resource: TabFormResource, path: FormWritePath
) -> FormFieldObservation:
    """Read description and local values in this owner turn using fixed sources.

    resource must be live; path is canonical model-field names and FixedList
    indexes, () for root.
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
        if isinstance(raw, dict) and isinstance(part, str) and part in raw:
            child = raw.get(part)
        elif isinstance(raw, list) and isinstance(part, int) and 0 <= part < len(raw):
            child = raw[part]
        else:
            present, raw = False, None
            break
        raw = child
    value = _preview_subtree(
        {
            leaf[len(path) :]: value
            for leaf, value in preview.values.items()
            if leaf[: len(path)] == path
        }
    )
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


def _preview_subtree(values: Mapping[FormWritePath, FormValue]) -> JsonValue:
    """Reassemble relative preview leaves without converting indexes to keys.

    Unavailable leaves are absent. Missing list positions remain null, so a
    later element keeps its original index. Conflicting object/list paths are
    an owner programming error, not a user draft error.
    """
    if () in values:
        return context_wire_value(values[()])
    grouped: dict[str | int, dict[FormWritePath, FormValue]] = {}
    for path, value in values.items():
        grouped.setdefault(path[0], {})[path[1:]] = value
    if not grouped:
        return None
    if all(isinstance(name, int) for name in grouped):
        positions = {
            name: value for name, value in grouped.items() if isinstance(name, int)
        }
        return [
            _preview_subtree(positions.get(index, {}))
            for index in range(max(positions) + 1)
        ]
    if any(not isinstance(name, str) for name in grouped):
        raise RuntimeError("Conflicting Form preview object/list paths")
    return {
        name: _preview_subtree(children)
        for name, children in grouped.items()
        if isinstance(name, str)
    }


class RecipeSelectorWire(TypedDict):
    """Generated selector identity, not inferred from its authored envelope.

    mode distinguishes explicit key/inline, null disabled, absent missing and
    unresolved draft. key is the actual key only in key mode, otherwise None.
    available_keys is the captured ordered menu; its size does not imply locks.
    """

    mode: Literal["key", "inline", "disabled", "missing", "unresolved"]
    key: str | None
    available_keys: list[str]


class RecipeSourceWire(TypedDict):
    """Nearest captured ref providing a logical value.

    selector_path identifies that logical selector, key its actual library key,
    source_kind its module/waveform namespace, and spec_path the true persistent
    field path from that Spec root (empty for the selector itself).
    """

    selector_path: list[str | int]
    key: str
    source_kind: Literal["module", "waveform"]
    spec_path: list[str | int]


class RecipeFieldObservation(FormFieldObservation):
    """Detached logical read; inherited metadata/errors use logical paths.

    input_present/input describe authored input only, not projected source data.
    value_status distinguishes available (including null), missing and unavailable
    due to evaluation/validation failure. selector is set only for generated
    selectors, source identifies the nearest supplying ref or None. readonly
    means a ref ancestor was traversed, not declaration-fixed or write authority.
    Root has target/selector/source=None and an ordinary logical subtree value.
    element, when present, is the FixedElements owner's same-read original
    index/name. Fields outside declared elements omit it.
    """

    value_status: Literal["available", "missing", "unavailable"]
    selector: RecipeSelectorWire | None
    source: RecipeSourceWire | None
    readonly: bool
    element: NotRequired[ElementIdentity]


class RecipeEditResult(TypedDict):
    """changed reports installed input, errors the complete logical Form preview.

    Ordinary invalid drafts may be changed=True; this does not certify cfg.
    Admission/guard failure raises instead of returning changed=False.
    """

    changed: bool
    errors: list[FormError]


class RecipeSelectionResult(TypedDict):
    """changed reports installation; field is a fresh same-turn target read.

    A guarded inline no-op returns changed=False. No cross-command transaction
    or validity guarantee is implied by either field.
    """

    changed: bool
    field: RecipeFieldObservation


class RecipeElementFieldsWire(TypedDict):
    """Requested fields for one original element in a read_elements response.

    index is the original complete-list index. fields contains detached logical
    observations in the request's relative-path order, including owner identity.
    """

    index: int
    fields: list[RecipeFieldObservation]


class RecipeElementsWire(TypedDict):
    """Logical subset of read_elements when relative fields were requested.

    elements includes all original elements, never just participants.
    Other aggregate input/preview/error fields remain on the full RPC response.
    """

    elements: list[RecipeElementFieldsWire]


def recipe_field_wire(observation: LogicalFieldObservation) -> RecipeFieldObservation:
    """Encode one adapter observation, preserving logical paths and source facts.

    Does not resolve sources or inspect authored selector envelopes. Ordinary
    values use the existing JSON/complex codec; containers are detached.
    """
    selector = observation.selector
    source = observation.source
    result: RecipeFieldObservation = {
        "description": description_wire(observation.description),
        "input_present": observation.input_present,
        "input": observation.input,
        "value": context_wire_value(observation.value),
        "errors": [
            {"path": list(path), "message": message}
            for path, message in (
                observation.errors.messages.items() if observation.errors else ()
            )
        ],
        "value_status": observation.value_status,
        "selector": {
            "mode": selector.mode,
            "key": selector.key,
            "available_keys": list(selector.available_keys),
        }
        if selector is not None
        else None,
        "source": {
            "selector_path": list(source.selector_path),
            "key": source.key,
            "source_kind": source.source_kind,
            "spec_path": list(source.spec_path),
        }
        if source is not None
        else None,
        "readonly": observation.readonly,
    }
    if observation.element is not None:
        result["element"] = observation.element.copy()
    return result
