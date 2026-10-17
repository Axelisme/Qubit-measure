"""Workflow-owned JSON representation of core single-layer form descriptions."""

from typing import TypedDict

from pydantic import JsonValue

from zcu_tools.gui.form import FormDescription, FormFieldDescription


class FormConstraintsReply(TypedDict):
    """Finite leaf hints, not validation: null means no declared hint.

    ge/gt bound numbers inclusively/exclusively below; le/lt bound them above.
    min_length/max_length bound string or whole-container length. pattern is a
    string regular expression. Arbitrary validators are not represented.
    """

    ge: int | float | None
    gt: int | float | None
    le: int | float | None
    lt: int | float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None


class FormCommandReply(TypedDict):
    """name identifies an available run_command action; label is display text."""

    name: str
    label: str


class FormInputCommandReply(FormCommandReply):
    """target is the replaced model; inputs are participating canonical paths."""

    target: list[str]
    inputs: list[list[str]]


class FormOptionReply(TypedDict):
    """One ordered option, without branch fields or validation schema.

    label is display text; value is detached partial JSON input to write at the
    field path; commands lists actions available when this option is selected.
    """

    label: str
    value: JsonValue
    commands: list[FormCommandReply]


class FormFieldReply(TypedDict):
    """One field's metadata, without descendants or child-name hints.

    path is the full canonical model-field path; name is its final segment.
    kind is the core control category (model, integer, number, complex, string,
    boolean, sequence, mapping, literal, enum, union or unknown). title is display
    text. description/unit/placeholder are optional explanatory strings.
    decimals is float display precision or null. hidden hides the row, not
    validation; group is a same-level display group or null; collapsed is the
    initial model fold hint. can_expand means the active structure can reach a
    model or selected option's fields without running child hooks; expansion may
    still return no children. nullable allows null; expression allows Expr input.
    choices is a detached legal-value list or null when undeclared. options is
    the effective ordered selector list, null for non-selectors, [] for an empty
    override. selected is the core's first matching index, or null for missing/
    no-match input. selection_fixed means the annotation permits one non-null
    selector branch at this level, without freezing inner choices. Null only
    enables/disables the field; it is not another branch, even while disabled.
    selection_fixed does not depend on menu size and carries no value.
    fixed says the declaration/selection fixes the leaf or complete option value;
    fixed_value is meaningful only when fixed, otherwise null. Both flags may be
    true. constraints carries leaf hints. commands lists field actions with
    target and participating inputs. selector=False hides the dropdown without
    changing options or active structure.
    """

    path: list[str]
    name: str
    kind: str
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
    options: list[FormOptionReply] | None
    selected: int | None
    selection_fixed: bool
    fixed: bool
    fixed_value: JsonValue
    constraints: FormConstraintsReply
    commands: list[FormInputCommandReply]
    selector: bool


class FormDescriptionReply(TypedDict):
    """Single-layer metadata at path from the same observed candidate input.

    path is the requested model-field path ([] is root). target is null only at
    root, otherwise that field's metadata. children lists direct metadata in
    declaration order, with canonical paths even through a flattened wrapper.
    Leaves, disabled nullable models and unmatched selectors have no children.
    """

    path: list[str]
    target: FormFieldReply | None
    children: list[FormFieldReply]


def _project_field(field: FormFieldDescription) -> FormFieldReply:
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
        "choices": None if field.choices is None else list(field.choices),
        "options": None
        if field.options is None
        else [
            {
                "label": option.label,
                "value": option.value,
                "commands": [
                    {"name": command.name, "label": command.label}
                    for command in option.commands
                ],
            }
            for option in field.options
        ],
        "selected": field.selected,
        "selector": field.selector,
        "commands": [
            {
                "name": item.name,
                "label": item.label,
                "target": list(item.target),
                "inputs": [list(path) for path in item.inputs],
            }
            for item in field.commands
        ],
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


def project_form_description(description: FormDescription) -> FormDescriptionReply:
    """Convert detached core description to workflow JSON without interpretation.

    description is the core's single-layer result for the chosen input and fixed
    env. Return path, optional target and direct children; options retain only
    label/value/commands. No validation, branch inference, capture or I/O occurs.
    Mutable JSON values remain detached from state by the core's contract.
    """
    return {
        "path": list(description.path),
        "target": None
        if description.target is None
        else _project_field(description.target),
        "children": [_project_field(child) for child in description.children],
    }
