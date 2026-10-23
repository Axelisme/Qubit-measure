"""Convert core metadata and preview values to one layer of Qt row data."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import (
    FormCommandDescription,
    FormElementDescription,
    FormFieldDescription,
    FormOptionDescription,
    FormPreview,
    FormValue,
    FormWritePath,
)

type _Kind = Literal[
    "text",
    "integer",
    "number",
    "complex",
    "boolean",
    "json",
    "model",
    "choice",
    "elements",
]


@dataclass(frozen=True)
class FieldView:
    """One drawn row, with no descendants or writable model mirror.

    path is the canonical field path; kind selects the Qt control. label,
    description, unit and placeholder are display text. collapsed is the initial
    fold hint, nullable permits null, and active means the model is enabled.
    can_expand is the core's active child capability, controlling fold titles.
    expression permits explicit Expr input. options/selected are the core's
    detached selector list and matching index (None for no list/no match).
    branch identifies the actual child structure independently of selection.
    selection/selection_present describe the explicit selector override, falling
    back to raw input only when no override exists.
    raw/present describe authored input; a missing fixed leaf displays its
    fixed value without writing it. value/has_value identify a preview leaf.
    decimals rounds display only. fixed marks a declaration-fixed scalar or
    option value, not read-only children or commands. group is the same-level
    display group name, or None for an ungrouped row. elements lists a
    FixedList row's picker entries by original index and name, else None.
    disabled projects core applicability independently of readonly/fixed.
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
    can_expand: bool
    expression: bool
    options: tuple[FormOptionDescription, ...] | None
    selected: int | None
    selection: JsonValue
    selection_present: bool
    branch: JsonValue
    raw: JsonValue
    present: bool
    value: FormValue
    has_value: bool
    decimals: int | None
    fixed: bool
    group: str | None
    commands: tuple[FormCommandDescription, ...]
    disabled: bool = False
    elements: tuple[FormElementDescription, ...] | None = None


def _control_presentation(
    field: FormFieldDescription,
) -> tuple[_Kind, tuple[FormOptionDescription, ...] | None]:
    """Choose the Qt control category and menu from core-owned metadata.

    A fixed option value fixes its tags, not its editable descendants or
    explicit commands. Only a fixed scalar without actions is plain text.
    No branch matching or annotation interpretation occurs here.
    """
    if field.kind == "elements":
        return "elements", None
    options = field.options if field.selector else None
    if (
        (field.fixed or field.readonly)
        and field.kind != "model"
        and not field.can_expand
        and (options is None or not any(option.commands for option in options))
    ):
        options = None
    if options is not None:
        return ("model" if field.can_expand else "choice"), options
    if field.kind == "model":
        return "model", options
    if field.kind in ("integer", "number", "complex", "boolean"):
        return field.kind, options
    if field.kind in ("sequence", "mapping"):
        return "json", options
    return "text", options


def present_fields(
    fields: tuple[FormFieldDescription, ...],
    children: Callable[[FormWritePath], tuple[FormFieldDescription, ...]],
) -> tuple[FormFieldDescription, ...]:
    """Apply Sweep layout and initial folds without changing canonical paths.

    fields is one visible model layer. children reads the current direct child
    metadata at a canonical path. Only an ordinary nonnullable sweep container
    is inspected eagerly: one expandable axis replaces its container row, with
    a combined title; multiple axes keep both levels. Axis commands and selector
    metadata remain attached to the axis path. Sweep containers and axes start
    open, modules start closed, and other rows retain their collapsed hint.
    """
    result: list[FormFieldDescription] = []
    for field in fields:
        if field.name == "modules":
            field = replace(field, collapsed=True)
        elif len(field.path) >= 2 and field.path[-2] == "sweep":
            field = replace(field, collapsed=False)
        if (
            field.name == "sweep"
            and field.kind == "model"
            and field.can_expand
            and not field.nullable
            and field.options is None
            and not field.commands
        ):
            axes = tuple(child for child in children(field.path) if not child.hidden)
            if axes and all(axis.can_expand and not axis.nullable for axis in axes):
                field = replace(field, collapsed=False)
                if len(axes) == 1:
                    axis = axes[0]
                    unit = f" ({axis.unit})" if axis.unit else ""
                    field = replace(
                        axis,
                        title=f"{field.title} · {axis.title}{unit}",
                        unit=None,
                        group=field.group,
                        collapsed=False,
                    )
        result.append(field)
    return tuple(result)


def project_field(
    field: FormFieldDescription,
    data: dict[str, JsonValue],
    preview: FormPreview,
    selections: Mapping[FormWritePath, JsonValue],
) -> FieldView:
    """Translate field's core metadata to controls using current data and preview.

    data is the full detached input tree, preview the owner's full evaluation,
    and selections the detached explicit selector overrides. Unmatched selectors
    display the override rather than the independently preserved raw child input.
    No validation, option matching, annotation resolution or defaulting occurs.
    Hidden metadata must be filtered by the caller. Only display categories
    are translated; the core decides active children and selected options.
    """
    current: JsonValue = data
    present = True
    for name in field.path:
        if isinstance(current, dict) and isinstance(name, str) and name in current:
            value = current.get(name)
        elif (
            isinstance(current, list)
            and isinstance(name, int)
            and 0 <= name < len(current)
        ):
            value = current[name]
        else:
            current = None
            present = False
            break
        current = value
    kind, options = _control_presentation(field)
    return FieldView(
        path=field.path,
        kind=kind,
        label=field.title,
        description=field.description or "",
        unit=field.unit or "",
        placeholder=field.placeholder or "",
        collapsed=field.collapsed,
        nullable=field.nullable,
        active=not field.nullable or (present and current is not None),
        can_expand=field.can_expand,
        expression=field.expression and not (field.fixed or field.readonly),
        options=options,
        selected=field.selected,
        selection=selections.get(field.path, current),
        selection_present=field.path in selections or present,
        branch=field.branch,
        raw=current if present or not field.fixed else field.fixed_value,
        present=present,
        value=preview.values.get(field.path),
        has_value=field.path in preview.values,
        decimals=field.decimals,
        fixed=field.fixed or field.readonly,
        group=field.group,
        commands=field.commands,
        elements=field.elements,
        disabled=field.disabled,
    )
