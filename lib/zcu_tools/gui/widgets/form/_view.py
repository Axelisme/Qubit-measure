"""Convert core metadata and preview values to one layer of Qt row data."""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import (
    FormFieldDescription,
    FormOptionDescription,
    FormPreview,
    FormValue,
    FormWritePath,
)

type _Kind = Literal[
    "text", "integer", "number", "complex", "boolean", "json", "model", "choice"
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
    raw/present describe authored input; a missing fixed leaf displays its
    fixed value without writing it. value/has_value identify a preview leaf.
    decimals rounds display only. fixed makes the input read-only. group is
    the same-level display group name, or None for an ungrouped row.
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
    raw: JsonValue
    present: bool
    value: FormValue
    has_value: bool
    decimals: int | None
    fixed: bool
    group: str | None


def project_field(
    field: FormFieldDescription, data: dict[str, JsonValue], preview: FormPreview
) -> FieldView:
    """Translate field's core metadata to controls using current data and preview.

    data is the full detached input tree, preview the owner's full evaluation.
    No validation, option matching, annotation resolution or defaulting occurs.
    Hidden metadata must be filtered by the caller. Only display categories
    are translated; the core decides active children and selected options.
    """
    current: JsonValue = data
    present = True
    for name in field.path:
        if not isinstance(current, dict) or name not in current:
            current = None
            present = False
            break
        current = current[name]
    kind: _Kind
    if field.options is not None and not field.fixed:
        kind = "model" if field.can_expand else "choice"
    elif field.kind == "model":
        kind = "model"
    elif field.kind in ("integer", "number", "complex", "boolean"):
        kind = field.kind
    elif field.kind in ("sequence", "mapping"):
        kind = "json"
    else:
        kind = "text"
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
        expression=field.expression and not field.fixed,
        options=None if field.fixed else field.options,
        selected=field.selected,
        raw=current if present or not field.fixed else field.fixed_value,
        present=present,
        value=preview.values.get(field.path),
        has_value=field.path in preview.values,
        decimals=field.decimals,
        fixed=field.fixed,
        group=field.group,
    )
