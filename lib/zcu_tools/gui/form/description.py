"""Detached single-layer descriptions shared by editors and wire adapters."""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from .model import FormWritePath

type FormFieldKind = Literal[
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
"""Input control category, not a validation schema; containers are whole leaves."""


@dataclass(frozen=True)
class FormConstraints:
    """Optional leaf hints, never a replacement for Pydantic validation.

    ge/gt are inclusive/exclusive lower numeric bounds; le/lt are upper bounds.
    min_length/max_length bound string or whole-container length. pattern is a
    string regular expression. None means no representable declared hint.
    Arbitrary validators and constraints inside containers are not described.
    """

    ge: int | float | None = None
    gt: int | float | None = None
    le: int | float | None = None
    lt: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class FormCommandDescription:
    """Available action metadata; no callable or captured sources are exposed.

    name is the identifier passed to run_command; label is display text.
    Both are nonempty and names are unique within their option or field.
    For field conversions, target is the replaced model's canonical path and
    inputs are participating canonical paths. Option actions leave both empty.
    """

    name: str
    label: str
    target: FormWritePath = ()
    inputs: tuple[FormWritePath, ...] = ()


@dataclass(frozen=True)
class FormOptionDescription:
    """One ordered selection, without children or a validation branch.

    label is display text. value is detached JSON partial input to write at
    the field path. commands lists the explicit actions available when this
    option is the first matching option. Values may be mutated by the caller
    without changing state, declarations or subsequent descriptions.
    """

    label: str
    value: JsonValue
    commands: tuple[FormCommandDescription, ...]


@dataclass(frozen=True)
class FormElementDescription:
    """One model-list element: current index, display label and field path.

    name comes from Display.element_label when it names a string value;
    otherwise it is the index text. It is presentation, not an identity rule.
    """

    index: int
    name: str
    path: FormWritePath


@dataclass(frozen=True)
class FormFieldDescription:
    """Metadata for one model field, deliberately without child-name hints.

    path is its full canonical model-field path, even through a flattened
    wrapper. name is path's final component. kind selects the input control
    category; union/unknown leaves retain validation in the model owner.
    title is the Display label, Field title or humanized field name.
    description/unit/placeholder are optional explanatory text. decimals is
    the declared float display precision, not input rounding. hidden hides a
    row, not its validation. group names a same-level display group or None;
    collapsed is a model's initial fold hint. can_expand means the current
    active structure reaches a model or selected option with fields; expansion
    may still return no children after hooks, fixed-field omission or wrapper
    flattening. It is false for whole leaves, disabled nullable models and
    unmatched/no-fields options. It never runs child hooks. nullable means
    null is allowed.
    expression indicates that an explicit Expr envelope can be evaluated.
    choices is the detached dynamic legal-value list, or None if undeclared;
    an empty tuple allows no value. options is the effective ordered selector
    list, or None for a non-selector; () preserves an explicit empty override.
    selected is the first option index matching the explicit in-memory selection,
    or raw input when no override exists. None means missing input/no match. No default option is guessed. selection_fixed means the
    annotation permits one non-null selector branch at this level, without
    freezing nested choices or fields. Null only enables/disables the field;
    it does not count as a branch, even while the field is disabled.
    selection_fixed is False without a selector and does not use dynamic menu
    size. fixed means the declaration or enclosing selection fixes
    this leaf or the complete option value; fixed_value is that detached JSON
    value (meaningful only when fixed, otherwise None). An outer model branch
    can be selection_fixed while inner choices keep fixed False. Both flags
    may be True. constraints contains finite leaf hints. commands lists explicit
    field actions with replacement target and participating inputs. selector=False
    hides the dropdown only, not its options or active structure. branch is the
    detached option value describing actual child structure, independently of
    selected; None denotes no matched structural option. A selection-only write
    changes selected without rebuilding branch or its local child drafts.
    A list[Model] field has kind "elements", can_expand and its element models
    as children; elements lists current indices and display labels. A child
    element path ends in its index. elements is None for every other field.
    readonly is Display's presentation hint, never mutation authorization.
    """

    path: FormWritePath
    name: str
    kind: FormFieldKind
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
    choices: tuple[JsonValue, ...] | None
    options: tuple[FormOptionDescription, ...] | None
    selected: int | None
    selection_fixed: bool
    fixed: bool
    fixed_value: JsonValue
    constraints: FormConstraints
    commands: tuple[FormCommandDescription, ...] = ()
    selector: bool = True
    readonly: bool = False
    branch: JsonValue = None
    elements: tuple[FormElementDescription, ...] | None = None


@dataclass(frozen=True)
class FormDescription:
    """Detached metadata at path plus at most one layer of children.

    path is the requested canonical model-field tuple; () means root. target
    is None only at root, otherwise the requested field metadata. children
    contains metadata only, in declaration order, with full canonical paths.
    A selected option with just one nonnullable, non-selector model wrapper
    lists that wrapper's fields instead; the wrapper path remains queryable.
    Leaves, disabled nullable models, refs and unmatched selectors have no
    children. This is input description, not validation or evaluated values.
    """

    path: FormWritePath
    target: FormFieldDescription | None
    children: tuple[FormFieldDescription, ...]
