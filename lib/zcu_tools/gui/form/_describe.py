"""Project a requested layer directly from annotations, never JSON Schema."""

from collections.abc import Mapping
from enum import Enum
from re import Pattern
from types import UnionType
from typing import TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from ._json import copy_json
from ._selection import Field, Level, literal_values, resolve, root_level, validate_path
from ._structure import (
    alternatives,
    branch_tags,
    expression_allowed,
    fixed_tag,
    model_types,
)
from .description import (
    FormCommandDescription,
    FormConstraints,
    FormDescription,
    FormFieldDescription,
    FormFieldKind,
    FormOptionDescription,
)
from .model import Display, FormEnv, FormModel, FormWritePath, InputCommand
from .options import Option
from .sweep import CenterSpan, LinRange, StepCenter, StepRange


def _kind(target: Field) -> FormFieldKind:
    members = tuple(
        item for item in alternatives(target.info.annotation) if item is not type(None)
    )
    if model_types(target.info.annotation):
        return "model"
    if literal_values(target.info.annotation) is not None:
        return (
            "enum"
            if any(
                isinstance(item, type) and issubclass(item, Enum) for item in members
            )
            else "literal"
        )
    if len(members) == 1:
        member = members[0]
        scalars: tuple[tuple[type, FormFieldKind], ...] = (
            (bool, "boolean"),
            (int, "integer"),
            (float, "number"),
            (str, "string"),
            (complex, "complex"),
        )
        scalar: FormFieldKind | None = next(
            (kind for annotation, kind in scalars if member is annotation), None
        )
        if scalar is not None:
            return scalar
        if get_origin(member) in (list, tuple):
            return "sequence"
        if get_origin(member) in (dict, Mapping):
            return "mapping"
    return "union" if len(members) > 1 else "unknown"


def _constraint_metadata(target: Field) -> list[object]:
    """Read inner-to-outer metadata through aliases and one non-null branch.

    Pydantic keeps nullable member constraints inside the annotation. Use its
    FieldInfo parser for Annotated, retaining outer field metadata precedence.
    Multiple non-null branches keep only their common outer metadata.
    """
    metadata: list[object] = list(target.info.metadata)
    annotation = target.info.annotation
    while annotation is not None:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
            continue
        if get_origin(annotation) in (Union, UnionType):
            members = [item for item in get_args(annotation) if item is not type(None)]
            if len(members) != 1:
                break
            annotation = members[0]
            continue
        inner = FieldInfo.from_annotation(annotation)
        metadata = [*inner.metadata, *metadata]
        if inner.annotation is annotation:
            break
        annotation = inner.annotation
    return metadata


def _number_hint(metadata: list[object], name: str) -> int | float | None:
    value = next(
        (
            getattr(item, name)
            for item in reversed(metadata)
            if getattr(item, name, None) is not None
        ),
        None,
    )
    return value if isinstance(value, int | float) else None


def _length_hint(metadata: list[object], name: str) -> int | None:
    value = _number_hint(metadata, name)
    return value if isinstance(value, int) else None


def _constraints(target: Field) -> FormConstraints:
    metadata = _constraint_metadata(target)
    pattern = next(
        (
            getattr(item, "pattern", None)
            for item in reversed(metadata)
            if getattr(item, "pattern", None) is not None
        ),
        None,
    )
    if isinstance(pattern, Pattern):
        pattern = pattern.pattern
    return FormConstraints(
        ge=_number_hint(metadata, "ge"),
        gt=_number_hint(metadata, "gt"),
        le=_number_hint(metadata, "le"),
        lt=_number_hint(metadata, "lt"),
        min_length=_length_hint(metadata, "min_length"),
        max_length=_length_hint(metadata, "max_length"),
        pattern=pattern if isinstance(pattern, str) else None,
    )


def _option_metadata(
    options: tuple[Option, ...] | None,
) -> tuple[FormOptionDescription, ...] | None:
    """Detach ordered author selections, exposing actions but never fields/callables."""
    if options is None:
        return None
    return tuple(
        FormOptionDescription(
            option.label,
            option.value,
            tuple(
                FormCommandDescription(command.name, command.label)
                for command in option.commands
            ),
        )
        for option in options
    )


def _fixed_value(target: Field) -> tuple[bool, JsonValue]:
    """Return whether a whole leaf/option value is fixed, and that value.

    Enclosing categorical pins take precedence over intrinsic single values
    and declaration-fixed complete option values. An unfixed field returns
    (False, None); a fixed null returns (True, None).
    """
    if target.pinned:
        return True, target.level.pinned[target.name]
    values = literal_values(target.info.annotation)
    if values is not None and len(values) == 1:
        return True, values[0]
    option = target.fixed_option
    if option is not None:
        return True, option.value
    return False, None


def _metadata(
    target: Field, tags: Mapping[type[BaseModel], str]
) -> FormFieldDescription:
    info = target.info
    display = next(
        (item for item in reversed(info.metadata) if isinstance(item, Display)),
        Display(),
    )
    if display.decimals is not None and float not in alternatives(info.annotation):
        raise ValueError(f"{target.path!r}: Display decimals requires a float field")
    fixed, fixed_value = _fixed_value(target)
    hidden_tag = fixed_tag(target.level.model, tags) == target.name
    text = target.name.replace("_", " ")
    return FormFieldDescription(
        path=target.path,
        name=target.name,
        kind=_kind(target),
        title=display.label or info.title or text[:1].upper() + text[1:],
        description=display.description
        if display.description is not None
        else info.description,
        unit=display.unit,
        placeholder=display.placeholder,
        decimals=display.decimals,
        hidden=display.hidden or hidden_tag or target.pinned,
        group=display.group,
        collapsed=display.collapsed,
        can_expand=target.active_model is not None,
        nullable=type(None) in alternatives(info.annotation),
        expression=expression_allowed(info, target.path) and target.choices is None,
        choices=None
        if target.choices is None
        else tuple(copy_json(value) for value in target.choices),
        options=_option_metadata(target.options),
        selected=target.selected,
        selection_fixed=target.selection_fixed,
        fixed=fixed,
        fixed_value=copy_json(fixed_value),
        constraints=_constraints(target),
        selector=display.selector,
        commands=tuple(
            FormCommandDescription(
                item.name,
                item.label,
                target.level.prefix,
                tuple((*target.level.prefix, name) for name in item.inputs),
            )
            for item in info.metadata
            if isinstance(item, InputCommand)
        ),
    )


def _children(
    level: Level, tags: Mapping[type[BaseModel], str]
) -> tuple[FormFieldDescription, ...]:
    return tuple(
        _metadata(target, tags)
        for name in level.model.model_fields
        if not (target := level.field(name)).pinned
    )


def _description_tags(roots: tuple[object, ...]) -> dict[type[BaseModel], str]:
    tags = branch_tags(roots)
    tags[LinRange] = "kind"
    tags[CenterSpan] = "kind"
    tags[StepRange] = "kind"
    tags[StepCenter] = "kind"
    return tags


def describe_field(target: Field, *, roots: tuple[object, ...]) -> FormFieldDescription:
    """Describe an already resolved field in its original parent context.

    target retains its Level, effective choices/options and enclosing selection.
    roots are the annotations from which that occurrence was reached, including
    a projected model when it is outside the original root graph. They identify
    structural branch tags; ordinary same-named fields remain ordinary fields.
    This reads metadata only, without evaluation, source lookup or mutation.
    Invalid display declarations raise ValueError, as canonical describe does.
    """
    return _metadata(target, _description_tags(roots))


def describe(
    model: type[FormModel],
    data: Mapping[str, JsonValue],
    path: FormWritePath,
    env: FormEnv,
) -> FormDescription:
    """Return target and one layer, resolving only active model ancestry.

    Reads declaration hooks with fixed env, never defaults, evaluators or
    source lookup. ValueError reports noncanonical, unknown or inactive paths.
    Structural tag identity is annotation-owned, independent of dynamic choices.
    """
    validate_path(path)
    tags = _description_tags((model,))
    if not path:
        return FormDescription(
            path, None, _children(root_level(model, data, env), tags)
        )
    target = resolve(model, data, path, env)
    metadata = describe_field(target, roots=(model,))
    level = target.children(env)
    if level is None:
        return FormDescription(path, metadata, ())
    children = _children(level, tags)
    if target.options is not None and len(children) == 1:
        wrapper = children[0]
        if wrapper.kind == "model" and not wrapper.nullable and wrapper.options is None:
            nested = level.field(wrapper.name).children(env)
            if nested is not None:
                children = _children(nested, tags)
    return FormDescription(path, metadata, children)
