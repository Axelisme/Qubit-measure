"""Resolve only the requested active ancestry and its local declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from ._json import default_json
from ._sources import local_choices
from ._structure import (
    alternatives,
    discriminator,
    element_model,
    fixed_list,
    intrinsic_tags,
    model_types,
)
from .model import (
    FormEnv,
    FormModel,
    FormPath,
    FormPathError,
    FormWritePath,
    InputCommand,
    InputConverter,
)
from .options import Option


def option_errors(
    model: type[FormModel],
    data: Mapping[str, JsonValue],
    env: FormEnv,
    selections: Mapping[FormWritePath, JsonValue],
) -> dict[FormPath, str]:
    """Validate current option membership without changing authored branches.

    Hooks use the normal fixed env. Only explicit overrides are checked;
    raw branches retain their annotation/value contract. An unavailable choice
    is a field value error, not a conversion or restoration failure.
    """
    errors: dict[FormPath, str] = {}
    for path in selections:
        target = resolve(model, data, path, env, selections)
        if target.options is None or target.selected is None:
            errors[path] = "Selection is not among current options"
    return errors


def literal_values(annotation: object) -> tuple[JsonValue, ...] | None:
    """Read intrinsic Literal/Enum values in order, including nullable null."""
    result: list[JsonValue] = []
    nullable = False
    for member in alternatives(annotation):
        if member is type(None):
            nullable = True
        elif get_origin(member) is Literal:
            result.extend(default_json(item) for item in get_args(member))
        elif isinstance(member, type) and issubclass(member, Enum):
            result.extend(default_json(item.value) for item in member)
        else:
            return None
    if nullable and None not in result:
        result.insert(0, None)
    return tuple(result)


def _declaration_fixes(annotation: object, value: JsonValue) -> bool:
    """Whether annotation fixes every leaf of a complete partial option value.

    Only single categorical values and single non-null model branches qualify.
    Missing object fields are not part of the option value being checked.
    No dynamic declarations, defaults or validation participate.
    """
    values = literal_values(annotation)
    if values is not None:
        return (
            len(values) == 1 and type(values[0]) is type(value) and values[0] == value
        )
    models = model_types(annotation)
    if (
        not isinstance(value, dict)
        or len(alternatives(annotation)) != 1
        or len(models) != 1
    ):
        return False
    fields = models[0].model_fields
    return all(
        name in fields and _declaration_fixes(fields[name].annotation, item)
        for name, item in value.items()
    )


def _option_path(model: type[BaseModel], path: FormPath) -> bool:
    if not path or not isinstance(path[0], str):
        return False
    info = model.model_fields.get(path[0])
    return info is not None and (
        len(path) == 1
        or any(_option_path(child, path[1:]) for child in model_types(info.annotation))
    )


def _local_options(
    model: type[BaseModel], env: FormEnv
) -> Mapping[FormPath, tuple[Option, ...]]:
    declared = model.options(env) if issubclass(model, FormModel) else {}
    for path in declared:
        if not _option_path(model, path):
            raise ValueError(f"Unknown option field path: {path!r}")
    return declared


def _automatic_options(
    info: FieldInfo, choices: tuple[JsonValue, ...] | None
) -> tuple[Option, ...] | None:
    if choices is not None:
        return tuple(Option(str(value), value) for value in choices)
    tag = discriminator(info)
    if tag is not None:
        options = tuple(
            Option(str(value), {tag: value}, fields=model)
            for model in model_types(info.annotation)
            for value in intrinsic_tags(model, tag)
        )
        if type(None) in alternatives(info.annotation):
            options = (Option("None", None), *options)
        return options
    values = literal_values(info.annotation)
    return (
        tuple(Option(str(value), value) for value in values)
        if values is not None
        else None
    )


@dataclass(frozen=True)
class Level:
    """One active model occurrence with inherited relative declarations.

    Construct via root_level or Field.children so local hooks have already
    overlaid inherited paths. data is raw input only; missing/invalid ordinary
    models use an empty mapping without generating defaults. pinned carries
    the enclosing selected option's partial value for structural-tag omission.
    variants retains enclosing annotation branches for command-result admission.
    A FixedList's level has items, its raw elements, and its fields are the
    element indexes. Its model is the element model, shared by every element.
    name_field is the FixedList's identity field, set on the list level and on
    each element level; None elsewhere.
    """

    model: type[BaseModel]
    prefix: FormWritePath
    data: Mapping[str, JsonValue]
    choices: Mapping[FormPath, tuple[JsonValue, ...]]
    options: Mapping[FormPath, tuple[Option, ...]]
    pinned: Mapping[str, JsonValue] = field(default_factory=dict)
    variants: tuple[type[BaseModel], ...] = ()
    selections: Mapping[FormWritePath, JsonValue] = field(default_factory=dict)
    converters: Mapping[FormPath, InputConverter] = field(default_factory=dict)
    items: tuple[JsonValue, ...] | None = None
    name_field: str | None = None

    @property
    def names(self) -> tuple[str | int, ...]:
        """Declared field names, or the element indexes of a FixedList level."""
        if self.items is not None:
            return tuple(range(len(self.items)))
        return tuple(self.model.model_fields)

    def has(self, name: str | int) -> bool:
        """Whether name is a declared field or an in-range element index."""
        if self.items is not None:
            return (
                isinstance(name, int)
                and not isinstance(name, bool)
                and 0 <= name < len(self.items)
            )
        return isinstance(name, str) and name in self.model.model_fields

    def value(self, name: str | int) -> JsonValue:
        """Raw input at name, or None when missing."""
        if self.items is not None:
            return (
                self.items[name] if self.has(name) and isinstance(name, int) else None
            )
        return self.data.get(name) if isinstance(name, str) else None

    def field(self, name: str | int) -> Field:
        """Resolve a declared direct field or element; undeclared names raise KeyError."""
        if self.items is not None:
            if not self.has(name):
                raise KeyError(name)
            return Field(
                self, name, FieldInfo.from_annotation(self.model), None, None, None
            )
        if not isinstance(name, str):
            raise KeyError(name)
        info = self.model.model_fields[name]
        choices = self.choices.get((name,))
        options = self.options.get((name,))
        if options is None:
            options = _automatic_options(info, choices)
        selected = next(
            (
                index
                for index, option in enumerate(options or ())
                if (name in self.data or (*self.prefix, name) in self.selections)
                and option.matches(
                    self.selections.get((*self.prefix, name), self.data.get(name))
                )
            ),
            None,
        )
        return Field(self, name, info, choices, options, selected)


@dataclass(frozen=True)
class Field:
    """A resolved field; describe and command execution share this selection."""

    level: Level
    name: str | int
    info: FieldInfo
    choices: tuple[JsonValue, ...] | None
    options: tuple[Option, ...] | None
    selected: int | None

    @property
    def path(self) -> FormWritePath:
        """Full model-field path, never a display alias."""
        return (*self.level.prefix, self.name)

    @property
    def element_name(self) -> str:
        """Identity of a FixedList element field; its index text when unnamed."""
        item = self.level.value(self.name)
        key = self.level.name_field
        name = item.get(key) if isinstance(item, dict) and key is not None else None
        return name if isinstance(name, str) and name else str(self.name)

    @property
    def converter(self) -> InputConverter | None:
        """One effective local converter; hooks override Annotated metadata."""
        return self.level.converters.get((self.name,)) or next(
            (item for item in self.info.metadata if isinstance(item, InputConverter)),
            None,
        )

    @property
    def branch_option(self) -> Option | None:
        """Option describing raw children, independently of current selection."""
        return next(
            (
                option
                for option in self.options or ()
                if self.name in self.level.data
                and option.matches(self.level.value(self.name))
            ),
            None,
        )

    @property
    def input_commands(self) -> tuple[InputCommand, ...]:
        """Mounted actions whose result branch the field annotation accepts."""
        model = self.active_model
        declared = (
            model.input_commands
            if model is not None and issubclass(model, FormModel)
            else ()
        )
        return tuple(
            item
            for item in (*self.info.metadata, *declared)
            if isinstance(item, InputCommand)
            and (
                item.result_tag is None
                or any(
                    item.result_tag[1] in intrinsic_tags(model, item.result_tag[0])
                    for model in model_types(self.info.annotation)
                )
            )
        )

    @property
    def pinned(self) -> bool:
        """Whether the enclosing option fixes a categorical leaf's value."""
        return self.name in self.level.pinned and (
            self.choices is not None or literal_values(self.info.annotation) is not None
        )

    @property
    def selection_fixed(self) -> bool:
        """Whether annotation permits one non-null selector branch at this level.

        Null only enables/disables the field. A single non-null model branch
        qualifies without freezing nested selectors or fields. A categorical
        leaf must have one non-null intrinsic value. No selector means False;
        dynamic menu size does not participate.
        """
        if self.options is None:
            return False
        values = literal_values(self.info.annotation)
        if values is not None:
            return len([value for value in values if value is not None]) == 1
        branches = [
            member
            for member in alternatives(self.info.annotation)
            if member is not type(None)
        ]
        return len(branches) == 1 and len(model_types(self.info.annotation)) == 1

    @property
    def fixed_option(self) -> Option | None:
        """First option whose complete value is fixed by annotation, or None.

        Check every partial-value leaf against single declared categorical
        values. This does not depend on input selection or dynamic menu size.
        """
        return next(
            (
                option
                for option in self.options or ()
                if _declaration_fixes(self.info.annotation, option.value)
            ),
            None,
        )

    @property
    def active_model(self) -> type[BaseModel] | None:
        """Return the structurally reachable child model without running hooks.

        Ordinary missing/invalid models retain their declared shape. Nullable
        models require an object; selectors use the raw child branch's matching
        option fields, independently of an overridden selection. A reachable model may have no displayed fields after expansion.
        """
        if fixed_list(self.info) is not None:
            return element_model(self.info)
        models = model_types(self.info.annotation)
        if not models:
            return None
        if self.options is not None:
            option = self.branch_option
            return None if option is None else option.fields
        if type(None) in alternatives(self.info.annotation) and not isinstance(
            self.level.value(self.name), dict
        ):
            return None
        return models[0] if len(models) == 1 else None

    def children(self, env: FormEnv) -> Level | None:
        """Resolve the active child model and read only that child's hooks.

        Return None for structurally unreachable children, as active_model
        defines. Apply inherited declarations and the child's local overrides
        with fixed env, without defaults or validation.
        """
        model = self.active_model
        if model is None:
            return None
        value = self.level.value(self.name)
        pinned = (
            self.level.pinned.get(self.name) if isinstance(self.name, str) else None
        )
        if self.branch_option is not None:
            pinned = self.branch_option.value
        # Declarations of a FixedList apply to each of its elements, so an
        # element inherits them without an index component.
        element = self.level.items is not None
        choices = {
            (path if element else path[1:]): values
            for path, values in self.level.choices.items()
            if element or (len(path) > 1 and path[0] == self.name)
        }
        options = {
            (path if element else path[1:]): values
            for path, values in self.level.options.items()
            if element or (len(path) > 1 and path[0] == self.name)
        }
        fixed = fixed_list(self.info)
        if fixed is not None:
            return Level(
                model,
                self.path,
                {},
                choices,
                options,
                variants=(model,),
                selections=self.level.selections,
                converters={
                    path[1:]: item
                    for path, item in self.level.converters.items()
                    if len(path) > 1 and path[0] == self.name
                },
                items=tuple(value) if isinstance(value, list) else (),
                name_field=fixed.name_field,
            )
        return Level(
            model,
            self.path,
            value if isinstance(value, dict) else {},
            {**choices, **local_choices(model, env)},
            {**options, **_local_options(model, env)},
            pinned if isinstance(pinned, dict) else {},
            model_types(self.info.annotation),
            self.level.selections,
            {
                **{
                    (path if element else path[1:]): value
                    for path, value in self.level.converters.items()
                    if element or (len(path) > 1 and path[0] == self.name)
                },
                **_local_converters(model, env),
            },
            name_field=self.level.name_field if element else None,
        )


def validate_path(path: object) -> None:
    """Reject non-tuple paths and components that are neither str nor int."""
    if not isinstance(path, tuple) or any(
        not isinstance(part, (str, int)) for part in path
    ):
        raise ValueError(f"Expected a model-field tuple: {path!r}")


def _local_converters(
    model: type[BaseModel], env: FormEnv
) -> Mapping[FormPath, InputConverter]:
    metadata = {
        (name,): item
        for name, info in model.model_fields.items()
        for item in info.metadata
        if isinstance(item, InputConverter)
    }
    return {
        **metadata,
        **(model.converters(env) if issubclass(model, FormModel) else {}),
    }


def root_level(
    model: type[FormModel],
    data: Mapping[str, JsonValue],
    env: FormEnv,
    selections: Mapping[FormWritePath, JsonValue] | None = None,
) -> Level:
    """Read only root hooks; nested declarations are read when reached."""
    return Level(
        model,
        (),
        data,
        local_choices(model, env),
        _local_options(model, env),
        variants=(model,),
        selections={} if selections is None else selections,
        converters=_local_converters(model, env),
    )


def resolve(
    model: type[FormModel],
    data: Mapping[str, JsonValue],
    path: FormWritePath,
    env: FormEnv,
    selections: Mapping[FormWritePath, JsonValue] | None = None,
) -> Field:
    """Resolve one nonroot canonical path through active children only.

    Unknown fields, inactive descendants and traversal into whole leaves raise
    FormPathError carrying the full requested path. The target itself may be
    invalid. Noncanonical path shapes still raise ValueError.
    """
    validate_path(path)
    if not path:
        raise ValueError(f"Expected a nonroot model-field tuple: {path!r}")
    level = root_level(model, data, env, selections)
    for index, name in enumerate(path):
        if not level.has(name):
            raise FormPathError(path, f"Unknown form field path: {path!r}")
        target = level.field(name)
        if index == len(path) - 1:
            return target
        children = target.children(env)
        if children is None:
            raise FormPathError(
                path, f"Inactive or whole-leaf form field path: {path!r}"
            )
        level = children
    raise FormPathError(path, f"Unknown form field path: {path!r}")
