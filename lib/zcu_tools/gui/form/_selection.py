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
from ._structure import alternatives, discriminator, intrinsic_tags, model_types
from .model import FormEnv, FormModel, FormPath, FormWritePath
from .options import Option


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
    """

    model: type[BaseModel]
    prefix: FormWritePath
    data: Mapping[str, JsonValue]
    choices: Mapping[FormPath, tuple[JsonValue, ...]]
    options: Mapping[FormPath, tuple[Option, ...]]
    pinned: Mapping[str, JsonValue] = field(default_factory=dict)

    def field(self, name: str) -> Field:
        """Resolve a direct field; unknown names raise with the full path."""
        path = (*self.prefix, name)
        info = self.model.model_fields.get(name)
        if info is None:
            raise ValueError(f"Unknown form field path: {path!r}")
        choices = self.choices.get((name,))
        options = self.options.get((name,))
        if options is None:
            options = _automatic_options(info, choices)
        selected = next(
            (
                index
                for index, option in enumerate(options or ())
                if name in self.data and option.matches(self.data[name])
            ),
            None,
        )
        return Field(self, name, info, choices, options, selected)


@dataclass(frozen=True)
class Field:
    """A resolved field; describe and command execution share this selection."""

    level: Level
    name: str
    info: FieldInfo
    choices: tuple[JsonValue, ...] | None
    options: tuple[Option, ...] | None
    selected: int | None

    @property
    def path(self) -> FormWritePath:
        """Full model-field path, never a display alias."""
        return (*self.level.prefix, self.name)

    @property
    def pinned(self) -> bool:
        """Whether the enclosing option fixes a categorical leaf's value."""
        return self.name in self.level.pinned and (
            self.choices is not None or literal_values(self.info.annotation) is not None
        )

    @property
    def active_model(self) -> type[BaseModel] | None:
        """Return the structurally reachable child model without running hooks.

        Ordinary missing/invalid models retain their declared shape. Nullable
        models require an object; selectors require a matching option with
        fields. A reachable model may have no displayed fields after expansion.
        """
        models = model_types(self.info.annotation)
        if not models:
            return None
        if self.options is not None:
            return None if self.selected is None else self.options[self.selected].fields
        if type(None) in alternatives(self.info.annotation) and not isinstance(
            self.level.data.get(self.name), dict
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
        value = self.level.data.get(self.name)
        pinned = self.level.pinned.get(self.name)
        if self.options is not None and self.selected is not None:
            pinned = self.options[self.selected].value
        choices = {
            path[1:]: values
            for path, values in self.level.choices.items()
            if len(path) > 1 and path[0] == self.name
        }
        options = {
            path[1:]: values
            for path, values in self.level.options.items()
            if len(path) > 1 and path[0] == self.name
        }
        return Level(
            model,
            self.path,
            value if isinstance(value, dict) else {},
            {**choices, **local_choices(model, env)},
            {**options, **_local_options(model, env)},
            pinned if isinstance(pinned, dict) else {},
        )


def validate_path(path: object) -> None:
    """Reject non-tuple paths and non-string components with ValueError."""
    if not isinstance(path, tuple) or any(not isinstance(part, str) for part in path):
        raise ValueError(f"Expected a model-field tuple: {path!r}")


def root_level(
    model: type[FormModel], data: Mapping[str, JsonValue], env: FormEnv
) -> Level:
    """Read only root hooks; nested declarations are read when reached."""
    return Level(model, (), data, local_choices(model, env), _local_options(model, env))


def resolve(
    model: type[FormModel],
    data: Mapping[str, JsonValue],
    path: FormWritePath,
    env: FormEnv,
) -> Field:
    """Resolve one nonroot canonical path through active children only.

    Unknown fields, inactive descendants and traversal into whole leaves raise
    ValueError naming the requested path. The target itself may be invalid.
    """
    validate_path(path)
    if not path:
        raise ValueError(f"Expected a nonroot model-field tuple: {path!r}")
    level = root_level(model, data, env)
    for index, name in enumerate(path):
        target = level.field(name)
        if index == len(path) - 1:
            return target
        children = target.children(env)
        if children is None:
            raise ValueError(f"Inactive or whole-leaf form field path: {path!r}")
        level = children
    raise ValueError(f"Unknown form field path: {path!r}")
