"""Component model declarations, independent of experiment definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

from pydantic import BaseModel

from .errors import MissingReferenceError, UnknownFieldError, UnknownKindError
from .schema import ComponentSchema, ModuleSlot, Ref


def _nested_model(annotation: object) -> type[BaseModel] | None:
    if get_origin(annotation) in (Union, UnionType):
        alternatives = tuple(
            item for item in get_args(annotation) if item is not type(None)
        )
        if len(alternatives) != 1:
            return None
        annotation = alternatives[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _validate_supported_model(model: type[BaseModel]) -> None:
    for validator in model.__pydantic_decorators__.model_validators.values():
        raise ValueError(
            f"{model.__name__}: {validator.info.mode} model validator is unsupported "
            "in component models; cross-field checks are not supported in this batch"
        )
    if model.model_post_init is not ComponentSchema.model_post_init:
        raise ValueError(
            f"{model.__name__}: custom model_post_init is unsupported in component models"
        )
    for name, field in model.model_fields.items():
        if any(isinstance(marker, ModuleSlot) for marker in field.metadata) and (
            get_origin(field.annotation) not in (dict, Mapping)
            or get_args(field.annotation) != (str, str)
        ):
            raise TypeError(
                f"{model.__name__}.{name}: ModuleSlot requires a non-null "
                "mapping of string slot names to string paths"
            )
        nested_model = _nested_model(field.annotation)
        if nested_model is not None:
            _validate_supported_model(nested_model)


def _validate_component_model(model: object) -> None:
    if not isinstance(model, type) or not issubclass(model, ComponentSchema):
        raise TypeError("Registered models must derive from ComponentSchema")
    if model.model_config.get("extra") != "forbid":
        raise TypeError("Registered component models must use extra=forbid")


def _validate_reference(model: type[BaseModel], reference: str) -> None:
    parts = reference.split(".")
    for index, name in enumerate(parts):
        field = model.model_fields.get(name)
        if field is None:
            break
        annotation = field.annotation
        if index == len(parts) - 1:
            types = tuple(
                item for item in get_args(annotation) if item is not type(None)
            )
            if annotation is str or (
                get_origin(annotation) in (Union, UnionType) and types == (str,)
            ):
                return
            break
        nested_model = _nested_model(annotation)
        if nested_model is None:
            break
        model = nested_model
    raise ValueError(f"Invalid component reference path: {reference!r}")


@dataclass(frozen=True)
class RoleSpec:
    """Required component kind pattern and optional resolved-role reference path."""

    kind: str
    via: str | None = None


class RoleRegistry:
    """Register notebook/adapter role names without loading experiment modules."""

    def __init__(self) -> None:
        self._roles: dict[str, RoleSpec] = {}
        self._shorthand: tuple[str, ...] = ()
        self._focus_kinds: tuple[str, ...] = ()

    def configure(
        self, *, shorthand: Sequence[str], focus_kinds: Sequence[str]
    ) -> None:
        """Declare notebook roles and kind patterns eligible for default focus.

        All shorthand roles must already be registered and unique. Kind patterns
        must be nonempty. Reject invalid declarations without changing either
        setting. Empty tuples disable shorthand or automatic focus respectively.
        """
        names, patterns = tuple(shorthand), tuple(focus_kinds)
        for name in names:
            self.get(name)
        if len(set(names)) != len(names):
            raise ValueError("Shorthand roles must be unique")
        if any(not pattern for pattern in patterns):
            raise ValueError("A focus kind pattern must not be empty")
        self._shorthand, self._focus_kinds = names, patterns

    @property
    def shorthand(self) -> tuple[str, ...]:
        """Return the declared ordered notebook roles."""
        return self._shorthand

    @property
    def focus_kinds(self) -> tuple[str, ...]:
        """Return the declared patterns eligible for automatic focus."""
        return self._focus_kinds

    def register(self, name: str, spec: RoleSpec) -> None:
        """Register a unique public role name; invalid declarations reserve nothing."""
        if not name.isidentifier() or name.startswith("_") or name == "components":
            raise ValueError(f"Invalid role name {name!r}")
        if name in self._roles:
            raise ValueError(f"Role {name!r} is already registered")
        if not spec.kind:
            raise ValueError("A role kind pattern must not be empty")
        if spec.via is not None:
            parts = spec.via.split(".")
            if len(parts) < 2 or any(
                not part.isidentifier() or part.startswith("_") for part in parts
            ):
                raise ValueError(f"Invalid role reference path {spec.via!r}")
        self._roles[name] = spec

    def unregister(self, name: str) -> None:
        """Remove a declaration; missing names raise KeyError."""
        del self._roles[name]

    def get(self, name: str) -> RoleSpec:
        """Return a registered declaration; unknown names raise ValueError."""
        try:
            return self._roles[name]
        except KeyError as cause:
            raise ValueError(f"Unknown role {name!r}") from cause


def _marked_references(model: type[BaseModel], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        if any(isinstance(marker, Ref) for marker in field.metadata):
            paths.append(path)
        nested_model = _nested_model(field.annotation)
        if nested_model is not None:
            paths.extend(_marked_references(nested_model, path))
    return tuple(paths)


class ComponentRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self._references: dict[str, tuple[str, ...]] = {}
        self.roles = RoleRegistry()

    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        """Register a unique document kind and its component schema.

        ``model`` must extend ComponentSchema with extra=forbid.
        ``references`` contains component-reference field paths. Annotated Ref
        fields are collected automatically, including nested paths. All
        model-level validators and custom model_post_init are rejected, including
        inherited and direct or nullable nested declarations. Cross-field checks
        are unsupported in this batch;
        field validators retain their conversions and canonical constraints.
        Defaults and default_factory retain Pydantic semantics.

        Raise ValueError for duplicates, invalid references or unsupported model
        hooks; raise TypeError for invalid model declarations. Failure does not
        reserve kind or execute a trial model.
        """
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        _validate_component_model(model)
        _validate_supported_model(model)
        reference_paths = tuple(
            dict.fromkeys((*references, *_marked_references(model)))
        )
        for reference in reference_paths:
            _validate_reference(model, reference)
        self._models[kind] = model
        self._references[kind] = reference_paths

    def unregister(self, kind: str) -> None:
        del self._models[kind]
        del self._references[kind]

    def get(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        try:
            return self._models[kind]
        except KeyError as cause:
            raise UnknownKindError(
                source, component, kind, tuple(get_close_matches(kind, self._models))
            ) from cause

    def check_fields(
        self, kind: str | type[BaseModel], fields: Mapping[str, object], *, path: str
    ) -> None:
        known_fields = (self.get(kind) if isinstance(kind, str) else kind).model_fields
        for name, value in fields.items():
            if name not in known_fields:
                raise UnknownFieldError(
                    f"{path}.{name}", name, tuple(get_close_matches(name, known_fields))
                )
            nested_model = _nested_model(known_fields[name].annotation)
            if isinstance(value, dict) and nested_model is not None:
                self.check_fields(nested_model, value, path=f"{path}.{name}")

    def validate_references(
        self, components: Mapping[str, ComponentSchema], *, source: Path
    ) -> None:
        """Reject supplied references that do not name a component in this document."""
        for name, component in components.items():
            self.get(component.kind, source=source, component=name)
            for field in self._references[component.kind]:
                target: object = component
                for part in field.split("."):
                    target = (
                        getattr(target, part, None)
                        if isinstance(target, BaseModel)
                        else None
                    )
                if isinstance(target, str) and target not in components:
                    raise MissingReferenceError(source, name, field, target)

    def references(self, kind: str) -> tuple[str, ...]:
        """Return registered reference paths, not arbitrary string fields."""
        self.get(kind)
        return self._references[kind]


component_registry = ComponentRegistry()
