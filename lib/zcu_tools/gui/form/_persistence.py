"""Lossless draft envelopes, identified by the trusted annotation contract."""

from enum import Enum
from hashlib import sha256
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from ._selection import literal_values
from ._structure import alternatives, discriminator, model_types, selected
from .model import FormModel, FormWritePath


class _Selection(BaseModel):
    """An explicit choice, independent of the branch stored in raw input."""

    model_config = ConfigDict(extra="forbid", strict=True)
    path: FormWritePath
    value: JsonValue


class _Draft(BaseModel):
    """Only data and a declaration digest cross the persistence boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    identity: str
    data: dict[str, JsonValue]
    selections: tuple[_Selection, ...]


def _qualified_name(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _annotation_identity(
    annotation: object, ancestors: tuple[type[BaseModel], ...]
) -> str:
    members = alternatives(annotation)
    if len(members) != 1:
        return repr(
            tuple(_annotation_identity(member, ancestors) for member in members)
        )
    member = members[0]
    if isinstance(member, type) and issubclass(member, BaseModel):
        return _model_identity(member, ancestors)
    if isinstance(member, type) and issubclass(member, Enum):
        return repr((_qualified_name(member), tuple(item.value for item in member)))
    origin = get_origin(member)
    if origin is Literal:
        return repr(("Literal", get_args(member)))
    if origin is not None:
        return repr(
            (
                str(origin),
                tuple(_annotation_identity(arg, ancestors) for arg in get_args(member)),
            )
        )
    return _qualified_name(member) if isinstance(member, type) else repr(member)


def _model_identity(
    model: type[BaseModel], ancestors: tuple[type[BaseModel], ...]
) -> str:
    name = _qualified_name(model)
    if model in ancestors:
        return name
    fields = tuple(
        (
            key,
            _annotation_identity(info.annotation, (*ancestors, model)),
            discriminator(info),
        )
        for key, info in sorted(model.model_fields.items())
    )
    return repr((name, fields))


def declaration_identity(model: type[FormModel]) -> str:
    """Hash nominal types, fields and branch tags, not defaults or callables.

    Equivalent regenerated classes have the same identity. Source contents,
    validators, converters and dynamic class objects are never serialized.
    The digest is a compatibility check, not an executable schema.
    """
    return sha256(_annotation_identity(model, ()).encode("utf-8")).hexdigest()


def encode(
    input_model: type[FormModel],
    data: dict[str, JsonValue],
    selections: dict[FormWritePath, JsonValue],
) -> str:
    """Encode an authored tree and independent choices without evaluation."""
    return _Draft(
        version=1,
        identity=declaration_identity(input_model),
        data=data,
        selections=tuple(
            _Selection(path=path, value=value)
            for path, value in sorted(selections.items())
        ),
    ).model_dump_json()


def decode(
    model: type[FormModel], text: str
) -> tuple[dict[str, JsonValue], dict[FormWritePath, JsonValue]]:
    """Reject mismatched declarations; return untouched raw input and choices."""
    try:
        draft = _Draft.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"Invalid JSON draft envelope: {exc}") from exc
    if draft.identity != declaration_identity(model):
        raise ValueError(
            "Saved draft type identity is incompatible with the trusted declaration"
        )
    selections: dict[FormWritePath, JsonValue] = {}
    for selection in draft.selections:
        if not selection.path or selection.path in selections:
            raise ValueError("Invalid or duplicate saved selection path")
        selections[selection.path] = selection.value
    _check_model(model, draft.data)
    for path, value in selections.items():
        info = _selection_field(model, draft.data, path)
        check_selection(info, value)
    return draft.data, selections


def _branch(info: FieldInfo, value: JsonValue) -> type[BaseModel] | None:
    models = model_types(info.annotation)
    if not models:
        return None
    if value is None and type(None) in alternatives(info.annotation):
        return None
    tag = discriminator(info)
    if tag is None and len(models) > 1:
        return None  # The declaration has no structural selector to reconstruct.
    branch = selected(info, value)
    if not isinstance(value, dict) or branch is None:
        raise ValueError("Saved branch identity is incompatible with its annotation")
    if tag is not None:
        values = literal_values(branch.model_fields[tag].annotation)
        if values is None or not any(
            type(value.get(tag)) is type(item) and value.get(tag) == item
            for item in values
        ):
            raise ValueError(
                "Saved branch tag identity is incompatible with its annotation"
            )
    return branch


def _check_model(model: type[BaseModel], data: dict[str, JsonValue]) -> None:
    for name, value in data.items():
        info = model.model_fields.get(name)
        if info is None:
            continue  # Unknown raw leaves remain visible invalid input, never dropped.
        branch = _branch(info, value)
        if branch is not None and isinstance(value, dict):
            _check_model(branch, value)


def _selection_field(
    model: type[BaseModel], data: dict[str, JsonValue], path: FormWritePath
) -> FieldInfo:
    name, *remaining = path
    if not isinstance(name, str) or (info := model.model_fields.get(name)) is None:
        raise ValueError(f"Saved selection path identity is incompatible: {path!r}")
    if not remaining:
        return info
    value = data.get(name)
    branch = _branch(info, value)
    if branch is None or not isinstance(value, dict):
        raise ValueError(f"Saved selection ancestry is incompatible: {path!r}")
    return _selection_field(branch, value, tuple(remaining))


def check_selection(info: FieldInfo, value: JsonValue) -> None:
    """Reject a choice outside annotation structure, without hooks or evaluation.

    info is the trusted target field; value is its partial explicit override.
    Model branch tags and provided children must fit the annotation. Ordinary
    scalar choices must have the declared type. Raise ValueError on mismatch.
    """
    branch = _branch(info, value)
    if branch is not None and isinstance(value, dict):
        for name, child in value.items():
            field = branch.model_fields.get(name)
            if field is None:
                raise ValueError(
                    f"Saved selection field identity is incompatible: {name!r}"
                )
            check_selection(field, child)
        return
    values = literal_values(info.annotation)
    if values is not None:
        if not any(type(value) is type(item) and value == item for item in values):
            raise ValueError("Saved selection Literal identity is incompatible")
        return
    try:
        TypeAdapter(info.annotation).validate_python(value, strict=True)
    except ValidationError as exc:
        raise ValueError(
            "Saved selection identity is incompatible with its annotation"
        ) from exc
