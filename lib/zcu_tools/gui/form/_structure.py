"""Pydantic field traversal and transactional candidate construction.

Reflection stays here: model fields, union tags and mapping annotations define
which paths can be edited. Sequences never expose writable child paths.
"""

from collections.abc import Mapping
from types import UnionType
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from ._json import copy_json, copy_object, default_json
from .model import FormEnv, FormModel, FormPath, FormWritePath


def unwrap(annotation: object) -> object:
    """Expand named aliases and Annotated metadata to the value annotation."""
    while isinstance(annotation, TypeAliasType) or get_origin(annotation) is Annotated:
        annotation = (
            annotation.__value__
            if isinstance(annotation, TypeAliasType)
            else get_args(annotation)[0]
        )
    return annotation


def alternatives(annotation: object) -> tuple[object, ...]:
    """Return union members, or the one unwrapped non-union annotation."""
    annotation = unwrap(annotation)
    if get_origin(annotation) in (Union, UnionType):
        return tuple(
            member for item in get_args(annotation) for member in alternatives(item)
        )
    return (annotation,)


def model_types(annotation: object) -> tuple[type[BaseModel], ...]:
    """Return BaseModel branches from an annotation without instantiating them."""
    return tuple(
        item
        for item in alternatives(annotation)
        if isinstance(item, type) and issubclass(item, BaseModel)
    )


def discriminator(info: FieldInfo) -> str | None:
    """Read a string tag through aliases/nullable wrappers; absent tags return None."""
    if isinstance(info.discriminator, str):
        return info.discriminator
    annotation = info.annotation
    while True:
        annotation = _expand_alias(annotation)
        origin = get_origin(annotation)
        if origin is Annotated:
            args = get_args(annotation)
            for metadata in args[1:]:
                if isinstance(metadata, FieldInfo) and isinstance(
                    metadata.discriminator, str
                ):
                    return metadata.discriminator
            annotation = args[0]
            continue
        if origin in (Union, UnionType):
            members = tuple(
                item for item in get_args(annotation) if item is not type(None)
            )
            if len(members) == 1:
                annotation = members[0]
                continue
        return None


def _expand_alias(annotation: object) -> object:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _tag(model: type[BaseModel], key: str) -> tuple[object, ...]:
    info = model.model_fields.get(key)
    if info is None or get_origin(info.annotation) is not Literal:
        return ()
    return get_args(info.annotation)


def discriminator_values(model: type[BaseModel], key: str) -> tuple[JsonValue, ...]:
    """Return intrinsic Literal tags for model's discriminator field key.

    Preserve their ordinary JSON types independently of dynamic choices.
    Missing/non-Literal fields or non-finite/non-JSON tags raise ValueError.
    """
    tags = _tag(model, key)
    if not tags:
        raise ValueError(f"Discriminator field {key!r} must have Literal tags")
    return tuple(default_json(tag) for tag in tags)


def selected(info: FieldInfo, value: JsonValue) -> type[BaseModel] | None:
    """Select a sole model or an explicitly tagged model; otherwise return None."""
    models = model_types(info.annotation)
    if len(models) == 1:
        return models[0]
    key = discriminator(info)
    if key is not None and isinstance(value, dict):
        return next(
            (model for model in models if value.get(key) in _tag(model, key)), None
        )
    return None


def choice_fields(model: type[BaseModel], path: FormPath) -> tuple[FieldInfo, ...]:
    """Resolve a declared model-field leaf path across possible model branches.

    Unknown paths, model sections and container key/index paths raise ValueError.
    Nullable and tagged branches are structural declarations, not active input.
    """
    if not path or any(not isinstance(part, str) for part in path):
        raise ValueError(f"Invalid choice field path: {path!r}")
    fields = _choice_fields(model, path)
    if not fields:
        raise ValueError(f"Unknown or non-leaf choice field path: {path!r}")
    return fields


def _choice_fields(model: type[BaseModel], path: FormPath) -> tuple[FieldInfo, ...]:
    name, *tail = path
    if not isinstance(name, str) or (info := model.model_fields.get(name)) is None:
        return ()
    children = model_types(info.annotation)
    if tail:
        return tuple(
            field for child in children for field in _choice_fields(child, tuple(tail))
        )
    if children and all(
        isinstance(t, type) and (t is type(None) or issubclass(t, BaseModel))
        for t in alternatives(info.annotation)
    ):
        return ()
    return (info,)


def static_defaults(model: type[BaseModel]) -> dict[str, JsonValue]:
    """Encode declared Pydantic defaults; required fields remain absent."""
    return {
        name: default_json(info.get_default(call_default_factory=True))
        for name, info in model.model_fields.items()
        if not info.is_required()
    }


def _branch_defaults(
    model: type[BaseModel], env: FormEnv, previous: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    if issubclass(model, FormModel):
        return copy_object(model.defaults(env, copy_object(previous)))
    return static_defaults(model)


def _field_at(
    model: type[BaseModel], data: dict[str, JsonValue], path: FormWritePath
) -> tuple[dict[str, JsonValue], str, FieldInfo]:
    if not path:
        raise ValueError(f"Invalid write path: {path!r}")
    name, *tail = path
    info = model.model_fields.get(name)
    if info is None:
        raise ValueError(f"Unknown field path: {path!r}")
    if not tail:
        return data, name, info
    value = data.get(name)
    child = selected(info, value)
    if child is not None:
        if name not in data and type(None) not in alternatives(info.annotation):
            value = static_defaults(child)
            data[name] = value
        if not isinstance(value, dict):
            raise ValueError(f"Inactive, missing or invalid model path: {path!r}")
        return _field_at(child, value, tuple(tail))
    raise ValueError(
        f"Not a writable child path (mapping, sequence or scalar): {path!r}"
    )


def _switch(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> bool:
    if len(path) < 2:
        return False
    parent, key, info = _field_at(model, data, path[:-1])
    tag_key = discriminator(info)
    if tag_key is None or path[-1] != tag_key:
        return False
    old = parent.get(key)
    if key in parent and old is None:
        raise ValueError(f"Activate null section before child write: {path!r}")
    if isinstance(old, dict) and old.get(tag_key) == value:
        return True
    branch = next(
        (item for item in model_types(info.annotation) if value in _tag(item, tag_key)),
        None,
    )
    if branch is None:
        raise ValueError(f"Unknown discriminator at path {path!r}: {value!r}")
    previous = old if isinstance(old, dict) else {}
    try:
        new = _branch_defaults(branch, env, previous)
    except PydanticCustomError as exc:
        from ._sources import prefix_error

        raise prefix_error(exc, path[:-1]) from exc
    new.update(
        {
            name: previous[name]
            for name in branch.model_fields
            if name != tag_key and name in previous
        }
    )
    new[tag_key] = value
    parent[key] = new
    return True


def build_candidate(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    batch: Mapping[FormWritePath, JsonValue],
    env: FormEnv,
) -> dict[str, JsonValue]:
    """Build a detached discriminator-first batch; structural/source errors raise."""
    paths = tuple(batch)
    for path in paths:
        if not path:
            raise ValueError(f"Invalid write path: {path!r}")
        if any(path != other and other[: len(path)] == path for other in paths):
            raise ValueError(f"Overlapping paths: {path!r}")
    copied = {path: copy_json(value) for path, value in batch.items()}
    result = copy_object(data)
    pending: dict[FormWritePath, JsonValue] = {}
    for path in sorted(copied, key=len):
        if not _switch(model, result, path, copied[path], env):
            pending[path] = copied[path]
    for path, value in pending.items():
        parent, key, info = _field_at(model, result, path)
        _require_leaf(info, value, path)
        parent[key] = value
    return result


def _require_leaf(info: FieldInfo, value: JsonValue, path: FormWritePath) -> None:
    types = alternatives(info.annotation)
    if model_types(info.annotation):
        if type(None) in types and (value is None or isinstance(value, dict)):
            return
        # Spec unions also admit scalar reference keys; those remain ordinary leaves.
        if not isinstance(value, dict) and any(
            t is str or t is int or t is float or t is bool or t is complex
            for t in types
        ):
            return
        raise ValueError(f"Whole model replacement is not a leaf path: {path!r}")


def _field_leaves(
    info: FieldInfo, value: JsonValue, path: FormWritePath
) -> dict[FormWritePath, JsonValue]:
    child = selected(info, value)
    if child is not None and isinstance(value, dict):
        return leaves(child, value, path)
    return {path: value}


def leaves(
    model: type[BaseModel], data: Mapping[str, JsonValue], prefix: FormWritePath = ()
) -> dict[FormWritePath, JsonValue]:
    """Collect active model leaves, retaining whole containers and scalar tags."""
    result: dict[FormWritePath, JsonValue] = {}
    for key, value in data.items():
        info = model.model_fields.get(key)
        if info is not None:
            result.update(_field_leaves(info, value, (*prefix, key)))
    return result


def _compatible(info: FieldInfo, value: JsonValue) -> bool:
    types = alternatives(info.annotation)
    if value is None:
        return type(None) in types
    if isinstance(value, dict) and any(
        t is int or t is float or t is complex for t in types
    ):
        if set(value) == {"expr"}:
            return isinstance(value["expr"], str) and any(
                t is int or t is float or t is complex for t in types
            )
        if set(value) == {"__complex__"} and complex in types:
            from zcu_tools.gui.cfg.codec import decode_complex

            try:
                decode_complex(value)
            except ValueError:
                return False
            return True
    return any(_compatible_type(t, value) for t in types)


def _compatible_type(annotation: object, value: JsonValue) -> bool:
    annotation = unwrap(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return isinstance(value, dict) and all(
            name in annotation.model_fields
            and _compatible(annotation.model_fields[name], item)
            for name, item in value.items()
        )
    if get_origin(annotation) is Literal:
        return value in get_args(annotation)
    if annotation is float or annotation is complex or annotation is int:
        number_type = int if annotation is int else (int, float)
        return isinstance(value, number_type) and not isinstance(value, bool)
    if annotation is bool or annotation is str:
        return isinstance(value, bool if annotation is bool else str)
    if get_origin(annotation) in (list, tuple, dict, Mapping):
        return (
            _compatible_mapping(annotation, value)
            if get_origin(annotation) in (dict, Mapping)
            else _compatible_sequence(annotation, value)
        )
    return False


def _compatible_mapping(annotation: object, value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    args = get_args(annotation)
    return all(
        _compatible(FieldInfo.from_annotation(args[1]), item) for item in value.values()
    )


def _compatible_sequence(annotation: object, value: JsonValue) -> bool:
    if not isinstance(value, list):
        return False
    args = get_args(annotation)
    if not args:
        return True
    if get_origin(annotation) is tuple and args[-1] is not Ellipsis:
        return len(value) == len(args) and all(
            _compatible(FieldInfo.from_annotation(t), v)
            for t, v in zip(args, value, strict=True)
        )
    return all(_compatible(FieldInfo.from_annotation(args[0]), item) for item in value)


def _restore_field(
    info: FieldInfo,
    current: JsonValue,
    saved: JsonValue,
    env: FormEnv,
    path: FormPath,
    dropped: list[FormPath],
) -> JsonValue:
    tag = discriminator(info)
    child = selected(info, saved)
    if isinstance(saved, dict) and tag is not None and tag not in saved:
        child = selected(info, current)
    if child is not None and isinstance(saved, dict):
        prior = current if isinstance(current, dict) else {}
        if tag is not None and tag in saved and saved[tag] != prior.get(tag):
            try:
                seed = _branch_defaults(child, env, prior)
            except PydanticCustomError as exc:
                from ._sources import prefix_error

                raise prefix_error(exc, path) from exc
            seed.update(
                {
                    name: prior[name]
                    for name in child.model_fields
                    if name != tag and name in prior
                }
            )
            seed[tag] = saved[tag]
        else:
            seed = (
                copy_object(prior)
                if isinstance(current, dict)
                else static_defaults(child)
            )
        return _restore_model(child, seed, saved, env, path, dropped)
    if isinstance(saved, dict) and model_types(info.annotation):
        dropped.append(path)
        return current
    if _compatible(info, saved):
        return copy_json(saved)
    dropped.append(path)
    return current


def _restore_model(
    model: type[BaseModel],
    current: dict[str, JsonValue],
    saved: Mapping[str, JsonValue],
    env: FormEnv,
    prefix: FormPath,
    dropped: list[FormPath],
) -> dict[str, JsonValue]:
    for name, value in saved.items():
        path = (*prefix, name)
        info = model.model_fields.get(name)
        if info is None:
            dropped.append(path)
        else:
            previous = current.get(name)
            restored = _restore_field(info, previous, value, env, path, dropped)
            if path not in dropped:
                current[name] = restored
    return current


def restore(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    saved: Mapping[str, JsonValue],
    env: FormEnv,
) -> tuple[dict[str, JsonValue], tuple[FormPath, ...]]:
    """Overlay type-compatible saved leaves, retaining model-invalid values."""
    dropped: list[FormPath] = []
    result = _restore_model(model, copy_object(data), saved, env, (), dropped)
    return result, tuple(dropped)
