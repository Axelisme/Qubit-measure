"""Pydantic field traversal and transactional candidate construction.

Reflection stays here: model fields, union tags and mapping annotations define
which paths can be edited. Only FixedList sequences expose indexed model paths.
"""

from collections.abc import Mapping
from enum import Enum
from types import UnionType
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from ._json import copy_json, copy_object, default_json
from .defaults import static_defaults
from .model import (
    FixedList,
    FormEnv,
    FormPath,
    FormWritePath,
    InputConverter,
    InputPolicy,
    Participants,
)


class StructureError(ValueError):
    """Invalid write structure, distinct from author/provider ValueErrors.

    path identifies the canonical batch write that failed, not a position inside
    its value. build_candidate preserves that full path through nested traversal
    and partial-object expansion. No candidate is installed on failure.
    """

    def __init__(self, message: str, *, path: FormWritePath) -> None:
        """Store the diagnostic message and the failing canonical write path."""
        super().__init__(message)
        self.path = path


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


def accepts_expression(annotation: object, path: FormPath) -> bool:
    """Whether a value leaf of this annotation is a text field taking Expr input.

    This is the core's one text-field rule: the description permits expression
    input. Decode and defaults evaluation accept an Expr envelope exactly
    where it returns True. Restoration retains raw for general validation. Union members
    are text values (int, float, complex, str), typed into a text field, or
    picked values (Literal, Enum, bool), chosen from a dropdown or checkbox.
    A text field has a text member and no picked member, possibly nullable.
    Two leaves this annotation check cannot see are not text fields: a
    leaf with declared choices (by path) and a field restricted by InputPolicy.

    path is the leaf's field path, used only in the error. A union with both
    text and picked members raises ValueError naming path: the editor has no
    control that types a value and picks one.
    """
    members = alternatives(annotation)
    picked = any(
        get_origin(item) is Literal
        or item is bool
        or (isinstance(item, type) and issubclass(item, Enum))
        for item in members
    )
    text = any(
        item is int or item is float or item is complex or item is str
        for item in members
    )
    if text and picked:
        dotted = ".".join(str(part) for part in path)
        raise ValueError(
            f"{dotted}: a union of text and picked values is not supported"
        )
    return text


def expression_allowed(info: FieldInfo, path: FormPath) -> bool:
    """Return the annotation capability restricted by explicit InputPolicy."""
    capable = accepts_expression(info.annotation, path)
    return capable and not any(
        isinstance(item, InputPolicy) and not item.expression for item in info.metadata
    )


def model_types(annotation: object) -> tuple[type[BaseModel], ...]:
    """Return BaseModel branches from an annotation without instantiating them."""
    return tuple(
        item
        for item in alternatives(annotation)
        if isinstance(item, type) and issubclass(item, BaseModel)
    )


def fixed_list(info: FieldInfo) -> FixedList | None:
    """Return the FixedList metadata declared on info, if any."""
    return next((m for m in info.metadata if isinstance(m, FixedList)), None)


def participants(info: FieldInfo) -> Participants | None:
    """Return the Participants metadata declared on info, if any."""
    return next((m for m in info.metadata if isinstance(m, Participants)), None)


def element_model(info: FieldInfo) -> type[BaseModel]:
    """Return the element model of a validated FixedList field."""
    (item,) = get_args(unwrap(info.annotation))
    return model_types(item)[0]


def _check_fixed_declaration(
    model: type[BaseModel], name: str, info: FieldInfo, path: FormPath
) -> None:
    """Reject FixedList/Participants declarations that cannot hold their contract."""
    dotted = ".".join(str(part) for part in (*path, name))
    fixed = fixed_list(info)
    if fixed is not None:
        annotation = unwrap(info.annotation)
        args = get_args(annotation)
        item = unwrap(args[0]) if len(args) == 1 else None
        if (
            get_origin(annotation) is not list
            or not isinstance(item, type)
            or not issubclass(item, BaseModel)
        ):
            raise ValueError(f"{dotted}: FixedList requires a list of one model")
        key = item.model_fields.get(fixed.name_field)
        if key is None or unwrap(key.annotation) is not str:
            raise ValueError(
                f"{dotted}: FixedList name_field {fixed.name_field!r} "
                f"must be a str field of {item.__name__}"
            )
    participating = participants(info)
    if participating is not None:
        target = _field_by_path(model, participating.list_path)
        if (
            unwrap(info.annotation) != list[str]
            or target is None
            or fixed_list(target) is None
        ):
            raise ValueError(
                f"{dotted}: Participants requires list[str] naming "
                f"the FixedList at {participating.list_path!r}"
            )


def _field_by_path(model: type[BaseModel], path: tuple[str, ...]) -> FieldInfo | None:
    """Resolve model-field names through sole-model fields, or None."""
    info: FieldInfo | None = None
    for position, name in enumerate(path):
        info = model.model_fields.get(name)
        if info is None:
            return None
        models = model_types(info.annotation)
        if position < len(path) - 1:
            if len(models) != 1:
                return None
            model = models[0]
    return info


def check_form_shape(model: type[BaseModel]) -> None:
    """Reject retired str-or-model selectors throughout a form's declaration.

    Inspect all branches, including inactive and nullable sections, before input
    or source hooks run. Ordinary models and model-only unions remain supported.
    """
    pending: list[tuple[object, FormPath]] = [(model, ())]
    seen: set[object] = set()
    while pending:
        annotation, path = pending.pop()
        members = alternatives(annotation)
        if str in members and model_types(annotation):
            dotted = ".".join(str(part) for part in path)
            raise ValueError(
                f"{dotted}: untagged str | Model input is not supported; "
                "use a tagged ref/inline union"
            )
        for member in members:
            if isinstance(member, type) and issubclass(member, BaseModel):
                if member not in seen:
                    seen.add(member)
                    for field_name, info in member.model_fields.items():
                        _check_fixed_declaration(member, field_name, info, path)
                        if (
                            sum(
                                isinstance(item, InputConverter)
                                for item in info.metadata
                            )
                            > 1
                        ):
                            raise ValueError("A field permits only one converter")
                    pending.extend(
                        (info.annotation, (*path, name))
                        for name, info in member.model_fields.items()
                    )
            elif (
                get_origin(member) in (list, tuple, dict, Mapping)
                and member not in seen
            ):
                seen.add(member)
                pending.extend((item, path) for item in get_args(member))


def discriminator(info: FieldInfo) -> str | None:
    """Read a string tag through aliases/nullable wrappers; absent tags return None."""
    if isinstance(info.discriminator, str):
        return info.discriminator
    return union_tag(info.annotation)


def union_tag(annotation: object) -> str | None:
    """Read the string tag declared in annotation itself, as discriminator does.

    Use this for an annotation outside a model field, such as a union alias;
    a field's own Field(discriminator=...) needs discriminator(info).
    """
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
    """Read native Literal tags through named aliases and unions of Literals."""
    info = model.model_fields.get(key)
    if info is None:
        return ()
    return tuple(
        tag
        for annotation in alternatives(info.annotation)
        if get_origin(annotation) is Literal
        for tag in get_args(annotation)
    )


def intrinsic_tags(model: type[BaseModel], key: str) -> tuple[JsonValue, ...]:
    """Read typed JSON tags from model's Literal declaration, not dynamic choices.

    Named aliases, Annotated wrappers and unions of Literals share the native
    tag reader. Non-JSON/non-finite values fail through the shared JSON codec.
    """
    return tuple(
        default_json(tag.value if isinstance(tag, Enum) else tag)
        for tag in _tag(model, key)
    )


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


def branch_tags(roots: tuple[object, ...]) -> dict[type[BaseModel], str]:
    """Map each tagged-union branch model reachable from roots to its tag key.

    roots are annotations, such as a form class or a tagged-union alias. Model
    fields are followed through every reached model, so a union nested in a
    branch (a module's waveform) contributes too; sequence and mapping items
    are not followed. A model branched on several keys keeps the first found.
    """
    tags: dict[type[BaseModel], str] = {}
    seen: set[type[BaseModel]] = set()
    # Each pending item is an annotation with the tag key that selects it.
    pending = [(root, union_tag(root)) for root in roots]
    while pending:
        annotation, key = pending.pop()
        for model in model_types(annotation):
            if key is not None:
                tags.setdefault(model, key)
            if model not in seen:
                seen.add(model)
                pending.extend(
                    (info.annotation, discriminator(info))
                    for info in model.model_fields.values()
                )
    return tags


def fixed_tag(
    model: type[BaseModel], tags: Mapping[type[BaseModel], str]
) -> str | None:
    """Return model's one-value structural tag field, or None.

    tags is a branch_tags result. The first class in model's MRO that tags
    lists gives the tag key, so a subclass of a branch keeps its tag. Return
    that key when model's field under it is a Literal of exactly one value;
    return None for an unlisted model or a tag field with several values.
    """
    key = next(
        (
            tags[base]
            for base in model.__mro__
            if issubclass(base, BaseModel) and base in tags
        ),
        None,
    )
    if key is None or len(_tag(model, key)) != 1:
        return None
    return key


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


def _field_at(
    model: type[BaseModel], data: dict[str, JsonValue], path: FormWritePath
) -> tuple[dict[str, JsonValue], str, FieldInfo]:
    """Resolve a writable field path to its containing object, name and metadata."""
    if not path:
        raise StructureError(f"Invalid write path: {path!r}", path=path)
    name, *tail = path
    if not isinstance(name, str) or (info := model.model_fields.get(name)) is None:
        raise StructureError(f"Unknown field path: {path!r}", path=path)
    if not tail:
        return data, name, info
    value = data.get(name)
    if fixed_list(info) is not None:
        index, *rest = tail
        if (
            not isinstance(value, list)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(value)
        ):
            raise StructureError(f"Unknown fixed element path: {path!r}", path=path)
        element = element_model(info)
        if not rest:
            raise StructureError(
                f"Fixed element is not a leaf path: {path!r}", path=path
            )
        item = value[index]
        if not isinstance(item, dict):
            raise StructureError(f"Invalid fixed element at path: {path!r}", path=path)
        return _field_at(element, item, tuple(rest))
    child = selected(info, value)
    if child is not None:
        if name not in data and type(None) not in alternatives(info.annotation):
            value = static_defaults(child)
            data[name] = value
        if not isinstance(value, dict):
            raise StructureError(
                f"Inactive, missing or invalid model path: {path!r}", path=path
            )
        return _field_at(child, value, tuple(tail))
    raise StructureError(
        f"Not a writable child path (mapping, sequence or scalar): {path!r}", path=path
    )


def _switch(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: FormWritePath,
    value: JsonValue,
    _env: FormEnv,
) -> bool:
    if len(path) < 2 or isinstance(path[-2], int):
        return False
    parent, key, info = _field_at(model, data, path[:-1])
    tag_key = discriminator(info)
    if tag_key is None or path[-1] != tag_key:
        return False
    old = parent.get(key)
    if key in parent and old is None:
        raise StructureError(
            f"Activate null section before child write: {path!r}", path=path
        )
    if isinstance(old, dict) and old.get(tag_key) == value:
        return True
    branch = next(
        (item for item in model_types(info.annotation) if value in _tag(item, tag_key)),
        None,
    )
    if branch is None:
        raise StructureError(
            f"Unknown discriminator at path {path!r}: {value!r}", path=path
        )
    new: dict[str, JsonValue] = {tag_key: value}
    parent[key] = new
    return True


def _select_object(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> bool:
    """Admit explicitly authored union content without generating defaults.

    Declared option/tag selections are handled by _transitions first. This
    structural fallback also admits invalid drafts outside dynamic option menus.
    """
    if not isinstance(value, dict):
        return False
    parent, key, info = _field_at(model, data, path)
    tag = discriminator(info)
    if tag is None:
        return False
    old = parent.get(key)
    branch = selected(info, value if tag in value else old)
    if branch is None:
        raise StructureError(f"Missing or unknown discriminator at {path!r}", path=path)
    seed = (
        copy_object(old)
        if isinstance(old, dict) and selected(info, old) is branch
        else {}
    )
    parent[key] = _partial_model(branch, seed, value, env)
    return True


def _partial_model(
    model: type[BaseModel],
    seed: dict[str, JsonValue],
    value: dict[str, JsonValue],
    env: FormEnv,
) -> dict[str, JsonValue]:
    """Expand partial model children to one candidate; containers remain leaves."""
    batch: dict[FormWritePath, JsonValue] = {}
    for name, item in value.items():
        info = model.model_fields.get(name)
        if info is None:
            raise StructureError(
                f"Unknown field in partial branch object: {name!r}", path=(name,)
            )
        if (
            isinstance(item, dict)
            and model_types(info.annotation)
            and discriminator(info) is None
        ):
            child = selected(info, item)
            if child is None:
                raise StructureError(
                    f"Ambiguous partial model field: {name!r}", path=(name,)
                )
            prior = seed.get(name)
            seed[name] = _partial_model(
                child,
                copy_object(prior) if isinstance(prior, dict) else {},
                item,
                env,
            )
        else:
            batch[(name,)] = item
    return build_candidate(model, seed, batch, env)


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
            raise StructureError(f"Invalid write path: {path!r}", path=path)
        if any(path != other and other[: len(path)] == path for other in paths):
            raise StructureError(f"Overlapping paths: {path!r}", path=path)
    copied = {path: copy_json(value) for path, value in batch.items()}
    result = copy_object(data)
    pending: dict[FormWritePath, JsonValue] = {}
    for path in sorted(copied, key=len):
        value = copied[path]
        try:
            if not (
                _switch(model, result, path, value, env)
                or _select_object(model, result, path, value, env)
            ):
                pending[path] = value
        except StructureError as exc:
            # Nested traversal/partial objects report local paths; retain the
            # original batch key so callers can identify the consuming write.
            exc.path = path
            raise
    for path, value in pending.items():
        try:
            parent, key, info = _field_at(model, result, path)
            _require_leaf(info, value, path)
            parent[key] = value
        except StructureError as exc:
            exc.path = path
            raise
    return result


def _require_leaf(info: FieldInfo, value: JsonValue, path: FormWritePath) -> None:
    types = alternatives(info.annotation)
    if model_types(info.annotation):
        if type(None) in types and (value is None or isinstance(value, dict)):
            return
        # Other mixed scalar/model annotations retain ordinary scalar leaves.
        if not isinstance(value, dict) and any(
            t is str or t is int or t is float or t is bool or t is complex
            for t in types
        ):
            return
        raise StructureError(
            f"Whole model replacement is not a leaf path: {path!r}", path=path
        )


def _field_leaves(
    info: FieldInfo, value: JsonValue, path: FormWritePath
) -> dict[FormWritePath, JsonValue]:
    child = selected(info, value)
    if child is not None and isinstance(value, dict):
        return leaves(child, value, path)
    if (
        fixed_list(info) is not None
        and isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
    ):
        element = element_model(info)
        result: dict[FormWritePath, JsonValue] = {}
        for index, item in enumerate(value):
            if isinstance(item, dict):
                result.update(leaves(element, item, (*path, index)))
        return result
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
