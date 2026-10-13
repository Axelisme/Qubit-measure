"""Pydantic field traversal and transactional candidate construction.

Reflection stays here: model fields, union tags and mapping annotations define
which paths can be edited. Sequences never expose writable child paths.
"""

from collections.abc import Mapping
from enum import Enum
from types import UnionType
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from ._json import copy_json, copy_object, default_json, expr_text
from .defaults import static_defaults
from .model import FormEnv, FormModel, FormPath, FormWritePath, ModuleRef


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

    This is the core's one text-field rule: the input schema offers an Expr
    alternative, and decode, defaults evaluation and compatible restore
    accept an Expr envelope, exactly where it returns True. Union members
    are text values (int, float, complex, str), typed into a text field, or
    picked values (Literal, Enum, bool), chosen from a dropdown or checkbox.
    A text field has a text member and no picked member, possibly nullable.
    A str-or-model key (see key_or_inline) is not a text field: an Expr
    envelope is an object, and an object selects the inline model. Two
    leaves this annotation check cannot see are not text fields either: a
    leaf with declared choices (by path) and a ModuleRef name (see
    library_key).

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
    return text and not key_or_inline(annotation)


def library_key(model: type[BaseModel], name: str) -> bool:
    """Whether field name of model holds a library key: a ModuleRef's name.

    A library key is a plain string, never an Expr envelope, like the key
    of a str-or-model field.
    """
    return issubclass(model, ModuleRef) and name == "name"


def model_types(annotation: object) -> tuple[type[BaseModel], ...]:
    """Return BaseModel branches from an annotation without instantiating them."""
    return tuple(
        item
        for item in alternatives(annotation)
        if isinstance(item, type) and issubclass(item, BaseModel)
    )


def key_or_inline(annotation: object) -> bool:
    """Whether annotation is an untagged union of str and model branches.

    Its JSON type is the structural selector: a string is a library key and
    an object is the inline model, a sole model or one branch of a tagged union.
    """
    return str in alternatives(annotation) and bool(model_types(annotation))


def discriminator(info: FieldInfo) -> str | None:
    """Read a string tag through aliases/nullable wrappers; absent tags return None.

    A str-or-model union reads the tag of its inline part, so an inline
    object selects its tagged branch like any tagged model field.
    """
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
            if str in members and len(members) > 1:
                # The key alternative selects no branch; the inline part may.
                members = tuple(item for item in members if item is not str)
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


def schema_branch_model(info: FieldInfo, tags: list[JsonValue]) -> type[BaseModel]:
    """Match emitted intrinsic JSON tags to info's declared union model.

    info must declare a discriminator field name for a model union. tags are its emitted
    x-form-discriminator-values, not dynamic choice enums. Declared Literal tags
    may use named aliases, Annotated wrappers or unions of Literals. Match both
    JSON type and value. Raise ValueError for a missing discriminator or unmatched tags;
    declared non-JSON/non-finite tags also fail through the shared JSON codec.
    """
    key = discriminator(info)
    if key is None:
        raise ValueError("Schema branch requires a string discriminator")
    for model in model_types(info.annotation):
        for tag in _tag(model, key):
            value = default_json(tag.value if isinstance(tag, Enum) else tag)
            if any(
                type(candidate) is type(value) and candidate == value
                for candidate in tags
            ):
                return model
    raise ValueError("Tagged schema branch does not match a declared model")


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


def _branch_defaults(
    model: type[BaseModel], env: FormEnv, previous: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    if issubclass(model, FormModel):
        from ._evaluate import evaluate_defaults

        return evaluate_defaults(model, model.defaults(env, copy_object(previous)), env)
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
    try:
        new = _branch_defaults(branch, env, {})
    except PydanticCustomError as exc:
        from ._sources import prefix_error

        raise prefix_error(exc, path[:-1]) from exc
    new[tag_key] = value
    parent[key] = new
    return True


def _switch_inline(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> bool:
    """Treat {} at a str-or-model field as the switch to its inline model.

    A key or unfilled value becomes the inline defaults; an inline object
    stays unchanged. Other values return False and remain leaf writes.
    """
    if value != {}:
        return False
    parent, key, info = _field_at(model, data, path)
    if not key_or_inline(info.annotation):
        return False
    if not isinstance(parent.get(key), dict):
        parent[key] = _inline_defaults(info, env, path)
    return True


def _select_object(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> bool:
    """Apply a union-position partial object as fresh branch selection plus leaves."""
    if not isinstance(value, dict):
        return False
    parent, key, info = _field_at(model, data, path)
    tag = discriminator(info)
    if tag is None and not key_or_inline(info.annotation):
        return False
    old = parent.get(key)
    branch = selected(info, value if tag in value else old)
    if branch is None:
        raise ValueError(f"Missing or unknown discriminator at {path!r}")
    try:
        if not isinstance(old, dict) or selected(info, old) is not branch:
            seed = _branch_defaults(branch, env, {})
        else:
            seed = copy_object(old)
        parent[key] = _partial_model(branch, seed, value, env)
    except PydanticCustomError as exc:
        from ._sources import prefix_error

        raise prefix_error(exc, path) from exc
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
            raise ValueError(f"Unknown field in partial branch object: {name!r}")
        if (
            isinstance(item, dict)
            and model_types(info.annotation)
            and discriminator(info) is None
            and not key_or_inline(info.annotation)
        ):
            child = selected(info, item)
            if child is None:
                raise ValueError(f"Ambiguous partial model field: {name!r}")
            prior = seed.get(name)
            try:
                seed[name] = _partial_model(
                    child,
                    copy_object(prior)
                    if isinstance(prior, dict)
                    else _branch_defaults(child, env, {}),
                    item,
                    env,
                )
            except PydanticCustomError as exc:
                from ._sources import prefix_error

                raise prefix_error(exc, (name,)) from exc
        else:
            batch[(name,)] = item
    return build_candidate(model, seed, batch, env)


def _inline_defaults(
    info: FieldInfo, env: FormEnv, path: FormWritePath
) -> dict[str, JsonValue]:
    """Defaults of a str-or-model field's inline model, never its old key.

    A tagged inline union starts at its first declared branch and tag. An
    untagged choice between several models has no default branch and raises.
    """
    branch, *others = model_types(info.annotation)
    tag_key = discriminator(info)
    if others and tag_key is None:
        raise ValueError(f"Ambiguous untagged inline model at path {path!r}")
    try:
        new = _branch_defaults(branch, env, {})
    except PydanticCustomError as exc:
        from ._sources import prefix_error

        raise prefix_error(exc, path) from exc
    if tag_key is not None:
        tag = _tag(branch, tag_key)[0]
        new[tag_key] = default_json(tag.value if isinstance(tag, Enum) else tag)
    return new


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
        value = copied[path]
        if not (
            _switch(model, result, path, value, env)
            or _switch_inline(model, result, path, value, env)
            or _select_object(model, result, path, value, env)
        ):
            pending[path] = value
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


def _compatible(info: FieldInfo, value: JsonValue, path: FormPath) -> bool:
    """Whether value has a shape info's annotation accepts; path names the leaf.

    A union of text and picked values raises ValueError (see
    accepts_expression).
    """
    text_field = accepts_expression(info.annotation, path)
    # Containers are whole leaves, but nested selectors still require ordinary tags.
    tag = discriminator(info)
    if tag is not None and isinstance(value, dict) and isinstance(value.get(tag), dict):
        return False
    types = alternatives(info.annotation)
    if value is None:
        return type(None) in types
    if isinstance(value, dict) and text_field:
        if expr_text(value) is not None:
            return True
        if set(value) == {"__complex__"} and complex in types:
            from zcu_tools.gui.cfg.codec import decode_complex

            try:
                decode_complex(value)
            except ValueError:
                return False
            return True
    return any(_compatible_type(t, value, path) for t in types)


def _compatible_type(annotation: object, value: JsonValue, path: FormPath) -> bool:
    annotation = unwrap(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return isinstance(value, dict) and all(
            name in annotation.model_fields
            and _compatible(annotation.model_fields[name], item, (*path, name))
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
            _compatible_mapping(annotation, value, path)
            if get_origin(annotation) in (dict, Mapping)
            else _compatible_sequence(annotation, value, path)
        )
    return False


def _compatible_mapping(annotation: object, value: JsonValue, path: FormPath) -> bool:
    if not isinstance(value, dict):
        return False
    item_info = FieldInfo.from_annotation(get_args(annotation)[1])
    return all(
        _compatible(item_info, item, (*path, key)) for key, item in value.items()
    )


def _compatible_sequence(annotation: object, value: JsonValue, path: FormPath) -> bool:
    if not isinstance(value, list):
        return False
    args = get_args(annotation)
    if not args:
        return True
    if get_origin(annotation) is tuple and args[-1] is not Ellipsis:
        return len(value) == len(args) and all(
            _compatible(FieldInfo.from_annotation(t), v, (*path, index))
            for index, (t, v) in enumerate(zip(args, value, strict=True))
        )
    item_info = FieldInfo.from_annotation(args[0])
    return all(
        _compatible(item_info, item, (*path, index)) for index, item in enumerate(value)
    )


def _restore_child_model(
    info: FieldInfo,
    current: JsonValue,
    saved: JsonValue,
    tag: str | None,
    path: FormPath,
) -> type[BaseModel] | None:
    """Select a saved branch without coercion; omitted tags keep the current branch."""
    child = selected(info, saved)
    if child is None and tag is None and isinstance(saved, dict):
        # Raw shape, not evaluation/Pydantic coercion, identifies an untagged
        # branch only when exactly one declared model is compatible.
        matches = tuple(
            model
            for model in model_types(info.annotation)
            if _compatible_type(model, saved, path)
        )
        if len(matches) == 1:
            child = matches[0]
    if isinstance(saved, dict) and tag is not None and tag not in saved:
        child = selected(info, current)
    return child


def _restore_branch_seed(
    child: type[BaseModel],
    current: JsonValue,
    saved: dict[str, JsonValue],
    tag: str | None,
    env: FormEnv,
    path: FormPath,
) -> dict[str, JsonValue]:
    """Inherit current fields, using branch defaults only when the tag changes."""
    prior = current if isinstance(current, dict) else {}
    if tag is None or tag not in saved or saved[tag] == prior.get(tag):
        return (
            copy_object(prior) if isinstance(current, dict) else static_defaults(child)
        )
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
    return seed


def _restore_field(
    info: FieldInfo,
    current: JsonValue,
    saved: JsonValue,
    env: FormEnv,
    path: FormPath,
    dropped: list[FormPath],
) -> JsonValue:
    tag = discriminator(info)
    child = _restore_child_model(info, current, saved, tag, path)
    if child is not None and isinstance(saved, dict):
        # Structural selectors cannot accept Expr or complex envelopes.
        if tag is not None and isinstance(saved.get(tag), dict):
            dropped.append((*path, tag))
            saved = {name: value for name, value in saved.items() if name != tag}
        seed = _restore_branch_seed(child, current, saved, tag, env, path)
        return _restore_model(child, seed, saved, env, path, dropped)
    if isinstance(saved, dict) and model_types(info.annotation):
        dropped.append(path)
        return current
    if _compatible(info, saved, path):
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
        if info is None or (library_key(model, name) and isinstance(value, dict)):
            # A library key is a plain string, so an envelope cannot restore.
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


def _object_at(
    data: dict[str, JsonValue], names: list[str]
) -> dict[str, JsonValue] | None:
    for name in names:
        value = data.get(name)
        if not isinstance(value, dict):
            return None
        data = value
    return data


def revert_leaf(
    model: type[BaseModel],
    candidate: dict[str, JsonValue],
    current: dict[str, JsonValue],
    path: FormPath,
) -> FormWritePath | None:
    """Give the active candidate leaf containing path its current input value.

    path may point inside a whole sequence/mapping leaf; that whole leaf reverts.
    current is the input before restoration and is not edited. Its value at the
    same leaf path is copied into candidate in place; when current has none, the
    leaf key is removed. Return the reverted leaf, or None when no active leaf of
    candidate contains path (model-wide paths and absent default fields).
    """
    leaf = next(
        (key for key in leaves(model, candidate) if path[: len(key)] == key), None
    )
    if leaf is None:
        return None
    *parents, name = leaf
    target = _object_at(candidate, parents)
    # leaves() descends only through objects; None would mean a traversal bug.
    if target is None:
        raise ValueError(f"Active leaf {leaf!r} is not inside input objects")
    source = _object_at(current, parents)
    if source is not None and name in source:
        target[name] = copy_json(source[name])
    else:
        del target[name]
    return leaf
