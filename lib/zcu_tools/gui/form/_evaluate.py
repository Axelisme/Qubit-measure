"""Input evaluation and Pydantic error translation at the form boundary."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from zcu_tools.gui.cfg.codec import decode_complex
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr

from ._json import copy_object, default_json
from ._sources import check_sources, error_path
from ._structure import (
    accepts_expression,
    alternatives,
    discriminator,
    key_or_inline,
    leaves,
    selected,
    unwrap,
)
from .model import (
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    FormPreview,
    FormValue,
    FormWritePath,
)


@dataclass
class _Selections:
    """Input-selected structure recorded by one decode, for error paths.

    tags maps a tagged model path to its selected string tag. inline holds the
    paths of str-or-model fields whose input is an object (the inline model).
    """

    tags: dict[FormPath, str] = field(default_factory=dict)
    inline: set[FormPath] = field(default_factory=set)


def _decode(
    info: FieldInfo,
    value: JsonValue,
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
    selections: _Selections,
) -> FormValue:
    annotation = _non_null_annotation(info.annotation)
    child = selected(info, value)
    if isinstance(value, dict) and key_or_inline(info.annotation):
        selections.inline.add(path)
    if child is not None and isinstance(value, dict):
        tag_key = discriminator(info)
        tag = value.get(tag_key) if tag_key is not None else None
        if isinstance(tag, str):
            selections.tags[path] = tag
        # Selectors determine shape before any value-leaf evaluation.
        fields = {key: item for key, item in value.items() if key != tag_key}
        ordinary = _decode_model(child, fields, env, path, errors, selections)
        if tag_key is not None and tag_key in value:
            ordinary[tag_key] = _ordinary(value[tag_key])
        return ordinary
    numeric = _numeric_annotation(annotation)
    if isinstance(value, dict) and (
        (accepts_expression(annotation) and "expr" in value) or numeric
    ):
        return _scalar(value, env, path, errors, numeric=numeric)
    if get_origin(annotation) in (dict, Mapping) and isinstance(value, dict):
        args = get_args(annotation)
        item_info = FieldInfo.from_annotation(args[1])
        return {
            key: _decode(item_info, item, env, (*path, key), errors, selections)
            for key, item in value.items()
        }
    if get_origin(annotation) in (list, tuple) and isinstance(value, list):
        return _decode_sequence(annotation, value, env, path, errors, selections)
    return _ordinary(value)


def _non_null_annotation(annotation: object) -> object:
    """Unwrap a nullable single type, retaining genuine unions for scalar dispatch."""
    types = tuple(item for item in alternatives(annotation) if item is not type(None))
    return unwrap(types[0]) if len(types) == 1 else unwrap(annotation)


def _numeric_annotation(annotation: object) -> bool:
    """Whether a scalar accepts numeric expressions or complex wire values."""
    return any(
        item is int or item is float or item is complex
        for item in alternatives(annotation)
    )


def _decode_sequence(
    annotation: object,
    value: list[JsonValue],
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
    selections: _Selections,
) -> list[FormValue]:
    """Evaluate sequence items by annotation; Pydantic checks shape afterwards."""
    args = get_args(annotation)
    fixed = get_origin(annotation) is tuple and args and args[-1] is not Ellipsis
    decoded: list[FormValue] = []
    for index, item in enumerate(value):
        item_type = (
            args[index]
            if fixed and index < len(args)
            else (args[0] if args and not fixed else object)
        )
        # Items can nest sequences, mappings, or tagged models; re-enter the full
        # decoder to preserve expression evaluation, tag collection, and error paths.
        decoded.append(
            _decode(
                FieldInfo.from_annotation(item_type),
                item,
                env,
                (*path, index),
                errors,
                selections,
            )
        )
    return decoded


def _scalar(
    value: dict[str, JsonValue],
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
    *,
    numeric: bool,
) -> FormValue:
    try:
        if "expr" in value:
            if set(value) != {"expr"} or not isinstance(value["expr"], str):
                kind = "Numeric" if numeric else "String"
                raise ValueError(f"{kind} expression requires {{expr: text}}")
            return evaluate_scalar_expr(value["expr"], env.parameters)
        if "__complex__" in value:
            return decode_complex(value)
    except (ExpressionError, ValueError) as exc:
        errors[path] = str(exc)
        return None
    return _ordinary(value)


def _ordinary(value: JsonValue) -> FormValue:
    if isinstance(value, list):
        return [_ordinary(item) for item in value]
    if isinstance(value, dict):
        return {key: _ordinary(item) for key, item in value.items()}
    return value


def _decode_model(
    model: type[BaseModel],
    data: Mapping[str, JsonValue],
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
    selections: _Selections,
) -> dict[str, FormValue]:
    return {
        key: _decode(info, value, env, (*path, key), errors, selections)
        if (info := model.model_fields.get(key)) is not None
        else _ordinary(value)
        for key, value in data.items()
    }


def _read(data: object, path: FormPath) -> object:
    for part in path:
        if isinstance(data, BaseModel) and isinstance(part, str):
            data = getattr(data, part)
        elif isinstance(data, Mapping):
            if part not in data:
                return None
            data = data[part]
        elif isinstance(data, (list, tuple)) and isinstance(part, int):
            data = data[part]
        else:
            return None
    return data


def value_at(data: BaseModel, path: FormPath) -> FormValue:
    """Copy an evaluated model leaf at its canonical field/key path."""
    return _model_value(_read(data, path))


def _model_value(value: object) -> FormValue:
    if value is None or isinstance(value, (str, int, float, complex, bool)):
        return value
    if isinstance(value, BaseModel):
        return {
            name: _model_value(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, tuple):
        return tuple(_model_value(item) for item in value)
    if isinstance(value, list):
        return [_model_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, FormValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Form value mapping keys must be strings")
            result[key] = _model_value(item)
        return result
    raise ValueError(f"Unsupported form value: {type(value).__name__}")


def _canonical(path: FormPath, selections: _Selections) -> FormPath:
    """Drop Pydantic's union labels from an error location.

    An inline str-or-model value is one member of a smart union, so its loc
    names that member once ('PulseSpec', 'tagged-union[...]'); a tagged
    model's loc names its tag once. Neither is a model-field path part.
    """
    result: FormPath = ()
    labelled: set[FormPath] = set()
    stripped: set[FormPath] = set()
    for part in path:
        if result in selections.inline and result not in labelled:
            labelled.add(result)
            continue
        if result not in stripped and selections.tags.get(result) == part:
            stripped.add(result)
            continue
        result = (*result, part)
    return result


def _active_leaf(value: BaseModel, path: FormPath) -> bool:
    current: object = value
    for name in path:
        if (
            isinstance(current, BaseModel)
            and isinstance(name, str)
            and name in type(current).model_fields
        ):
            current = getattr(current, name)
        elif isinstance(current, Mapping):
            if name not in current:
                return False
            current = current[name]
        elif isinstance(current, (list, tuple)):
            if not isinstance(name, int) or not 0 <= name < len(current):
                return False
            current = current[name]
        else:
            return False
    return True


def result[F: FormModel](
    model: type[F], data: dict[str, JsonValue], env: FormEnv
) -> F | FormErrors:
    """Evaluate one detached input and return its model or canonical errors."""
    errors: dict[FormPath, str] = {}
    selections = _Selections()
    ordinary = _decode_model(model, data, env, (), errors, selections)
    return _validate(model, data, ordinary, env, errors, selections)


def preview(
    model: type[FormModel], data: dict[str, JsonValue], env: FormEnv
) -> FormPreview:
    """Share decoding/validation while retaining non-failing leaves for display."""
    errors: dict[FormPath, str] = {}
    selections = _Selections()
    ordinary = _decode_model(model, data, env, (), errors, selections)
    failed_decoding = tuple(errors)
    validated = _validate(model, data, ordinary, env, errors, selections)
    if not isinstance(validated, FormErrors):
        complete = {
            key: default_json(getattr(validated, key)) for key in model.model_fields
        }
        return FormPreview(
            {path: value_at(validated, path) for path in leaves(model, complete)}, None
        )
    values = {
        path: _model_value(_read(ordinary, path))
        for path in leaves(model, data)
        if not any(failed[: len(path)] == path for failed in failed_decoding)
    }
    return FormPreview(values, validated)


def _validate[F: FormModel](
    model: type[F],
    data: dict[str, JsonValue],
    ordinary: dict[str, FormValue],
    env: FormEnv,
    errors: dict[FormPath, str],
    selections: _Selections,
) -> F | FormErrors:
    try:
        options = check_sources(model, data, env, validate_paths=True)
    except PydanticCustomError as exc:
        errors[error_path(exc)] = exc.message()
        options = {}
    if errors:
        return FormErrors(errors)
    try:
        validated = model.model_validate(ordinary)
    except ValidationError as exc:
        for error in exc.errors():
            path = _canonical(error["loc"], selections)
            if path in selections.inline and error["type"] == "string_type":
                # The key alternative rejects every inline object; the inline
                # model's own errors already describe the failure.
                continue
            if error["type"] == "form_field":
                context = error.get("ctx") or {}
                directed = context.get("path", ())
                if isinstance(directed, tuple):
                    path = (
                        *path,
                        *(item for item in directed if isinstance(item, (str, int))),
                    )
            errors[path] = error["msg"]
        return FormErrors(errors)
    for path, values in options.items():
        if not _active_leaf(validated, path):
            continue
        current = default_json(_read(validated, path))
        if current not in values:
            errors[path] = "Value is not among current choices"
    return FormErrors(errors) if errors else validated


def evaluate_defaults(
    model: type[FormModel], data: Mapping[str, JsonValue], env: FormEnv
) -> dict[str, JsonValue]:
    """Return a detached copy of defaults data with Expr leaves turned into values.

    data is what model.defaults returned. Each scalar model-field leaf holding
    exactly {"expr": text}, in model or nested/selected-branch sections, is
    evaluated once with env.parameters and replaced by its JSON value. A leaf
    is removed instead when its expression raises ExpressionError (a missing
    path, an unbound alias, a non-numeric value), when its value is not finite
    JSON, or when result(env) then reports an error inside that leaf. Its
    field is then unfilled: a required field reports "Field required", and a
    field with a static Pydantic default validates with that default. Other
    exceptions from evaluation, such as unavailable captured sources,
    propagate. Envelopes inside whole sequences or mappings, malformed
    envelopes and every other value are copied unchanged. data is not edited.
    """
    evaluated = copy_object(data)
    # Leaf path -> the object in evaluated that holds that leaf.
    substituted: dict[FormWritePath, dict[str, JsonValue]] = {}
    _evaluate_section(model, evaluated, env, (), None, substituted)
    if not substituted:
        return evaluated
    checked = result(model, evaluated, env)
    if isinstance(checked, FormErrors):
        for leaf, parent in substituted.items():
            if any(path[: len(leaf)] == leaf for path in checked.messages):
                del parent[leaf[-1]]
    return evaluated


def _evaluate_section(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    env: FormEnv,
    prefix: FormWritePath,
    tag_key: str | None,
    substituted: dict[FormWritePath, dict[str, JsonValue]],
) -> None:
    """Evaluate one model object's Expr leaves in place; tag_key is its selector."""
    for name, value in tuple(data.items()):
        info = model.model_fields.get(name)
        if info is None or name == tag_key:
            continue
        path = (*prefix, name)
        child = selected(info, value)
        if child is not None and isinstance(value, dict):
            _evaluate_section(child, value, env, path, discriminator(info), substituted)
            continue
        if not (
            isinstance(value, dict)
            and set(value) == {"expr"}
            and isinstance(text := value["expr"], str)
            and accepts_expression(info.annotation)
        ):
            continue
        try:
            evaluated = evaluate_scalar_expr(text, env.parameters)
        except ExpressionError:
            del data[name]
            continue
        try:
            data[name] = default_json(evaluated)
        except ValueError:
            # Non-finite results have no JSON value to show or store.
            del data[name]
            continue
        substituted[path] = data
