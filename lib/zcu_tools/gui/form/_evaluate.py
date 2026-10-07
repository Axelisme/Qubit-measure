"""Input evaluation and Pydantic error translation at the form boundary."""

from collections.abc import Mapping
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from zcu_tools.gui.cfg.codec import decode_complex
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr

from ._sources import check_sources, error_path
from ._structure import alternatives, discriminator, selected, unwrap
from .model import FormEnv, FormErrors, FormModel, FormPath, FormValue


def _decode(
    info: FieldInfo,
    value: JsonValue,
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
    tags: dict[FormPath, str],
) -> FormValue:
    types = tuple(
        item for item in alternatives(info.annotation) if item is not type(None)
    )
    annotation = unwrap(types[0]) if len(types) == 1 else unwrap(info.annotation)
    child = selected(info, value)
    if child is not None and isinstance(value, dict):
        if (tag_key := discriminator(info)) is not None:
            tag = value.get(tag_key)
            if isinstance(tag, str):
                tags[path] = tag
        return _decode_model(child, value, env, path, errors, tags)
    types = alternatives(annotation)
    if isinstance(value, dict) and any(
        item is int or item is float or item is complex for item in types
    ):
        return _scalar(value, env, path, errors)
    if get_origin(annotation) in (dict, Mapping) and isinstance(value, dict):
        args = get_args(annotation)
        item_info = FieldInfo.from_annotation(args[1])
        return {
            key: _decode(item_info, item, env, (*path, key), errors, tags)
            for key, item in value.items()
        }
    if get_origin(annotation) in (list, tuple) and isinstance(value, list):
        args = get_args(annotation)
        decoded: list[FormValue] = []
        for index, item in enumerate(value):
            fixed = (
                get_origin(annotation) is tuple and args and args[-1] is not Ellipsis
            )
            item_type = (
                args[index]
                if fixed and index < len(args)
                else (args[0] if args and not fixed else object)
            )
            decoded.append(
                _decode(
                    FieldInfo.from_annotation(item_type),
                    item,
                    env,
                    (*path, index),
                    errors,
                    tags,
                )
            )
        return decoded
    return _ordinary(value)


def _scalar(
    value: dict[str, JsonValue],
    env: FormEnv,
    path: FormPath,
    errors: dict[FormPath, str],
) -> FormValue:
    try:
        if "expr" in value:
            if set(value) != {"expr"} or not isinstance(value["expr"], str):
                raise ValueError("Numeric expression requires {expr: text}")
            return evaluate_scalar_expr(value["expr"], env.md)
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
    tags: dict[FormPath, str],
) -> dict[str, FormValue]:
    return {
        key: _decode(info, value, env, (*path, key), errors, tags)
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


def _canonical(path: FormPath, tags: Mapping[FormPath, str]) -> FormPath:
    result: FormPath = ()
    stripped: set[FormPath] = set()
    for part in path:
        if result not in stripped and tags.get(result) == part:
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


def _choice_values(
    current: object, options: tuple[JsonValue, ...]
) -> tuple[FormValue, ...]:
    return tuple(
        decode_complex(value)
        if isinstance(current, complex)
        and isinstance(value, dict)
        and "__complex__" in value
        else _ordinary(value)
        for value in options
    )


def result[F: FormModel](
    model: type[F], data: dict[str, JsonValue], env: FormEnv
) -> F | FormErrors:
    """Evaluate one detached input and return its model or canonical errors."""
    errors: dict[FormPath, str] = {}
    tags: dict[FormPath, str] = {}
    ordinary = _decode_model(model, data, env, (), errors, tags)
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
            path = _canonical(error["loc"], tags)
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
        current = _read(validated, path)
        if current not in _choice_values(current, values):
            errors[path] = "Value is not among current choices"
    return FormErrors(errors) if errors else validated
