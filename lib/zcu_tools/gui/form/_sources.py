"""Path-directed captured capability checks, without materialization or I/O."""

from collections.abc import Mapping
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from ._json import default_json
from ._structure import alternatives, choice_fields, selected
from .model import FormEnv, FormModel, FormPath, ModuleRef, field_error


def error_path(error: PydanticCustomError) -> FormPath:
    """Read a form_field context path; other custom errors are model-wide."""
    context = error.context or {}
    path = context.get("path", ())
    if not isinstance(path, tuple) or not all(
        isinstance(item, (str, int)) for item in path
    ):
        return ()
    return tuple(item for item in path if isinstance(item, (str, int)))


def prefix_error(error: PydanticCustomError, prefix: FormPath) -> PydanticCustomError:
    """Rebase a branch/source error and include its consuming path in the text."""
    path = (*prefix, *error_path(error))
    return field_error(
        path, f"{'.'.join(str(part) for part in path)}: {error.message()}"
    )


def _reference(data: Mapping[str, JsonValue], env: FormEnv, prefix: FormPath) -> None:
    name = data.get("name")
    if not isinstance(name, str):
        return
    path = (*prefix, "name")
    label = ".".join(str(part) for part in path)
    if env.library is None:
        raise field_error(path, f"{label}: Missing captured library")
    try:
        env.library.module_spec(name)
    except (ValueError, KeyError) as exc:
        raise field_error(path, f"{label}: {exc}") from exc


def _children(
    info: FieldInfo, value: JsonValue, prefix: FormPath
) -> tuple[tuple[type[BaseModel], dict[str, JsonValue], FormPath], ...]:
    child = selected(info, value)
    if child is not None and isinstance(value, dict):
        return ((child, value, prefix),)
    annotation = next(
        (t for t in alternatives(info.annotation) if t is not type(None)), None
    )
    args = get_args(annotation)
    if not args:
        return ()
    children: list[tuple[type[BaseModel], dict[str, JsonValue], FormPath]] = []
    if get_origin(annotation) in (dict, Mapping) and isinstance(value, dict):
        for key, item in value.items():
            children.extend(
                _children(FieldInfo.from_annotation(args[1]), item, (*prefix, key))
            )
    elif get_origin(annotation) in (list, tuple) and isinstance(value, list):
        for index, item in enumerate(value):
            item_type = (
                args[index]
                if get_origin(annotation) is tuple
                and args[-1] is not Ellipsis
                and index < len(args)
                else args[0]
            )
            children.extend(
                _children(FieldInfo.from_annotation(item_type), item, (*prefix, index))
            )
    return tuple(children)


def local_choices(
    model: type[BaseModel], env: FormEnv, *, validate_paths: bool = True
) -> Mapping[FormPath, tuple[JsonValue, ...]]:
    """Read one model's local hook; ordinary BaseModels have no local options.

    Invalid model-field leaf declarations raise ValueError when validate_paths
    is true. Hook source errors propagate for the caller to prefix as needed.
    """
    if not issubclass(model, FormModel):
        return {}
    options = model.choices(env)
    if validate_paths:
        for path in options:
            choice_fields(model, path)
    return options


def check_sources(
    model: type[BaseModel],
    data: Mapping[str, JsonValue],
    env: FormEnv,
    prefix: FormPath = (),
    *,
    validate_paths: bool = False,
) -> dict[FormPath, tuple[JsonValue, ...]]:
    """Collect choices and require every used captured capability.

    Missing fields with static defaults still consume their declared sources.
    Invalid ordinary drafts are not validated here. Directed source failures
    raise PydanticCustomError; library roots are read as Specs, never cfgs.
    validate_paths rejects invalid local choices declarations with ValueError;
    enclosing container keys/indexes only prefix errors, not declarations.
    Nested local options override enclosing options at the same leaf path.
    """
    options: dict[FormPath, tuple[JsonValue, ...]] = {}
    try:
        for path, values in local_choices(
            model, env, validate_paths=validate_paths
        ).items():
            options[(*prefix, *path)] = values
    except PydanticCustomError as exc:
        raise prefix_error(exc, prefix) from exc
    if issubclass(model, ModuleRef):
        _reference(data, env, prefix)
    for name, info in model.model_fields.items():
        value = data.get(name)
        if name not in data and not info.is_required():
            value = default_json(info.get_default(call_default_factory=True))
        for child, child_data, path in _children(info, value, (*prefix, name)):
            options.update(
                check_sources(
                    child, child_data, env, path, validate_paths=validate_paths
                )
            )
    return options
