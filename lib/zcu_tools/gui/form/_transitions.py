"""Transactional selector writes using the same field installer as commands."""

from collections.abc import Mapping

from pydantic import JsonValue

from ._commands import install_field, observation
from ._fixed import require_same_identity
from ._selection import Field, resolve
from ._sources import check_sources
from ._structure import StructureError, build_candidate, model_types
from .commands import FormCommandResult, readonly_input
from .model import (
    FormCommandError,
    FormConversionError,
    FormEnv,
    FormInput,
    FormModel,
    FormPathError,
    FormWritePath,
)
from .options import Option


def _tag_value(value: JsonValue, path: FormWritePath) -> tuple[bool, JsonValue]:
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _tag_target(
    input: FormInput[FormModel],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> tuple[Field, Option] | None:
    """Lift a structural-tag write to its declared selector/converter."""
    matches: list[tuple[Field, Option]] = []
    for depth in range(len(path), 0, -1):
        try:
            target = resolve(
                input.model, input.data, path[:depth], env, input.selections
            )
        except FormPathError:
            continue
        for option in target.options or ():
            present, tag = _tag_value(option.value, path[depth:])
            if present and (Option("tag", tag).matches(value)):
                matches.append((target, option))
                break
    return next(
        (pair for pair in matches if pair[0].converter is not None),
        matches[0] if matches else None,
    )


def _select(
    input: FormInput[FormModel],
    target: Field,
    option: Option,
    env: FormEnv,
) -> FormInput[FormModel]:
    converter = target.converter
    if converter is not None:
        snapshot, context = observation(input, target, env)
        from pydantic_core import PydanticCustomError

        from ._sources import prefix_error

        try:
            converted = converter.run(snapshot, readonly_input(option.value), context)
        except PydanticCustomError as exc:
            raise prefix_error(exc, target.path) from exc
        except FormConversionError:
            raise
        except FormCommandError as exc:
            raise FormConversionError(str(exc)) from exc
        try:
            return install_field(
                input,
                target,
                FormCommandResult(option.value, converted.structure, converted.input),
                env,
            )
        except FormCommandError as exc:
            raise FormConversionError(str(exc)) from exc
    if model_types(target.info.annotation):
        selections = input.selections
        if (
            target.branch_option is not None
            and target.branch_option.value == option.value
        ):
            selections.pop(target.path, None)
        else:
            selections[target.path] = option.value
        return FormInput(input.model, input.data, selections=selections)
    data = build_candidate(input.model, input.data, {target.path: option.value}, env)
    return FormInput(input.model, data, selections=input.selections)


def _remaining(value: JsonValue, selection: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    result = dict(value)
    if isinstance(selection, dict):
        for key, selected in selection.items():
            if key not in result:
                continue
            if isinstance(result[key], dict) and isinstance(selected, dict):
                child = _remaining(result[key], selected)
                if child:
                    result[key] = child
                else:
                    del result[key]
            elif result[key] == selected:
                del result[key]
    return result


def _write_one(
    input: FormInput[FormModel],
    path: FormWritePath,
    value: JsonValue,
    env: FormEnv,
) -> FormInput[FormModel]:
    tag = _tag_target(input, path, value, env)
    if tag is not None:
        target, option = tag
        if (
            len(path) > len(target.path)
            and target.name in target.level.data
            and target.level.value(target.name) is None
        ):
            raise StructureError(
                f"Activate null section before child write: {path!r}", path=path
            )
        candidate = _select(input, target, option, env)
        _, selected = _tag_value(option.value, path[len(target.path) :])
        return _write_children(candidate, path, _remaining(value, selected), env)
    try:
        target = resolve(input.model, input.data, path, env, input.selections)
    except FormPathError:
        # Some declared selector branches intentionally hide their tag/key
        # children from describe. Structural admission still owns leaf writes.
        data = build_candidate(input.model, input.data, {path: value}, env)
        return FormInput(input.model, data, selections=input.selections)
    option = next(
        (option for option in target.options or () if option.matches(value)), None
    )
    if option is not None:
        candidate = _select(input, target, option, env)
        return _write_children(candidate, path, _remaining(value, option.value), env)
    data = build_candidate(input.model, input.data, {path: value}, env)
    return FormInput(input.model, data, selections=input.selections)


def _write_children(
    input: FormInput[FormModel],
    path: FormWritePath,
    values: Mapping[str, JsonValue],
    env: FormEnv,
) -> FormInput[FormModel]:
    candidate = input
    for name, value in values.items():
        child_path = (*path, name)
        try:
            target = resolve(
                candidate.model, candidate.data, child_path, env, candidate.selections
            )
        except FormPathError as exc:
            raise StructureError(str(exc), path=path) from exc
        if (
            isinstance(value, dict)
            and target.active_model is not None
            and target.options is None
        ):
            candidate = _write_children(candidate, child_path, value, env)
        else:
            candidate = _write_one(candidate, child_path, value, env)
    return candidate


def prepare_write[F: FormModel](
    input: FormInput[F],
    batch: Mapping[FormWritePath, JsonValue],
    env: FormEnv,
) -> FormInput[F]:
    """Prepare all selections and leaves atomically; never notify or mutate."""
    paths = tuple(batch)
    for path in paths:
        if not path or any(
            path != other and other[: len(path)] == path for other in paths
        ):
            raise StructureError(f"Invalid or overlapping path: {path!r}", path=path)
    candidate: FormInput[FormModel] = input
    for path in sorted(
        paths,
        key=lambda path: (
            _tag_target(input, path, batch[path], env) is None,
            len(path),
        ),
    ):
        try:
            candidate = _write_one(candidate, path, batch[path], env)
        except StructureError as exc:
            exc.path = path
            raise
    require_same_identity(input.model, input.data, candidate.data)
    check_sources(input.model, candidate.data, env)
    return FormInput(input.model, candidate.data, selections=candidate.selections)
