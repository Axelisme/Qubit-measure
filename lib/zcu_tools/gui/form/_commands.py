"""Prepare and admit complete mounted-field command replacements."""

from collections.abc import Mapping
from copy import deepcopy

from pydantic import BaseModel, JsonValue, create_model
from pydantic.fields import FieldInfo

from ._admission import check_change
from ._evaluate import result, value_at
from ._json import copy_json
from ._selection import Field, resolve
from ._sources import check_sources
from ._structure import (
    StructureError,
    alternatives,
    discriminator,
    intrinsic_tags,
    list_model,
    model_types,
    selected,
)
from .commands import (
    FormCommandContext,
    FormCommandResult,
    FormCommandSnapshot,
    readonly_input,
)
from .model import (
    FormCommandError,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormPathError,
    FormValue,
    FormWritePath,
)
from .options import Option


def _evaluate(target: Field, path: FormWritePath, env: FormEnv) -> FormValue:
    """Evaluate an authored field alone, retaining its constraints/declarations."""
    if target.name not in target.level.data:
        raise FormCommandError(f"Cannot convert: missing participating input {path!r}")

    class LocalField(FormModel):
        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            return {("value",): target.choices} if target.choices is not None else {}

        @classmethod
        def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
            return {("value",): target.options} if target.options is not None else {}

    shell = create_model(
        "CommandParticipant",
        __base__=LocalField,
        value=(target.info.annotation, deepcopy(target.info)),
    )
    evaluated = result(shell, {"value": target.level.value(target.name)}, env)
    if isinstance(evaluated, FormErrors):
        raise FormCommandError(
            f"Cannot convert participating input {path!r}: {evaluated.messages}"
        )
    return value_at(evaluated, ("value",))


def _check_object(
    model: type[BaseModel], data: dict[str, JsonValue], path: FormWritePath
) -> None:
    """Check model shape only; ordinary invalid leaf values remain drafts."""
    for name, value in data.items():
        info = model.model_fields.get(name)
        if info is None:
            raise StructureError(
                f"{'.'.join(map(str, (*path, name)))}: unknown field; "
                f"expected fields: {', '.join(model.model_fields)}",
                path=(*path, name),
            )
        _check_field(info, value, (*path, name))


def _model_structure_error(info: FieldInfo, path: FormWritePath) -> StructureError:
    expected = "expected a model object"
    key = discriminator(info)
    if key is not None:
        tags = [
            tag
            for model in model_types(info.annotation)
            for tag in intrinsic_tags(model, key)
        ]
        expected += f" with {key!r} in {tags!r}"
    if type(None) in alternatives(info.annotation):
        expected += " or null"
    return StructureError(f"{'.'.join(map(str, path))}: {expected}", path=path)


def _check_field(info: FieldInfo, value: JsonValue, path: FormWritePath) -> None:
    models = model_types(info.annotation)
    if models:
        if value is None:
            if type(None) not in alternatives(info.annotation):
                raise _model_structure_error(info, path)
            return
        branch = selected(info, value)
        if not isinstance(value, dict) or branch is None:
            raise _model_structure_error(info, path)
        key = discriminator(info)
        if key is not None and value.get(key) not in intrinsic_tags(branch, key):
            raise _model_structure_error(info, path)
        _check_object(branch, value, path)
    elif (element := list_model(info)) is not None and isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise FormCommandError("Replacement returned an illegal list element")
            _check_object(element, item, (*path, index))
    elif isinstance(value, dict) and info.annotation in (
        int,
        float,
        complex,
        str,
        bool,
    ):
        if set(value) not in ({"expr"}, {"__complex__"}):
            raise FormCommandError(
                "Command returned a model structure for a scalar field"
            )


def _merge_selection(
    data: dict[str, JsonValue], selection: dict[str, JsonValue]
) -> None:
    for key, value in selection.items():
        child = data.get(key)
        if isinstance(child, dict) and isinstance(value, dict):
            _merge_selection(child, value)
        else:
            data[key] = value


def _replacement(target: Field, output: FormCommandResult) -> JsonValue:
    value = copy_json(output.input)
    selection = copy_json(output.selection)
    if selection is not None:
        if target.options is None or not any(
            option.matches(selection) for option in target.options
        ):
            raise FormCommandError("Command returned an unavailable selection")
        if isinstance(selection, dict) and isinstance(value, dict):
            _merge_selection(value, selection)
        elif value != selection:
            raise FormCommandError("Command selection disagrees with its leaf input")
    branch = selected(target.info, value) if isinstance(value, dict) else None
    if output.structure is not None:
        if (
            branch is None
            or not issubclass(branch, output.structure)
            or branch.model_fields.keys() != output.structure.model_fields.keys()
        ):
            raise FormCommandError(
                "Command returned a structure outside the field annotation"
            )
    elif branch is not None:
        raise FormCommandError("Command must name its model structure")
    _check_field(target.info, value, target.path)
    return value


def prepare_command[F: FormModel](
    input: FormInput[F], path: FormWritePath, name: str, env: FormEnv
) -> FormInput[F]:
    """Run a declared command without state access, then admit its local result."""
    data = input.data
    target = resolve(input.model, data, path, env, input.selections)
    option = (
        target.options[target.selected]
        if target.options is not None and target.selected is not None
        else None
    )
    commands = (
        *target.input_commands,
        *(option.commands if option is not None else ()),
    )
    command = next((item for item in commands if item.name == name), None)
    if command is None:
        raise FormCommandError(f"Unavailable field command {name!r} at {path!r}")
    snapshot, context = observation(input, target, env)
    output = command.run(snapshot, context)
    candidate = install_field(input, target, output, env)
    check_change(input, candidate, (path,))
    return candidate


def prepare_write[F: FormModel](
    input: FormInput[F], batch: Mapping[FormWritePath, JsonValue], env: FormEnv
) -> FormInput[F]:
    """Prepare exact nonoverlapping replacements, checking the final candidate."""
    paths = tuple(batch)
    if any(
        not path or any(path != other and other[: len(path)] == path for other in paths)
        for path in paths
    ):
        raise FormCommandError("Replacement paths must be nonempty and nonoverlapping")
    candidate = input
    for path, value in batch.items():
        target = resolve(input.model, input.data, path, env, input.selections)
        try:
            candidate = install_field(
                candidate,
                target,
                FormCommandResult(
                    None,
                    selected(target.info, value) if isinstance(value, dict) else None,
                    value,
                ),
                env,
                validate_sources=False,
            )
        except StructureError as exc:
            raise StructureError(
                str(exc), path=path, field_path=exc.field_path
            ) from exc
        except FormCommandError as exc:
            raise StructureError(str(exc), path=path) from exc
    check_sources(input.model, candidate.data, env, validate_paths=True)
    check_change(input, candidate, paths)
    return candidate


def observation(
    input: FormInput[FormModel],
    target: Field,
    env: FormEnv,
) -> tuple[FormCommandSnapshot, FormCommandContext]:
    """Observe one field with narrow evaluation, without executing a callable."""
    data = input.data
    path = target.path
    option = (
        target.options[target.selected]
        if target.options is not None and target.selected is not None
        else None
    )
    snapshot = FormCommandSnapshot(
        readonly_input(option.value if option is not None else None),
        selected(target.info, target.level.value(target.name)),
        readonly_input(target.level.value(target.name)),
        target.name in target.level.data,
    )

    def evaluate(relative: FormWritePath) -> FormValue:
        try:
            participant = resolve(input.model, data, (*path, *relative), env)
            return _evaluate(participant, relative, env)
        except FormPathError as exc:
            raise FormCommandError(
                f"Cannot convert participating input {relative!r}: {exc}"
            ) from exc

    return snapshot, FormCommandContext(evaluate)


def install_field[F: FormModel](
    input: FormInput[F],
    target: Field,
    output: FormCommandResult,
    env: FormEnv,
    *,
    validate_sources: bool = True,
) -> FormInput[F]:
    """Admit a complete replacement; batch preparation defers source checking."""
    data = input.data
    path = target.path
    try:
        replacement = _replacement(target, output)
    except FormCommandError:
        raise
    except ValueError as exc:
        raise FormCommandError(str(exc)) from exc
    parent: JsonValue = data
    for part in path[:-1]:
        match parent, part:
            case dict() as obj, str() as key:
                if key not in obj:
                    obj[key] = {}
                parent = obj[key]
            case list() as items, int() as index:
                parent = items[index]
            case _:
                raise FormCommandError("Command target is not an active container")
    match parent, path[-1]:
        case dict() as obj, str() as key:
            obj[key] = replacement
        case list() as items, int() as index:
            items[index] = replacement
        case _:
            raise FormCommandError("Command target is not an active field")
    if validate_sources:
        check_sources(input.model, data, env, validate_paths=True)
    selections = {
        key: value
        for key, value in input.selections.items()
        if key[: len(path)] != path
    }
    return FormInput(input.model, data, selections=selections)
