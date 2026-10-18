"""Prepare and admit complete mounted-field command replacements."""

from collections.abc import Mapping
from copy import deepcopy

from pydantic import BaseModel, JsonValue, create_model
from pydantic.fields import FieldInfo

from ._evaluate import result, value_at
from ._json import copy_json
from ._selection import Field, resolve
from ._sources import check_sources
from ._structure import (
    alternatives,
    discriminator,
    intrinsic_tags,
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
    evaluated = result(shell, {"value": target.level.data[target.name]}, env)
    if isinstance(evaluated, FormErrors):
        raise FormCommandError(
            f"Cannot convert participating input {path!r}: {evaluated.messages}"
        )
    return value_at(evaluated, ("value",))


def _check_object(model: type[BaseModel], data: dict[str, JsonValue]) -> None:
    """Check model shape only; ordinary invalid leaf values remain drafts."""
    for name, value in data.items():
        info = model.model_fields.get(name)
        if info is None:
            raise FormCommandError(f"Command returned unknown field {name!r}")
        _check_field(info, value)


def _check_field(info: FieldInfo, value: JsonValue) -> None:
    models = model_types(info.annotation)
    if models:
        if value is None:
            if type(None) not in alternatives(info.annotation):
                raise FormCommandError(
                    "Command cannot remove a nonnullable model structure"
                )
            return
        branch = selected(info, value)
        if not isinstance(value, dict) or branch is None:
            raise FormCommandError("Command returned an illegal model structure")
        key = discriminator(info)
        if key is not None and value.get(key) not in intrinsic_tags(branch, key):
            raise FormCommandError("Command returned an illegal selection")
        _check_object(branch, value)
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
    _check_field(target.info, value)
    return value


def prepare_command[F: FormModel](
    input: FormInput[F], path: FormWritePath, name: str, env: FormEnv
) -> FormInput[F]:
    """Run a declared command without state access, then admit its local result."""
    data = input.data
    target = resolve(input.model, data, path, env)
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
    snapshot = FormCommandSnapshot(
        readonly_input(option.value if option is not None else None),
        selected(target.info, target.level.data.get(target.name)),
        readonly_input(target.level.data.get(target.name)),
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

    output = command.run(snapshot, FormCommandContext(evaluate))
    try:
        replacement = _replacement(target, output)
    except ValueError as exc:
        raise FormCommandError(str(exc)) from exc
    parent = data
    for part in path[:-1]:
        child = parent[part]
        if not isinstance(child, dict):
            raise FormCommandError("Command target is not an active object")
        parent = child
    parent[path[-1]] = replacement
    check_sources(input.model, data, env, validate_paths=True)
    return FormInput(input.model, data)
