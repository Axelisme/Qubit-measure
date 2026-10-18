"""Explicit option selections, separate from exact raw-value writes."""

from pydantic import JsonValue
from pydantic_core import PydanticCustomError

from ._admission import check_change
from ._commands import install_field, observation
from ._selection import Field, resolve
from ._sources import check_sources, prefix_error
from ._structure import model_types
from .commands import FormCommandResult, readonly_input
from .model import (
    FormCommandError,
    FormConversionError,
    FormEnv,
    FormInput,
    FormModel,
    FormWritePath,
)
from .options import Option


def _select(
    input: FormInput[FormModel], target: Field, option: Option, env: FormEnv
) -> FormInput[FormModel]:
    converter = target.converter
    if converter is not None:
        snapshot, context = observation(input, target, env)
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
    candidate = install_field(
        input, target, FormCommandResult(None, None, option.value), env
    )
    selections = candidate.selections
    selections[target.path] = option.value
    return FormInput(input.model, candidate.data, selections=selections)


def prepare_selection[F: FormModel](
    input: FormInput[F], path: FormWritePath, selection: JsonValue, env: FormEnv
) -> FormInput[F]:
    """Select a listed option and admit its complete candidate without installing."""
    target = resolve(input.model, input.data, path, env, input.selections)
    option = next(
        (item for item in target.options or () if item.matches(selection)), None
    )
    if option is None:
        raise FormConversionError(f"Unavailable selection at {path!r}: {selection!r}")
    candidate = _select(input, target, option, env)
    check_sources(input.model, candidate.data, env, validate_paths=True)
    check_change(input, candidate, (path,))
    return FormInput(input.model, candidate.data, selections=candidate.selections)
