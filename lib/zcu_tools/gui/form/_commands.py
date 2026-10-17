"""Install explicit parent conversions without ordinary branch defaults."""

from ._json import copy_object
from ._selection import Field
from ._sources import check_sources
from .model import (
    FormCommandError,
    FormEnv,
    FormInput,
    FormModel,
    FormState,
)


def run_input_command[F: FormModel](
    state: FormState[F], target: Field, name: str, env: FormEnv
) -> bool:
    """Run matching field metadata; False leaves option command resolution in charge."""
    command = next(
        (item for item in target.input_commands if item.name == name),
        None,
    )
    if command is None:
        return False
    converted = command.run(copy_object(target.level.data), env)
    try:
        replacement = copy_object(converted)
    except ValueError as exc:
        raise FormCommandError(f"Cannot convert coordinates: {exc}") from exc
    candidate = state.input.data
    if not target.level.prefix:
        candidate = replacement
    else:
        parent = candidate
        for part in target.level.prefix[:-1]:
            child = parent[part]
            if not isinstance(child, dict):
                raise ValueError("Command parent is not an object")
            parent = child
        parent[target.level.prefix[-1]] = replacement
    check_sources(state.input.model, candidate, env, validate_paths=True)
    state.install_input(FormInput(state.input.model, candidate))
    return True
