"""Named, detached selection values and explicit option actions."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from ._json import copy_json
from .model import FormEnv


@dataclass(frozen=True)
class OptionCommand:
    """An explicit action available only while its option matches current input.

    name is a nonempty identifier unique within the option; label is nonempty
    display text. run receives the caller's fixed FormEnv and returns detached
    partial JSON input to write at the option's field, not a whole form batch.
    It must not mutate state or captured sources. Source failures propagate
    before any write; ordinary invalid returned values remain editable drafts.
    The description exposes only name and label, never the callable.
    """

    name: str
    label: str
    run: Callable[[FormEnv], JsonValue]

    def __post_init__(self) -> None:
        """Reject empty names/labels or a non-callable implementation."""
        if not self.name or not self.label or not callable(self.run):
            raise ValueError("Option command needs a name, label and callable")


class Option:
    """One named selection, detached from its author and captured sources.

    label is nonempty display text. value is finite JSON partial input: null,
    scalar or object (whole JSON sequences are also legal leaf values). fields
    is an optional BaseModel class describing editable object children, not an
    instance or library entry; it is legal only with an object value. Core
    describes its direct children only on expansion, omitting tags fixed by value.
    The selection list never serializes fields or descendants.
    commands are explicit actions with unique names, available on this option.
    Invalid JSON, fields or duplicate command names raise ValueError here.
    The copied value property never exposes mutable storage.
    """

    def __init__(
        self,
        label: str,
        value: JsonValue,
        fields: type[BaseModel] | None = None,
        commands: tuple[OptionCommand, ...] = (),
    ) -> None:
        """Copy a selection value and validate its presentation/action contract."""
        if not label:
            raise ValueError("Option label must be nonempty")
        self._value = copy_json(value)
        if fields is not None and not isinstance(value, dict):
            raise ValueError("Option fields require an object value")
        if len({command.name for command in commands}) != len(commands):
            raise ValueError("Option command names must be unique")
        self.label = label
        self.fields = fields
        self.commands = tuple(commands)

    @property
    def value(self) -> JsonValue:
        """Return a fresh copy of the JSON partial input written on selection."""
        return copy_json(self._value)

    def matches(self, current: JsonValue) -> bool:
        """Whether current has every leaf of value, with equal JSON types/values.

        Extra object leaves in current are ignored. Empty objects match only
        objects. Scalars/null and whole arrays compare exactly without numeric
        coercion (True is not 1). Consumers select the first matching option;
        no match means display current plus '(missing)', never change input.
        This method does not read sources or evaluate expressions.
        """
        return _matches(self._value, current)


def _matches(
    value: JsonValue, current: JsonValue, *, partial_objects: bool = True
) -> bool:
    if isinstance(value, dict):
        return (
            isinstance(current, dict)
            and (partial_objects or value.keys() == current.keys())
            and all(
                key in current
                and _matches(child, current[key], partial_objects=partial_objects)
                for key, child in value.items()
            )
        )
    if isinstance(value, list):
        return (
            isinstance(current, list)
            and len(value) == len(current)
            # Arrays are whole leaves, including objects nested inside them.
            and all(
                _matches(a, b, partial_objects=False)
                for a, b in zip(value, current, strict=True)
            )
        )
    return type(value) is type(current) and value == current
