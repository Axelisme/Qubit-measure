"""Readonly command observations and explicit annotation-bound replacements."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, JsonValue

from .model import FormValue, FormWritePath

type ReadonlyFormInput = (
    str
    | int
    | float
    | bool
    | None
    | tuple[ReadonlyFormInput, ...]
    | Mapping[str, ReadonlyFormInput]
)
"""Deeply immutable JSON observation; arrays are tuples and objects are mappings."""


def readonly_input(value: JsonValue) -> ReadonlyFormInput:
    """Freeze detached JSON recursively, including expression envelopes."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: readonly_input(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(readonly_input(item) for item in value)
    return value


@dataclass(frozen=True)
class FormCommandSnapshot:
    """One mounted field before a command, independent of model validity.

    selection is the selected option's partial value, or None for no matching
    option/no selector (including a null selection). structure is the active
    annotation model, or None for a leaf/inactive model. input is raw authored
    JSON, recursively readonly; present distinguishes missing input from null.
    No field aliases writable state, UI or captured sources.
    """

    selection: ReadonlyFormInput
    structure: type[BaseModel] | None
    input: ReadonlyFormInput
    present: bool

    def copy_input(self) -> JsonValue:
        """Return an editable detached copy, without exposing snapshot storage."""
        return _mutable_input(self.input)


def _mutable_input(value: ReadonlyFormInput) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _mutable_input(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_input(item) for item in value]
    return value


@dataclass(frozen=True)
class FormCommandContext:
    """Narrow evaluation capability for one fixed command observation.

    read evaluates an authored path relative to the mounted field; () means
    that field itself. evaluate never supplies a missing value from defaults,
    evaluates unrelated fields, refreshes sources or installs input. Missing,
    unreachable or invalid participants raise FormCommandError. The host must
    consume the context synchronously in the same guarded turn.
    """

    read: Callable[[FormWritePath], FormValue]

    def evaluate(self, path: FormWritePath) -> FormValue:
        """Evaluate only path in the command's fixed authored observation."""
        return self.read(path)


@dataclass(frozen=True)
class FormConversionResult:
    """Complete new child branch and raw input for the caller's selection.

    structure is an annotation-admitted model (None for a leaf/null). input is
    the complete authored subtree, including structural tags. No defaults are
    supplied by Core; missing children stay missing and invalid values survive.
    """

    structure: type[BaseModel] | None
    input: JsonValue


@dataclass(frozen=True)
class FormCommandResult:
    """Complete replacement for the mounted field, never a parent write batch.

    selection explicitly supplies the new option value, or None without a
    selector. For model selectors its partial object overlays input (typically
    the discriminator). structure names an annotation-admitted model or its
    declared base; None denotes a leaf/null. input is the entire replacement
    raw subtree, not a partial patch. Missing children stay missing. Unknown
    structures/fields and mismatched selections are rejected before install;
    ordinary invalid leaf values remain editable drafts.
    """

    selection: JsonValue
    structure: type[BaseModel] | None
    input: JsonValue
