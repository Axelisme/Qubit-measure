"""Front-end-neutral instrument names and their no-dot validation rule."""

from dataclasses import dataclass
from typing import Annotated, TypeAliasType, get_args

from pydantic import AfterValidator
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class DeviceNameHint:
    """Mark a string as an instrument name, without lookup or validation."""


def validate_device_name(name: str) -> str:
    """Return name unchanged, or raise ValueError identifying an invalid name.

    Names must be nonblank and contain no ASCII dot, which separates component
    paths and flattened record fields. All other strings remain valid, including
    surrounding whitespace. This performs no lookup, trimming, or I/O.
    """
    if not name.strip():
        raise ValueError(f"Device name {name!r} must not be empty or whitespace-only")
    if "." in name:
        raise ValueError(f"Device name {name!r} must not contain '.'")
    return name


DeviceName = Annotated[str, DeviceNameHint(), AfterValidator(validate_device_name)]
"""An instrument-name string without dots; no lookup or serialization changes."""


def is_device_name(info: FieldInfo) -> bool:
    """Read the marker through nullable/Annotated aliases, without changing info."""
    if any(isinstance(item, DeviceNameHint) for item in info.metadata):
        return True
    pending: list[object] = [info.annotation]
    while pending:
        item = pending.pop()
        if isinstance(item, DeviceNameHint):
            return True
        if isinstance(item, TypeAliasType):
            pending.append(item.__value__)
        else:
            pending.extend(get_args(item))
    return False
