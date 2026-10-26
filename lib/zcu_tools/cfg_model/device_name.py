"""Front-end-neutral instrument names and their no-dot validation rule."""

from dataclasses import dataclass
from typing import Annotated, TypeAliasType, get_args

from pydantic import AfterValidator
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class DeviceNameHint:
    """Mark a string as an instrument name, without lookup or validation."""


def validate_device_name(name: str) -> str:
    """Return name unchanged, or raise ValueError naming it if it contains a dot.

    Dots separate component paths and flattened record fields. All other string
    names, including empty strings, remain valid. This performs no lookup or I/O.
    """
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
