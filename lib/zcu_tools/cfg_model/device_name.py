"""Front-end-neutral instrument-name annotation, with ordinary string validation."""

from dataclasses import dataclass
from typing import Annotated, TypeAliasType, get_args

from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class DeviceNameHint:
    """Mark a string as an instrument name, without lookup or validation."""


DeviceName = Annotated[str, DeviceNameHint()]
"""An instrument name stored and validated as an unrestricted ordinary string."""


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
