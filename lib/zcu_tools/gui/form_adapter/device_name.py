"""Instrument-name hints projected into non-validating scalar suggestions."""

from typing import Annotated

from pydantic.fields import FieldInfo

from zcu_tools.cfg_model import DeviceName as CoreDeviceName
from zcu_tools.cfg_model import is_device_name
from zcu_tools.gui.form import FormEnv, InputSuggestions, Option


def device_options(env: FormEnv) -> tuple[Option, ...]:
    """List instrument names from the fixed capture, without driver queries."""
    return tuple(Option(name, name) for name in env.devices)


def device_suggestions(info: FieldInfo) -> InputSuggestions | None:
    """Translate a DeviceName marker, including nullable names, to suggestions."""
    return InputSuggestions(device_options) if is_device_name(info) else None


DeviceName = Annotated[CoreDeviceName, InputSuggestions(device_options)]
"""Handwritten FormModel instrument name; accepts unlisted strings unchanged."""
