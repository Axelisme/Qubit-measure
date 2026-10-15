"""Writeback proposals, opaque drafts, app controls and the Qt review widget.

Proposal imports do not load app services or Qt. Controls and views load only
when requested, so adapter authors can use the proposal interface independently.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .items import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
)

if TYPE_CHECKING:
    from .control import WritebackControlFacet, WritebackControlPort, WritebackPane
    from .service import (
        WritebackDraft,
        WritebackEdit,
        WritebackService,
        WritebackValues,
        WritebackWritten,
    )
    from .widget import WritebackWidget

__all__ = [
    "ModuleWriteback",
    "ParameterWriteback",
    "SampleWriteback",
    "WritebackItem",
    "WritebackControlFacet",
    "WritebackControlPort",
    "WritebackPane",
    "WritebackDraft",
    "WritebackEdit",
    "WritebackService",
    "WritebackValues",
    "WritebackWritten",
    "WritebackWidget",
]

_EXPORT_MODULES = {
    "WritebackControlFacet": ".control",
    "WritebackControlPort": ".control",
    "WritebackPane": ".control",
    "WritebackDraft": ".service",
    "WritebackEdit": ".service",
    "WritebackService": ".service",
    "WritebackValues": ".service",
    "WritebackWritten": ".service",
    "WritebackWidget": ".widget",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
