"""Writeback proposals, opaque drafts, app controls and the Qt review widget.

Proposal imports do not load app services or Qt. Controls and views load only
when requested, so adapter authors can use the proposal interface independently.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .items import (
    ModuleWriteback,
    ParameterWriteback,
    RecordWriteback,
    WritebackItem,
)

if TYPE_CHECKING:
    from .contracts import (
        WritebackAcceptance,
        WritebackEdit,
        WritebackSelection,
        WritebackValues,
        WritebackWritten,
    )
    from .control import (
        CapturedWritebackPane,
        WritebackControlFacet,
        WritebackControlPort,
        WritebackPane,
    )
    from .service import WritebackDraft, WritebackService
    from .widget import WritebackWidget

__all__ = [
    "ModuleWriteback",
    "ParameterWriteback",
    "RecordWriteback",
    "WritebackItem",
    "WritebackControlFacet",
    "WritebackControlPort",
    "WritebackPane",
    "WritebackDraft",
    "WritebackEdit",
    "CapturedWritebackPane",
    "WritebackSelection",
    "WritebackAcceptance",
    "WritebackService",
    "WritebackValues",
    "WritebackWritten",
    "WritebackWidget",
]

_EXPORT_MODULES = {
    "WritebackControlFacet": ".control",
    "WritebackControlPort": ".control",
    "WritebackPane": ".control",
    "CapturedWritebackPane": ".control",
    "WritebackSelection": ".contracts",
    "WritebackAcceptance": ".contracts",
    "WritebackDraft": ".service",
    "WritebackEdit": ".contracts",
    "WritebackService": ".service",
    "WritebackValues": ".contracts",
    "WritebackWritten": ".contracts",
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
