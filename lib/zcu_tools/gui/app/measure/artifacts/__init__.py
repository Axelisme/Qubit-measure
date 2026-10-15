"""Public artifact handover, save commands, tracker and save-center entry points.

State and application composition use this package, not its implementation files.
Qt widgets and save commands load only when requested; adapter handover does not
construct the app, a writer or a widget.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from zcu_tools.experiment.artifacts import (
    AnalysisItem,
    ArtifactRef,
    DataItem,
    FigureItem,
)
from zcu_tools.resources.entry import ArtifactKey

from .handover import ArtifactHandover
from .tracker import (
    ArtifactDeclaration,
    ArtifactSectionSnapshot,
    ArtifactSnapshot,
    ArtifactTracker,
    MemberSnapshot,
    OutputSnapshot,
    SaveStatus,
)

if TYPE_CHECKING:
    from .control import SaveControlFacet, SaveControlPort, SavedRunInfo
    from .save import SaveService
    from .save_center import ArtifactSaveCenter

__all__ = [
    "AnalysisItem",
    "ArtifactRef",
    "DataItem",
    "FigureItem",
    "ArtifactKey",
    "ArtifactHandover",
    "ArtifactDeclaration",
    "ArtifactSectionSnapshot",
    "ArtifactSnapshot",
    "ArtifactTracker",
    "MemberSnapshot",
    "OutputSnapshot",
    "SaveStatus",
    "SaveControlFacet",
    "SaveControlPort",
    "SavedRunInfo",
    "SaveService",
    "ArtifactSaveCenter",
]

_EXPORT_MODULES = {
    "SaveControlFacet": ".control",
    "SaveControlPort": ".control",
    "SavedRunInfo": ".control",
    "SaveService": ".save",
    "ArtifactSaveCenter": ".save_center",
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
