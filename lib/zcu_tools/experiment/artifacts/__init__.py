"""Qt-free artifact value types and shared save orchestration.

Load owners on demand so analysis documents do not import persistence services.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_document import AnalysisDocument, encode_analysis_document
    from .items import (
        AnalysisItem,
        ArtifactItem,
        ArtifactRef,
        DataItem,
        DataPreparationFailure,
        FigureItem,
        SaveArtifact,
    )
    from .saving import OutputResult, SaveReport, save_artifacts

__all__ = [
    "AnalysisDocument",
    "encode_analysis_document",
    "AnalysisItem",
    "ArtifactItem",
    "ArtifactRef",
    "DataItem",
    "DataPreparationFailure",
    "FigureItem",
    "OutputResult",
    "SaveArtifact",
    "SaveReport",
    "save_artifacts",
]

_EXPORT_MODULES = {
    "AnalysisDocument": ".analysis_document",
    "encode_analysis_document": ".analysis_document",
    "AnalysisItem": ".items",
    "ArtifactItem": ".items",
    "ArtifactRef": ".items",
    "DataItem": ".items",
    "DataPreparationFailure": ".items",
    "FigureItem": ".items",
    "SaveArtifact": ".items",
    "OutputResult": ".saving",
    "SaveReport": ".saving",
    "save_artifacts": ".saving",
}


def __getattr__(name: str) -> object:
    """Load the selected artifact type or saver without importing other owners."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List artifact exports without loading their owners."""
    return sorted(set(globals()) | set(__all__))
