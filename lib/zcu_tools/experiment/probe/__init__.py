"""Independent offline stages, loaded only when their public names are selected.

Importing generators does not load acquisition or simulation dependencies.
Stage callables and result types retain their defining modules' contracts.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._comparison import compare_roundtrip
    from .analysis import AnalysisStageResult, analysis_stage
    from .run_save import RunStageResult, SaveStageResult, run_stage, save_stage

__all__ = [
    "AnalysisStageResult",
    "RunStageResult",
    "SaveStageResult",
    "analysis_stage",
    "compare_roundtrip",
    "run_stage",
    "save_stage",
]

_EXPORT_MODULES = {
    "AnalysisStageResult": ".analysis",
    "RunStageResult": ".run_save",
    "SaveStageResult": ".run_save",
    "analysis_stage": ".analysis",
    "compare_roundtrip": "._comparison",
    "run_stage": ".run_save",
    "save_stage": ".run_save",
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
