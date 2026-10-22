from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import config
    from .acquisition import (
        AcquisitionEvidence,
        capture_software_provenance,
        new_run_id,
    )
    from .artifacts import (
        AnalysisItem,
        ArtifactItem,
        ArtifactRef,
        DataItem,
        DataPreparationFailure,
        FigureItem,
        OutputResult,
        SaveArtifact,
        SaveReport,
        save_artifacts,
    )
    from .axes_spec import (
        IDENTITY,
        MHZ_TO_HZ,
        US_TO_S,
        AxesSpec,
        Axis,
        GroupedAxesSpec,
        GroupedLoadData,
        LoadedVariableData,
        VariableAxisSpec,
        VariableSpec,
        VariableZSpec,
        ZSpec,
    )
    from .base import PersistableExperiment
    from .cfg_model import ExpCfgModel
    from .interfaces import (
        PersistableRecordExperiment,
        RecordExperiment,
        SynchronousExperiment,
    )
    from .native_persistence import (
        CfgUnavailableWarning,
        PreparedRunData,
        RestoredRun,
        RunSnapshot,
        load_run,
        load_run_with_metadata,
        native_schemas,
        prepare_run_data,
        restore_run,
        save_run,
    )
    from .records import AnalysisRecord, RunRecord
    from .scope import Scope, ScriptCancelled

__all__ = [
    "Scope",
    "ScriptCancelled",
    "config",
    "AcquisitionEvidence",
    "capture_software_provenance",
    "new_run_id",
    "ExpCfgModel",
    "RunRecord",
    "RunSnapshot",
    "save_run",
    "load_run",
    "native_schemas",
    "PreparedRunData",
    "RestoredRun",
    "CfgUnavailableWarning",
    "restore_run",
    "prepare_run_data",
    "load_run_with_metadata",
    "AnalysisRecord",
    "RecordExperiment",
    "PersistableRecordExperiment",
    "ArtifactRef",
    "ArtifactItem",
    "DataItem",
    "FigureItem",
    "AnalysisItem",
    "DataPreparationFailure",
    "SaveArtifact",
    "OutputResult",
    "SaveReport",
    "save_artifacts",
    "SynchronousExperiment",
    "PersistableExperiment",
    "Axis",
    "ZSpec",
    "AxesSpec",
    "VariableAxisSpec",
    "VariableZSpec",
    "VariableSpec",
    "LoadedVariableData",
    "GroupedLoadData",
    "GroupedAxesSpec",
    "IDENTITY",
    "MHZ_TO_HZ",
    "US_TO_S",
]

_EXPORT_MODULES = {
    "Scope": ".scope",
    "ScriptCancelled": ".scope",
    "config": ".config",
    "AcquisitionEvidence": ".acquisition",
    "capture_software_provenance": ".acquisition",
    "new_run_id": ".acquisition",
    "ExpCfgModel": ".cfg_model",
    "RunRecord": ".records",
    "RunSnapshot": ".native_persistence",
    "save_run": ".native_persistence",
    "load_run": ".native_persistence",
    "native_schemas": ".native_persistence",
    "PreparedRunData": ".native_persistence",
    "RestoredRun": ".native_persistence",
    "CfgUnavailableWarning": ".native_persistence",
    "restore_run": ".native_persistence",
    "prepare_run_data": ".native_persistence",
    "load_run_with_metadata": ".native_persistence",
    "AnalysisRecord": ".records",
    "RecordExperiment": ".interfaces",
    "PersistableRecordExperiment": ".interfaces",
    "ArtifactRef": ".artifacts",
    "ArtifactItem": ".artifacts",
    "DataItem": ".artifacts",
    "FigureItem": ".artifacts",
    "AnalysisItem": ".artifacts",
    "DataPreparationFailure": ".artifacts",
    "SaveArtifact": ".artifacts",
    "OutputResult": ".artifacts",
    "SaveReport": ".artifacts",
    "save_artifacts": ".artifacts",
    "SynchronousExperiment": ".interfaces",
    "PersistableExperiment": ".base",
    "Axis": ".axes_spec",
    "ZSpec": ".axes_spec",
    "AxesSpec": ".axes_spec",
    "VariableAxisSpec": ".axes_spec",
    "VariableZSpec": ".axes_spec",
    "VariableSpec": ".axes_spec",
    "LoadedVariableData": ".axes_spec",
    "GroupedLoadData": ".axes_spec",
    "GroupedAxesSpec": ".axes_spec",
    "IDENTITY": ".axes_spec",
    "MHZ_TO_HZ": ".axes_spec",
    "US_TO_S": ".axes_spec",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = module if name == "config" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
