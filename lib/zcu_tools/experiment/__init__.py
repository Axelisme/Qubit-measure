from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import config
    from .axes_spec import (
        IDENTITY,
        MHZ_TO_HZ,
        US_TO_S,
        AxesSpec,
        Axis,
        GroupedAxesSpec,
        GroupedLoadData,
        LoadedRoleData,
        RoleAxisSpec,
        RoleSpec,
        RoleZSpec,
        ZSpec,
    )
    from .base import PersistableExperiment
    from .cfg_model import ExpCfgModel
    from .interfaces import RecordExperiment, SynchronousExperiment
    from .records import AnalysisRecord, RunRecord

__all__ = [
    "config",
    "ExpCfgModel",
    "RunRecord",
    "AnalysisRecord",
    "RecordExperiment",
    "SynchronousExperiment",
    "PersistableExperiment",
    "Axis",
    "ZSpec",
    "AxesSpec",
    "RoleAxisSpec",
    "RoleZSpec",
    "RoleSpec",
    "LoadedRoleData",
    "GroupedLoadData",
    "GroupedAxesSpec",
    "IDENTITY",
    "MHZ_TO_HZ",
    "US_TO_S",
]

_EXPORT_MODULES = {
    "config": ".config",
    "ExpCfgModel": ".cfg_model",
    "RunRecord": ".records",
    "AnalysisRecord": ".records",
    "RecordExperiment": ".interfaces",
    "SynchronousExperiment": ".interfaces",
    "PersistableExperiment": ".base",
    "Axis": ".axes_spec",
    "ZSpec": ".axes_spec",
    "AxesSpec": ".axes_spec",
    "RoleAxisSpec": ".axes_spec",
    "RoleZSpec": ".axes_spec",
    "RoleSpec": ".axes_spec",
    "LoadedRoleData": ".axes_spec",
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
