from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "AbsExpAdapter": (".adapter", "AbsExpAdapter"),
    "AnalyzeParam": (".adapter", "AnalyzeParam"),
    "CfgNodeSpec": (".adapter", "CfgNodeSpec"),
    "CfgNodeValue": (".adapter", "CfgNodeValue"),
    "CfgSchema": (".adapter", "CfgSchema"),
    "CfgSectionSpec": (".adapter", "CfgSectionSpec"),
    "CfgSectionValue": (".adapter", "CfgSectionValue"),
    "DeviceManager": (".device_manager", "DeviceManager"),
    "DirectValue": (".adapter", "DirectValue"),
    "EvalValue": (".adapter", "EvalValue"),
    "ExpContext": (".adapter", "ExpContext"),
    "IOManager": (".io_manager", "IOManager"),
    "MainWindow": (".ui", "MainWindow"),
    "MetaDictWriteback": (".adapter", "MetaDictWriteback"),
    "ModuleRefSpec": (".adapter", "ModuleRefSpec"),
    "ModuleRefValue": (".adapter", "ModuleRefValue"),
    "ModuleWriteback": (".adapter", "ModuleWriteback"),
    "MultiSweepSpec": (".adapter", "MultiSweepSpec"),
    "MultiSweepValue": (".adapter", "MultiSweepValue"),
    "Registry": (".registry", "Registry"),
    "SavePaths": (".adapter", "SavePaths"),
    "ScalarSpec": (".adapter", "ScalarSpec"),
    "ScalarValue": (".adapter", "ScalarValue"),
    "State": (".state", "State"),
    "SweepSpec": (".adapter", "SweepSpec"),
    "SweepValue": (".adapter", "SweepValue"),
    "TabState": (".state", "TabState"),
    "WaveformRefSpec": (".adapter", "WaveformRefSpec"),
    "WaveformRefValue": (".adapter", "WaveformRefValue"),
    "WaveformWriteback": (".adapter", "WaveformWriteback"),
    "WritebackItem": (".adapter", "WritebackItem"),
    "make_default_value": (".adapter", "make_default_value"),
    "schema_to_dict": (".adapter", "schema_to_dict"),
}

__all__ = (
    "AbsExpAdapter",
    "AnalyzeParam",
    "CfgNodeSpec",
    "CfgNodeValue",
    "CfgSchema",
    "CfgSectionSpec",
    "CfgSectionValue",
    "DeviceManager",
    "DirectValue",
    "EvalValue",
    "ExpContext",
    "IOManager",
    "MainWindow",
    "MetaDictWriteback",
    "ModuleRefSpec",
    "ModuleRefValue",
    "ModuleWriteback",
    "MultiSweepSpec",
    "MultiSweepValue",
    "Registry",
    "SavePaths",
    "ScalarSpec",
    "ScalarValue",
    "State",
    "SweepSpec",
    "SweepValue",
    "TabState",
    "WaveformRefSpec",
    "WaveformRefValue",
    "WaveformWriteback",
    "WritebackItem",
    "make_default_value",
    "schema_to_dict",
)

if TYPE_CHECKING:
    from .adapter import (
        AbsExpAdapter,
        AnalyzeParam,
        CfgNodeSpec,
        CfgNodeValue,
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ExpContext,
        MetaDictWriteback,
        ModuleRefSpec,
        ModuleRefValue,
        ModuleWriteback,
        MultiSweepSpec,
        MultiSweepValue,
        SavePaths,
        ScalarSpec,
        ScalarValue,
        SweepSpec,
        SweepValue,
        WaveformRefSpec,
        WaveformRefValue,
        WaveformWriteback,
        WritebackItem,
        make_default_value,
        schema_to_dict,
    )
    from .device_manager import DeviceManager
    from .io_manager import IOManager
    from .registry import Registry
    from .state import State, TabState
    from .ui import MainWindow


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
