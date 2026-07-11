"""Lazy public exports for the measure-gui service package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.session.ports import OperationConflictError
    from zcu_tools.gui.session.services.connection import SoCConnectionService
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.services.device import (
        ConnectDeviceRequest,
        DeviceService,
        DeviceSnapshot,
        DeviceStatus,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.services.predictor import PredictorService
    from zcu_tools.gui.session.services.startup import (
        StartupConnectionRequest,
        StartupProjectRequest,
        StartupService,
    )

    from .analyze import AnalyzeService
    from .app_services import AppServices, build_app_services
    from .arb_waveform import ArbWaveformService
    from .caretaker import (
        AppSnapshotCodec,
        RestoreOutcome,
        SingleFileCaretaker,
        create_persistence_caretaker,
    )
    from .guard import (
        AnalyzePermit,
        GuardError,
        GuardService,
        LoadPermit,
        RunPermit,
        SavePermit,
        WritebackPermit,
    )
    from .load import LoadService, LoadTabResultOutcome
    from .operation_gate import OperationGate, OperationKind
    from .persistence_types import (
        APP_STATE_VERSION,
        DEFAULT_LEFT_PANEL_WIDTH,
        AppPersistedState,
        PersistedDeviceEntry,
        PersistedSession,
        PersistedStartup,
        PersistedTab,
        PersistenceError,
    )
    from .ports import RestoreIssue, RestoreReport
    from .post_analyze import PostAnalyzeService
    from .run import RunService
    from .save import SaveService
    from .tab import TabService, TabSnapshot
    from .workspace import WorkspaceService
    from .writeback import WritebackService

__all__ = [
    "AnalyzeService",
    "ArbWaveformService",
    "AppServices",
    "build_app_services",
    "SoCConnectionService",
    "PredictorService",
    "ContextService",
    "DeviceService",
    "ConnectDeviceRequest",
    "DeviceSnapshot",
    "DeviceStatus",
    "DisconnectDeviceRequest",
    "SetupDeviceRequest",
    "OperationConflictError",
    "OperationGate",
    "OperationKind",
    "PostAnalyzeService",
    "GuardError",
    "GuardService",
    "LoadPermit",
    "RunPermit",
    "SavePermit",
    "AnalyzePermit",
    "WritebackPermit",
    "LoadService",
    "LoadTabResultOutcome",
    "RunService",
    "SaveService",
    "AppSnapshotCodec",
    "SingleFileCaretaker",
    "create_persistence_caretaker",
    "RestoreOutcome",
    "AppPersistedState",
    "PersistedSession",
    "PersistedTab",
    "PersistedDeviceEntry",
    "PersistedStartup",
    "PersistenceError",
    "APP_STATE_VERSION",
    "DEFAULT_LEFT_PANEL_WIDTH",
    "StartupConnectionRequest",
    "StartupProjectRequest",
    "StartupService",
    "TabService",
    "TabSnapshot",
    "RestoreIssue",
    "RestoreReport",
    "WorkspaceService",
    "WritebackService",
]

_EXPORT_MODULES: dict[str, str] = {
    "AnalyzeService": ".analyze",
    "ArbWaveformService": ".arb_waveform",
    "AppServices": ".app_services",
    "build_app_services": ".app_services",
    "SoCConnectionService": "zcu_tools.gui.session.services.connection",
    "PredictorService": "zcu_tools.gui.session.services.predictor",
    "ContextService": "zcu_tools.gui.session.services.context",
    "DeviceService": "zcu_tools.gui.session.services.device",
    "ConnectDeviceRequest": "zcu_tools.gui.session.services.device",
    "DeviceSnapshot": "zcu_tools.gui.session.services.device",
    "DeviceStatus": "zcu_tools.gui.session.services.device",
    "DisconnectDeviceRequest": "zcu_tools.gui.session.services.device",
    "SetupDeviceRequest": "zcu_tools.gui.session.services.device",
    "OperationConflictError": "zcu_tools.gui.session.ports",
    "OperationGate": ".operation_gate",
    "OperationKind": ".operation_gate",
    "PostAnalyzeService": ".post_analyze",
    "GuardError": ".guard",
    "GuardService": ".guard",
    "LoadPermit": ".guard",
    "RunPermit": ".guard",
    "SavePermit": ".guard",
    "AnalyzePermit": ".guard",
    "WritebackPermit": ".guard",
    "LoadService": ".load",
    "LoadTabResultOutcome": ".load",
    "RunService": ".run",
    "SaveService": ".save",
    "AppSnapshotCodec": ".caretaker",
    "SingleFileCaretaker": ".caretaker",
    "create_persistence_caretaker": ".caretaker",
    "RestoreOutcome": ".caretaker",
    "AppPersistedState": ".persistence_types",
    "PersistedSession": ".persistence_types",
    "PersistedTab": ".persistence_types",
    "PersistedDeviceEntry": ".persistence_types",
    "PersistedStartup": ".persistence_types",
    "PersistenceError": ".persistence_types",
    "APP_STATE_VERSION": ".persistence_types",
    "DEFAULT_LEFT_PANEL_WIDTH": ".persistence_types",
    "StartupConnectionRequest": "zcu_tools.gui.session.services.startup",
    "StartupProjectRequest": "zcu_tools.gui.session.services.startup",
    "StartupService": "zcu_tools.gui.session.services.startup",
    "TabService": ".tab",
    "TabSnapshot": ".tab",
    "RestoreIssue": ".ports",
    "RestoreReport": ".ports",
    "WorkspaceService": ".workspace",
    "WritebackService": ".writeback",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
