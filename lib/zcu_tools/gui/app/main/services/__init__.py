"""Services encapsulating domain logic, decoupled from the Controller Façade."""

from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.gui.session.services.connection import ConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)

from .analyze import AnalyzeService
from .app_services import AppServices, build_app_services
from .caretaker import PersistenceCaretaker, RestoreOutcome
from .guard import (
    AnalyzePermit,
    GuardError,
    GuardService,
    RunPermit,
    SavePermit,
    WritebackPermit,
)
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
from .run import RunService
from .save import SaveResultOutcome, SaveService
from .startup import StartupConnectionRequest, StartupProjectRequest, StartupService
from .tab import TabService, TabSnapshot
from .workspace import WorkspaceService
from .writeback import WritebackService

__all__ = [
    "AnalyzeService",
    "AppServices",
    "build_app_services",
    "ConnectionService",
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
    "GuardError",
    "GuardService",
    "RunPermit",
    "SavePermit",
    "AnalyzePermit",
    "WritebackPermit",
    "RunService",
    "SaveResultOutcome",
    "SaveService",
    "PersistenceCaretaker",
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
