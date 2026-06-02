"""Services encapsulating domain logic, decoupled from the Controller Façade."""

from .analyze import AnalyzeService
from .app_services import AppServices, build_app_services
from .connection import ConnectionService
from .context import ContextService
from .device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from .guard import (
    AnalyzePermit,
    GuardError,
    GuardService,
    RunPermit,
    SavePermit,
    WritebackPermit,
)
from .operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
    OperationLease,
    OperationOutcome,
)
from .run import RunService
from .save import SaveBothOutcome, SaveService
from .session_persistence import (
    SESSION_VERSION,
    PersistedSession,
    PersistedTab,
    SessionPersistenceError,
    SessionPersistenceService,
)
from .startup import StartupConnectionRequest, StartupProjectRequest, StartupService
from .startup_persistence import (
    DEFAULT_LEFT_PANEL_WIDTH,
    STARTUP_VERSION,
    PersistedDeviceEntry,
    PersistedStartup,
    StartupPersistenceError,
    StartupPersistenceService,
)
from .tab import TabService, TabSnapshot
from .workspace import RestoreIssue, RestoreReport, WorkspaceService
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
    "OperationLease",
    "OperationOutcome",
    "GuardError",
    "GuardService",
    "RunPermit",
    "SavePermit",
    "AnalyzePermit",
    "WritebackPermit",
    "RunService",
    "SaveBothOutcome",
    "SaveService",
    "PersistedSession",
    "PersistedTab",
    "SESSION_VERSION",
    "SessionPersistenceError",
    "SessionPersistenceService",
    "PersistedDeviceEntry",
    "PersistedStartup",
    "DEFAULT_LEFT_PANEL_WIDTH",
    "STARTUP_VERSION",
    "StartupPersistenceError",
    "StartupPersistenceService",
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
