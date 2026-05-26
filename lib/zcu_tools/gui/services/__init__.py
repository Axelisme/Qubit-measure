"""Services encapsulating domain logic, decoupled from the Controller Façade."""

from .analyze import AnalyzeService
from .connection import ConnectionService
from .context import ContextService
from .device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetDeviceValueRequest,
    SetupDeviceRequest,
)
from .operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
    OperationLease,
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
from .startup_persistence import (
    STARTUP_VERSION,
    PersistedDeviceEntry,
    PersistedStartup,
    StartupPersistenceError,
    StartupPersistenceService,
)
from .tab import TabService
from .writeback import WritebackService

__all__ = [
    "AnalyzeService",
    "ConnectionService",
    "ContextService",
    "DeviceService",
    "ConnectDeviceRequest",
    "DeviceSnapshot",
    "DeviceStatus",
    "DisconnectDeviceRequest",
    "SetDeviceValueRequest",
    "SetupDeviceRequest",
    "OperationConflictError",
    "OperationGate",
    "OperationKind",
    "OperationLease",
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
    "STARTUP_VERSION",
    "StartupPersistenceError",
    "StartupPersistenceService",
    "TabService",
    "WritebackService",
]
