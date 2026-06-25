"""Services encapsulating domain logic, decoupled from the Controller Façade."""

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
from .arb_waveform import ArbWaveformService
from .app_services import AppServices, build_app_services
from .caretaker import PersistenceCaretaker, RestoreOutcome
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
from .save import SaveResultOutcome, SaveService
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
