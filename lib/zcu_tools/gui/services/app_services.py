from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .analyze import AnalyzeService
from .cfg_editor import CfgEditorService
from .connection import ConnectionService
from .context import ContextService
from .device import DeviceService
from .guard import GuardService
from .operation_gate import OperationGate
from .run import RunService
from .save import SaveService
from .session_persistence import SessionPersistenceService
from .startup import StartupService
from .startup_persistence import StartupPersistenceService
from .tab import TabService
from .tab_view import TabViewService
from .view_query import ViewQueryService
from .workspace import WorkspaceService
from .writeback import WritebackService

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.io_manager import IOManager
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.runner import AnalyzeRunner, Runner, SaveDataRunner
    from zcu_tools.gui.services.cfg_editor import CfgEditorHost
    from zcu_tools.gui.services.view_query import _ViewQueryTarget
    from zcu_tools.gui.state import State


@dataclass(frozen=True)
class AppServices:
    """Immutable bundle of constructed domain services.

    Owns service construction and wiring so the Controller does not. The
    Controller holds this bundle and exposes services through its façade; the
    remote dispatch path reaches the same instances through the Controller.
    A single OperationGate is shared across the run / connection / device
    services so hardware exclusion is global.
    """

    operation_gate: OperationGate
    guard: GuardService
    view_query: ViewQueryService
    device: DeviceService
    connection: ConnectionService
    context: ContextService
    tab: TabService
    run: RunService
    analyze: AnalyzeService
    save: SaveService
    writeback: WritebackService
    tab_view: TabViewService
    workspace: WorkspaceService
    startup: StartupService
    cfg_editor: CfgEditorService


def build_app_services(
    *,
    state: "State",
    bus: "EventBus",
    registry: "Registry",
    io_manager: "IOManager",
    runner: "Runner",
    analyze_runner: "AnalyzeRunner",
    save_runner: "SaveDataRunner",
    view_provider: "Callable[[], _ViewQueryTarget]",
    cfg_editor_ctrl: "CfgEditorHost",
) -> AppServices:
    """Construct and wire every domain service into a frozen bundle.

    ``cfg_editor_ctrl`` is the Controller itself (the CfgEditor session's
    LiveModel env + ModuleLibrary registration surface). It is only stored, never
    called during construction, so passing the still-initialising Controller is
    safe and keeps the bundle complete (no service built outside this function).
    """
    operation_gate = OperationGate()
    device = DeviceService(bus, state, operation_gate)
    context = ContextService(state, io_manager, bus)
    tab = TabService(state, registry)
    writeback = WritebackService(state, bus)
    return AppServices(
        operation_gate=operation_gate,
        guard=GuardService(state),
        view_query=ViewQueryService(view_provider),
        device=device,
        connection=ConnectionService(state, bus, operation_gate),
        context=context,
        tab=tab,
        run=RunService(state, runner, bus, operation_gate),
        analyze=AnalyzeService(state, analyze_runner, bus),
        save=SaveService(state, save_runner, bus),
        writeback=writeback,
        tab_view=TabViewService(state, writeback),
        workspace=WorkspaceService(state, tab, SessionPersistenceService(), bus),
        startup=StartupService(
            context, device, StartupPersistenceService(), state, bus
        ),
        cfg_editor=CfgEditorService(
            cfg_editor_ctrl,
            ml_port=cfg_editor_ctrl,
            version_bump=cfg_editor_ctrl.bump_editor_version,
            version_drop=cfg_editor_ctrl.drop_editor_version,
            bus=bus,
        ),
    )
