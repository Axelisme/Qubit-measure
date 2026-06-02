from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .analyze import AnalyzeService
from .cfg_editor import CfgEditorService
from .connection import ConnectionService
from .context import ContextService
from .device import DeviceService
from .guard import GuardService
from .operation_gate import OperationGate
from .progress import ProgressService
from .run import RunService
from .save import SaveService
from .session_persistence import SessionPersistenceService
from .startup import StartupService
from .startup_persistence import StartupPersistenceService
from .tab import TabService
from .tab_view import TabViewService
from .workspace import WorkspaceService
from .writeback import WritebackService

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.io_manager import IOManager
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.runner import AnalyzeRunner, Runner, SaveDataRunner
    from zcu_tools.gui.services.cfg_editor import CfgEditorHost
    from zcu_tools.gui.services.ports import ProgressTransport
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
    progress: ProgressService
    guard: GuardService
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
    cfg_editor_ctrl: "CfgEditorHost",
    progress_transport: "ProgressTransport",
) -> AppServices:
    """Construct and wire every domain service into a frozen bundle.

    ``cfg_editor_ctrl`` is the Controller itself (the CfgEditor session's
    LiveModel env + ModuleLibrary registration surface). It is only stored, never
    called during construction, so passing the still-initialising Controller is
    safe and keeps the bundle complete (no service built outside this function).
    """
    operation_gate = OperationGate()
    progress = ProgressService(progress_transport)
    device = DeviceService(bus, state, operation_gate, progress=progress)
    context = ContextService(state, io_manager, bus)
    tab = TabService(state, registry)
    # cfg_editor owns the per-tab and per-writeback-item cfg models; WritebackService
    # builds/reads/tears those down, so it is built after cfg_editor (single-
    # direction command edge — cfg_editor never calls writeback, ADR-0007).
    cfg_editor = CfgEditorService(
        cfg_editor_ctrl,
        read_port=cfg_editor_ctrl,
        write_port=cfg_editor_ctrl,
        version_bump=cfg_editor_ctrl.bump_editor_version,
        version_drop=cfg_editor_ctrl.drop_editor_version,
        bus=bus,
    )
    writeback = WritebackService(state, bus, cfg_editor, write_port=cfg_editor_ctrl)
    return AppServices(
        operation_gate=operation_gate,
        progress=progress,
        guard=GuardService(state),
        device=device,
        connection=ConnectionService(state, bus, operation_gate),
        context=context,
        tab=tab,
        run=RunService(state, runner, bus, operation_gate, writeback, progress),
        analyze=AnalyzeService(state, analyze_runner, bus, writeback, operation_gate),
        save=SaveService(state, save_runner, bus),
        writeback=writeback,
        tab_view=TabViewService(state, writeback),
        workspace=WorkspaceService(state, tab, SessionPersistenceService(), bus),
        startup=StartupService(
            context, device, StartupPersistenceService(), state, bus
        ),
        cfg_editor=cfg_editor,
    )
