from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.session.operation_handles import OperationHandles

from .analyze import AnalyzeService
from .background import BackgroundService
from .cfg_editor import CfgEditorService
from .connection import ConnectionService
from .context import ContextService
from .device import DeviceService
from .guard import GuardService
from .operation_gate import OperationGate
from .progress import ProgressService
from .run import RunService
from .save import SaveService
from .startup import StartupService
from .tab import TabService
from .workspace import WorkspaceService
from .writeback import WritebackService

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.event_bus import EventBus
    from zcu_tools.gui.app.main.io_manager import IOManager
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.state import State

    from .cfg_editor import CfgEditorHost
    from .ports import ProgressTransport


@dataclass(frozen=True)
class AppServices:
    """Immutable bundle of constructed domain services.

    Owns service construction and wiring so the Controller does not. The
    Controller holds this bundle and exposes services through its façade; the
    remote dispatch path reaches the same instances through the Controller.
    A single OperationGate (Exclusion) is shared across run / connection / device
    so hardware exclusion is global; a single OperationHandles (the async
    Handle / Cancel facet) is shared across those plus analyze (handle-only, no
    exclusion) so operation_id / await / cancel are uniform (ADR-0019).
    """

    operation_gate: OperationGate
    handles: OperationHandles
    background: BackgroundService
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
    workspace: WorkspaceService
    startup: StartupService
    cfg_editor: CfgEditorService


def build_app_services(
    *,
    state: "State",
    bus: "EventBus",
    registry: "Registry",
    io_manager: "IOManager",
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
    handles = OperationHandles()
    background = BackgroundService()
    progress = ProgressService(progress_transport)
    device = DeviceService(
        bus, state, operation_gate, handles, background, progress=progress
    )
    context = ContextService(state, io_manager, bus)
    # cfg_editor owns the per-tab and per-writeback-item cfg models; WritebackService
    # builds/reads/tears those down, so it is built after cfg_editor (single-
    # direction command edge — cfg_editor never calls writeback, ADR-0004).
    cfg_editor = CfgEditorService(
        cfg_editor_ctrl,
        read_port=cfg_editor_ctrl,
        write_port=cfg_editor_ctrl,
        version_bump=cfg_editor_ctrl.bump_editor_version,
        version_drop=cfg_editor_ctrl.drop_editor_version,
        bus=bus,
    )
    writeback = WritebackService(state, bus, cfg_editor, write_port=cfg_editor_ctrl)
    # TabService composes the tab render model and needs the writeback query port
    # (built above) — built after writeback (read-model dependency, ADR-0005).
    tab = TabService(state, registry, writeback)
    return AppServices(
        operation_gate=operation_gate,
        handles=handles,
        background=background,
        progress=progress,
        guard=GuardService(state),
        device=device,
        connection=ConnectionService(state, bus, operation_gate, handles),
        context=context,
        tab=tab,
        run=RunService(
            state, background, bus, operation_gate, handles, writeback, progress
        ),
        analyze=AnalyzeService(state, background, bus, writeback, handles),
        save=SaveService(state, background, bus),
        writeback=writeback,
        workspace=WorkspaceService(state, tab, bus),
        startup=StartupService(context, device, state),
        cfg_editor=cfg_editor,
    )
