from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.connection import SoCConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.services.startup import StartupService

from .analyze import AnalyzeService
from .arb_waveform import ArbWaveformService
from .cfg_editor import CfgEditorService
from .guard import GuardService
from .load import LoadService
from .operation_gate import OperationGate
from .post_analyze import PostAnalyzeService
from .run import RunService
from .save import SaveService
from .tab import TabService
from .workspace import WorkspaceService
from .writeback import WritebackService

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.services.io_manager import IOManager

    from .cfg_editor import CfgEditorHost


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
    A single OperationRunner (the kind-agnostic lifecycle mechanism, ADR-0026 §1)
    is shared by run / analyze / post-analyze / device-setup operations.
    """

    operation_gate: OperationGate
    handles: OperationHandles
    background: BackgroundRunner
    progress: ProgressService
    guard: GuardService
    device: DeviceService
    device_control: DeviceControlPort
    soc_connection: SoCConnectionService
    predictor: PredictorService
    predictor_control: PredictorControlPort
    context: ContextService
    tab: TabService
    load: LoadService
    run: RunService
    analyze: AnalyzeService
    post_analyze: PostAnalyzeService
    save: SaveService
    writeback: WritebackService
    workspace: WorkspaceService
    startup: StartupService
    cfg_editor: CfgEditorService
    arb_waveform: ArbWaveformService


def build_app_services(
    *,
    state: State,
    bus: EventBus,
    registry: Registry,
    io_manager: IOManager,
    cfg_editor_ctrl: CfgEditorHost,
    progress_transport: ProgressTransport,
    project_root: str,
) -> AppServices:
    """Construct and wire every domain service into a frozen bundle.

    ``cfg_editor_ctrl`` is the Controller itself (the CfgEditor session's
    LiveModel env + ModuleLibrary registration surface). It is only stored, never
    called during construction, so passing the still-initialising Controller is
    safe and keeps the bundle complete (no service built outside this function).
    """
    operation_gate = OperationGate()
    handles = OperationHandles()
    background = BackgroundRunner()
    progress = ProgressService(progress_transport)
    # OperationRunner: the kind-agnostic lifecycle mechanism (ADR-0026 §1).
    # Shared by run / FIT-analyze / post-analyze / device ops. Interactive analyze
    # does not use runner (main-thread-user-paced, stage2c_spec.md).
    runner = OperationRunner(operation_gate, handles, progress, background)
    session = build_session_services(
        state=state,
        bus=bus,
        gate=operation_gate,
        handles=handles,
        background=background,
        progress=progress,
        io_manager=io_manager,
        runner=runner,
        project_root=project_root,
    )
    context = session.context
    device = session.device
    arb_waveform = ArbWaveformService(state)
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
        device_control=session.device_control,
        soc_connection=session.soc_connection,
        predictor=session.predictor,
        predictor_control=session.predictor_control,
        context=context,
        tab=tab,
        load=LoadService(state, bus, writeback),
        run=RunService(state, runner, bus, handles, writeback),
        analyze=AnalyzeService(state, runner, bus, writeback, handles),
        # Second analysis layer (post_analysis cap) — handle-only off-main worker,
        # same runner/handles as the primary analyze (ADR-0019, ADR-0026 §1).
        post_analyze=PostAnalyzeService(state, runner, bus, handles),
        save=SaveService(state, background, bus),
        writeback=writeback,
        workspace=WorkspaceService(state, tab, bus),
        startup=session.startup,
        cfg_editor=cfg_editor,
        arb_waveform=arb_waveform,
    )
