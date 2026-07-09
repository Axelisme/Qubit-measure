from __future__ import annotations

from collections.abc import Callable
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
from .operation_control import OperationControlFacet
from .operation_gate import OperationGate
from .post_analyze import PostAnalyzeService
from .run import RunService
from .run_analyze_control import RunAnalyzeControlFacet
from .save import SaveService
from .save_control import SaveControlFacet
from .tab import TabService
from .tab_control import TabControlFacet
from .workspace import WorkspaceService
from .writeback import WritebackService

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.services.io_manager import IOManager
    from zcu_tools.gui.session.setup_control import SetupControlPort

    from .cfg_editor import CfgEditorHost
    from .operation_control import OperationControlPort
    from .run_analyze_control import RunAnalyzeControlPort, RunAnalyzeRenderHost
    from .save_control import SaveControlPort
    from .tab_control import TabControlPort


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
    operation_control: OperationControlPort
    handles: OperationHandles
    background: BackgroundRunner
    progress: ProgressService
    guard: GuardService
    device: DeviceService
    device_control: DeviceControlPort
    soc_connection: SoCConnectionService
    predictor: PredictorService
    predictor_control: PredictorControlPort
    progress_control: ProgressControlPort
    context: ContextService
    context_control: ContextControlPort
    setup_control: SetupControlPort
    tab: TabService
    tab_control: TabControlPort
    run_analyze_control: RunAnalyzeControlPort
    load: LoadService
    run: RunService
    analyze: AnalyzeService
    post_analyze: PostAnalyzeService
    save: SaveService
    save_control: SaveControlPort
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
    notify_info: Callable[[str], None],
    render_host: Callable[[], RunAnalyzeRenderHost | None],
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
    workspace = WorkspaceService(state, tab, bus)
    tab_control = TabControlFacet(
        state=state,
        tab=tab,
        workspace=workspace,
        bus=bus,
    )
    guard = GuardService(state)
    load = LoadService(state, bus, writeback)
    run = RunService(state, runner, bus, handles, writeback)
    analyze = AnalyzeService(state, runner, bus, writeback, handles)
    post_analyze = PostAnalyzeService(state, runner, bus, handles)
    save = SaveService(state, background, bus)
    run_analyze_control = RunAnalyzeControlFacet(
        state=state,
        bus=bus,
        guard=guard,
        tab=tab,
        load=load,
        run=run,
        analyze=analyze,
        post_analyze=post_analyze,
        render_host=render_host,
    )
    operation_control = OperationControlFacet(handles=handles, progress=progress)
    save_control = SaveControlFacet(
        state=state,
        bus=bus,
        guard=guard,
        tab=tab,
        save=save,
        notify_info=notify_info,
    )
    return AppServices(
        operation_gate=operation_gate,
        operation_control=operation_control,
        handles=handles,
        background=background,
        progress=progress,
        guard=guard,
        device=device,
        device_control=session.device_control,
        soc_connection=session.soc_connection,
        predictor=session.predictor,
        predictor_control=session.predictor_control,
        progress_control=session.progress_control,
        context=context,
        context_control=session.context_control,
        setup_control=session.setup_control,
        tab=tab,
        tab_control=tab_control,
        run_analyze_control=run_analyze_control,
        load=load,
        run=run,
        analyze=analyze,
        # Second analysis layer (post_analysis cap) — handle-only off-main worker,
        # same runner/handles as the primary analyze (ADR-0019, ADR-0026 §1).
        post_analyze=post_analyze,
        save=save,
        save_control=save_control,
        writeback=writeback,
        workspace=workspace,
        startup=session.startup,
        cfg_editor=cfg_editor,
        arb_waveform=arb_waveform,
    )
