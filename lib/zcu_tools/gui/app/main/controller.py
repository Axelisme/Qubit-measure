from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

logger = logging.getLogger(__name__)

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import AwaitResult, FeedbackInbox
from zcu_tools.gui.session.services.connection import (
    ConnectRequest,
    LoadPredictorRequest,
    PredictCurveRequest,
    PredictCurveResult,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictMatrixCurveResult,
)
from zcu_tools.gui.session.services.device import (
    ActiveDeviceOperation,
    DeviceEntry,
    DeviceSetupSnapshot,
)
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.mock_flux import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapter import (
    AnalysisMode,
    AnalyzeRequest,
    CfgSchema,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    SavePaths,
    SocCfgHandle,
    WritebackItem,
)
from .events.tab import TabContentChangedPayload, TabInteractionChangedPayload
from .registry import Registry
from .role_catalog import RoleCatalog
from .services import (
    AppPersistedState,
    ConnectDeviceRequest,
    DeviceSnapshot,
    DisconnectDeviceRequest,
    PersistedStartup,
    PersistenceCaretaker,
    PersistenceError,
    RestoreReport,
    SaveResultOutcome,
    SetupDeviceRequest,
    StartupConnectionRequest,
    StartupProjectRequest,
    TabSnapshot,
    build_app_services,
)
from .services.cfg_lowering import lower_module, lower_waveform
from .services.ports import ContextWrites
from .services.remote.dialogs import DialogName
from .state import State

if TYPE_CHECKING:
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.ports import ProgressTransport

    from .adapters.qt_shutdown_driver import QtShutdownDriver
    from .guard import AnalyzePermit


# A View has two distinct down-channels from the Controller (ADR-0013):
#   - diagnostics (error / info) — fanned out to *every* attached View;
#   - render help (pbar / live container) — pulled from the *one* View that has
#     a real canvas.
# So the interface splits by channel, not lumped into a single "View".
# Render/snapshot/dialog *queries* are pulled by the RemoteControlAdapter
# through its own ``render_view`` (see RenderView), not by the Controller.

Severity = Literal["error", "info"]

# FLUX-AWARE-MOCK: the provisioning now lives in the shared session layer
# (gui/session/services/mock_flux) so both measure and autofluxdep reuse one
# implementation. Re-exported here for the existing tests that import the names.
_FAKE_FLUX_DEVICE_NAME = FAKE_FLUX_DEVICE_NAME
_FAKE_FLUX_INITIAL_VALUE = FAKE_FLUX_INITIAL_VALUE


class DiagnosticSink(Protocol):
    """A View that can receive a Controller diagnostic (error / info).

    Fanned out to every attached sink. Each View renders it its own way — the
    Qt window pops a dialog (error) or status bar (info); the remote adapter
    enqueues a diagnostic line. Diagnostics never go through EventBus (a channel
    that reports a fault must not be the faulty channel — ADR-0013).
    """

    def notify_diagnostic(
        self, severity: Severity, title: str, message: str
    ) -> None: ...


class RenderHost(Protocol):
    """The single canvas-bearing View the Controller asks for run/analyze Qt
    artefacts (the live figure container). Progress bars no longer come from the
    View — they are minted by ProgressService, bound to the operation."""

    def make_live_container(self, tab_id: str) -> FigureContainer | None:
        """The tab's single right-pane figure container, shared by run, analyze,
        and post-analysis (the container always shows the most recently produced
        figure). Headless Views may return None."""
        ...

    def mount_interactive_analysis(
        self,
        tab_id: str,
        session_factory: Callable[[InteractiveHost], InteractiveSession],
        on_finish: Callable[[InteractiveSession], None],
    ) -> None:
        """Mount an interactive analysis on a tab's analysis area (INTERACTIVE
        adapters). The View builds an ``InteractiveHost`` (its canvas + worker
        pool), calls ``session_factory(host)`` to get the adapter's session,
        renders its actions / forwards canvas events, and on Done calls
        ``on_finish(session)``. The View knows nothing of the interaction."""
        ...

    # The View's current left-panel width — the only persistence value sourced
    # from the View (the splitter widget); captured at flush time.
    def current_left_panel_width(self) -> int: ...


class RenderView(Protocol):
    """Pure-read View surface the RemoteControlAdapter pulls from (snapshot /
    screenshot / dialog management). Held by the adapter, not the Controller."""

    def get_view_snapshot(self) -> dict[str, object]: ...
    def take_figure_screenshot(self, tab_id: str) -> bytes: ...
    def take_dialog_screenshot(self, dialog_name: Any) -> bytes: ...
    def open_dialog(self, name: DialogName) -> None: ...
    def close_dialog(self, name: DialogName) -> None: ...
    def list_open_dialogs(self) -> list[DialogName]: ...
    def register_dialog(self, name: DialogName, dialog: Any) -> None: ...
    def request_shutdown(self) -> None: ...


class ViewProtocol(DiagnosticSink, RenderHost, RenderView, Protocol):
    """A full Qt View (``MainWindow``) implements all three channels."""


class Controller:
    """Façade for the GUI application. Delegates to domain services."""

    def __init__(
        self,
        state: State,
        registry: Registry,
        io_manager: IOManager,
        view: ViewProtocol | None,
        bus: EventBus,
        role_catalog: RoleCatalog | None = None,
        progress_transport: ProgressTransport | None = None,
        project_root: str | None = None,
    ) -> None:
        self._state = state
        # Base directory the default per-qubit result/database paths are anchored
        # under. The entry script injects the repo root (Path(__file__).parent...)
        # so a .bat launcher that cd's into script/ does not scope defaults under
        # script/. None falls back to cwd — fine for tests / direct `python -m`
        # runs from the repo root, where cwd already IS the repo root.
        import os

        self._project_root = project_root if project_root is not None else os.getcwd()
        # Views attached as diagnostic sinks (fan-out target). A full Qt View is
        # also the single RenderHost (run/analyze Qt artefacts). The remote
        # adapter is a diagnostic-only View — it holds its own RenderView.
        self._diag_sinks: list[DiagnosticSink] = []
        self._render_host: RenderHost | None = None
        if view is not None:
            self.add_view(view)
        self._bus = bus
        # Catalog of experiment-role templates (gui interface, populated by
        # experiment/v2_gui at startup). Optional so tests can construct a bare
        # Controller; create_from_role fails fast when absent.
        self._role_catalog = role_catalog

        # Construct and wire every domain service into an immutable bundle, then
        # alias them onto self for the façade's call sites.
        # ``cfg_editor_ctrl=self`` lets build_app_services build CfgEditorService
        # in the same bundle: the session only stores the Controller (as its
        # LiveModel env + ModuleLibrary registration surface) and never calls it
        # during construction, so passing the still-initialising self is safe.
        # The progress transport is the Qt marshal (a driven adapter). Default to
        # the Qt one so GUI/agent processes (which run a Qt event loop) work
        # without the entry point wiring it; tests inject a synchronous fake.
        transport: ProgressTransport
        if progress_transport is not None:
            transport = progress_transport
        else:
            from zcu_tools.gui.session.adapters.qt_progress_transport import (
                QtProgressTransport,
            )

            transport = QtProgressTransport()
        services = build_app_services(
            state=state,
            bus=bus,
            registry=registry,
            io_manager=io_manager,
            cfg_editor_ctrl=self,
            progress_transport=transport,
        )
        self._services = services
        self._operation_gate = services.operation_gate
        self._operation_handles = services.handles
        # ADR-0023: session-scoped feedback inbox (thread-safe; not in State).
        # Wired into OperationHandles so await_outcome can deliver feedback as a
        # second wakeup source without touching main-thread-owned State.
        self._feedback_inbox = FeedbackInbox()
        self._operation_handles.set_feedback_inbox(self._feedback_inbox)
        self._background_svc = services.background
        self._progress_svc = services.progress
        self._guard_svc = services.guard
        self._dev_svc = services.device
        self._conn_svc = services.connection
        self._ctx_svc = services.context
        self._tab_svc = services.tab
        self._run_svc = services.run
        self._analyze_svc = services.analyze
        self._post_analyze_svc = services.post_analyze
        self._save_svc = services.save
        self._writeback_svc = services.writeback
        self._workspace_svc = services.workspace
        self._startup_svc = services.startup
        self._cfg_editor_svc = services.cfg_editor
        self._agent_chat_svc = services.agent_chat
        # App-level PersistenceCaretaker, injected by run_app via attach_caretaker
        # (None in bare-Controller tests that don't exercise persistence).
        self._caretaker: PersistenceCaretaker | None = None
        # Lazily built on first begin_shutdown so the Controller stays importable
        # without a Qt event loop (tests construct a bare Controller). The driver
        # is a Qt adapter owning the QTimer that pumps the Qt-free coordinator.
        self._shutdown_driver: QtShutdownDriver | None = None

        self._run_svc.run_finished.connect(self._on_run_finished)
        self._run_svc.run_failed.connect(self._on_run_failed)
        self._analyze_svc.analyze_finished.connect(self._on_analyze_finished)
        self._analyze_svc.analyze_failed.connect(self._on_analyze_failed)
        self._post_analyze_svc.post_analyze_finished.connect(
            self._on_post_analyze_finished
        )
        self._post_analyze_svc.post_analyze_failed.connect(self._on_post_analyze_failed)
        self._save_svc.save_finished.connect(self._on_save_finished)
        self._save_svc.save_failed.connect(self._on_save_failed)
        self._save_svc.save_result_finished.connect(self._on_save_result_finished)
        self._dev_svc.setup_finished.connect(self._on_device_setup_finished)
        self._dev_svc.setup_failed.connect(self._on_device_setup_failed)
        self._dev_svc.setup_cancelled.connect(self._on_device_setup_cancelled)
        self._dev_svc.device_connected.connect(self._on_device_connected)
        self._dev_svc.device_disconnected.connect(self._on_device_disconnected)
        self._dev_svc.operation_failed.connect(self._on_device_operation_failed)
        # FLUX-AWARE-MOCK: the mock flux source provisioning (SOC_CHANGED hook +
        # fake_flux register/reconnect/ramp) is owned by the shared session-layer
        # MockFluxProvisioner (built in build_session_services), so this controller
        # carries no provisioning code of its own.

    def add_view(self, view: ViewProtocol) -> None:
        """Attach a full Qt View. It is a diagnostic sink (fan-out target) and
        the single RenderHost / RenderView (run/analyze Qt artefacts + pure-read
        render queries). Attaching a second replaces the render role (last
        writer wins); diagnostic sinks accumulate."""
        if view not in self._diag_sinks:
            self._diag_sinks.append(view)
        self._render_host = view

    def add_diagnostic_sink(self, sink: DiagnosticSink) -> None:
        """Attach a diagnostic-only View (e.g. the remote adapter): it receives
        error/info fan-out but is not a RenderHost."""
        if sink not in self._diag_sinks:
            self._diag_sinks.append(sink)

    def remove_diagnostic_sink(self, sink: DiagnosticSink) -> None:
        if sink in self._diag_sinks:
            self._diag_sinks.remove(sink)

    def _notify(self, severity: Severity, title: str, message: str) -> None:
        """Fan a diagnostic out to every attached View (ADR-0013). Never via
        EventBus — diagnostics must not depend on the channel they report on.
        Also records to AgentChatService so the transcript shows GUI events."""
        for sink in list(self._diag_sinks):
            sink.notify_diagnostic(severity, title, message)
        # Best-effort — transcript is display-only; never let it block diagnostics.
        try:
            self._agent_chat_svc.record_diagnostic(severity, title, message)
        except Exception:
            logger.exception("AgentChatService.record_diagnostic raised; ignoring")

    def _info(self, message: str) -> None:
        """Transient status diagnostic (no title) — Qt status bar / wire line."""
        self._notify("info", "", message)

    def _report_persistence_error(self, title: str, error: Exception) -> None:
        self._notify("error", title, str(error))

    def _on_run_finished(self, tab_id: str, _result: object) -> None:
        # State is already updated in RunService. Only adapters that do
        # analysis (mode != NONE) are routed into analyze-params init; the NONE
        # 2D sweeps (flux_dep / power_dep) have no analyze step, and their base
        # ``get_analyze_params`` is a Fast-Fail guard — never call it for them.
        if (
            self._state.get_tab(tab_id).adapter.capabilities.analysis
            is not AnalysisMode.NONE
        ):
            self._tab_svc.initialize_tab_analyze_params(tab_id)
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))

    def _on_run_failed(self, _tab_id: str, error: Exception) -> None:
        self._notify("error", "Run failed", str(error))

    def _on_analyze_finished(self, tab_id: str, _result: object) -> None:
        # A fresh primary analyze result seeds the post-analysis params (mirrors
        # how a finished run seeds the analyze params). Only adapters declaring
        # post_analysis are routed there; the base get_post_analyze_params is a
        # Fast-Fail guard. The primary result is already in State (AnalyzeService).
        if self._state.get_tab(tab_id).adapter.capabilities.post_analysis:
            self._tab_svc.initialize_tab_post_analyze_params(tab_id)
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))

    def _on_analyze_failed(self, _tab_id: str, error: Exception) -> None:
        self._notify("error", "Analyze failed", str(error))

    def _on_post_analyze_finished(self, tab_id: str, _result: object) -> None:
        # The post result + figure are already in State (PostAnalyzeService);
        # emit the content event so the View refreshes the Post sub-tab.
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))

    def _on_post_analyze_failed(self, _tab_id: str, error: Exception) -> None:
        self._notify("error", "Post-analysis failed", str(error))

    def _on_save_finished(self, tab_id: str, data_path: str) -> None:
        del tab_id
        self._info(f"Data saved to {data_path}")

    def _on_save_failed(self, tab_id: str, data_path: str, error: Exception) -> None:
        del tab_id, data_path
        self._notify("error", "Save data failed", str(error))

    def _on_save_result_finished(self, tab_id: str, outcome: SaveResultOutcome) -> None:
        del tab_id
        if outcome.data_error is None and outcome.image_error is None:
            self._info(
                f"Data saved to {outcome.data_path}; "
                f"image saved to {outcome.image_path}"
            )
            return
        if outcome.data_error is None:
            self._info(
                f"Data saved to {outcome.data_path}; image failed: {outcome.image_error}"
            )
            return
        if outcome.image_error is None:
            self._info(
                f"Data failed: {outcome.data_error}; "
                f"image saved to {outcome.image_path}"
            )
            return
        self._notify(
            "error",
            "Save Result failed",
            f"Data failed: {outcome.data_error}\nImage failed: {outcome.image_error}",
        )

    def _on_device_setup_finished(self, name: str) -> None:
        self._info(f"Device setup completed: {name}")

    def _on_device_setup_failed(self, name: str, error: str) -> None:
        self._notify(
            "error",
            f"Device setup failed: {name}",
            error,
        )

    def _on_device_setup_cancelled(self, name: str) -> None:
        self._info(f"Device setup cancelled: {name}")

    def _on_device_connected(self, req: ConnectDeviceRequest) -> None:
        # Persistence is a projection of device State, driven by StartupService's
        # DEVICE_CHANGED subscription — this handler only presents UI feedback.
        self._info(f"Device connected: {req.name}")

    def _on_device_disconnected(self, req: DisconnectDeviceRequest) -> None:
        self._info(f"Device disconnected: {req.name}")

    def _on_device_operation_failed(self, name: str, error: str) -> None:
        self._notify(
            "error",
            f"Device operation failed: {name}",
            error,
        )

    def get_bus(self) -> EventBus:
        return self._bus

    def attach_caretaker(self, caretaker: PersistenceCaretaker) -> None:
        """Wire the app-level PersistenceCaretaker (built by run_app). The
        Controller is the Memento Originator; the Caretaker owns disk I/O."""
        self._caretaker = caretaker

    # -- Memento Originator (PersistOriginatorPort) --------------------------

    def capture_persisted_state(self) -> AppPersistedState:
        """Snapshot the whole app state into a memento (no disk). Composes the
        startup prefs + device projection + view's left-panel width and the live
        tabs into one immutable ``AppPersistedState``."""
        startup = self._startup_svc.capture_startup(
            left_panel_width=self._capture_left_panel_width()
        )
        session = self._workspace_svc.capture_session()
        return AppPersistedState(startup=startup, session=session)

    def restore_persisted_state(self, state: AppPersistedState) -> RestoreReport:
        """Dispatch a memento back to the sub-owners: seed startup prefs +
        register remembered devices, then rebuild tabs. Returns the session's
        per-tab restore report (presented to the user by ``restore_all``)."""
        self._startup_svc.restore_startup(state.startup)
        report = self._workspace_svc.apply_session(state.session)
        self._present_restore_report(report)
        return report

    # -- lifecycle façade (run_app startup / MainWindow close) ---------------

    def restore_all(self, *, load: bool = True) -> None:
        assert self._caretaker is not None, "caretaker not attached"
        outcome = self._caretaker.restore_all(load=load)
        if outcome.load_error is not None:
            self._report_persistence_error(
                "Settings restore failed", outcome.load_error
            )

    def persist_all(self) -> None:
        assert self._caretaker is not None, "caretaker not attached"
        try:
            self._caretaker.flush()
        except PersistenceError as exc:
            self._report_persistence_error("Settings save failed", exc)

    def _capture_left_panel_width(self) -> int:
        from .services.persistence_types import DEFAULT_LEFT_PANEL_WIDTH

        host = self._render_host
        if host is None:
            return DEFAULT_LEFT_PANEL_WIDTH
        return host.current_left_panel_width()

    def _present_restore_report(self, report: RestoreReport) -> None:
        if report.rejected_tabs:
            self._notify(
                "error",
                "Some session tabs were not restored",
                "\n".join(
                    f"{issue.subject}: {issue.message}"
                    for issue in report.rejected_tabs
                ),
            )

    # ------------------------------------------------------------------
    # ExpTab operations (TabService)
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        return self._workspace_svc.new_tab(adapter_name)

    def close_tab(self, tab_id: str) -> None:
        self._workspace_svc.close_tab(tab_id)

    def set_active_tab(self, tab_id: str) -> None:
        self._workspace_svc.set_active_tab(tab_id)

    # ------------------------------------------------------------------
    # Run flow (RunService & ContextService)
    # ------------------------------------------------------------------

    def has_project(self) -> bool:
        return self._ctx_svc.has_project()

    def has_context(self) -> bool:
        return self._ctx_svc.has_context()

    def has_startup_context(self) -> bool:
        return self._ctx_svc.has_startup_context()

    def has_active_context(self) -> bool:
        return self._ctx_svc.is_active_context()

    def get_running_tab_id(self) -> str | None:
        return self._state.running_tab_id

    def has_soc(self) -> bool:
        return self._conn_svc.has_soc()

    def get_soc_info(self) -> dict[str, object]:
        """Hardware summary of the connected SoC (QICK soccfg): a compact
        per-channel description (generator/readout type, converter port, sample
        rate, max pulse/buffer length) + structured cfg with the full detail.
        Raises if no SoC is connected (→ precondition_failed)."""
        import json

        from zcu_tools.program import describe_soc

        soccfg = self._conn_svc.get_soccfg()
        if soccfg is None:
            raise RuntimeError("No SoC connected")
        return {
            "description": describe_soc(soccfg),
            "cfg": json.loads(soccfg.dump_cfg()),
            "is_mock": self._conn_svc.is_mock_soc(),
        }

    def resources_versions(self) -> dict[str, int]:
        """Full resource-version snapshot (the resources.versions RPC payload)."""
        return self._state.version.snapshot()

    def start_run(self, tab_id: str) -> int:
        # GuardService proves context readiness + committed-cfg validity + soc
        # capability, and carries the worker payload in the permit. Both clients
        # acquire through the same path so guard logic cannot drift. Returns the
        # operation token (handle for operation.await).
        #
        # Terminal: this only launches the worker. The outcome lands on the Qt
        # main thread in ``RunService._on_run_finished`` / ``_on_run_failed``,
        # which bump ``tab:<id>:result`` (finished only), release the RUN lease
        # exactly-once, and emit ``RUN_FINISHED`` carrying the outcome in its
        # payload (finished / failed / cancelled).
        permit = self._guard_svc.acquire_run_permit(tab_id)
        # Headless (no Qt RenderHost, e.g. a pure-agent process): RunService
        # tolerates a None live container. The progress factory is minted by
        # RunService from ProgressService (bound to the operation), not the View.
        host = self._render_host
        live_container = host.make_live_container(tab_id) if host is not None else None
        return self._run_svc.start_run(permit, live_container)

    def cancel_run(self) -> None:
        self._run_svc.cancel_run()

    # ------------------------------------------------------------------
    # Shutdown coordination (cancel-all + wait, ADR-0003)
    # ------------------------------------------------------------------

    def active_operation_count(self) -> int:
        """How many operations (run / device / connect) are live right now.

        The View reads this before closing to decide whether to confirm with the
        user (a non-zero count means closing will cancel work in progress).
        Counts all live operations (run / device / connect AND analyze /
        interactive) — Handles owns the lifecycle (ADR-0019)."""
        return self._operation_handles.live_count()

    def begin_shutdown(self, on_closed: Callable[[], None]) -> None:
        """Cancel every live operation, wait (with a timeout) for them to stop,
        then run ``on_closed`` — the View's actual teardown.

        Qt-free façade: the QTimer-driven coordinator lives in a driving adapter
        (ADR-0005), built lazily here so the Controller stays importable without
        a Qt loop. ``on_closed`` always runs on the main thread."""
        if self._shutdown_driver is None:
            from .adapters.qt_shutdown_driver import QtShutdownDriver

            self._shutdown_driver = QtShutdownDriver(self._operation_handles)
        self._shutdown_driver.begin(on_closed)

    def get_operation_progress(self, operation_id: int) -> tuple:
        """Live progress bars for one operation by id (run / device setup alike).
        The wire's operation.progress handler reads this; the mcp poll folds it
        into its reply."""
        return self._progress_svc.bars_for_operation(operation_id)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_analyze_result(tab_id)

    # ------------------------------------------------------------------
    # Analyze flow (TabService)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, analyze_params_instance: object) -> int:
        """Start analyze; returns the operation token to await.

        FIT runs on a worker. INTERACTIVE mounts a live picker on the main thread
        (no worker) and the lease is held until the user clicks Done — see
        ``_start_interactive_analyze``.
        """
        permit = self._guard_svc.acquire_analyze_permit(tab_id)
        tab = self._state.get_tab(tab_id)
        if tab.adapter.capabilities.analysis is AnalysisMode.INTERACTIVE:
            return self._start_interactive_analyze(
                tab_id, permit, analyze_params_instance
            )
        host = self._render_host
        figure_container = (
            host.make_live_container(tab_id) if host is not None else None
        )
        return self._analyze_svc.start_analyze(
            permit, analyze_params_instance, figure_container
        )

    def _start_interactive_analyze(
        self, tab_id: str, permit: AnalyzePermit, analyze_params_instance: object
    ) -> int:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        # Acquire the handle (marks analyzing) first, then ask the View to mount
        # the interactive canvas. The View builds the InteractiveHost (its canvas
        # + worker pool), gets the adapter's session, and wires Done back to
        # AnalyzeService.finish_interactive — which runs the same terminal path as
        # a FIT result.
        token = self._analyze_svc.start_interactive(permit)
        host = self._render_host
        if host is not None:
            host.mount_interactive_analysis(
                tab_id,
                lambda ihost: tab.adapter.setup_interactive_analysis(req, ihost),
                lambda session: self._analyze_svc.finish_interactive(tab_id, session),
            )
        return token

    # ------------------------------------------------------------------
    # Post-analysis flow (second layer; PostAnalyzeService)
    # ------------------------------------------------------------------

    def start_post_analyze(
        self, tab_id: str, post_analyze_params_instance: object
    ) -> int:
        """Start the second-layer analysis on a tab's primary analyze result.

        Returns the operation token. PostAnalyzeService gates on tab-busy + a
        primary analyze result existing. The post figure live-plots into the same
        shared right-pane container as run/analyze (the container shows the most
        recent figure)."""
        host = self._render_host
        figure_container = (
            host.make_live_container(tab_id) if host is not None else None
        )
        return self._post_analyze_svc.start_post_analyze(
            tab_id, post_analyze_params_instance, figure_container
        )

    def get_post_analyze_result(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_post_analyze_result(tab_id)

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._tab_svc.update_tab_post_analyze_param_instance(tab_id, instance)

    def run_background(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        """InteractiveHostEnv (ADR-0019): run a short interactive compute off-main
        via BackgroundService's pool, delivering the result to ``on_done`` on the
        main thread. The interactive host has no error channel, so a failure is
        logged (the user keeps the current picker state)."""
        self._background_svc.submit(
            compute,
            run_in_pool=True,
            on_done=on_done,
            on_error=lambda exc: logger.warning(
                "interactive run_background failed: %r", exc
            ),
        )

    # ------------------------------------------------------------------
    # Writeback (TabService)
    # ------------------------------------------------------------------

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        """Recompute the tab's writeback proposals (read-only, no permit).

        Returns [] when the tab has no run/analyze result yet.
        """
        return list(self._writeback_svc.get_tab_writeback_items(tab_id))

    def apply_writeback(self, tab_id: str) -> list[str]:
        """Apply the tab's persistent writeback draft as-is (no recompute)."""
        permit = self._guard_svc.acquire_writeback_permit(tab_id)
        return self._writeback_svc.apply_tab_writeback(permit)

    def set_writeback_item(self, tab_id: str, session_id: str, **changes: Any) -> None:
        """Edit a persistent writeback item (selected / target_name / value)."""
        self._guard_svc.acquire_writeback_permit(tab_id)
        self._writeback_svc.set_item_field(tab_id, session_id, **changes)

    # ------------------------------------------------------------------
    # Save (TabService)
    # ------------------------------------------------------------------

    def _resolve_save_paths(self, tab_id: str) -> SavePaths:
        paths = self._tab_svc.get_tab_save_paths(tab_id)
        if paths is None:
            raise RuntimeError(
                f"Tab {tab_id!r} has no save paths configured — "
                "set paths via the Save panel or update_tab_save_paths()."
            )
        return paths

    def save_data(
        self, tab_id: str, data_path: str | None = None, comment: str = ""
    ) -> str:
        """Start the (async) data save; returns the path the saver will write
        (``.hdf5`` + uniqueness suffix already applied), known synchronously."""
        permit = self._guard_svc.acquire_save_permit(tab_id)
        resolved = data_path or self._resolve_save_paths(tab_id).data_path
        return self._save_svc.start_save_data(permit, resolved, comment=comment)

    def save_image(self, tab_id: str, image_path: str | None = None) -> str:
        """Save the image synchronously; returns the written image path."""
        permit = self._guard_svc.acquire_save_permit(tab_id)
        resolved = image_path or self._resolve_save_paths(tab_id).image_path
        self._save_svc.save_image_sync(permit, resolved)
        self._info(f"Image saved to {resolved}")
        return resolved

    def save_post_image(self, tab_id: str, image_path: str | None = None) -> str:
        """Save the post-analysis figure synchronously; returns the written path.

        The post sub-tab's own Save Image — targets ``tab.post_figure`` (distinct
        from the primary ``save_image``). Reuses the save permit (active context +
        run result); the View additionally gates the button on a post result."""
        permit = self._guard_svc.acquire_save_permit(tab_id)
        resolved = image_path or self._resolve_save_paths(tab_id).image_path
        self._save_svc.save_post_image_sync(permit, resolved)
        self._info(f"Post-analysis image saved to {resolved}")
        return resolved

    def save_result(
        self,
        tab_id: str,
        data_path: str | None = None,
        image_path: str | None = None,
        comment: str = "",
    ) -> tuple[str, str]:
        """Save image (sync) + data (async); returns ``(data_path, image_path)``
        the saver will write — the data path with its ``.hdf5``/suffix applied."""
        permit = self._guard_svc.acquire_save_permit(tab_id)
        paths = self._resolve_save_paths(tab_id)
        resolved_data = data_path or paths.data_path
        resolved_image = image_path or paths.image_path
        written_data = self._save_svc.start_save_result(
            permit, resolved_data, resolved_image, comment=comment
        )
        return written_data, resolved_image

    # ------------------------------------------------------------------
    # Context / IO (ContextService)
    # ------------------------------------------------------------------

    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        # Applies the project to the active context AND records it as the
        # remembered prefs (in State); persisted to disk only at close.
        self._startup_svc.apply_project(req)
        return True

    def use_context(self, label: str) -> None:
        self._ctx_svc.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        """Create a new flux context, optionally bound to a flux device.

        ``bind_device`` (a connected device name) decides the flux unit/value:
        the unit comes from the device-type whitelist (Fast-Fail if the device
        is unknown or its type is not whitelisted) and the value is *read* from
        the device's current state (never set). ``bind_device=None`` makes an
        unbound context (unit="none", no value). ``clone_from`` is the label of
        an existing context to clone its ml/md from. The new context's label is
        derived automatically by ``ExperimentManager`` — the agent cannot name
        it directly.
        """
        if bind_device is not None:
            unit = self._dev_svc.get_device_unit_strict(bind_device)
            value = self._dev_svc.get_device_value_for_new_context(bind_device)
        else:
            unit, value = "none", None
        self._ctx_svc.new_context(value=value, unit=unit, clone_from=clone_from)

    def get_active_context_label(self) -> str | None:
        return self._ctx_svc.get_active_context_label()

    def get_flux_dir(self) -> str | None:
        return self._ctx_svc.get_flux_dir()

    def get_context_labels(self) -> list[str]:
        return self._ctx_svc.get_context_labels()

    def get_current_md(self) -> MetaDict:
        return self._ctx_svc.get_current_md()

    def set_md_attr(self, key: str, value: Any) -> None:
        self._ctx_svc.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._ctx_svc.del_md_attr(key)

    def get_current_ml(self) -> ModuleLibrary:
        return self._ctx_svc.get_current_ml()

    def set_ml_module_from_schema(self, name: str, schema: CfgSchema) -> None:
        self._ctx_svc.apply_ml_writes(
            {},
            {name: schema},
            {},
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=False,
        )

    def del_ml_module(self, name: str) -> None:
        self._ctx_svc.del_ml_module(name)

    def set_ml_waveform_from_schema(self, name: str, schema: CfgSchema) -> None:
        self._ctx_svc.apply_ml_writes(
            {},
            {},
            {name: schema},
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=False,
        )

    def apply_writes(self, writes: ContextWrites) -> None:
        self._ctx_svc.apply_ml_writes(
            writes.md,
            writes.ml_modules,
            writes.ml_waveforms,
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=True,
        )

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._ctx_svc.coerce_md_value(key, text)

    def del_ml_waveform(self, name: str) -> None:
        self._ctx_svc.del_ml_waveform(name)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_waveform(old, new)

    # ------------------------------------------------------------------
    # Role templates — one-shot "create blank ml entry from a named role"
    # (shared by inspect UI and ml.create_from_role RPC). Editing afterwards
    # goes through the normal modify path (inspect / editor.open(from_name)).
    # ------------------------------------------------------------------

    def get_role_catalog(self) -> RoleCatalog:
        if self._role_catalog is None:
            raise RuntimeError("No role catalog is wired up.")
        return self._role_catalog

    def get_exp_context(self) -> ExpContext:
        return self._ctx_svc.get_exp_context()

    def create_from_role(self, item_kind: str, role_id: str, name: str) -> None:
        """Seed a blank ml module/waveform from a named role and register it.

        The role's eval-aware factory produces md-linked defaults; lowering
        against the live md turns those into the md's current concrete values
        (ModuleLibrary stores concrete numbers, never md references).
        """
        from zcu_tools.gui.app.main.adapter import CfgSchema
        from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES
        from zcu_tools.gui.app.main.specs import make_waveform_spec_by_style

        if not name:
            raise RuntimeError("Entry name must not be empty.")
        entry = self.get_role_catalog().get(role_id)
        if entry.item_kind != item_kind:
            raise RuntimeError(
                f"Role {role_id!r} is a {entry.item_kind}, not a {item_kind}."
            )
        # create = new entry; a name clash is an error (the user/agent meant to
        # add, not silently overwrite an existing entry — register_module would
        # overwrite). Editing an existing entry goes through the modify path.
        self._require_new_ml_name(item_kind, name)

        ctx = self.get_exp_context()
        ref = entry.make_value(ctx)
        value = ref.value
        discriminator = self._discriminator_of(value, item_kind)
        if item_kind == "module":
            spec = _MODULE_SPEC_FACTORIES[discriminator]()
        else:
            spec = make_waveform_spec_by_style(discriminator)
        # ADR-0006: hand the un-lowered CfgSchema to the single write authority;
        # ContextService lowers (against live md) + registers. No UI-side lowering.
        schema = CfgSchema(spec=spec, value=value)
        if item_kind == "module":
            self.set_ml_module_from_schema(name, schema)
        else:
            self.set_ml_waveform_from_schema(name, schema)

    @staticmethod
    def _discriminator_of(value: Any, item_kind: str) -> str:
        """Read the type/style discriminator off a role factory's value."""
        key = "type" if item_kind == "module" else "style"
        field = value.fields.get(key)
        disc = getattr(field, "value", None)
        if not isinstance(disc, str):
            raise RuntimeError(
                f"Role value has no usable {key!r} discriminator (got {disc!r})."
            )
        return disc

    def has_ml_entry(self, item_kind: str, name: str) -> bool:
        """Whether an ml module/waveform of this name already exists."""
        ml = self.get_current_ml()
        store = ml.modules if item_kind == "module" else ml.waveforms
        return name in store

    def _require_new_ml_name(self, item_kind: str, name: str) -> None:
        """Fail fast if a create would overwrite an existing ml entry."""
        if self.has_ml_entry(item_kind, name):
            raise RuntimeError(
                f"A {item_kind} named {name!r} already exists; "
                "use Modify to change it, or pick a different name."
            )

    # ------------------------------------------------------------------
    # CfgEditor sessions (CfgEditorService) — headless ml editing for RPC
    # ------------------------------------------------------------------

    def open_cfg_editor(
        self,
        item_kind: str,
        *,
        discriminator: str | None = None,
        from_name: str | None = None,
        gc: bool = True,
        owner_key: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Open a CfgEditor session. The ``editor.open`` RPC only uses
        ``from_name`` (edit an existing entry); ``discriminator`` (blank seed)
        remains an internal seam — creating a blank goes through
        ``create_from_role`` (``<disc>:blank`` roles), not the RPC.
        """
        return self._cfg_editor_svc.open(
            item_kind,
            discriminator=discriminator,
            from_name=from_name,
            gc=gc,
            owner_key=owner_key,
        )

    def open_seeded_cfg_editor(
        self, seed: Any, *, gc: bool = False, owner_key: str | None = None
    ) -> tuple[str, list[dict[str, object]]]:
        """Open a service-owned cfg model seeded from an existing CfgSchema.

        Used by UI surfaces (tab cfg / writeback item) that own a non-ml-entry
        draft; the owning widget then ``attach``es to ``get_cfg_editor_root``.
        """
        return self._cfg_editor_svc.open_seeded(seed, gc=gc, owner_key=owner_key)

    def get_cfg_editor_root(self, editor_id: str) -> Any:
        """Return the service-owned LiveModel for a widget to ``attach`` to."""
        return self._cfg_editor_svc.get_root(editor_id)

    def teardown_cfg_editor(self, editor_id: str) -> None:
        """Tear down a UI-owned (gc=False) cfg-editor session + its LiveModel."""
        self._cfg_editor_svc.teardown(editor_id)

    def editor_id_for_owner(self, owner_key: str) -> str | None:
        return self._cfg_editor_svc.editor_id_for_owner(owner_key)

    def set_cfg_editor_change_listener(self, listener: Any) -> None:
        """Wire the per-session push listener (remote layer injects this)."""
        self._cfg_editor_svc.set_change_listener(listener)

    def bump_editor_version(self, editor_id: str) -> None:
        """Bump an editor session's draft version (editor.commit guard input).

        Symmetric teardown is ``drop_editor_version`` (called from
        ``CfgEditorService._remove``): a session that ends must drop its key, or
        a stale dependency would spuriously match a retained version.
        """
        self._state.version.bump(f"editor:{editor_id}")

    def drop_editor_version(self, editor_id: str) -> None:
        """Forget an editor session's version on teardown (symmetric to tab/
        device drop). A stale dependency on a gone editor then reads version 0,
        so the guard treats it as stale rather than spuriously matching."""
        self._state.version.drop_prefix(f"editor:{editor_id}")

    def cfg_editor_set_field(
        self, editor_id: str, path: str, value: object
    ) -> dict[str, object]:
        return self._cfg_editor_svc.set_field(editor_id, path, value)

    def owner_of_editor(self, editor_id: str) -> str | None:
        """The owner_key a cfg-editor session is keyed to (tab_id for tab cfg)."""
        return self._cfg_editor_svc.owner_of_editor(editor_id)

    def cfg_editor_get(
        self,
        editor_id: str,
        under: str | None = None,
        verbosity: str = "full",
    ) -> list[dict[str, object]] | list[str]:
        return self._cfg_editor_svc.get(editor_id, under=under, verbosity=verbosity)

    def commit_cfg_editor(self, editor_id: str, name: str) -> None:
        self._cfg_editor_svc.commit(editor_id, name)

    def discard_cfg_editor(self, editor_id: str) -> None:
        self._cfg_editor_svc.discard(editor_id)

    def discard_cfg_editors(self, editor_ids: list[str]) -> None:
        self._cfg_editor_svc.discard_for_client(editor_ids)

    # ------------------------------------------------------------------
    # Device (DeviceService)
    # ------------------------------------------------------------------

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        return self._dev_svc.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        return self._dev_svc.start_disconnect_device(req)

    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    def get_device_value_for_new_context(self, name: str) -> float | None:
        return self._dev_svc.get_device_value_for_new_context(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def poll_device_info(self, name: str) -> None:
        # Dialog-scoped off-main live-read (best-effort); result flows back via
        # DEVICE_CHANGED. DeviceService owns the worker/main-thread split.
        self._dev_svc.poll_device_info(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        return self._dev_svc.start_setup_device(req)

    def get_active_device_setups(self) -> tuple[DeviceSetupSnapshot, ...]:
        return self._dev_svc.get_active_device_setups()

    def cancel_device_operation(self, name: str) -> None:
        self._dev_svc.cancel_device_operation(name)

    def start_reconnect_device(self, name: str) -> None:
        self._dev_svc.start_reconnect_device(name)

    def forget_device(self, name: str) -> None:
        # Removing the device from State emits DEVICE_CHANGED, which re-projects
        # the remembered-device set onto disk via StartupService.
        self._dev_svc.forget_device(name)

    def is_memory_device(self, name: str) -> bool:
        return self._dev_svc.is_memory_device(name)

    def get_memory_device_address(self, name: str) -> str | None:
        """Return the persisted address for a memory-only device, or None."""
        return self._dev_svc.get_memory_device_address(name)

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._dev_svc.get_device_snapshot(name)

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        return self._dev_svc.get_active_device_operations()

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        """A View subscribes (by its own tab_id / device_name) to progress
        changes for that owner; returns a disposer. The listener fires whenever
        the owner's live operation's bars change (and across operation rotation),
        and re-reads via ``progress_bars``."""
        return self._progress_svc.attach_by_owner(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        """Live (handle_id, ProgressBarModel) pairs for the owner's current
        operation (empty if none live)."""
        return self._progress_svc.bars_for_owner(owner_id)

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Block until an async operation settles or a wakeup condition fires.

        Returns an AwaitResult with reason in {'completed', 'user_feedback',
        'timeout'}. Never returns None (kept for type-checker clarity during
        migration). Runs on an off-main IO thread — only touches the thread-safe
        OperationHandles and FeedbackInbox, no main-thread-owned state. (ADR-0023)
        """
        return self._operation_handles.await_outcome(operation_id, timeout)

    def get_feedback_inbox(self) -> FeedbackInbox:
        """Return the session-scoped user-feedback inbox (ADR-0023).

        Thread-safe: the main-thread GUI widget calls ``inbox.post(text)``
        and the IO worker thread reads it via ``await_outcome``. Always non-None
        after __init__ completes.
        """
        return self._feedback_inbox

    def has_pending_wait(self) -> bool:
        """True when at least one operation is live (pending) — i.e. an await
        call is potentially blocking. The feedback widget reads this to decide
        whether to show 'will be delivered now' vs 'will take effect at next wait'.
        """
        return self._operation_handles.live_count() > 0

    def get_agent_chat(self):
        """Return the session-scoped AgentChatService (transcript + observers).

        Used by AgentChatDialog to register listeners and by RemoteControlAdapter
        to record activity entries. Always non-None after __init__ completes.
        """
        from .services.agent_chat import AgentChatService

        assert isinstance(self._agent_chat_svc, AgentChatService)
        return self._agent_chat_svc

    def get_agent_session(self):
        """Return an AgentSessionPort backend (kept for backwards compat).

        Delegates to ``new_agent_session()`` or ``_cli_agent_session()``
        depending on the backend mode.  Existing callers that do not yet use
        the picker can continue using this method.
        """
        return self.new_agent_session()

    # ------------------------------------------------------------------
    # B1b-2: backend mode + session factory methods
    # ------------------------------------------------------------------

    def agent_backend_mode(self) -> Literal["cli", "independent"]:
        """Return the active agent backend mode.

        Reads ``ZCU_AGENT_BACKEND`` from the environment (default=independent).
        Callers (dialog, tests) should call this instead of reading os.environ
        directly — one place to override the default.
        """
        import os as _os

        raw = _os.environ.get("ZCU_AGENT_BACKEND", "independent").lower().strip()
        if raw == "cli":
            return "cli"
        return "independent"

    def _build_agent_callbacks(self):  # type: ignore[no-untyped-def]
        """Build the shared callback bundle for any agent session backend.

        Returns ``(on_update, on_state_changed, on_process_error, has_pending_wait)``
        — the four arguments consumed by both ``_RunnerCallbacks`` (AgentRunner)
        and the ``IndependentAgentSession`` keyword arguments.
        """
        from .services.agent_runner import (
            AssistantTextUpdate,
            ResultUpdate,
            SystemInitUpdate,
            ToolUseUpdate,
        )

        def _on_update(updates):  # type: ignore[no-untyped-def]
            # TranscriptUpdate variants → record_* on AgentChatService.
            # Tool *results* are intentionally not recorded: they are verbose
            # payloads (raw JSON) that only add noise — the assistant prose
            # summarises outcomes. Tool *uses* show the name only (no payload).
            chat = self.get_agent_chat()
            for update in updates:
                if isinstance(update, AssistantTextUpdate):
                    chat.record_assistant(update.text)
                elif isinstance(update, ToolUseUpdate):
                    chat.record_tool_use(update.tool_name)
                elif isinstance(update, SystemInitUpdate):
                    chat.record_system(update.session_id)
                elif isinstance(update, ResultUpdate):
                    chat.record_result(
                        update.is_error,
                        update.result_text,
                        update.total_cost_usd,
                        update.terminal_reason,
                    )
                # RateLimitUpdate is informational; not recorded.

        def _on_state_changed(state):  # type: ignore[no-untyped-def]
            # Sync embedded-active flag with runner liveness.  Sticky across the
            # whole session (incl. idle between turns) so the queued activity tap
            # stays suppressed and does not double-record tool calls already in
            # the stream-json transcript. Only a terminal "stopped" re-enables.
            chat = self.get_agent_chat()
            chat.set_embedded_active(state != "stopped")

        def _on_process_error(msg: str) -> None:
            self._notify("error", "Agent process error", msg)

        return _on_update, _on_state_changed, _on_process_error, self.has_pending_wait

    def new_agent_session(self):  # type: ignore[return]
        """Return a freshly built AgentSessionPort backend.

        In ``independent`` mode (default): returns an ``IndependentAgentSession``
        that has NOT been started yet — the dialog calls ``start()`` on the first
        Send (decision E).

        In ``cli`` mode (``ZCU_AGENT_BACKEND=cli``): returns an ``AgentRunner``
        (QProcess, bound child process; the original B0/B1a behaviour).
        """
        from .services.ports import AgentSessionPort

        on_update, on_state_changed, on_process_error, has_pending_wait = (
            self._build_agent_callbacks()
        )

        if self.agent_backend_mode() == "cli":
            from .services.agent_runner import AgentRunner, _RunnerCallbacks

            callbacks = _RunnerCallbacks(
                on_update=on_update,
                on_state_changed=on_state_changed,
                on_process_error=on_process_error,
                has_pending_wait=has_pending_wait,
            )
            session: AgentSessionPort = AgentRunner(callbacks, parent=None)
            return session

        # Independent mode (default).
        from .services.independent_agent_session import IndependentAgentSession

        session = IndependentAgentSession(
            on_update=on_update,
            on_state_changed=on_state_changed,
            on_process_error=on_process_error,
            has_pending_wait=has_pending_wait,
            parent=None,
        )
        return session

    def attach_agent_session(self, record):  # type: ignore[no-untyped-def]
        """Build an ``IndependentAgentSession`` and attach it to ``record``.

        Returns a session whose poll-tail starts at offset=0 so the full log
        history is replayed through the callbacks.  The dialog calls this when
        the user clicks Attach/Resume in the picker.
        """
        from .services.agent_session_registry import AgentSessionRecord
        from .services.independent_agent_session import IndependentAgentSession
        from .services.ports import AgentSessionPort

        on_update, on_state_changed, on_process_error, has_pending_wait = (
            self._build_agent_callbacks()
        )
        session = IndependentAgentSession(
            on_update=on_update,
            on_state_changed=on_state_changed,
            on_process_error=on_process_error,
            has_pending_wait=has_pending_wait,
            parent=None,
        )
        rec: AgentSessionRecord = record
        session.attach(rec)
        result: AgentSessionPort = session
        return result

    def list_agent_sessions(self):  # type: ignore[return]
        """Return all agent session records sorted by ``created`` (oldest first).

        Applies stale-running self-heal (dead pid → stopped) on each record.
        Always returns a list (empty when no sessions exist).
        """
        from .services.agent_session_registry import AgentSessionRecord, list_records

        result: list[AgentSessionRecord] = list_records()
        return result

    def remove_agent_session(self, session_id: str) -> None:
        """Remove a session record from the registry (decision C).

        For stopped sessions: deletes the record file.  For running sessions:
        also stops the supervisor before deleting (best-effort; if the process
        is already gone the stop call is a no-op).
        """
        from .services.agent_session_registry import read_record, remove_record
        from .services.agent_supervisor import stop_supervisor

        record = read_record(session_id)
        if record is not None and record.get("status") == "running":
            try:
                stop_supervisor(record["pid"])
            except Exception:
                logger.exception(
                    "controller.remove_agent_session: stop_supervisor failed "
                    "for pid=%s; continuing removal",
                    record.get("pid"),
                )
        remove_record(session_id)

    # ------------------------------------------------------------------
    # Startup application workflow (StartupService)
    # ------------------------------------------------------------------

    def get_project_root(self) -> str:
        """The base directory default result/database paths are anchored under
        (the repo root, injected by the entry script). Setup dialog + startup RPC
        derive defaults through ``derive_project_paths`` against this, NOT cwd, so
        a .bat launcher that cd's into script/ still scopes under the repo root."""
        return self._project_root

    def get_persisted_startup(self) -> PersistedStartup:
        """The remembered startup prefs (for the setup dialog's prefill). Reads
        State.startup_prefs — no disk I/O (the Caretaker loads at startup)."""
        return self._startup_svc.get_persisted()

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        # Updates the in-State prefs only; persisted at close by the Caretaker.
        self._startup_svc.remember_connection(req)

    # ------------------------------------------------------------------
    # Connection / Predictor (ConnectionService)
    # ------------------------------------------------------------------

    def start_connect(self, req: ConnectRequest) -> int:
        return self._conn_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Bind the single connection observer without exposing the service.

        Single-observer model: only one observer (the currently-open SetupDialog)
        should hear connection outcomes at a time. SetupDialog is re-created on
        every open (``MainWindow._make_dialog`` → popped from the registry on
        ``finished``), so a prior dialog's bound methods would otherwise stay
        connected and leak. We therefore drop **all** existing slots on these
        signals before connecting the new observer — a no-arg ``disconnect()``
        removes every connection — guaranteeing exactly the latest observer.
        """
        for signal in (
            self._conn_svc.connection_finished,
            self._conn_svc.connection_failed,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # no existing connections
        self._conn_svc.connection_finished.connect(on_finished)
        self._conn_svc.connection_failed.connect(on_failed)

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._conn_svc.load_predictor(req)

    def clear_predictor(self) -> None:
        self._conn_svc.clear_predictor()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._conn_svc.predict_freq(req)

    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult:
        return self._conn_svc.predict_freq_curve(req)

    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        return self._conn_svc.predict_matrix_element_curve(req)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._conn_svc.get_soccfg()

    def get_predictor(self) -> FluxoniumPredictor | None:
        return self._conn_svc.get_predictor()

    def get_predictor_info(self) -> dict | None:
        return self._conn_svc.get_predictor_info()

    # ------------------------------------------------------------------
    # View query interface (TabService) — strict APIs; callers must check
    # has_tab() and short-circuit if false. State / TabService raise KeyError
    # on unknown tab_id, which is a fatal contract violation.
    # ------------------------------------------------------------------

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def list_tab_ids(self) -> list[str]:
        return self._state.list_tab_ids()

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def get_tab_cfg_schema(self, tab_id: str) -> CfgSchema:
        return self._state.get_tab(tab_id).cfg_schema

    def get_tab_result(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_result(tab_id)

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab_svc.get_snapshot(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        """Auto-commit boundary for tab CfgFormWidget.

        Writes the latest form draft into ``State.cfg_schema`` as the committed
        truth. Cfg edits do not change ``TabInteractionState`` (run / analyze /
        save availability), so no ``TAB_INTERACTION_CHANGED`` is emitted here;
        the form's own ``validity_changed`` signal drives any UI refresh.

        Do not call from dialog / writeback local LiveModel paths — those keep
        their drafts off of ``State`` until their own Apply boundary.

        Terminal: → ``TabService.update_tab_cfg`` → ``State.update_tab_cfg_schema``,
        which bumps ``tab:<id>:cfg`` and emits no event.
        """
        self._tab_svc.update_tab_cfg(tab_id, schema)

    def reset_tab_cfg(self, tab_id: str) -> CfgSchema:
        """Regenerate the tab's cfg to the adapter's default and commit it.

        Discards the whole current cfg: builds a fresh default CfgSchema from the
        adapter (under the live context) and writes it back as the committed
        truth, returning it so the caller can re-seed its cfg form.

        Running gate: editing/replacing a running tab's cfg is forbidden (the
        worker captured the cfg at launch) — fail fast, mirroring the
        ``tab.update_cfg`` RPC guard.
        """
        if self._state.running_tab_id == tab_id:
            raise RuntimeError(
                f"tab {tab_id!r} is currently running; cancel the run before "
                "resetting cfg"
            )
        adapter_name = self._tab_svc.get_tab_adapter_name(tab_id)
        schema = self._tab_svc.make_default_cfg(adapter_name)
        self._tab_svc.update_tab_cfg(tab_id, schema)
        return schema

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._tab_svc.update_tab_analyze_param_instance(tab_id, instance)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._tab_svc.update_tab_save_path_overrides(tab_id, data_path, image_path)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def get_adapter_names(self) -> list[str]:
        return self._tab_svc.list_adapter_names()

    def get_adapter_cfg_spec(self, adapter_name: str):
        """Static cfg spec of an adapter (no tab/context needed)."""
        return self._tab_svc.adapter_cfg_spec(adapter_name)

    def get_adapter_analyze_params(self, adapter_name: str) -> list[dict]:
        """Static analyze-params field spec of an adapter ([] if unsupported)."""
        return self._tab_svc.adapter_analyze_params(adapter_name)

    def get_adapter_guide(self, adapter_name: str) -> dict:
        """Static human-facing orientation guide of an adapter (no tab needed)."""
        return self._tab_svc.adapter_guide(adapter_name)
