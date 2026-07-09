from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

logger = logging.getLogger(__name__)

from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.controller_mixin import SessionControllerMixin
from zcu_tools.gui.session.notify_handles import NotifyHandles, NotifyResult
from zcu_tools.gui.session.operation_handles import AwaitResult
from zcu_tools.gui.session.services.connection import (
    ConnectRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.device import (
    ActiveDeviceOperation,
    DeviceSetupSnapshot,
)
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.mock_flux import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
)

from .adapter import (
    AnalysisMode,
    AnalyzeRequest,
    CfgSchema,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    SavePaths,
    WritebackItem,
)
from .events.tab import TabContentChangedPayload, TabInteractionChangedPayload
from .registry import Registry
from .role_catalog import RoleCatalog
from .services import (
    AppPersistedState,
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    LoadTabResultOutcome,
    PersistenceCaretaker,
    PersistenceError,
    RestoreReport,
    SaveResultOutcome,
    StartupProjectRequest,
    TabSnapshot,
    build_app_services,
)
from .services.cfg_lowering import lower_module, lower_waveform
from .services.ports import ContextWrites
from .services.remote.dialogs import DialogName
from .state import State

if TYPE_CHECKING:
    from zcu_tools.gui.session.adapters.qt_shutdown_driver import QtShutdownDriver
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.setup_control import SetupControlPort
    from zcu_tools.meta_tool import ArbWaveformData, ArbWaveformInfo

    from .services.guard import AnalyzePermit
    from .services.tab_control import TabControlPort


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

    def unmount_interactive_analysis(self, tab_id: str) -> None:
        """Tear down a tab's mounted interactive picker (the dual of
        ``mount_interactive_analysis``). Called when an interactive analyze is
        cancelled — the picker widget has no settle path of its own, so the
        Controller drives its removal. A no-op when nothing is mounted."""
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
    def take_window_screenshot(self) -> bytes: ...
    def open_dialog(self, name: DialogName) -> None: ...
    def close_dialog(self, name: DialogName) -> None: ...
    def list_open_dialogs(self) -> list[DialogName]: ...
    def register_dialog(self, name: DialogName, dialog: Any) -> None: ...
    def request_shutdown(self) -> None: ...
    def open_notify_prompt(self, token: int, message: str, timeout: float) -> None:
        """Open a non-modal NotifyUserDialog for the given token / message.

        Called from the main thread (notify.open handler). The dialog mints no
        token — it receives one so reply/dismiss/timeout route back correctly.
        timeout is the display timeout in seconds (QTimer in the dialog).
        """
        ...

    def refresh_feedback_widget(self) -> None:
        """Re-evaluate and mount/unmount the docked feedback panel.

        Called by RemoteControlAdapter._on_client_count_changed() on the Qt
        main thread when a control client connects or disconnects, so the
        panel tracks both op-count changes and agent-presence changes.
        """
        ...


class ViewProtocol(DiagnosticSink, RenderHost, RenderView, Protocol):
    """A full Qt View (``MainWindow``) implements all three channels."""


class Controller(SessionControllerMixin):
    """Façade for the GUI application. Delegates to domain services.

    The identical shared controller forwards live in SessionControllerMixin
    (read through the _*_svc accessors assigned in __init__); only the methods whose
    body diverges from autofluxdep are kept here (apply_startup_project /
    get_project_root / get_bus, plus everything app-specific). Shared setup,
    context / inspect, and value-md-ml consumers use their own control facets
    instead of this facade.
    """

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
            project_root=self._project_root,
        )
        self._services = services
        self._operation_gate = services.operation_gate
        self._operation_handles = services.handles
        # ADR-0025: cross-thread interaction now uses per-op OperationChannel;
        # FeedbackInbox and set_feedback_inbox are removed.
        self._background_svc = services.background
        self._progress_svc = services.progress
        self._guard_svc = services.guard
        self._dev_svc = services.device
        self._device_control = services.device_control
        self._soc_svc: SoCConnectionService = services.soc_connection
        self._pred_svc = services.predictor
        self._predictor_control = services.predictor_control
        self._progress_control = services.progress_control
        self._ctx_svc = services.context
        self._context_control = services.context_control
        self._setup_control = services.setup_control
        self._tab_svc = services.tab
        self._tab_control = services.tab_control
        self._load_svc = services.load
        self._run_svc = services.run
        self._analyze_svc = services.analyze
        self._post_analyze_svc = services.post_analyze
        self._save_svc = services.save
        self._writeback_svc = services.writeback
        self._workspace_svc = services.workspace
        self._startup_svc = services.startup
        self._cfg_editor_svc = services.cfg_editor
        self._arb_waveform_svc = services.arb_waveform
        # Notify prompt registry (Stage 4b): independent of OperationHandles;
        # tokens minted on the main thread, consumed off-main (ADR-0025).
        self._notify_handles: NotifyHandles = NotifyHandles()
        # App-level PersistenceCaretaker, injected by runtime behavior via
        # attach_caretaker (None in bare-Controller tests that don't exercise
        # persistence).
        self._caretaker: PersistenceCaretaker | None = None
        # Lazily built on first begin_shutdown so the Controller stays importable
        # without a Qt event loop (tests construct a bare Controller). The driver
        # is a Qt adapter owning the QTimer that pumps the Qt-free coordinator.
        self._shutdown_driver: QtShutdownDriver | None = None
        # Injected by RemoteControlAdapter on start()/stop() so the View can
        # gate widgets on whether an MCP control client is connected (ADR-0025).
        # None means no control socket is running → treat as no client connected.
        self._agent_connected_query: Callable[[], bool] | None = None

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
        EventBus — diagnostics must not depend on the channel they report on."""
        for sink in list(self._diag_sinks):
            sink.notify_diagnostic(severity, title, message)

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

    @property
    def device_control(self) -> DeviceControlPort:
        return self._device_control

    @property
    def predictor_control(self) -> PredictorControlPort:
        return self._predictor_control

    @property
    def progress_control(self) -> ProgressControlPort:
        return self._progress_control

    @property
    def context_control(self) -> ContextControlPort:
        return self._context_control

    @property
    def setup_control(self) -> SetupControlPort:
        return self._setup_control

    @property
    def tab_control(self) -> TabControlPort:
        return self._tab_control

    def attach_caretaker(self, caretaker: PersistenceCaretaker) -> None:
        """Wire the app-level PersistenceCaretaker.

        The Controller is the Memento Originator; the Caretaker owns disk I/O.
        """
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

    # -- lifecycle façade (runtime startup / MainWindow close) ---------------

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
        return self._tab_control.new_tab(adapter_name)

    def close_tab(self, tab_id: str) -> None:
        self._tab_control.close_tab(tab_id)

    def set_active_tab(self, tab_id: str) -> None:
        self._tab_control.set_active_tab(tab_id)

    def reorder_tabs(self, tab_ids: list[str]) -> None:
        self._tab_control.reorder_tabs(tab_ids)

    def get_active_tab_id(self) -> str | None:
        return self._tab_control.get_active_tab_id()

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
        return self._tab_control.get_running_tab_id()

    def has_soc(self) -> bool:
        return self._soc_svc.has_soc()

    def get_soc_info(self, include_cfg: bool = False) -> dict[str, object]:
        """Hardware summary of the connected SoC (QICK soccfg): a compact
        per-channel description (generator/readout type, converter port, sample
        rate, max pulse/buffer length) + ``is_mock``. The structured cfg (the full
        ~2 KB QICK config) is only computed and included when ``include_cfg`` is
        true — the common reader (overview assembly) needs only is_mock, so the
        cfg deserialization is opt-in rather than paid on every call.
        Raises if no SoC is connected (→ precondition_failed)."""
        from zcu_tools.program import describe_soc

        soccfg = self._soc_svc.get_soccfg()
        if soccfg is None:
            raise RuntimeError("No SoC connected")
        info: dict[str, object] = {
            "description": describe_soc(soccfg),
            "is_mock": self._soc_svc.is_mock_soc(),
        }
        if include_cfg:
            import json

            info["cfg"] = json.loads(soccfg.dump_cfg())
        return info

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

    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        permit = self._guard_svc.acquire_load_permit(tab_id)
        outcome = self._load_svc.load_result(permit, data_path)
        tab = self._state.get_tab(tab_id)
        has_analyze_params = False
        if tab.adapter.capabilities.analysis is not AnalysisMode.NONE:
            self._tab_svc.initialize_tab_analyze_params(tab_id)
            has_analyze_params = True
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))
        return replace(outcome, has_analyze_params=has_analyze_params)

    def cancel_run(self) -> bool:
        # Best-effort: True when a live run was signalled, False on a no-op
        # (no run in flight). The worker's true terminal is observed via the
        # run handle (ADR-0019 / ADR-0026 §8).
        return self._run_svc.cancel_run()

    def cancel_analyze(self, tab_id: str) -> bool:
        """Cancel an in-flight INTERACTIVE analyze on ``tab_id``.

        Interactive analyze and run are separate operations (ADR-0019): cancel_run
        only trips the run's stop_event and never touches the analyze handle or
        ``is_analyzing``. The interactive picker has no worker / stop_event, so its
        only non-Done settle path is here — tear down the mounted picker widget,
        then settle the analyze handle as cancelled (clears ``is_analyzing`` so the
        tab can close). Returns False (graceful no-op) when the tab has no
        in-flight interactive analyze.

        The View teardown runs unconditionally first so a stale mounted widget
        never lingers, even if the service side is already settled.
        """
        host = self._render_host
        if host is not None:
            host.unmount_interactive_analysis(tab_id)
        return self._analyze_svc.cancel_interactive(tab_id)

    def _active_operation(self) -> tuple[int | None, str | None]:
        """Return (token, tag) for the single foreground in-flight operation.

        Applies the same taxonomy as cancel_active_operation / send_feedback:
        run > interactive analyze > device (measure-gui drives one foreground
        op at a time). Returns (None, None) when no operation is active.

        token is sourced from the respective service's active-token accessor;
        it may itself be None if the service is active but the token was not
        captured (edge case during startup races) — callers must tolerate
        token=None paired with a non-None tag.
        """
        running = self._state.running_tab_id
        if running is not None:
            return self._run_svc.active_token, "run"
        tab = self._analyze_svc.active_interactive_tab()
        if tab is not None:
            return self._analyze_svc.active_interactive_token(), f"analyze:{tab}"
        ops = self.get_active_device_operations()
        if ops:
            name = ops[0].device_name
            return self._dev_svc.active_operation_token(name), f"device:{name}"
        return None, None

    def cancel_active_operation(self) -> str | None:
        """Cancel the single in-flight operation the docked feedback panel
        represents; returns a short tag of what was cancelled (or None for a
        no-op). The ONLY place that maps "the active op" to the right cancel —
        op-taxonomy lives here, not in the View. Priority run > interactive
        analyze > device (measure-gui drives one foreground op at a time)."""
        running = self._state.running_tab_id
        if running is not None:
            self.cancel_run()
            return "run"
        tab = self._analyze_svc.active_interactive_tab()
        if tab is not None:
            self.cancel_analyze(tab)
            return f"analyze:{tab}"
        ops = self.get_active_device_operations()
        if ops:
            name = ops[0].device_name
            self._dev_svc.cancel_device_operation(name)
            return f"device:{name}"
        return None

    def can_cancel_active_operation(self) -> bool:
        """True when the active foreground operation has a cancel hook.

        Used by FeedbackPanel to gate the 'Send & Stop' button: ops
        without a cancel hook (connect / FIT-analyze / device connect-
        disconnect) should not show Stop (ADR-0025 §Stop-gating, ADR-0019).
        Returns False when no operation is active.
        """
        token, _tag = self._active_operation()
        if token is None:
            return False
        return self._operation_handles.has_cancel_hook(token)

    def send_feedback(self, message: str, *, stop: bool = False) -> str | None:
        """User->agent feedback from the GUI (ADR-0025).

        Routes the message to the active operation's OperationChannel using
        the taxonomy from _active_operation() (run > interactive > device):
        - ``stop=False``: ``handles.message(token, text)`` — pure nudge, op
          continues running; agent receives user_feedback (non-terminal).
        - ``stop=True``: UI teardown first (unmount view for interactive), then
          ``handles.stop(token, reason=text)`` — enqueues Stop BEFORE triggering
          the cancel hook (ADR-0025 ordering invariant preserved).

        Returns the taxonomy tag of what was cancelled (stop=True only), or None.
        """
        token, tag = self._active_operation()
        if tag is None:
            return None

        # Interactive analyze requires View teardown before the hook fires;
        # run and device have no UI unmount step.
        if stop and tag.startswith("analyze:"):
            tab = tag[len("analyze:") :]
            host = self._render_host
            if host is not None:
                host.unmount_interactive_analysis(tab)

        if stop:
            if token is not None:
                self._operation_handles.stop(token, reason=message)
            return tag
        else:
            if token is not None:
                self._operation_handles.message(token, message)
            return None

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

    def set_agent_connected_query(self, query: Callable[[], bool] | None) -> None:
        """Inject or clear the has-live-control-client predicate.

        Called by RemoteControlAdapter.start() / stop() so the View can gate
        the feedback widget on agent presence (ADR-0025 C3). None means the
        control socket is not running; the predicate then returns False.
        """
        self._agent_connected_query = query

    def has_agent_connected(self) -> bool:
        """Return True if at least one MCP control client is connected.

        The View calls this inside _refresh_feedback_widget() to gate display
        (ADR-0025 C3: show only when op live AND agent connected). Always
        returns False when no RemoteControlAdapter has been started.
        """
        q = self._agent_connected_query
        return q() if q is not None else False

    def begin_shutdown(self, on_closed: Callable[[], None]) -> None:
        """Cancel every live operation, wait (with a timeout) for them to stop,
        then run ``on_closed`` — the View's actual teardown.

        Qt-free façade: the QTimer-driven coordinator lives in a driving adapter
        (ADR-0005), built lazily here so the Controller stays importable without
        a Qt loop. ``on_closed`` always runs on the main thread."""
        if self._shutdown_driver is None:
            from zcu_tools.gui.session.adapters.qt_shutdown_driver import (
                QtShutdownDriver,
            )

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
        via BackgroundRunner's pool, delivering the result to ``on_done`` on the
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

    def apply_writeback(self, tab_id: str) -> dict[str, Any]:
        """Apply the tab's persistent writeback draft as-is (no recompute).

        Returns ``{applied_ids, written}`` echoing what was actually written
        (see WritebackService.apply_tab_writeback)."""
        permit = self._guard_svc.acquire_writeback_permit(tab_id)
        return self._writeback_svc.apply_tab_writeback(permit)

    def set_writeback_item(
        self, tab_id: str, session_id: str, **changes: Any
    ) -> dict[str, object]:
        """Edit a persistent writeback item (selected / target_name / metadict
        value / module-waveform cfg edits). Returns the aggregated
        ``{valid, removed, added}`` of any applied cfg edits."""
        self._guard_svc.acquire_writeback_permit(tab_id)
        return self._writeback_svc.set_item_field(tab_id, session_id, **changes)

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

    def apply_startup_project(self, req: StartupProjectRequest) -> dict[str, str]:
        # Applies the project to the active context AND records it as the
        # remembered prefs (in State); persisted to disk only at close.
        # apply_project always mutates and either succeeds or raises (no no-op
        # outcome), so we echo the resolved project rather than a bool.
        # StartupService owns result-scope resolution so callers cannot inject
        # arbitrary result/database paths.
        return self._startup_svc.apply_project(req).as_wire_dict()

    def get_flux_dir(self) -> str | None:
        return self._ctx_svc.get_flux_dir()

    def set_ml_module_from_schema(self, name: str, schema: CfgSchema) -> None:
        self._ctx_svc.apply_ml_writes(
            {},
            {name: schema},
            {},
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=False,
        )

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

    # ------------------------------------------------------------------
    # Arbitrary waveform assets (qubit-scoped repository)
    # ------------------------------------------------------------------

    def list_arb_waveforms(self) -> list[str]:
        return self._arb_waveform_svc.list_data_keys()

    def list_arb_waveform_infos(self) -> list[ArbWaveformInfo]:
        return self._arb_waveform_svc.list_infos()

    def inspect_arb_waveform(self, data_key: str) -> dict[str, object]:
        info = self._arb_waveform_svc.inspect(data_key)
        return {
            "data_key": info.data_key,
            "duration": info.duration,
            "mtime": info.mtime,
        }

    def load_arb_waveform_data(self, data_key: str) -> ArbWaveformData:
        return self._arb_waveform_svc.load_data(data_key)

    def get_arb_waveform_preview(self, data_key: str) -> dict[str, object]:
        return self._arb_waveform_svc.get_preview(data_key)

    def set_arb_waveform(
        self, data_key: str, recipe: Any, *, overwrite: bool = False
    ) -> dict[str, object]:
        return self._arb_waveform_svc.set_formula(
            data_key,
            recipe,
            overwrite=overwrite,
        )

    def delete_arb_waveform(self, data_key: str) -> None:
        self._arb_waveform_svc.delete(data_key)

    def rename_arb_waveform(self, old_data_key: str, new_data_key: str) -> None:
        self._arb_waveform_svc.rename(old_data_key, new_data_key)

    # ------------------------------------------------------------------
    # Role templates — one-shot "create blank ml entry from a named role"
    # (shared by inspect UI and ml.create_from_role RPC). Editing afterwards
    # goes through the normal modify path (inspect / editor.new(from_name)).
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
        """Open a CfgEditor session. The ``editor.new`` RPC only uses
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

    def commit_cfg_editor(self, editor_id: str, name: str) -> None:
        self._cfg_editor_svc.commit(editor_id, name)

    def discard_cfg_editor(self, editor_id: str) -> None:
        self._cfg_editor_svc.discard(editor_id)

    def discard_cfg_editors(self, editor_ids: list[str]) -> None:
        self._cfg_editor_svc.discard_for_client(editor_ids)

    # ------------------------------------------------------------------
    # Device (DeviceService)
    # ------------------------------------------------------------------

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_value_for_new_context(self, name: str) -> float | None:
        return self._dev_svc.get_device_value_for_new_context(name)

    def get_active_device_setups(self) -> tuple[DeviceSetupSnapshot, ...]:
        return self._dev_svc.get_active_device_setups()

    def get_memory_device_address(self, name: str) -> str | None:
        """Return the persisted address for a memory-only device, or None."""
        return self._dev_svc.get_memory_device_address(name)

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        return self._dev_svc.get_active_device_operations()

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Block until an async operation settles or a wakeup condition fires.

        Returns an AwaitResult with reason in {'completed', 'user_feedback',
        'timeout'}. Never returns None (kept for type-checker clarity during
        migration). Runs on an off-main IO thread — only touches the thread-safe
        OperationHandles / OperationChannel, no main-thread-owned state. (ADR-0025)
        """
        return self._operation_handles.await_outcome(operation_id, timeout)

    # ------------------------------------------------------------------
    # Notify-user prompt (Stage 4b, ADR-0025 two-RPC design)
    #
    # Producer side (main thread): open_notify_prompt / reply_notify /
    #   dismiss_notify / timeout_notify
    # Consumer side (off-main IO worker): await_notify
    # ------------------------------------------------------------------

    def open_notify_prompt(self, message: str, timeout: float) -> int:
        """Mint a notify token and open a non-modal prompt dialog. Main thread.

        Returns the token the caller must pass to await_notify and to the
        dialog so its reply/dismiss/timeout callbacks route correctly.
        """
        token = self._notify_handles.open()
        host = self._render_host
        if host is not None and hasattr(host, "open_notify_prompt"):
            host.open_notify_prompt(token, message, timeout)  # type: ignore[union-attr]
        return token

    def reply_notify(self, token: int, text: str) -> None:
        """Deliver a user reply to the notify channel. Main thread (dialog callback)."""
        self._notify_handles.reply(token, text)

    def dismiss_notify(self, token: int) -> None:
        """Deliver a dismiss signal to the notify channel. Main thread (dialog callback)."""
        self._notify_handles.dismiss(token)

    def timeout_notify(self, token: int) -> None:
        """Deliver a timeout signal to the notify channel. Main thread (QTimer callback)."""
        self._notify_handles.timeout(token)

    def await_notify(self, token: int, timeout: float) -> NotifyResult:
        """Block until the notify prompt settles. Off-main IO thread only.

        timeout is the backstop: longer than the dialog's QTimer so the dialog
        fires first. Returns NotifyResult with reason in {'reply', 'dismiss',
        'timeout'} — never raises on timeout/dismiss (ADR-0025 §6).
        """
        return self._notify_handles.await_result(token, timeout)

    # ------------------------------------------------------------------
    # Startup application workflow (StartupService)
    # ------------------------------------------------------------------

    def get_project_root(self) -> str:
        """The base directory default result/database paths are anchored under
        (the repo root, injected by the entry script). Setup dialog + startup RPC
        derive defaults through ``derive_project_paths`` against this, NOT cwd, so
        a .bat launcher that cd's into script/ still scopes under the repo root."""
        return self._project_root

    # ------------------------------------------------------------------
    # Connection / Predictor (SoCConnectionService + PredictorService)
    # ------------------------------------------------------------------

    def connect_sync(self, req: ConnectRequest) -> None:
        """Connect the SoC synchronously (blocks until connected + side effects
        applied). The wire ``soc.connect`` path uses this so the handler can return
        the SoC summary directly; the GUI's connect button keeps using the async
        start_connect + signals. Both share SoCConnectionService._apply_connection,
        so the post-connect side effects (State write, soc version bump,
        SocChangedPayload → FLUX-AWARE-MOCK provisioning) are identical."""
        self._soc_svc.connect_sync(req)

    def get_predictor(self) -> FluxoniumPredictor | None:
        return self._pred_svc.get_predictor()

    # ------------------------------------------------------------------
    # View query interface (TabService) — strict APIs; callers must check
    # has_tab() and short-circuit if false. State / TabService raise KeyError
    # on unknown tab_id, which is a fatal contract violation.
    # ------------------------------------------------------------------

    def has_tab(self, tab_id: str) -> bool:
        return self._tab_control.has_tab(tab_id)

    def list_tab_ids(self) -> list[str]:
        return self._tab_control.list_tab_ids()

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._tab_control.get_tab_adapter_name(tab_id)

    def get_tab_cfg_schema(self, tab_id: str) -> CfgSchema:
        return self._tab_control.get_tab_cfg_schema(tab_id)

    def get_tab_result(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_result(tab_id)

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab_control.get_tab_snapshot(tab_id)

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
        self._tab_control.update_tab_cfg(tab_id, schema)

    def reset_tab_cfg(self, tab_id: str) -> CfgSchema:
        """Regenerate the tab's cfg to the adapter's default and commit it.

        Discards the whole current cfg: builds a fresh default CfgSchema from the
        adapter (under the live context) and writes it back as the committed
        truth, returning it so the caller can re-seed its cfg form.

        Running gate: editing/replacing a running tab's cfg is forbidden (the
        worker captured the cfg at launch) — fail fast, mirroring the
        ``tab.update_cfg`` RPC guard.
        """
        return self._tab_control.reset_tab_cfg(tab_id)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._tab_svc.update_tab_analyze_param_instance(tab_id, instance)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._tab_control.update_tab_save_paths(tab_id, data_path, image_path)

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
