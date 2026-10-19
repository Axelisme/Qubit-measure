from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from pydantic import JsonValue

from zcu_tools.cfg_model import Template, TemplateDef
from zcu_tools.gui.app.measure.catalog import ExperimentCatalogLoader
from zcu_tools.gui.app.measure.services.experiment_reload import (
    ReloadPreview,
    ReloadReport,
)
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.form import FormErrors, FormWritePath
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.gui.session.controller_mixin import SessionControllerMixin
from zcu_tools.gui.session.events import (
    DeviceOperationFinishedPayload,
    DeviceSetupFinishedPayload,
    GatePresence,
)
from zcu_tools.gui.session.library_entry import (
    LibraryEntryInvalid,
    LibraryKind,
    ModuleEntryForm,
    WaveformEntryForm,
    new_library_entry,
    open_library_entry,
)
from zcu_tools.gui.session.operation_handles import AwaitResult
from zcu_tools.gui.session.services.connection import (
    ConnectRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.device import ActiveDeviceOperation
from zcu_tools.gui.session.services.predictor import ParamsFor
from zcu_tools.gui.session.types import ParameterContext

from .adapter import AdapterGuide, SessionEnv
from .events.completion import (
    AnalyzeFailedPayload,
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from .events.run import RunFinishedPayload
from .events.tab import (
    TabContentChangedPayload,
    TabContentFact,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from .registry import Registry
from .remote.dialogs import DialogName
from .services import (
    AppPersistedState,
    LoadTabResultOutcome,
    PersistenceError,
    TabSnapshot,
    build_app_services,
)
from .state import State
from .ui.interactive_frontend import InteractiveFrontend, InteractiveFrontendEnv

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.device.base import BaseDevice
    from zcu_tools.experiment.artifacts import SaveReport
    from zcu_tools.gui.session.adapters.qt_shutdown_driver import QtShutdownDriver
    from zcu_tools.gui.session.alias_presets import AliasPresets
    from zcu_tools.gui.session.arb_waveform import ArbWaveformPort
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.persistence import SingleFileCaretaker
    from zcu_tools.gui.session.ports import (
        DeviceRegistryPort,
        DriverFactoryPort,
        OwnerScheduler,
        ProgressTransport,
    )
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.setup_control import SetupControlPort
    from zcu_tools.gui.session.types import SocCfgHandle, SocHandle

    from .artifacts import SaveControlPort
    from .services import AppServices
    from .services.operation_control import OperationControlPort
    from .services.run_analyze_control import RunAnalyzeControlPort
    from .services.tab_control import TabControlPort
    from .services.tab_form import TabFormResources
    from .writeback import WritebackControlPort, WritebackPane


logger = logging.getLogger(__name__)


class TemplateMetadata(TypedDict):
    """Discovery fields: unique role_id, dropdown label, family and name suggestion.

    item_kind is module or waveform. default_name is an authored suggestion,
    potentially containing component placeholders, or empty for no suggestion.
    """

    role_id: str
    label: str
    item_kind: LibraryKind
    default_name: str


# A View has two distinct down-channels from the Controller (ADR-0068):
#   - diagnostics (error / info) — fanned out to *every* attached View;
#   - render help (pbar / live container) — pulled from the *one* View that has
#     a real canvas.
# So the interface splits by channel, not lumped into a single "View".
# Render/snapshot/dialog *queries* are pulled by the RemoteControlAdapter
# through its own ``render_view`` (see RenderView), not by the Controller.

Severity = Literal["error", "info"]


class DiagnosticSink(Protocol):
    """A View that can receive a Controller diagnostic (error / info).

    Fanned out to every attached sink. Each View renders it its own way — the
    Qt window pops a dialog (error) or status bar (info); the remote adapter
    enqueues a diagnostic line. Diagnostics never go through EventBus (a channel
    that reports a fault must not be the faulty channel — ADR-0068).
    """

    def notify_diagnostic(
        self, severity: Severity, title: str, message: str
    ) -> None: ...


class RenderHost(Protocol):
    """Canvas-bearing View for run/analyze figure routing (ADR-0067).

    Each (tab_id, subtab) pane owns a stable FigureContainer for its lifetime (S2).
    The worker captures the pane's container at start; switching the visible pane
    does not change the routing target. Save/Guide have no container (placeholder).
    """

    def make_run_container(self, tab_id: str) -> FigureContainer | None:
        """Run pane FigureContainer (live plot, view-only)."""
        ...

    def make_analysis_container(self, tab_id: str) -> FigureContainer | None:
        """Analysis pane FigureContainer (canonical)."""
        ...

    def make_post_analysis_container(self, tab_id: str) -> FigureContainer | None:
        """Post-Analysis pane FigureContainer (canonical)."""
        ...

    def get_figure_container(self, tab_id: str, pane: str) -> FigureContainer | None:
        """Pane-aware container access (pane in run|analysis|post_analysis)."""
        ...

    def mount_interactive_analysis(
        self,
        tab_id: str,
        frontend_factory: Callable[[InteractiveFrontendEnv], InteractiveFrontend],
    ) -> None:
        """Mount one plugin frontend; the service retains its committed session."""
        ...

    def unmount_interactive_analysis(
        self, tab_id: str, *, restore_result: bool = False
    ) -> None:
        """Remove the plugin widget; completed analysis may restore its figure."""
        ...

    def interactive_presentation(self, tab_id: str) -> tuple[Figure, bool] | None: ...
    def discard_interactive_preview(self, tab_id: str) -> None: ...

    # Pure Qt appearance is captured from the View at layout flush time.
    def current_left_panel_width(self) -> int: ...

    def current_window_size(self) -> tuple[int, int]:
        """Return current width and height in Qt logical pixels on the owner thread."""
        ...

    def resize_window(self, width: int, height: int) -> None:
        """Resize the caller-owned window to positive logical-pixel dimensions."""
        ...


class ViewSnapshot(TypedDict):
    """Visible measure-window facts exposed to remote readers.

    active_tab_id is the focused State-owned tab ID, or None.
    tab_ids lists State-owned IDs in State order, excluding ghost widgets.
    entry_label is the visible entry name, or "No entry open".
    point_label is the visible SavedPoint label, "Draft (in-memory)", or
    "No point selected". It never derives from component focus.
    status is the visible status-bar text.
    open_dialogs lists the named dialogs currently open, using wire names.
    """

    active_tab_id: str | None
    tab_ids: list[str]
    entry_label: str
    point_label: str
    status: str
    open_dialogs: list[str]


class RenderView(Protocol):
    """View reads and explicit presentation commands for RemoteControlAdapter.

    Held by the adapter, not the Controller. Domain completion events do not
    select panes; agent follow uses the explicit selection command.
    """

    def select_tab_pane(
        self, tab_id: str, pane: Literal["run", "analysis", "post_analysis", "data"]
    ) -> None: ...

    def get_view_snapshot(self) -> ViewSnapshot:
        """Read visible entry/point and State-owned tab facts without mutation."""
        ...

    def interactive_presentation(self, tab_id: str) -> tuple[Figure, bool] | None:
        """Read an active frontend's figure and preview flag, if mounted."""
        ...

    def take_figure_screenshot_for_subtab(self, tab_id: str, subtab_id: str) -> bytes:
        """Pane-qualified figure screenshot (run|analysis|post_analysis).

        Run reads the live FigureContainer (view-only); analysis/post read
        canonical State figures. Explicit subtab routing avoids visible-tab
        inference.
        """
        ...

    def take_dialog_screenshot(self, dialog_name: Any) -> bytes: ...
    def take_window_screenshot(self) -> bytes: ...
    def open_dialog(self, name: DialogName) -> None: ...
    def close_dialog(self, name: DialogName) -> None: ...
    def list_open_dialogs(self) -> list[DialogName]: ...
    def request_shutdown(self) -> None: ...


class ViewProtocol(DiagnosticSink, RenderHost, RenderView, Protocol):
    """A full Qt View (``MainWindow``) implements all three channels."""


def _save_summary(report: SaveReport) -> str:
    """One status line per Save: file counts and run id; paths stay in the Data tab."""
    total = len(report.outputs)
    failed = sum(item.status == "failed" for item in report.outputs)
    if failed:
        return (
            f"Saved {total - failed} of {total} files; {failed} failed "
            f"for run {report.run_id}"
        )
    return f"Saved {total} files for run {report.run_id}"


class Controller(SessionControllerMixin):
    """Façade for the GUI application. Delegates to domain services.

    The identical shared controller forwards live in SessionControllerMixin
    (read through the _*_svc accessors assigned in __init__); app-specific methods
    are kept here (get_project_root / get_bus, plus everything app-specific). Shared setup,
    context / inspect, and value-md-ml consumers use their own control facets
    instead of this facade.
    """

    def __init__(
        self,
        state: State,
        registry: Registry,
        view: ViewProtocol | None,
        bus: EventBus,
        progress_transport: ProgressTransport | None = None,
        project_root: str | None = None,
        catalog_loader: ExperimentCatalogLoader | None = None,
        owner_scheduler: OwnerScheduler | None = None,
        params_for: ParamsFor | None = None,
        *,
        alias_presets: AliasPresets | None = None,
        device_registry: DeviceRegistryPort | None = None,
        driver_factory: DriverFactoryPort | None = None,
    ) -> None:
        """Compose services for state/registry with an optional view and adapters.

        driver_factory creates device drivers; None selects the real hardware
        factory. Tests may inject a hardware-free constructor. device_registry
        supplies registered drivers, which close_hardware eventually closes.

        ``alias_presets`` maps names to ordered caller-owned alias rows. None
        or omission means no catalog and an empty initial table. A supplied
        mapping must contain "default" (possibly empty), or raise ValueError.
        Invalid table syntax raises InvalidInputError. Default rows initialize
        entries without saved aliases; saved tables, including empty ones, win.
        Softlinks and empty bindings stay authored. The session
        uses the role registry's unique focus candidate once for the primary
        empty row on the first nonempty component snapshot; later point changes
        do not reset rows. Explicit table edits disable that initialization.
        Invalid syntax, duplicates or forward links raise
        InvalidInputError(alias_invalid). No device or cache is used to bind rows.
        """
        self.control_identity = ControlIdentity()
        self._state = state
        # Base directory the default per-qubit result/database paths are anchored
        # under. The entry script injects the repo root (Path(__file__).parent...)
        # so a .bat launcher that cd's into scripts/ does not scope defaults under
        # scripts/. None falls back to cwd — fine for tests / direct `python -m`
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
        self._registry = registry

        # Construct and wire every domain service into an immutable bundle, then
        # alias them onto self for the façade's call sites.
        # The progress transport is the Qt marshal (a driven adapter). Default to
        # the Qt one so GUI/agent processes (which run a Qt event loop) work
        # without the entry point wiring it; tests inject a synchronous fake.
        transport = progress_transport
        if transport is None:
            from zcu_tools.gui.session.adapters.qt_progress_transport import (
                QtProgressTransport,
            )

            transport = QtProgressTransport()
        if owner_scheduler is None:
            from zcu_tools.gui.session.adapters.qt_owner_scheduler import (
                QtOwnerScheduler,
            )

            owner_scheduler = QtOwnerScheduler()
        self.owner_scheduler = owner_scheduler
        services = build_app_services(
            state=state,
            bus=bus,
            registry=registry,
            progress_transport=transport,
            resource_versions=self.resources_versions,
            render_host=lambda: self._render_host,
            owner_scheduler=owner_scheduler,
            project_root=self._project_root,
            catalog_loader=catalog_loader,
            params_for=params_for,
            alias_presets=alias_presets,
            device_registry=device_registry,
            driver_factory=driver_factory,
        )
        self._bind_services(services)
        # App-level PersistenceCaretaker, injected by runtime behavior via
        # attach_caretaker (None in bare-Controller tests that don't exercise
        # persistence).
        self._caretaker: SingleFileCaretaker[AppPersistedState, None] | None = None
        # Lazily built on first begin_shutdown so the Controller stays importable
        # without a Qt event loop (tests construct a bare Controller). The driver
        # is a Qt adapter owning the QTimer that pumps the Qt-free coordinator.
        self._shutdown_driver: QtShutdownDriver | None = None
        self._hardware_close_attempted = False
        bus.subscribe(RunFinishedPayload, self._on_run_finished)
        bus.subscribe(TabInteractionChangedPayload, self._on_tab_interaction_changed)
        bus.subscribe(AnalyzeFailedPayload, self._on_analyze_failed)
        bus.subscribe(SaveDataFinishedPayload, self._on_save_data_finished)
        bus.subscribe(SaveArtifactsFinishedPayload, self._on_save_artifacts_finished)
        bus.subscribe(DeviceSetupFinishedPayload, self._on_device_setup_finished)
        bus.subscribe(
            DeviceOperationFinishedPayload, self._on_device_operation_finished
        )

    def _bind_services(self, services: AppServices) -> None:
        """Retain the bundle and name the service instances used by the facade."""
        self._services = services
        self._operation_gate = services.operation_gate
        self._operation_control = services.operation_control
        self._operation_handles = services.handles
        # ADR-0066: cross-thread interaction now uses per-op OperationChannel;
        # FeedbackInbox and set_feedback_inbox are removed.
        self._background_svc = services.background
        self._progress_svc = services.progress
        self._guard_svc = services.guard
        self._dev_svc = services.device
        self._device_control = services.device_control
        self._soc_svc: SoCConnectionService = services.soc_connection
        self._predictor_control = services.predictor_control
        self._progress_control = services.progress_control
        self._ctx_svc = services.context
        self._context_control = services.context_control
        self._setup_control = services.setup_control
        self._tab_forms = services.tab_forms
        self._tab_svc = services.tab
        self._tab_control = services.tab_control
        self._run_analyze_control = services.run_analyze_control
        self._load_svc = services.load
        self._run_svc = services.run
        self._analyze_svc = services.analyze
        self._post_analyze_svc = services.post_analyze
        self._save_svc = services.save
        self._save_control = services.save_control
        self._writeback_svc = services.writeback
        self._writeback_control = services.writeback_control
        self._workspace_svc = services.workspace
        self._settings_svc = services.settings
        self._arb_waveform_svc = services.arb_waveform

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
        """Fan a diagnostic out to every attached View (ADR-0068). Never via
        EventBus — diagnostics must not depend on the channel they report on."""
        for sink in list(self._diag_sinks):
            sink.notify_diagnostic(severity, title, message)

    def _info(self, message: str) -> None:
        """Transient status diagnostic (no title) — Qt status bar / wire line."""
        self._notify("info", "", message)

    def _report_persistence_error(self, title: str, error: Exception) -> None:
        self._notify("error", title, str(error))

    def _on_run_finished(self, payload: RunFinishedPayload) -> None:
        if payload.outcome == "failed":
            self._notify("error", "Run failed", payload.error_message or "Run failed")
            return
        tab_id = payload.tab_id
        if (
            payload.outcome == "cancelled"
            and self._state.get_tab(tab_id).run.result is None
        ):
            return
        preparation = self._tab_svc.prepare_result_analysis(tab_id)
        self._bus.emit(
            TabContentChangedPayload(
                tab_id=tab_id,
                fact=TabContentFact.RUN_RESULT_COMMITTED,
            )
        )
        if preparation.error is not None:
            self._notify(
                "error",
                "Analysis preparation failed",
                f"Run result retained. Analysis preparation failed: {preparation.error}",
            )

    def _on_analyze_finished(self, tab_id: str) -> None:
        # A fresh primary analyze result seeds the post-analysis params (mirrors
        # how a finished run seeds the analyze params). Only adapters declaring
        # post_analysis are routed there; the base get_post_analyze_params is a
        # Fast-Fail guard. The primary result is already in State (AnalyzeService).
        if self._state.get_tab(tab_id).adapter.capabilities.post_analysis:
            self._tab_svc.initialize_tab_post_analyze_params(tab_id)
        self._bus.emit(
            TabContentChangedPayload(
                tab_id=tab_id,
                fact=TabContentFact.PRIMARY_ANALYSIS_COMMITTED,
            )
        )

    def _on_post_analyze_finished(self, tab_id: str) -> None:
        # The post result + figure are already in State (PostAnalyzeService);
        # emit the content event so the View refreshes the Post sub-tab.
        self._bus.emit(
            TabContentChangedPayload(
                tab_id=tab_id,
                fact=TabContentFact.POST_ANALYSIS_COMMITTED,
            )
        )

    def _on_tab_interaction_changed(
        self, payload: TabInteractionChangedPayload
    ) -> None:
        if payload.fact is TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED:
            self._on_analyze_finished(payload.tab_id)
        elif payload.fact is TabInteractionFact.POST_ANALYZE_SUCCEEDED:
            self._on_post_analyze_finished(payload.tab_id)

    def _on_analyze_failed(self, payload: AnalyzeFailedPayload) -> None:
        title = (
            "Analyze failed" if payload.stage == "primary" else "Post-analysis failed"
        )
        self._notify("error", title, payload.error_message)

    def _on_save_artifacts_finished(
        self, outcome: SaveArtifactsFinishedPayload
    ) -> None:
        if outcome.report is not None:
            self._info(_save_summary(outcome.report))
        if outcome.error is not None:
            detail = ""
            if outcome.report is not None:
                detail = "\n".join(
                    f"{item.output.path}: {item.status}"
                    + (f" ({item.error})" if item.error is not None else "")
                    for item in outcome.report.outputs
                )
            self._notify("error", "Save failed", detail or outcome.error)

    def _on_save_data_finished(self, outcome: SaveDataFinishedPayload) -> None:
        if outcome.report is not None:
            self._info(_save_summary(outcome.report))
        if outcome.error is not None:
            self._notify("error", "Save data failed", outcome.error)

    def _on_device_setup_finished(self, payload: DeviceSetupFinishedPayload) -> None:
        if payload.outcome == "finished":
            self._info(f"Device setup completed: {payload.name}")
        elif payload.outcome == "cancelled":
            self._info(f"Device setup cancelled: {payload.name}")
        else:
            self._notify(
                "error",
                f"Device setup failed: {payload.name}",
                payload.error_message or "Device setup failed",
            )

    def _on_device_operation_finished(
        self, payload: DeviceOperationFinishedPayload
    ) -> None:
        if payload.success:
            self._info(f"Device {payload.action}ed: {payload.name}")
            return
        self._notify(
            "error",
            f"Device operation failed: {payload.name}",
            payload.error_message or "Device operation failed",
        )

    def get_bus(self) -> EventBus:
        return self._bus

    def get_hardware_gate_presence(self) -> tuple[GatePresence, ...]:
        return self._operation_gate.snapshot()

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
    def cfg_resources(self) -> TabFormResources:
        """Return the tab Form owner pool; lookup rejects retired tab identities."""
        return self._tab_forms

    @property
    def tab_control(self) -> TabControlPort:
        return self._tab_control

    @property
    def run_analyze_control(self) -> RunAnalyzeControlPort:
        return self._run_analyze_control

    @property
    def operation_control(self) -> OperationControlPort:
        return self._operation_control

    @property
    def save_control(self) -> SaveControlPort:
        return self._save_control

    @property
    def writeback_control(self) -> WritebackControlPort:
        return self._writeback_control

    def attach_caretaker(
        self, caretaker: SingleFileCaretaker[AppPersistedState, None]
    ) -> None:
        """Wire the app-level PersistenceCaretaker.

        The Controller is the Memento Originator; the Caretaker owns disk I/O.
        """
        self._caretaker = caretaker

    # -- Layout SnapshotOriginator -----------------------------------------

    def capture_persisted_state(self) -> AppPersistedState:
        """Capture only Qt appearance on the owner thread, without disk I/O.

        The returned dimensions and panel width are detached values. Without a
        view use default window dimensions. Session resources and drafts have
        their own live owners and are not part of this layout snapshot.
        """
        width, height = (
            (1280, 750)
            if self._render_host is None
            else self._render_host.current_window_size()
        )
        return AppPersistedState(
            window_width=width,
            window_height=height,
            left_panel_width=self._capture_left_panel_width(),
        )

    def restore_persisted_state(self, state: AppPersistedState) -> None:
        """Apply a layout-only snapshot on the owner thread, without disk I/O.

        state supplies positive Qt logical-pixel dimensions. Panel width seeds
        subsequently created tabs; an attached render host resizes immediately.
        This method does not create tabs, connect drivers or change sources.
        The caller retains ownership of the snapshot. Host failures propagate.
        """
        self._settings_svc.restore_left_panel_width(state.left_panel_width)
        if self._render_host is not None:
            self._render_host.resize_window(state.window_width, state.window_height)

    def restore_all(self, *, load: bool = True) -> None:
        """Load pure Qt appearance on the owner thread using the attached caretaker.

        load=False uses fresh default dimensions without reading disk. Present
        load/codec errors through diagnostic sinks; unexpected failures propagate.
        No domain session state is restored. Attach a caretaker before calling.
        """
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
        return self._ctx_svc.get_entry_info() is not None

    def has_context(self) -> bool:
        return self._ctx_svc.has_context()

    def has_active_context(self) -> bool:
        return self._ctx_svc.is_active_context()

    def get_running_tab_id(self) -> str | None:
        return self._tab_control.get_running_tab_id()

    def install_soc(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        """Install or replace a caller-owned board/config on the Qt owner thread.

        Clear the remote endpoint and publish the new pair. Run/connect conflicts
        raise OperationConflictError. The caller retains board cleanup ownership.
        """
        self._soc_svc.install_soc(soc, soccfg)

    def uninstall_soc(self) -> None:
        """Clear the board/config on the Qt owner thread without closing it.

        Publish the empty connection. Run/connect conflicts raise
        OperationConflictError; external resource cleanup stays with the caller.
        """
        self._soc_svc.uninstall_soc()

    def has_soc(self) -> bool:
        return self._soc_svc.has_soc()

    def get_soc_info(self, *, include_cfg: bool = False) -> dict[str, object]:
        """Read the connected board's GUI-owned compact or full hardware view."""
        from .soc_view import project_soc_info

        return project_soc_info(
            self._soc_svc.get_soccfg(),
            is_mock=self._soc_svc.is_mock_soc(),
            endpoint=self._soc_svc.connected_endpoint(),
            include_cfg=include_cfg,
        )

    def resources_versions(self) -> dict[str, int]:
        """Full resource-version snapshot (the resources.versions RPC payload)."""
        return self._state.version.snapshot()

    def start_run(
        self, tab_id: str, *, aliases: Mapping[str, str] | None = None
    ) -> int:
        """Run displayed cfg values; nonempty aliases require an explicit Reset first.

        See RunAnalyzeControlFacet.start_run for admission and failure semantics.
        """
        return self._run_analyze_control.start_run(tab_id, aliases=aliases)

    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        return self._run_analyze_control.load_tab_result(tab_id, data_path)

    def cancel_run(self) -> bool:
        return self._run_analyze_control.cancel_run()

    def cancel_analyze(self, tab_id: str) -> bool:
        return self._run_analyze_control.cancel_analyze(tab_id)

    # ------------------------------------------------------------------
    # Shutdown coordination (cancel-all + wait, ADR-0066)
    # ------------------------------------------------------------------

    def prepare_experiment_reload(self) -> ReloadPreview:
        return self._services.experiment_reload.prepare_reload()

    def reload_experiments(self, preview: ReloadPreview) -> ReloadReport:
        return self._services.experiment_reload.reload_confirmed(preview)

    def retry_experiment_reload(self) -> ReloadReport:
        return self._services.experiment_reload.retry_failed_reload()

    def retry_skipped_experiment_tabs(self) -> ReloadReport:
        return self._services.experiment_reload.retry_skipped_tabs()

    def can_retry_experiment_reload(self) -> bool:
        return self._services.experiment_reload.can_retry_reload

    def skipped_experiment_tab_count(self) -> int:
        return self._services.experiment_reload.skipped_count

    def active_operation_count(self) -> int:
        """How many operations (run / device / connect) are live right now.

        The View reads this before closing to decide whether to confirm with the
        user (a non-zero count means closing will cancel work in progress).
        Counts all live operations (run / device / connect AND analyze /
        interactive) — Handles owns the lifecycle (ADR-0066)."""
        return self._operation_handles.live_count()

    def close(self, timeout_ms: int = 5000) -> None:
        """Permanently stop background admission and drain workers and deliveries.

        Call on the Qt owner thread before releasing this Controller or its View.
        ``timeout_ms`` is the drain wait in milliseconds. A failed drain raises
        TimeoutError and leaves admission closed; retain the owner and retry close.
        Successful repeated calls are harmless. This does not cancel operations,
        decide whether to discard data, disconnect devices, or establish RF safety;
        use begin_shutdown for operation cancellation before final teardown.
        """
        self._background_svc.stop_accepting()
        if not self._background_svc.quiesce(timeout_ms):
            raise TimeoutError("Controller background work did not drain in time")

    def register_device(self, name: str, driver: BaseDevice[Any]) -> None:
        """Adopt a constructed driver through DeviceService on the owner thread.

        ``name`` is a nonblank session name and ``driver`` supplies its own info.
        Replaces an idle remembered row or adds a nonremembered connected row.
        Run/mutation conflicts, duplicate live names and driver/registry failures
        propagate without publishing a candidate. Ownership transfers only on
        success; GUI teardown closes registered drivers after drain and callbacks.
        """
        if self._hardware_close_attempted:
            raise FailedPreconditionError("GUI hardware has already closed")
        self._dev_svc.register_device(name, driver)

    def start_startup(self, path: Path) -> int:
        """Run trusted path's startup(ctx) off-owner after the Qt loop starts.

        Owner-only; returns an operation token. Script failures publish a
        path/line diagnostic and retain earlier commands. Repeated start raises
        RuntimeError. The shared background drain includes this work.
        """
        return self._services.startup.start(path, self._services.startup_context)

    def finish_close(self, timeout_ms: int = 5000) -> None:
        """Drain, run script callbacks in reverse order, then close managed hardware.

        Owner-only. timeout_ms bounds the normal drain; TimeoutError propagates
        without callbacks or hardware cleanup, allowing a later retry. Each
        callback/driver failure is logged and isolated. Successful repeated calls
        do not repeat callbacks or driver close attempts. SoC remains caller-owned.
        """
        self.close(timeout_ms)
        self._services.startup.close_callbacks()
        self.close_hardware()

    def close_hardware(self) -> None:
        """Close managed drivers once, then unpublish the caller-owned SoC.

        Call on the Qt owner thread only after successful close() drain and any
        caller on-close callbacks. Every driver gets one attempt; successful rows
        become MEMORY_ONLY and failed rows stay CONNECTED with error. Aggregate
        close errors are logged, not raised, so they cannot block GUI teardown.
        The SoC has no close contract. Its creator still owns external cleanup.
        Repeated calls do not retry failed driver closes or emit duplicate facts.
        """
        if self._hardware_close_attempted:
            return
        self._hardware_close_attempted = True
        try:
            self._dev_svc.close_all_devices()
        except ExceptionGroup:
            logger.exception("Failed to close GUI-managed devices")
        finally:
            self._soc_svc.uninstall_soc()

    def begin_shutdown(self, on_closed: Callable[[], None]) -> None:
        """Cancel every live operation, wait (with a timeout) for them to stop,
        then run ``on_closed`` — the View's actual teardown.

        Qt-free façade: the QTimer-driven coordinator lives in a driving adapter
        (ADR-0067), built lazily here so the Controller stays importable without
        a Qt loop. ``on_closed`` always runs on the main thread."""
        access = self._services.experiment_access
        was_shutting_down = access.shutting_down
        access.shutting_down = True
        try:
            if self._shutdown_driver is None:
                from zcu_tools.gui.session.adapters.qt_shutdown_driver import (
                    QtShutdownDriver,
                )

                self._shutdown_driver = QtShutdownDriver(self._operation_handles)
            self._shutdown_driver.begin(on_closed)
        except Exception:
            access.shutting_down = was_shutting_down
            raise

    def abort_shutdown(self) -> None:
        """Resume entry points after the settled close confirmation is declined.

        This does not restart cancelled operations or reopen a failed catalog.
        """
        self._services.experiment_access.shutting_down = False

    def get_operation_progress(
        self, operation_id: int
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._operation_control.get_operation_progress(operation_id)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._run_analyze_control.get_tab_analyze_result(tab_id)

    # ------------------------------------------------------------------
    # Analyze flow (TabService)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, analyze_params_instance: object) -> int:
        return self._run_analyze_control.analyze(tab_id, analyze_params_instance)

    # ------------------------------------------------------------------
    # Post-analysis flow (second layer; PostAnalyzeService)
    # ------------------------------------------------------------------

    def start_post_analyze(
        self, tab_id: str, post_analyze_params_instance: object
    ) -> int:
        return self._run_analyze_control.start_post_analyze(
            tab_id, post_analyze_params_instance
        )

    def get_post_analyze_result(self, tab_id: str) -> object | None:
        return self._run_analyze_control.get_post_analyze_result(tab_id)

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._tab_svc.update_tab_post_analyze_param_instance(tab_id, instance)
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=TabInteractionFact.POST_ANALYZE_PARAMS_CHANGED,
            )
        )

    def run_background(
        self,
        compute: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Run captured interactive computation off-main; deliver on the owner loop."""
        self._background_svc.submit(
            compute,
            run_in_pool=True,
            on_done=on_done,
            on_error=on_error,
        )

    # ------------------------------------------------------------------
    # Pane-qualified writeback commands
    # ------------------------------------------------------------------

    def get_writeback_item_form_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str
    ) -> ModuleEntryForm:
        """Admit access to a module item's shared editing Form for this pane.

        Every call checks writeback permission, tab idle and current draft
        ownership. Call again immediately before each state.write or
        state.run_command; retaining a Form does not retain admission.
        See WritebackService.get_item_form for the Form's editing effects.
        """
        return self._writeback_control.get_writeback_item_form_for_pane(
            tab_id, pane, session_id
        )

    def set_writeback_item_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str, **changes: Any
    ) -> None:
        self._writeback_control.set_writeback_item_for_pane(
            tab_id, pane, session_id, **changes
        )

    def apply_writeback_for_pane(
        self,
        tab_id: str,
        pane: WritebackPane,
        *,
        item_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        return self._writeback_control.apply_writeback_for_pane(
            tab_id, pane, item_ids=item_ids
        )

    def get_writeback_summaries_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, tuple[str | None, str | None]]:
        return self._writeback_control.get_writeback_summaries_for_pane(tab_id, pane)

    def get_writeback_applied_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, bool]:
        return self._writeback_control.get_writeback_applied_for_pane(tab_id, pane)

    # ------------------------------------------------------------------
    # Context and IO operations via ContextService
    # ------------------------------------------------------------------

    def read_context(
        self, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext:
        """Fix point/library/storage and aliases through the context owner.

        aliases maps alias names to temporary concrete components, preserving
        authored dependent links without changing global state. Reads perform
        no I/O or kind checks and retain no live callback. Missing/unbound $
        aliases fail by name on use; EMPTY/source/invalid override errors
        propagate. See ContextControlPort.read_context for the full contract.
        """
        return self._context_control.read_context(aliases=aliases)

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        """Commit one library structure command at the supplied fixed capture.

        Put/replace carry the persistent Spec to write. ContextService owns
        target validation, the single resource transaction and publication.
        Failed commands leave editor teardown to the caller.
        """
        self._context_control.apply_library_change(parameters, change)

    # ------------------------------------------------------------------
    # Arbitrary waveform assets (qubit-scoped repository)
    # ------------------------------------------------------------------

    @property
    def arb_waveforms(self) -> ArbWaveformPort:
        return self._arb_waveform_svc

    # ------------------------------------------------------------------
    # Templates: one-shot create for context.ml_create_from_role.
    # ------------------------------------------------------------------

    def get_template(self, template_id: str) -> TemplateDef:
        """Return the Registry definition; an unknown ID raises KeyError."""
        return self._registry.templates[template_id]

    def template_kind(self, template_id: str) -> LibraryKind:
        """Return the definition's library family; an unknown ID raises KeyError."""
        definition = self.get_template(template_id)
        return "module" if "type" in definition.module.model_fields else "waveform"

    def list_templates(self) -> list[TemplateMetadata]:
        """Return ordered discovery metadata with unchanged wire field names.

        default_name is the authored suggestion, including component
        placeholders; UI callers expand it against their selected capture.
        An empty Registry catalog returns an empty list.
        """
        return [
            {
                "role_id": template_id,
                "label": definition.label,
                "item_kind": self.template_kind(template_id),
                "default_name": definition.default_name,
            }
            for template_id, definition in self._registry.templates.items()
        ]

    def get_session_env(self) -> SessionEnv:
        return self._ctx_svc.get_session_env()

    def create_from_template(self, item_kind: str, template_id: str, name: str) -> None:
        """Create a library root from a template using one fixed parameter capture.

        Kind must match the template; the concrete name must be nonempty and
        unused in its library family. Template, capture, Form validation,
        conflict and I/O failures propagate without a write. Expr seeds use
        the fixed alias-aware capture, and bound Slots keep child library keys.
        The new state uses template_input's specialized model.
        """
        if not name:
            raise FailedPreconditionError("Entry name must not be empty.")
        template_kind = self.template_kind(template_id)
        if item_kind != "module" and item_kind != "waveform":
            raise FailedPreconditionError(f"Unknown library kind: {item_kind!r}")
        if template_kind != item_kind:
            raise FailedPreconditionError(
                f"Template {template_id!r} is a {template_kind}, not a {item_kind}."
            )
        parameters = self._context_control.read_context()
        keys = (
            parameters.library.module_keys()
            if item_kind == "module"
            else parameters.library.waveform_keys()
        )
        if name in keys:
            raise FailedPreconditionError(
                f"A {item_kind} named {name!r} already exists."
            )
        entry = new_library_entry(
            parameters,
            Template(template_id),
            self._registry.templates,
            arb_names=self._arb_waveform_svc.list_data_keys(),
        )
        self.apply_library_change(parameters, entry.put(name))

    def has_ml_entry(self, item_kind: str, name: str) -> bool:
        """Query a concrete root in the selected library; invalid kind raises."""
        library = self._context_control.read_context().library
        if item_kind == "module":
            return name in library.module_keys()
        if item_kind == "waveform":
            return name in library.waveform_keys()
        raise FailedPreconditionError(f"Unknown library kind: {item_kind!r}")

    # ------------------------------------------------------------------
    # Library entry editing through the shared Form entry
    # ------------------------------------------------------------------

    def open_library_entry(
        self, item_kind: LibraryKind, name: str
    ) -> ModuleEntryForm | WaveformEntryForm:
        """Open a selected library root as a Form at one fixed capture.

        The capture is read once from the selected point; the form evaluates
        expressions against it and submissions target it. Arb waveform data
        options are the asset keys listed now. Child library keys stay keys.
        No selection, an unknown kind or a missing root raises an ExpectedError.
        The caller owns the returned form; writing and discarding it need no
        controller call.
        """
        return open_library_entry(
            self._context_control.read_context(),
            item_kind,
            name,
            arb_names=self._arb_waveform_svc.list_data_keys(),
        )

    def commit_library_entry(
        self, entry: ModuleEntryForm | WaveformEntryForm, name: str
    ) -> None:
        """Put entry's current Spec at root name in its captured library.

        Invalid input raises LibraryEntryInvalid without writing. Target,
        conflict and I/O failures from apply_library_change propagate.
        """
        self.apply_library_change(entry.parameters, entry.put(name))

    def replace_library_entry(
        self, entry: ModuleEntryForm | WaveformEntryForm, source: str, new_name: str
    ) -> None:
        """Atomically replace root source with entry's Spec, renaming to new_name.

        Failures follow commit_library_entry; an occupied different new_name
        or a missing source is rejected with the library unchanged.
        """
        self.apply_library_change(entry.parameters, entry.replace(source, new_name))

    def edit_library_entry(
        self,
        item_kind: LibraryKind,
        name: str,
        batch: Mapping[FormWritePath, JsonValue],
        *,
        save_as: str | None = None,
    ) -> FormErrors | None:
        """Apply one Form write batch to library root name and write it if valid.

        The root opens as with open_library_entry. batch maps model-field leaf
        paths to Form input values, such as {"expr": text} for an expression
        or a plain key string for a child library entry, and is written in one
        FormState.write. A valid result is put at save_as, which must be a
        new root of this kind, or at name when save_as is None; save_as leaves
        name unchanged. Returns None after the write, or the form's FormErrors
        with the library unchanged. Unknown or overlapping paths, an existing
        save_as root, an unknown kind or a missing root raise
        InvalidInputError before any write. Target, conflict and I/O failures
        from apply_library_change propagate.
        """
        entry = self.open_library_entry(item_kind, name)
        if save_as is not None:
            library = entry.parameters.library
            keys = (
                library.module_keys()
                if item_kind == "module"
                else library.waveform_keys()
            )
            if save_as in keys:
                raise InvalidInputError(
                    f"{item_kind} destination already exists: {save_as!r}"
                )
        try:
            entry.state.write(batch, entry.env)
        except ValueError as exc:
            raise InvalidInputError(str(exc), reason_code="form_errors") from exc
        try:
            self.commit_library_entry(entry, name if save_as is None else save_as)
        except LibraryEntryInvalid as exc:
            return exc.errors
        return None

    # ------------------------------------------------------------------
    # Device operations via DeviceService
    # ------------------------------------------------------------------

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_value_for_new_context(self, name: str) -> float | None:
        return self._dev_svc.get_device_value_for_new_context(name)

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        return self._dev_svc.get_active_device_operations()

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        return self._operation_control.await_operation(operation_id, timeout)

    # ------------------------------------------------------------------
    # Project root and settings
    # ------------------------------------------------------------------

    def get_project_root(self) -> str:
        """Return the injected base directory for result and Database roots.

        Session composition passes its explicit roots to the entry owner;
        opening an entry never derives chip/qubit identity from the current cwd.
        """
        return self._project_root

    # ------------------------------------------------------------------
    # Connection and predictor operations via SoCConnectionService and PredictorService
    # ------------------------------------------------------------------

    def connect_sync(self, req: ConnectRequest) -> None:
        """Connect the SoC synchronously (blocks until connected + side effects
        applied). The wire ``soc.connect`` path uses this so the handler can return
        the SoC summary directly; the GUI's connect button keeps using the async
        start_connect + completion facts. Both share SoCConnectionService._apply_connection,
        so the post-connect side effects (State write, soc version bump,
        SocChangedPayload → FLUX-AWARE-MOCK provisioning) are identical."""
        self._soc_svc.connect_sync(req)

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

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab_control.get_tab_snapshot(tab_id)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._tab_svc.update_tab_analyze_param_instance(tab_id, instance)
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=TabInteractionFact.ANALYZE_PARAMS_CHANGED,
            ),
        )

    def get_adapter_names(self) -> list[str]:
        return self._tab_svc.list_adapter_names()

    def get_adapter_guide(self, adapter_name: str) -> AdapterGuide:
        """Read Markdown and resource directory for a registered adapter name.

        No tab is needed. Unknown names raise KeyError; adapter construction,
        guide I/O and decoding errors propagate unchanged.
        """
        return self._tab_svc.adapter_guide(adapter_name)
