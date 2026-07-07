"""MainWindow — the top-level View for the v2_gui framework.

Implements ViewProtocol; all state lives in Controller/State, never here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.main.adapter import AnalysisMode
from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.app.main.services.load import LoadDataError
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.plotting import set_shutting_down
from zcu_tools.gui.project import nearest_existing
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
)
from zcu_tools.gui.widgets import DialogPresenter, DialogRefStore, QtDialogPresenter

logger = logging.getLogger(__name__)


from qtpy.QtGui import QCloseEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .exp_tab_widget import ExpTabWidget, TabActions
from .feedback_dock import FeedbackDockController
from .main_dialog_registry import MainDialogRegistry

if TYPE_CHECKING:
    from qtpy.QtWidgets import QDialog  # type: ignore[attr-defined]

    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services import TabSnapshot


# ---------------------------------------------------------------------------
# MainWindow — implements ViewProtocol
# ---------------------------------------------------------------------------


class _MainWindowTabActions:
    """Adapt the tab action port to MainWindow's existing private handlers."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window

    def refresh_interaction(self, tab_id: str) -> None:
        self._window.refresh_tab_interaction(tab_id)

    def run_or_stop(self, tab_id: str) -> None:
        self._window._on_run_stop_clicked(tab_id)

    def load_data(self, tab_id: str) -> None:
        self._window._on_load_data_clicked(tab_id)

    def analyze(self, tab_id: str) -> None:
        self._window._on_analyze_clicked(tab_id)

    def post_analyze(self, tab_id: str) -> None:
        self._window._on_post_analyze_clicked(tab_id)

    def apply_writeback(self, tab_id: str) -> None:
        self._window._on_writeback_inline_apply(tab_id)

    def save_data(self, tab_id: str) -> None:
        self._window._on_save_data_clicked(tab_id)

    def save_image(self, tab_id: str) -> None:
        self._window._on_save_image_clicked(tab_id)

    def save_result(self, tab_id: str) -> None:
        self._window._on_save_result_clicked(tab_id)

    def save_post_image(self, tab_id: str) -> None:
        self._window._on_post_save_image_clicked(tab_id)


class MainWindow(QMainWindow):
    """Top-level window; implements ViewProtocol for Controller callbacks."""

    def __init__(
        self,
        controller: Controller,
        *,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        super().__init__()
        self._ctrl = controller
        self._tab_widgets: dict[str, ExpTabWidget] = {}
        self._tab_actions: TabActions = _MainWindowTabActions(self)
        self._dialog_refs = DialogRefStore()
        self._dialog_presenter = dialog_presenter or QtDialogPresenter(
            self._dialog_refs
        )
        self._dialog_registry = MainDialogRegistry(self._ctrl, parent=self)
        self._bus_subs = EventSubscriptions()
        # True once _perform_close has begun the actual teardown, so the second
        # closeEvent (triggered by _perform_close's self.close()) passes straight
        # through instead of re-entering the cancel-and-wait coordination.
        self._closing = False
        self.setWindowTitle("ZCU Qubit Measure — v2 GUI")
        self.resize(1280, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- toolbar ---
        toolbar = QHBoxLayout()
        self._new_tab_btn = QPushButton("New Tab ▾")
        self._new_tab_btn.clicked.connect(self._on_new_tab_requested)
        toolbar.addWidget(self._new_tab_btn)
        toolbar.addStretch()

        setup_btn = QPushButton("Setup…")
        setup_btn.clicked.connect(self._on_setup_clicked)
        toolbar.addWidget(setup_btn)

        devices_btn = QPushButton("Devices…")
        devices_btn.clicked.connect(self._on_devices_clicked)
        toolbar.addWidget(devices_btn)

        predictor_btn = QPushButton("Predictor…")
        predictor_btn.clicked.connect(self._on_predictor_clicked)
        toolbar.addWidget(predictor_btn)

        inspect_btn = QPushButton("Inspect…")
        inspect_btn.clicked.connect(self._on_inspect_clicked)
        toolbar.addWidget(inspect_btn)

        main_layout.addLayout(toolbar)

        # --- context / predictor status bar ---
        ctx_bar = QHBoxLayout()
        ctx_bar.addWidget(QLabel("Context:"))
        self._ctx_label = QLabel("(none)")
        ctx_bar.addWidget(self._ctx_label)
        ctx_bar.addSpacing(24)
        ctx_bar.addWidget(QLabel("Predictor:"))
        self._predictor_label = QLabel("none")
        ctx_bar.addWidget(self._predictor_label)
        ctx_bar.addStretch()
        main_layout.addLayout(ctx_bar)

        # --- tab widget ---
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        tab_bar = self._tabs.tabBar()
        assert tab_bar is not None
        tab_bar.tabMoved.connect(self._on_tab_moved)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_current_tab_changed)
        main_layout.addWidget(self._tabs, stretch=1)

        # --- status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # EventBus subscriptions
        bus = self._ctrl.get_bus()
        self._bus_subs.subscribe(
            bus, TabInteractionChangedPayload, self._on_bus_tab_interaction_changed
        )
        self._bus_subs.subscribe(bus, RunStartedPayload, self._on_bus_run_started)
        self._bus_subs.subscribe(bus, RunFinishedPayload, self._on_bus_run_finished)
        self._bus_subs.subscribe(
            bus, ContextSwitchedPayload, self._on_bus_context_switched
        )
        self._bus_subs.subscribe(bus, MlChangedPayload, self._on_bus_ml_changed)
        self._bus_subs.subscribe(bus, TabAddedPayload, self._on_bus_tab_added)
        self._bus_subs.subscribe(bus, TabClosedPayload, self._on_bus_tab_closed)
        self._bus_subs.subscribe(
            bus, TabContentChangedPayload, self._on_bus_tab_content_changed
        )
        self._bus_subs.subscribe(
            bus, PredictorChangedPayload, self._on_bus_predictor_changed
        )
        self._bus_subs.subscribe(bus, SocChangedPayload, self._on_bus_soc_changed)
        # Device setup ops: bus events let us track op start/finish for the
        # feedback widget (B1 approach — refresh at every op count change).
        self._bus_subs.subscribe(
            bus, DeviceSetupStartedPayload, self._on_bus_device_setup_started
        )
        self._bus_subs.subscribe(
            bus, DeviceSetupFinishedPayload, self._on_bus_device_setup_finished
        )
        # Device connect/disconnect ops: DeviceChangedPayload fires when the
        # device state changes (after op start and after op finish), covering
        # both starts and completions of connect/disconnect ops.
        self._bus_subs.subscribe(bus, DeviceChangedPayload, self._on_bus_device_changed)

        # Docked feedback panel (built after bus wiring; mounted under the
        # target tab's figure only while the C3 gate holds).
        self._feedback_dock = FeedbackDockController(
            self._ctrl,
            parent=self,
            tab_by_id=lambda tab_id: self._tab_widgets.get(tab_id),
            running_tab_id=self._ctrl.get_running_tab_id,
            active_tab_id=self._ctrl.get_active_tab_id,
        )

        # Cleanup on destroy
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        self._bus_subs.unsubscribe_all()

    def _on_bus_tab_interaction_changed(
        self, payload: TabInteractionChangedPayload
    ) -> None:
        tab_id = payload.tab_id
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self.refresh_tab_writeback(tab_id, snapshot)
        self.refresh_tab_interaction(tab_id, snapshot)
        interaction = snapshot.interaction
        if interaction is not None and not (
            interaction.is_running
            or interaction.is_analyzing
            or interaction.is_saving_data
        ):
            if interaction.has_analyze_result:
                self.refresh_tab_figure(tab_id, snapshot)
            if interaction.has_post_analyze_result:
                self.refresh_tab_post_figure(tab_id, snapshot)
        # Interactive analyze start/finish both emit TabInteractionChangedPayload;
        # refresh widget visibility and stop-gating on every change.
        self._refresh_feedback_widget()

    def _on_bus_run_started(self, payload: RunStartedPayload) -> None:
        # Run lock now held by this tab.
        self.refresh_run_lock(payload.tab_id)
        self._refresh_feedback_widget()

    def _on_bus_run_finished(self, payload: RunFinishedPayload) -> None:
        # Run lock released.
        self.refresh_run_lock(None)
        self._refresh_feedback_widget()
        # Auto-switch to the second tab on a normal finish — RUN_FINISHED carries
        # the outcome directly, so the decision lives here. A stopped run
        # (outcome=cancelled) may leave a partial result, but the user interrupted
        # it on purpose; don't yank them away. RunService writes the result to
        # State before emitting RUN_FINISHED, so has_run_result is already set.
        # The second tab holds analysis widgets AND the Save section; for a
        # non-analysis adapter (flux_dep / power_dep) the analysis widgets are
        # hidden but Save stays — switching there lands the user on Save, which is
        # exactly what they want after a 2D sweep.
        if payload.outcome != "finished":
            return
        tab_w = self._tab_widgets.get(payload.tab_id)
        if tab_w is None:
            return
        snapshot = self._ctrl.get_tab_snapshot(payload.tab_id)
        assert snapshot.interaction is not None  # render snapshot fills live fields
        if snapshot.interaction.has_run_result:
            tab_w._left_tabs.setCurrentIndex(1)

    def _on_bus_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        # cfg EvalValue refresh is the CfgEditorService's job now (it owns the
        # models — ADR-0008); here we only refresh the surrounding tab panels.
        self.refresh_context_panel()
        for tab_id in list(self._tab_widgets):
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self.refresh_tab_writeback(tab_id, snapshot)
            self.refresh_tab_save_paths(tab_id, snapshot)
            self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_ml_changed(self, payload: MlChangedPayload) -> None:
        del payload
        # cfg EvalValue refresh is the CfgEditorService's job now (ADR-0008);
        # here we only refresh the surrounding tab panels.
        for tab_id in list(self._tab_widgets):
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self.refresh_tab_writeback(tab_id, snapshot)
            self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_tab_added(self, payload: TabAddedPayload) -> None:
        tab_id = payload.tab_id
        adapter_name = payload.adapter_name
        logger.info("_on_bus_tab_added: tab_id=%r adapter=%r", tab_id, adapter_name)
        if tab_id in self._tab_widgets:
            return

        tab_label = adapter_name
        tab_w = ExpTabWidget(
            tab_id, self._ctrl, dialog_presenter=self._dialog_presenter
        )
        self._tab_widgets[tab_id] = tab_w
        self._tabs.addTab(tab_w, tab_label)
        self._tabs.setCurrentWidget(tab_w)

        # Bring the whole tab widget to life from one render snapshot (seed every
        # sub-view + wire controller signals) — the whole-tab analogue of
        # CfgFormWidget.attach.
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self._new_tab_btn.setEnabled(True)
        tab_w.attach(snapshot, self._tab_actions)

    def _on_bus_tab_closed(self, payload: TabClosedPayload) -> None:
        tab_id = payload.tab_id
        logger.info("_on_bus_tab_closed: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.pop(tab_id, None)
        if tab_w is not None:
            tab_w.detach()
            index = self._tabs.indexOf(tab_w)
            if index >= 0:
                self._tabs.removeTab(index)
            tab_w.deleteLater()

        self.refresh_run_lock(self._ctrl.get_running_tab_id())

    def _on_bus_tab_content_changed(self, payload: TabContentChangedPayload) -> None:
        tab_id = payload.tab_id
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self.refresh_tab_analyze_form(tab_id, snapshot)
        self.refresh_tab_post_analyze_form(tab_id, snapshot)
        self.refresh_tab_writeback(tab_id, snapshot)
        self.refresh_tab_save_paths(tab_id, snapshot)
        self.refresh_tab_figure(tab_id, snapshot)
        self.refresh_tab_post_figure(tab_id, snapshot)
        # The auto-switch to Analysis lives in _on_bus_run_finished (it needs the
        # run outcome); content refresh here is outcome-agnostic.
        self.refresh_tab_interaction(tab_id, snapshot)
        # FIT analyze finish emits TabContentChangedPayload; refresh widget.
        self._refresh_feedback_widget()

    def _on_bus_predictor_changed(self, payload: PredictorChangedPayload) -> None:
        del payload
        self.refresh_predictor_panel()

    def _on_bus_soc_changed(self, payload: SocChangedPayload) -> None:
        del payload
        self.refresh_run_lock(self._ctrl.get_running_tab_id())
        # SoC connect succeeds → op count drops; refresh widget visibility.
        self._refresh_feedback_widget()

    def _on_bus_device_setup_started(self, payload: DeviceSetupStartedPayload) -> None:
        del payload
        self._refresh_feedback_widget()

    def _on_bus_device_setup_finished(
        self, payload: DeviceSetupFinishedPayload
    ) -> None:
        del payload
        self._refresh_feedback_widget()

    def _on_bus_device_changed(self, payload: DeviceChangedPayload) -> None:
        # Fires on device connect/disconnect start and on completion (state
        # change after op finish). Refreshes the widget on every op boundary.
        del payload
        self._refresh_feedback_widget()

    # ------------------------------------------------------------------
    # Docked feedback panel (ADR-0025 C3)
    # ------------------------------------------------------------------

    def _refresh_feedback_widget(self) -> None:
        """Mount/unmount the feedback panel (internal bus-handler trampoline).

        Idempotent: all bus handlers that may change op-count or agent-presence
        call this; the decision is centralized in refresh_feedback_widget().
        """
        self.refresh_feedback_widget()

    def refresh_feedback_widget(self) -> None:
        """Mount/unmount the docked feedback panel on op count + agent presence.

        Called by both bus handlers (op count change) and
        RemoteControlAdapter._on_client_count_changed() (agent presence change).
        Both callers run on the Qt main thread — no thread guard needed.
        """
        self._feedback_dock.refresh()

    # ------------------------------------------------------------------
    # ViewProtocol implementation
    # ------------------------------------------------------------------

    def _set_tab_running(
        self,
        tab_w: ExpTabWidget,
        snapshot: TabSnapshot,
    ) -> None:
        tab_w.update_interaction_state(snapshot)

    def refresh_tab_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        assert current.interaction is not None  # render snapshot fills live fields
        assert current.capabilities is not None  # render snapshot fills live fields
        # Non-analysis adapters (flux_dep / power_dep 2D sweeps) intentionally
        # have no analyze params after a run — there is no analyze form to fill,
        # so skip before the Fast-Fail below (which guards the *analysis* adapter
        # contract: a run result must carry initialized params).
        if current.capabilities.analysis is AnalysisMode.NONE:
            return
        if not current.interaction.has_run_result:
            return
        if current.analyze_params is None:
            raise RuntimeError("Run result has no initialized analyze parameters")
        tab_w.populate_analyze_params(current.analyze_params)
        tab_w.analyze_form.populate_values(current.analyze_params)

    def refresh_tab_post_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        assert current.capabilities is not None  # render snapshot fills live fields
        # Only post-analysis adapters have a post form; for the rest there is
        # nothing to fill. When the primary analyze result is invalidated the post
        # params are cleared (State), so there is no instance to populate — the
        # gate (update_interaction_state) disables the empty form.
        if not current.capabilities.post_analysis:
            return
        if current.post_analyze_params is None:
            return
        tab_w.populate_post_analyze_params(current.post_analyze_params)
        tab_w.post_analyze_form.populate_values(current.post_analyze_params)

    def refresh_tab_writeback(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        tab_w.update_writeback_items(list(current.writeback_items))

    def refresh_tab_save_paths(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        save_paths = current.save_paths
        if save_paths is not None:
            tab_w.set_save_paths(save_paths.data_path, save_paths.image_path)

    def refresh_tab_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        figure = current.figure
        if figure is not None:
            self.show_analysis_image(tab_id, figure)

    def refresh_tab_post_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        post_figure = current.post_figure
        if post_figure is not None:
            # Post + analyze share one container; ``refresh_tab_figure`` runs just
            # before this and renders ``tab.figure``, so when a post figure exists
            # it is drawn last and the shared container shows the most recent
            # (post) figure. On invalidation (post_figure is None) there is nothing
            # to do: the shared container already shows the primary figure.
            self.show_post_analysis_image(tab_id, post_figure)

    def refresh_run_lock(self, running_tab_id: str | None) -> None:
        logger.debug("refresh_run_lock: running_tab_id=%r", running_tab_id)
        self._new_tab_btn.setEnabled(True)
        for tab_id, tab_w in self._tab_widgets.items():
            if self._ctrl.has_tab(tab_id):
                self._set_tab_running(tab_w, self._ctrl.get_tab_snapshot(tab_id))
        # Progress no longer cleared here — ProgressService.discard_operation on
        # the run's terminal path drops the container and notifies the tab's
        # listener, which re-renders to empty.

    def refresh_tab_interaction(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None or not self._ctrl.has_tab(tab_id):
            return
        self._set_tab_running(tab_w, snapshot or self._ctrl.get_tab_snapshot(tab_id))

    def refresh_context_panel(self) -> None:
        label = self._ctrl.get_active_context_label()
        if label is not None:
            # file-backed flux context is active
            self._ctx_label.setText(label)
            self._ctx_label.setStyleSheet("")
        elif self._ctrl.has_startup_context():
            # startup context (in-memory, no file sync)
            self._ctx_label.setText(
                "Startup context (in-memory) — set up project for persistence"
            )
            self._ctx_label.setStyleSheet("color: blue;")
        elif self._ctrl.has_project():
            self._ctx_label.setText(
                "Project set — select a context to enable Run/Analyze/Save"
            )
            self._ctx_label.setStyleSheet("color: orange;")
        else:
            self._ctx_label.setText("No project set — use Project… to configure")
            self._ctx_label.setStyleSheet("color: gray;")
        for tab_id, tab_w in self._tab_widgets.items():
            if self._ctrl.has_tab(tab_id):
                self._set_tab_running(tab_w, self._ctrl.get_tab_snapshot(tab_id))

    def refresh_inspect_panel(self) -> None:
        inspect = self._dialog_registry.dialog(DialogName.INSPECT)
        if inspect is not None and inspect.isVisible():
            # InspectDialog defines ``refresh``; cast through ``Any`` to avoid
            # importing the concrete class in the hot signature surface.
            from typing import cast

            cast(Any, inspect).refresh()

    def refresh_predictor_panel(self) -> None:
        info = self._ctrl.predictor_control.get_predictor_info()
        if info is None:
            self._predictor_label.setText("none")
            self._predictor_label.setStyleSheet("")
        else:
            flux_bias = info["flux_bias"]
            self._predictor_label.setText(f"loaded (flux_bias={flux_bias:.4g})")
            self._predictor_label.setStyleSheet("color: green;")

    def make_live_container(self, tab_id: str) -> Any:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return None
        tab_w.reset_plot()  # clear prior liveplot before new run/analyze
        return tab_w._figure_container

    def mount_interactive_analysis(
        self,
        tab_id: str,
        session_factory: Callable[[Any], Any],
        on_finish: Callable[[Any], None],
    ) -> None:
        """RenderHost impl: mount an INTERACTIVE adapter's live picker in the tab's
        plot stack. Build the host widget, hand it to the adapter to set up the
        session, render the session's actions, and wire Done -> on_finish(session).
        The View knows nothing of the interaction (lines / flux); it just forwards
        events and shows session.info_text()."""
        from zcu_tools.gui.app.main.ui.interactive_analysis import (
            InteractiveAnalysisWidget,
        )

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.reset_plot()
        # The Controller satisfies InteractiveHostEnv (run_background via bg's
        # pool); the widget pulls only that one capability through the port.
        widget = InteractiveAnalysisWidget(self._ctrl)
        session = session_factory(widget)  # the widget IS the InteractiveHost
        widget.bind(session, on_done=lambda: on_finish(session))
        tab_w._plot_stack.addWidget(widget)
        tab_w._plot_stack.setCurrentWidget(widget)

    def unmount_interactive_analysis(self, tab_id: str) -> None:
        """RenderHost impl: remove the tab's mounted interactive picker (dual of
        ``mount_interactive_analysis``). The picker widget is added straight to the
        plot stack (it is not a FigureContainer canvas), so ``reset_plot`` cannot
        reach it — this is the only teardown path for a cancelled interactive
        analyze. A no-op when no picker is mounted, and idempotent."""
        from zcu_tools.gui.app.main.ui.interactive_analysis import (
            InteractiveAnalysisWidget,
        )

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        stack = tab_w._plot_stack
        for i in range(stack.count()):
            widget = stack.widget(i)
            if isinstance(widget, InteractiveAnalysisWidget):
                stack.removeWidget(widget)
                widget.deleteLater()
        # Revert the visible pane to the placeholder; the cancelled analyze leaves
        # no figure of its own behind.
        stack.setCurrentWidget(tab_w._plot_placeholder)

    def current_left_panel_width(self) -> int:
        """RenderHost impl: the active tab's left-panel width (the single
        persistence value sourced from the View). Falls back to the default when
        no tab is open."""
        from zcu_tools.gui.app.main.state import DEFAULT_LEFT_PANEL_WIDTH

        current = self._tabs.currentWidget()
        if isinstance(current, ExpTabWidget):
            return current._splitter_left_saved
        return DEFAULT_LEFT_PANEL_WIDTH

    def notify_diagnostic(self, severity: str, title: str, message: str) -> None:
        """DiagnosticSink impl (ADR-0013): render a Controller diagnostic the Qt
        way — error pops a modal dialog, info goes to the status bar."""
        if severity == "error":
            self.show_error_dialog(title or "Error", message)
        else:
            self.show_status_message(message)

    def show_status_message(self, message: str) -> None:
        logger.info("status: %s", message)
        self._status_bar.showMessage(message)

    def show_error_dialog(self, title: str, message: str) -> None:
        self._dialog_presenter.critical(self, title, message)

    def show_plot(self, tab_id: str, fig: Any) -> None:  # Phase 11
        logger.debug("show_plot: tab_id=%r fig=%s", tab_id, type(fig).__name__)

    def show_analysis_image(self, tab_id: str, fig: Any) -> None:
        logger.debug("show_analysis_image: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.show_analysis_figure(fig)

    def show_post_analysis_image(self, tab_id: str, fig: Any) -> None:
        # Post figures share the primary right-pane container (the container shows
        # the most recently produced figure), so this routes through the same
        # render path as the analyze figure.
        logger.debug("show_post_analysis_image: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.show_analysis_figure(fig)

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_new_tab_requested(self) -> None:
        menu = QMenu(self)
        submenus: dict[tuple[str, ...], QMenu] = {}

        def _get_or_create_submenu(path: tuple[str, ...]) -> QMenu:
            cached = submenus.get(path)
            if cached is not None:
                return cached
            if len(path) == 1:
                parent_menu = menu
            else:
                parent_menu = _get_or_create_submenu(path[:-1])
            sub_menu = parent_menu.addMenu(path[-1])
            if sub_menu is None:
                raise RuntimeError(f"Failed to create submenu: {'/'.join(path)}")
            submenus[path] = sub_menu
            return sub_menu

        for name in self._ctrl.get_adapter_names():
            parts = tuple(name.split("/"))
            if len(parts) == 1:
                action = menu.addAction(parts[0])
                action.setData(name)  # type: ignore[union-attr]
                continue
            parent_menu = _get_or_create_submenu(parts[:-1])
            action = parent_menu.addAction(parts[-1])
            action.setData(name)  # type: ignore[union-attr]

        action = menu.exec(
            self._new_tab_btn.mapToGlobal(  # type: ignore[assignment]
                self._new_tab_btn.rect().bottomLeft()
            )
        )
        if action is None:
            return
        adapter_name = action.data()
        if not adapter_name:
            return

        self._ctrl.new_tab(adapter_name)

    def _on_tab_close_requested(self, index: int) -> None:
        tab_w = self._tabs.widget(index)
        if not isinstance(tab_w, ExpTabWidget):
            return
        tab_id = tab_w.tab_id
        logger.info("_on_tab_close_requested: tab_id=%r", tab_id)
        self._ctrl.close_tab(tab_id)

    def _on_tab_moved(self, from_index: int, to_index: int) -> None:
        logger.debug("_on_tab_moved: from=%d to=%d", from_index, to_index)
        tab_ids: list[str] = []
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, ExpTabWidget) and self._ctrl.has_tab(widget.tab_id):
                tab_ids.append(widget.tab_id)
        self._ctrl.reorder_tabs(tab_ids)

    def _on_current_tab_changed(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if not isinstance(widget, ExpTabWidget):
            return
        if not self._ctrl.has_tab(widget.tab_id):
            return
        self._ctrl.set_active_tab(widget.tab_id)
        # The active tab is the feedback panel's target when nothing is running;
        # re-evaluate so a visible panel follows the user to the new tab.
        self._refresh_feedback_widget()

    def _resolve_tab_widget(self, tab_id: str, action: str) -> ExpTabWidget | None:
        """Look up the widget; log + bail if tab_id is unknown to the controller."""
        if not self._ctrl.has_tab(tab_id):
            logger.warning("%s: unknown tab_id=%r — ignoring", action, tab_id)
            return None
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            logger.warning(
                "%s: tab_id=%r missing in view registry — ignoring", action, tab_id
            )
            return None
        return tab_w

    def _on_run_stop_clicked(self, tab_id: str) -> None:
        tab_w = self._resolve_tab_widget(tab_id, "_on_run_stop_clicked")
        if tab_w is None:
            return
        interaction = self._ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None  # render snapshot fills live fields
        if interaction.is_running:
            logger.info("_on_run_stop_clicked: stop requested tab_id=%r", tab_id)
            self._ctrl.cancel_run()
            return
        logger.info("_on_run_stop_clicked: run requested tab_id=%r", tab_id)
        if not tab_w.cfg_form.is_valid():
            reason = tab_w.cfg_form.first_invalid_reason()
            if reason:
                msg = f"Config invalid: {reason}"
            else:
                msg = "Config has unset fields — fill required values before running"
            logger.warning("_on_run_stop_clicked: blocked — %s", msg)
            self.show_status_message(msg)
            return
        self._ctrl.start_run(tab_id)

    def _on_analyze_clicked(self, tab_id: str) -> None:
        logger.info("_on_analyze_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_analyze_clicked")
        if tab_w is None:
            return
        self._ctrl.analyze(tab_id, tab_w.read_analyze_params())

    def _load_data_dialog_start_dir(self) -> str:
        return nearest_existing(self._ctrl.get_exp_context().database_path)

    def _on_load_data_clicked(self, tab_id: str) -> None:
        logger.info("_on_load_data_clicked: tab_id=%r", tab_id)
        if self._resolve_tab_widget(tab_id, "_on_load_data_clicked") is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load data file",
            self._load_data_dialog_start_dir(),
            "HDF5 files (*.hdf5 *.h5);;All files (*)",
        )
        if not path:
            return
        try:
            self._ctrl.load_tab_result(tab_id, path)
        except LoadDataError as exc:
            logger.warning(
                "_on_load_data_clicked rejected data file: tab_id=%r path=%r reason=%s",
                tab_id,
                path,
                exc.reason_code,
            )
            self.show_error_dialog("Load data failed", str(exc))
            return
        except Exception as exc:
            logger.exception("_on_load_data_clicked failed: tab_id=%r", tab_id)
            self.show_error_dialog("Load data failed", str(exc))
            return
        self.show_status_message(f"Loaded data from {path}")

    def _on_post_analyze_clicked(self, tab_id: str) -> None:
        logger.info("_on_post_analyze_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_post_analyze_clicked")
        if tab_w is None:
            return
        self._ctrl.start_post_analyze(tab_id, tab_w.read_post_analyze_params())

    def _on_writeback_inline_apply(self, tab_id: str) -> None:
        logger.info("_on_writeback_inline_apply: tab_id=%r", tab_id)
        if not self._ctrl.has_tab(tab_id):
            logger.warning(
                "_on_writeback_inline_apply: unknown tab_id=%r — ignoring", tab_id
            )
            return
        applied_ids = self._ctrl.apply_writeback(tab_id)
        if applied_ids:
            self.show_status_message(f"Writeback applied: {', '.join(applied_ids)}")

    def _on_save_data_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_data_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_data_clicked")
        if tab_w is None:
            return
        path = tab_w.get_data_path()
        self._ctrl.save_data(tab_id, path, comment=tab_w.get_comment())

    def _on_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_image_clicked")
        if tab_w is None:
            return
        path = tab_w.get_image_path()
        self._ctrl.save_image(tab_id, path)

    def _on_post_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_post_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_post_save_image_clicked")
        if tab_w is None:
            return
        path = tab_w.get_post_image_path()
        self._ctrl.save_post_image(tab_id, path)

    def _on_save_result_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_result_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_result_clicked")
        if tab_w is None:
            return
        data_path = tab_w.get_data_path()
        image_path = tab_w.get_image_path()
        self._ctrl.save_result(
            tab_id, data_path, image_path, comment=tab_w.get_comment()
        )

    def _on_setup_clicked(self) -> None:
        self.open_dialog(DialogName.SETUP)

    def _on_devices_clicked(self) -> None:
        self.open_dialog(DialogName.DEVICE)

    def _on_predictor_clicked(self) -> None:
        self.open_dialog(DialogName.PREDICTOR)

    def _on_inspect_clicked(self) -> None:
        self.open_dialog(DialogName.INSPECT)

    # ------------------------------------------------------------------
    # Dialog API — single entry point shared by UI clicks and remote control
    # ------------------------------------------------------------------

    def open_dialog(self, name: DialogName) -> None:
        """Open the named dialog non-modally, or raise it if already open.

        Idempotent: a second ``open_dialog(name)`` while the dialog exists
        raises it if visible, or shows a hidden persistent instance.
        """
        self._dialog_registry.open(name)

    def close_dialog(self, name: DialogName) -> None:
        """Close or hide the named dialog if it is currently open."""
        self._dialog_registry.close(name)

    def list_open_dialogs(self) -> list[DialogName]:
        """Return dialogs that are currently visible on screen."""
        return self._dialog_registry.visible_names()

    def register_dialog(self, name: DialogName, dialog: QDialog) -> None:
        """Register a dialog that was constructed outside ``open_dialog``.

        ``app.py`` uses this for the bootstrap startup dialog so the remote
        ``dialog.list_open`` query and ``dialog.close STARTUP`` work
        uniformly. The caller is responsible for ``setAttribute
        (WA_DeleteOnClose)`` and for ``open()`` / ``show()`` — this helper
        only wires the registry cleanup on ``finished``.
        """
        self._dialog_registry.register(name, dialog)

    # ------------------------------------------------------------------
    # Remote view query helpers
    # ------------------------------------------------------------------

    def get_view_snapshot(self) -> dict[str, object]:
        """Capture the visible window state as a JSON-friendly dict.

        tab_ids and active_tab_id are sourced from State (via ctrl.list_tab_ids)
        rather than _tab_widgets, so ghost widget entries that have diverged from
        State never leak into the projection (ADR-0013 view = second SSOT reader).
        """
        # State.tabs is the single SSOT for which tabs exist.
        state_tab_ids: list[str] = self._ctrl.list_tab_ids()
        state_tab_id_set = set(state_tab_ids)

        # Resolve the active tab through the widget hierarchy but only accept it
        # when the corresponding id is also known to State.
        active_id: str | None = None
        if self._tabs.count() > 0:
            current = self._tabs.currentWidget()
            for tid, tab_w in self._tab_widgets.items():
                if tab_w is current and tid in state_tab_id_set:
                    active_id = tid
                    break

        return {
            "active_tab_id": active_id,
            "tab_ids": state_tab_ids,
            "context_label": self._ctx_label.text() if self._ctx_label else "",
            "predictor_label": (
                self._predictor_label.text() if self._predictor_label else ""
            ),
            "status": self._status_bar.currentMessage() if self._status_bar else "",
            "open_dialogs": [name.value for name in self.list_open_dialogs()],
        }

    def take_figure_screenshot(self, tab_id: str) -> bytes:
        """Render a tab's figure to PNG bytes at the fixed export size.

        Renders the live figure via savefig (not ``canvas.grab()``) so the
        screenshot has the same window-independent geometry as a saved image,
        rather than tracking the current widget pixel size.
        """
        from matplotlib.figure import Figure

        from zcu_tools.gui.app.main.figure_export import render_figure_png

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            raise RuntimeError(f"unknown tab_id: {tab_id!r}")
        canvas = tab_w._plot_stack.currentWidget()
        if canvas is None or canvas is tab_w._plot_placeholder:
            raise RuntimeError(f"tab {tab_id!r} has no figure yet")
        figure = getattr(canvas, "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError(f"tab {tab_id!r} canvas has no matplotlib figure")
        return render_figure_png(figure)

    def take_dialog_screenshot(self, dialog_name: DialogName) -> bytes:
        """Grab a currently-open dialog and return raw PNG bytes."""
        return self._dialog_registry.take_screenshot(dialog_name)

    def take_window_screenshot(self) -> bytes:
        """Grab the WHOLE main window (client area + child widgets) as PNG bytes.

        ``self.grab()`` renders this QMainWindow and its child widgets, so it
        captures the docked feedback panel and the left-edge handle that ride on
        the client area and are invisible to the per-dialog grab. Same
        main-thread Qt path as take_dialog_screenshot.
        """
        from qtpy.QtCore import QBuffer, QIODevice  # type: ignore[attr-defined]

        pixmap = self.grab()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buf, "PNG"):
            raise RuntimeError("Qt failed to encode the main window as PNG")
        return bytes(buf.data().data())  # type: ignore[arg-type]

    def open_notify_prompt(self, token: int, message: str, timeout: float) -> None:
        """Open a non-modal NotifyUserDialog for an agent-initiated prompt.

        Called on the main thread by Controller.open_notify_prompt (via the
        notify.open RPC handler). The dialog is the timeout SSOT: its QTimer
        fires ctrl.timeout_notify rather than the consumer backstop.
        """
        from .notify_dialog import NotifyUserDialog

        dlg = NotifyUserDialog(token, message, timeout, self._ctrl, parent=self)
        self._dialog_refs.open_transient(dlg)

    def request_shutdown(self) -> None:
        """Programmatic close (the app.shutdown RPC). Runs on the Qt main thread
        via the remote dispatch marshal. Does the same work as a user's
        window-close — cancel every live operation, wait for them to stop,
        persist session, tear down remote, quit — but without the interactive
        confirmation (no user to answer it).

        The cancel-and-wait is deferred to the next event-loop turn so the
        triggering RPC's reply is written back before the remote service tears
        down (else the agent's app.shutdown would race the socket teardown)."""
        from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))

    def _perform_close(self, a0: QCloseEvent | None = None) -> None:
        """The actual teardown: persist session, stop remote, accept the close.
        Runs once every cancelled operation has settled (or timed out), driven by
        the Controller's shutdown coordinator. Shared by closeEvent (user) and
        request_shutdown (RPC)."""
        self._closing = True
        self._cleanup_bus_subscriptions()
        self._ctrl.persist_all()
        set_shutting_down(True)
        # Tear down remote control before the Qt main loop exits so any in-flight
        # RPC sees a clean shutdown (timeout / EPIPE) rather than a dead Controller.
        remote = getattr(self, "remote_control_service", None)
        if remote is not None:
            remote.stop()
            self.remote_control_service = None
        if a0 is not None:
            super().closeEvent(a0)
        else:
            self.close()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        # Second pass: _perform_close → self.close() re-enters here once teardown
        # has begun; accept it straight through.
        if self._closing:
            if a0 is not None:
                super().closeEvent(a0)
            return
        # A user window-close cancels every live operation, then closes once they
        # stop (or a timeout forces it). Confirm first if work is in progress —
        # closing will interrupt it. The wait is asynchronous, so ignore this
        # event now; the coordinator drives _perform_close when ready.
        active = self._ctrl.active_operation_count()
        if active > 0:
            if a0 is None:
                return
            confirmed = self._dialog_presenter.confirm(
                self,
                "Operations in progress",
                f"Cancel {active} operation(s) in progress and close once they stop?",
                default=False,
            )
            if not confirmed:
                a0.ignore()
                return
            a0.ignore()
        elif a0 is not None:
            a0.ignore()
        # Defer begin_shutdown to the next event-loop turn (mirrors
        # request_shutdown). When idle, the shutdown coordinator settles
        # synchronously and calls _perform_close → self.close(), which would
        # otherwise re-enter this closeEvent within its own stack — Qt does not
        # honour a self.close() issued from inside a closeEvent handler, so the
        # first click would appear to do nothing. The singleShot breaks out of
        # this stack first.
        from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))
