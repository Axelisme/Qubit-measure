"""MainWindow — the autofluxdep-gui shell.

A QMainWindow holding a measure-gui-style top session row, a left/right split
(``NodeListPane`` + ``NodeDetailPane``), and a global flux progress bar in the
central bottom row. It wires the edit↔run state switch and owns the liveplot
integration:

- At **Run start** it allocates each provider's sweep Result (``Controller
  .prepare_run_results``) and, for every Result, builds a bare matplotlib
  ``Figure`` + the provider's ``Plotter`` + a ``FigureCanvasQTAgg`` — all
  sweep-lived, so auto-follow can show any provider's plot at any time.
- It calls ``Controller.start_run`` as an OperationRunner client, passing a
  ``notify(name, idx)`` callback. The shared BackgroundRunner fills the Result
  rows in place and fires ``notify``. The controller-owned run relay emits
  EventBus run payloads on the Qt main thread, while ``_RunBridge`` fans those
  payloads into UI signals and keeps ``notify`` row redraws coalesced. A
  main-thread slot then calls ``plotter.update(result, idx)`` — all drawing stays
  on the main thread (ADR-0017: the worker never touches matplotlib).

The run acquires against a flux-aware MockSoc (offline) or real hardware; Setup
builds a MockSoc + FakeDevice + a SimplePredictor.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QCloseEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import (
    FLUX_PROGRESS_LABEL,
    RUN_PROGRESS_OWNER_ID,
    Controller,
)
from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunContinuedPayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunPausedPayload,
    RunPauseRequestedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.profiling import PerfStats, elapsed_ms, perf_now
from zcu_tools.gui.app.autofluxdep.services.persistence_types import (
    PersistedPredictorDialogState,
)
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SessionPayload,
    SocChangedPayload,
)
from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import DialogPresenter, DialogRefStore, QtDialogPresenter

from .node_detail import NodeDetailPane
from .node_list import NodeListPane

if TYPE_CHECKING:
    from zcu_tools.gui.session.ui.predictor_dialog import (
        PredictorDialog,
        PredictorDialogState,
    )

logger = logging.getLogger(__name__)
_ProgressSnapshot = tuple[int, str, int]


class _RunBridge(QObject):
    """Projects main-thread run events and coalesced row notifications to UI signals.

    The controller emits EventBus run-domain payloads on the Qt main thread.
    This bridge fans them out as UI-facing signals. ``row_updated`` stays as the
    worker round-hook path: it carries only a provider name + flux index and is
    coalesced locally so high-frequency redraws never become EventBus traffic.
    """

    run_started = Signal()
    run_pause_requested = Signal()
    run_paused = Signal(int)
    run_continued = Signal(int)
    node_entered = Signal(str, int)
    point_done = Signal(int)
    run_finished = Signal()
    run_stopped = Signal()
    run_failed = Signal(str)
    row_updated = Signal(str, int, float)

    def __init__(self, ctrl: Controller) -> None:
        super().__init__()
        self._row_lock = threading.Lock()
        self._pending_rows: set[tuple[str, int]] = set()
        self._dirty_rows: set[tuple[str, int]] = set()
        self._bus_subs = EventSubscriptions()
        bus = ctrl.bus
        self._bus_subs.subscribe(bus, RunStartedPayload, self._on_run_started_payload)
        self._bus_subs.subscribe(
            bus, RunPauseRequestedPayload, self._on_run_pause_requested_payload
        )
        self._bus_subs.subscribe(bus, RunPausedPayload, self._on_run_paused_payload)
        self._bus_subs.subscribe(
            bus, RunContinuedPayload, self._on_run_continued_payload
        )
        self._bus_subs.subscribe(bus, NodeEnteredPayload, self._on_node_entered)
        self._bus_subs.subscribe(bus, PointDonePayload, self._on_point_done_payload)
        self._bus_subs.subscribe(bus, RunFinishedPayload, self._on_run_finished_payload)
        self._bus_subs.subscribe(bus, RunStoppedPayload, self._on_run_stopped_payload)
        self._bus_subs.subscribe(bus, RunFailedPayload, self._on_run_failed_payload)
        self.destroyed.connect(self.teardown)

    def teardown(self, *_args: object) -> None:
        self._bus_subs.unsubscribe_all()

    def _on_run_started_payload(self, _payload: RunStartedPayload) -> None:
        self.run_started.emit()

    def _on_run_pause_requested_payload(
        self, _payload: RunPauseRequestedPayload
    ) -> None:
        self.run_pause_requested.emit()

    def _on_run_paused_payload(self, payload: RunPausedPayload) -> None:
        self.run_paused.emit(payload.next_flux_idx)

    def _on_run_continued_payload(self, payload: RunContinuedPayload) -> None:
        self.run_continued.emit(payload.next_flux_idx)

    def _on_node_entered(self, p: NodeEnteredPayload) -> None:
        self.node_entered.emit(p.name, p.idx)

    def _on_point_done_payload(self, payload: PointDonePayload) -> None:
        self.point_done.emit(payload.idx)

    def _on_run_finished_payload(self, _payload: RunFinishedPayload) -> None:
        self.run_finished.emit()

    def _on_run_stopped_payload(self, _payload: RunStoppedPayload) -> None:
        self.run_stopped.emit()

    def _on_run_failed_payload(self, payload: RunFailedPayload) -> None:
        self.run_failed.emit(payload.message)

    def notify(self, name: str, idx: int) -> None:
        """The worker-thread notify callback — re-emits as a queued Qt signal."""
        key = (name, idx)
        with self._row_lock:
            if key in self._pending_rows:
                self._dirty_rows.add(key)
                return
            self._pending_rows.add(key)
        self.row_updated.emit(name, idx, perf_now())

    def row_rendered(self, name: str, idx: int) -> None:
        """Main-thread callback after a queued row update has been consumed."""
        key = (name, idx)
        with self._row_lock:
            if key in self._dirty_rows:
                self._dirty_rows.remove(key)
                emit_again = True
            else:
                self._pending_rows.discard(key)
                emit_again = False

        if emit_again:
            QTimer.singleShot(
                0,
                lambda n=name, i=idx: self.row_updated.emit(n, i, perf_now()),
            )


class MainWindow(QMainWindow):
    """autofluxdep-gui main window: node list + node detail + flux progress."""

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget | None = None,
        *,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._progress_control = ctrl.progress_control
        self._bus_subs = EventSubscriptions()
        self._dialog_refs = DialogRefStore()
        self._dialog_presenter = dialog_presenter or QtDialogPresenter(
            self._dialog_refs
        )
        # per-provider sweep-lived liveplot state: name -> (canvas, plotter)
        self._plots: dict[str, tuple[QWidget, Any]] = {}
        self._row_update_perf = PerfStats("main.row_update", logger, slow_ms=30.0)
        self._progress_perf = PerfStats("main.progress_render", logger, slow_ms=20.0)
        self._progress_collect_perf = PerfStats(
            "main.progress_collect", logger, slow_ms=20.0
        )
        self._progress_flux_perf = PerfStats(
            "main.progress_flux_bar", logger, slow_ms=20.0
        )
        self._progress_stack_perf = PerfStats(
            "main.progress_round_stack", logger, slow_ms=20.0
        )
        self._progress_stack_step_perf: dict[str, PerfStats] = {}
        self._build_plot_perf = PerfStats("main.build_plots", logger, slow_ms=100.0)
        self._run_active = False
        self._run_paused = False
        self._closing = False
        self._close_prompt_open = False
        self._close_after_run_terminal = False
        self._force_close_prompt_open = False
        self._active_run_node_name: str | None = None
        self._live_predictor_flux_idx: int | None = None
        self._auto_follow_navigation = False
        self._flux_progress_snapshot: _ProgressSnapshot | None = None
        # Predictor needs a typed handle for live-mode calls; setup/devices/inspect
        # lifetimes are owned by DialogRefStore.
        self._predictor_dialog: PredictorDialog | None = None
        self.setWindowTitle("autofluxdep-gui")
        self.resize(1100, 800)

        # Hidden park for canvases not currently shown. Every Node's Plotter
        # redraws each run point (even the off-screen ones); a parentless canvas
        # becomes a top-level window the moment it draws, so it would flash as a
        # stray window. Parenting every canvas here (hidden) keeps it off-screen
        # but never a window. Parented to the MainWindow (shares its lifetime),
        # never shown.
        self._canvas_park = QWidget(self)
        self._canvas_park.hide()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- top session row (same global-session placement as measure-gui) ---
        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Context:"))
        self._ctx_label = QLabel("(none)")
        session_row.addWidget(self._ctx_label)
        session_row.addSpacing(24)
        session_row.addWidget(QLabel("Setup:"))
        self._setup_label = QLabel("no SoC")
        session_row.addWidget(self._setup_label)
        session_row.addSpacing(24)
        session_row.addWidget(QLabel("Predictor:"))
        self._predictor_label = QLabel("none")
        session_row.addWidget(self._predictor_label)
        session_row.addStretch()

        self._setup_btn = QPushButton("Setup…")
        self._setup_btn.clicked.connect(self._on_setup_clicked)
        session_row.addWidget(self._setup_btn)

        self._devices_btn = QPushButton("Devices…")
        self._devices_btn.clicked.connect(self._on_devices_clicked)
        session_row.addWidget(self._devices_btn)

        self._predictor_btn = QPushButton("Predictor…")
        self._predictor_btn.clicked.connect(self._on_predictor_clicked)
        session_row.addWidget(self._predictor_btn)

        self._inspect_btn = QPushButton("Inspect…")
        self._inspect_btn.clicked.connect(self._on_inspect)
        session_row.addWidget(self._inspect_btn)

        main_layout.addLayout(session_row)

        split = QSplitter()
        self._list = NodeListPane(ctrl, dialog_presenter=self._dialog_presenter)
        self._detail = NodeDetailPane()
        self._detail.set_canvas_park(self._canvas_park)
        split.addWidget(self._list)
        split.addWidget(self._detail)
        split.setStretchFactor(1, 1)
        main_layout.addWidget(split, stretch=1)

        # global flux progress as a central bottom row, matching window width
        self._round_progress = ProgressStack()
        self._round_progress.set_profile_callback(self._record_progress_stack_step)
        main_layout.addWidget(self._round_progress)
        self._progress_unsub = self._progress_control.attach_progress(
            RUN_PROGRESS_OWNER_ID, self._on_run_progress_changed
        )

        self._progress = LightweightProgressBar()
        self._progress.setFormat("flux %v/%m")
        self._progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        main_layout.addWidget(self._progress)

        # selection → right pane follows
        self._list.selection_changed.connect(self._on_select)
        self._list.run_requested.connect(self._start)
        self._list.pause_requested.connect(self._pause)
        self._list.continue_requested.connect(self._continue)
        self._list.restart_requested.connect(self._restart)
        self._list.abort_requested.connect(self._stop)
        self._list.sample_export_requested.connect(self._export_sample_table)
        self._list.auto_follow_changed.connect(self._on_auto_follow_changed)
        self._detail.user_tab_changed.connect(self._on_user_detail_tab_changed)

        # run bridge (worker thread → main thread)
        self._bridge = _RunBridge(ctrl)
        self._bridge.run_started.connect(self._on_run_started)
        self._bridge.run_pause_requested.connect(self._on_run_pause_requested)
        self._bridge.run_paused.connect(self._on_run_paused)
        self._bridge.run_continued.connect(self._on_run_continued)
        self._bridge.node_entered.connect(self._on_node_entered)
        self._bridge.point_done.connect(self._on_point_done)
        self._bridge.row_updated.connect(self._on_row_updated)
        self._bridge.run_finished.connect(self._on_run_done)
        self._bridge.run_stopped.connect(self._on_run_done)
        self._bridge.run_failed.connect(self._on_run_failed)

        # Shared session changes refresh the top status row, flux source picker, and
        # any materialized cfg editor LiveModels.
        self._bus_subs.subscribe(ctrl.bus, SocChangedPayload, self._on_soc_changed)
        self._bus_subs.subscribe(
            ctrl.bus, DeviceChangedPayload, self._on_device_changed
        )
        self._bus_subs.subscribe(
            ctrl.bus, PredictorChangedPayload, self._on_predictor_changed
        )
        self._bus_subs.subscribe(
            ctrl.bus, ContextSwitchedPayload, self._on_context_switched
        )
        self._bus_subs.subscribe(ctrl.bus, MdChangedPayload, self._on_md_changed)
        self._bus_subs.subscribe(ctrl.bus, MlChangedPayload, self._on_ml_changed)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

        self._refresh_toolbar_buttons()
        self._refresh_session_status()
        self._on_select(self._list.selected_index)

    def restore_workflow_view(self) -> None:
        """Refresh list/detail/flux widgets after controller-level restore."""
        self._list.refresh_from_state()
        self._on_select(self._list.selected_index)

    def _perform_close(self) -> None:
        """Final teardown after the shutdown coordinator has drained operations."""
        self._closing = True
        self._cleanup_bus_subscriptions()
        self._bridge.teardown()
        self._progress_unsub()
        self._list.teardown()
        self._detail.teardown()
        self._capture_predictor_dialog_state()
        self._ctrl.persist_all()
        self._ctrl.quiesce_background()
        self.close()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if self._closing:
            if a0 is not None:
                super().closeEvent(a0)
            return

        if self._ctrl.is_running:
            if a0 is not None:
                a0.ignore()
            self._confirm_running_close()
            return

        if self._ctrl.is_paused:
            if a0 is not None:
                a0.ignore()
            self._confirm_paused_close()
            return

        active = self._ctrl.active_operation_count()
        if active > 0:
            if a0 is not None:
                a0.ignore()
            self._confirm_active_operations_close(active)
            return
        if a0 is not None:
            a0.ignore()
        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))

    def _confirm_active_operations_close(self, active: int) -> None:
        """Ask before cancelling non-run operations during window close."""
        if self._close_prompt_open:
            return
        self._close_prompt_open = True
        self._dialog_presenter.confirm_async(
            self,
            "Operations in progress",
            f"Cancel {active} operation(s) in progress and close once they stop?",
            on_decision=self._on_active_operations_close_decision,
            default=False,
        )

    def _on_active_operations_close_decision(self, confirmed: bool) -> None:
        self._close_prompt_open = False
        if not confirmed or self._closing:
            return
        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))

    def _confirm_running_close(self) -> None:
        """Ask before turning a running close into a terminal stop-and-close."""
        if self._close_after_run_terminal or self._close_prompt_open:
            return
        self._close_prompt_open = True
        self._dialog_presenter.confirm_async(
            self,
            "Run in progress",
            "Stop the running autofluxdep sweep and close once it stops?",
            on_decision=self._on_running_close_decision,
            default=False,
        )

    def _on_running_close_decision(self, confirmed: bool) -> None:
        self._close_prompt_open = False
        if not confirmed or self._closing:
            return
        self._close_after_run_terminal = True
        if not self._ctrl.is_running:
            self._perform_close()
            return
        if not self._ctrl.stop_run(reason="Window close requested terminal stop"):
            self._perform_close()
            return
        QTimer.singleShot(30_000, self._show_force_close_prompt)

    def _confirm_paused_close(self) -> None:
        """Ask before discarding same-process continue state on close."""
        if self._close_prompt_open:
            return
        self._close_prompt_open = True
        self._dialog_presenter.confirm_async(
            self,
            "Paused run",
            "Finalize the paused run as stopped and close? Continue will no longer be available.",
            on_decision=self._on_paused_close_decision,
            default=False,
        )

    def _on_paused_close_decision(self, confirmed: bool) -> None:
        self._close_prompt_open = False
        if not confirmed or self._closing:
            return
        try:
            self._ctrl.finalize_paused_run_as_stopped()
        except Exception:
            return
        self._perform_close()

    def _show_force_close_prompt(self) -> None:
        """Offer an explicit force-close if cooperative terminal stop is too slow."""
        if (
            self._closing
            or not self._close_after_run_terminal
            or not self._ctrl.is_running
            or self._force_close_prompt_open
        ):
            return
        self._force_close_prompt_open = True
        self._dialog_presenter.destructive_confirm(
            self,
            "Run still stopping",
            "The run has not stopped after 30 seconds. Force Close may leave the "
            "artifact at the last flushed state.",
            action_text="Force Close",
            on_decision=self._on_force_close_decision,
            default=False,
        )

    def _on_force_close_decision(self, confirmed: bool) -> None:
        self._force_close_prompt_open = False
        if confirmed and self._close_after_run_terminal and not self._closing:
            self._perform_close()

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        self._bus_subs.unsubscribe_all()

    # --- selection ---

    def _on_select(self, row: int) -> None:
        nodes = self._ctrl.state.nodes
        node = nodes[row] if 0 <= row < len(nodes) else None
        build_edit_form = not (self._run_active and self._auto_follow_navigation)
        self._detail.show_node(
            self._ctrl,
            node,
            row,
            build_edit_form=build_edit_form,
        )
        # show this Node's live canvas (if a run built one)
        canvas = None
        if node is not None and node.name in self._plots:
            canvas = self._plots[node.name][0]
        self._detail.show_run_canvas(canvas)
        if not self._auto_follow_navigation:
            self._disable_auto_follow_from_user_action()

    # --- inspect (non-modal context inspector) ---

    def _on_setup_clicked(self) -> None:
        self.open_setup_dialog(startup_mode=False)

    def open_setup_dialog(self, *, startup_mode: bool = False) -> None:
        from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

        if self._raise_existing_dialog("setup") is not None:
            return

        # Non-blocking open() keeps the Qt event loop (and the control socket)
        # alive while the dialog is visible. WA_DeleteOnClose + instance ref
        # prevent premature GC; finished clears the ref and refreshes state.
        dlg = SetupDialog(self._ctrl.setup_control, self, startup_mode=startup_mode)

        def _on_finished(_status: int) -> None:
            self._refresh_session_dependents()

        self._dialog_refs.open_named("setup", dlg, on_finished=_on_finished)

    def _on_devices_clicked(self) -> None:
        from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

        if self._raise_existing_dialog("devices") is not None:
            return

        # The shared device dialog manages instruments through the narrow device
        # facet; context lookup is injected separately for eval-mode fields.
        dlg = DeviceDialog(
            self._ctrl.device_control,
            md_provider=self._ctrl.context_control.get_current_md,
            parent=self,
        )

        def _on_finished(_status: int) -> None:
            self._refresh_session_dependents()

        self._dialog_refs.open_named("devices", dlg, on_finished=_on_finished)

    def _on_predictor_clicked(self) -> None:
        from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

        if self._raise_existing_dialog("predictor") is not None:
            self._sync_predictor_dialog_live_state()
            return

        # The shared predictor dialog loads a FluxoniumPredictor into the active
        # context; the run reads exp_context.predictor.
        dlg = PredictorDialog(
            self._ctrl.predictor_control,
            self,
            device=self._ctrl.device_control,
            persistent_on_close=True,
        )
        dlg.restore_state(self._shared_predictor_dialog_state())
        dlg.state_changed.connect(self._capture_predictor_dialog_state)

        def _on_finished(_status: int) -> None:
            self._capture_predictor_dialog_state()
            self._predictor_dialog = None
            self._refresh_session_status()

        self._predictor_dialog = dlg
        self._sync_predictor_dialog_live_state()
        self._dialog_refs.open_named("predictor", dlg, on_finished=_on_finished)

    def _shared_predictor_dialog_state(self) -> PredictorDialogState:
        from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialogState

        state = self._ctrl.get_predictor_dialog_state()
        return PredictorDialogState(
            tracked_transitions=state.tracked_transitions,
            tab_index=state.tab_index,
            params_path_text=state.params_path_text,
        )

    def _capture_predictor_dialog_state(self) -> None:
        dlg = self._predictor_dialog
        if dlg is None:
            return
        state = dlg.capture_state()
        self._ctrl.set_predictor_dialog_state(
            PersistedPredictorDialogState(
                tracked_transitions=state.tracked_transitions,
                tab_index=state.tab_index,
                params_path_text=state.params_path_text,
            )
        )

    def _on_inspect(self) -> None:
        """Open the shared context inspector, or raise it if already open.

        autofluxdep reuses ``InspectDialogBase`` as-is (the measure-only ml
        create/modify path drags the CfgEditor and is deliberately excluded —
        Phase 160c). The dialog subscribes to the shared session event bus, so it
        auto-refreshes on every md/ml edit without this window pushing to it.
        A single instance is kept alive (non-modal, WA_DeleteOnClose); a second
        request just raises the existing one.
        """
        if self._raise_existing_dialog("inspect") is not None:
            self._sync_inspect_dialog_read_only()
            return

        from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase

        class _ReadOnlyAwareInspectDialog(InspectDialogBase):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._autofluxdep_read_only = False
                super().__init__(*args, **kwargs)

            def set_read_only(self, read_only: bool) -> None:
                self._autofluxdep_read_only = bool(read_only)
                self._apply_read_only_state()

            def _on_edit_key_changed(self, text: str) -> None:
                super()._on_edit_key_changed(text)
                self._apply_read_only_state()

            def _on_ml_item_changed(self, current: Any, previous: Any) -> None:
                super()._on_ml_item_changed(current, previous)
                self._apply_read_only_state()

            def _apply_read_only_state(self) -> None:
                if not hasattr(self, "_set_btn"):
                    return
                read_only = self._autofluxdep_read_only
                self._edit_key.setEnabled(not read_only)
                self._edit_value.setEnabled(not read_only)
                if read_only:
                    self._set_btn.setEnabled(False)
                    self._delete_btn.setEnabled(False)
                    self._rename_ml_btn.setEnabled(False)
                    self._del_ml_btn.setEnabled(False)
                    return
                has_key = bool(self._edit_key.text().strip())
                self._set_btn.setEnabled(has_key)
                self._delete_btn.setEnabled(has_key)
                has_ml_selection = self._current_ml_item_data() is not None
                self._rename_ml_btn.setEnabled(has_ml_selection)
                self._del_ml_btn.setEnabled(has_ml_selection)

        dlg = _ReadOnlyAwareInspectDialog(
            self._ctrl.context_control, self._ctrl.get_bus(), parent=self
        )

        self._sync_inspect_dialog_read_only()
        self._dialog_refs.open_named("inspect", dlg)

    def _raise_existing_dialog(self, key: str) -> QDialog | None:
        dialog = self._dialog_refs.get(key)
        if dialog is None:
            return None
        if not dialog.isVisible():
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    # --- run lifecycle ---

    def _start(self) -> None:
        try:
            self._build_plots()
            self._apply_flux_progress_snapshot(
                (
                    max(1, len(self._ctrl.state.flux_values)),
                    "flux %v/%m",
                    0,
                )
            )
            self._ctrl.start_run(notify=self._bridge.notify)
        except Exception as exc:
            logger.exception("autofluxdep run failed to start")
            self._clear_plots()
            self._ctrl.clear_run_products()
            self._reset_run_ui()
            self._dialog_presenter.warning(self, "Run failed to start", str(exc))

    def _clear_plots(self) -> None:
        self._detail.show_run_canvas(None)
        for canvas, _ in self._plots.values():
            canvas.setParent(None)
            canvas.deleteLater()
        self._plots = {}

    def _build_plots(self) -> None:
        """Allocate Results + build each provider's Figure / Plotter / canvas.

        Main-thread, Run start. Mirrors CONTEXT.md's Ownership: the main thread
        builds the empty Result containers (via the controller) and the
        UI-owned Plotters/canvases bound to them; the worker then fills rows.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        profile_start = perf_now()
        # tear down any previous run's canvases (detach the shown one first, then
        # destroy every canvas — they are discarded, never drawn again)
        self._clear_plots()

        results = self._ctrl.prepare_run_results()
        for node in self._ctrl.state.nodes:
            if not node.enabled:
                continue
            result = results.get(node.name)
            if result is None:
                continue  # a provider without a Result (none in the prototype)
            figure = Figure(figsize=(5, 4), tight_layout=True)
            # parent the canvas to the hidden park so it is never a top-level
            # window — only the selected one is re-parented into the run tab.
            canvas = FigureCanvasQTAgg(figure)
            canvas.setParent(self._canvas_park)
            plotter = node.builder.make_plotter(figure)
            self._plots[node.name] = (canvas, plotter)
        # show the currently selected Node's fresh canvas
        self._on_select(self._list.selected_index)
        self._build_plot_perf.record(
            elapsed_ms(profile_start),
            detail=f"nodes={len(self._plots)}",
        )

    def _stop(self) -> None:
        self._ctrl.stop_run()

    def _pause(self) -> None:
        self._ctrl.request_pause()

    def _continue(self) -> None:
        self._ctrl.continue_run()

    def _restart(self) -> None:
        try:
            self._ctrl.finalize_paused_run_as_stopped()
            self._start()
        except Exception as exc:
            logger.exception("autofluxdep run failed to restart")
            self._dialog_presenter.warning(self, "Restart failed", str(exc))

    def _export_sample_table(self) -> None:
        from qtpy.QtWidgets import QFileDialog  # type: ignore[attr-defined]

        default_path = self._ctrl.default_sample_table_path()
        if default_path is None:
            self._list.refresh_run_availability()
            self._dialog_presenter.warning(
                self, "Export sample", "No finished run is exportable."
            )
            return

        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export sample table",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        if not filepath:
            return

        try:
            result = self._ctrl.export_sample_table(filepath)
        except Exception as exc:
            logger.exception("autofluxdep sample table export failed")
            self._list.refresh_run_availability()
            self._dialog_presenter.warning(self, "Export sample failed", str(exc))
            return

        self._list.refresh_run_availability()
        self._dialog_presenter.information(
            self,
            "Sample exported",
            f"Exported {result.row_count} sample row(s) to:\n{result.path}",
        )

    def _on_run_started(self) -> None:
        self._enter_active_run_ui(start_idx=0)

    def _on_run_continued(self, next_flux_idx: int) -> None:
        self._enter_active_run_ui(start_idx=next_flux_idx)

    def _enter_active_run_ui(self, *, start_idx: int) -> None:
        self._close_mutation_dialogs()
        self._run_active = True
        self._run_paused = False
        self._active_run_node_name = None
        self._live_predictor_flux_idx = (
            start_idx if self._ctrl.state.flux_values else None
        )
        self._list.set_run_state("running")
        auto_follow = self._ctrl.get_auto_follow_tabs()
        self._detail.set_running(True, switch_tab=auto_follow)
        self._refresh_toolbar_buttons()
        self._sync_predictor_dialog_live_state()
        self._sync_inspect_dialog_read_only()

    def _on_run_pause_requested(self) -> None:
        self._close_mutation_dialogs()
        self._list.set_run_state("pausing")
        self._sync_inspect_dialog_read_only()

    def _on_run_paused(self, next_flux_idx: int) -> None:
        self._close_mutation_dialogs()
        self._run_active = True
        self._run_paused = True
        self._active_run_node_name = None
        self._live_predictor_flux_idx = next_flux_idx
        self._list.set_run_state("paused")
        self._detail.set_running(True, switch_tab=False)
        self._refresh_toolbar_buttons()
        self._sync_predictor_dialog_live_state()
        self._sync_inspect_dialog_read_only()

    def _close_mutation_dialogs(self) -> None:
        for key in ("setup", "devices"):
            dialog = self._dialog_refs.get(key)
            if dialog is not None:
                dialog.close()

    def _on_node_entered(self, name: str, idx: int) -> None:
        """Auto-follow: a provider started → select it + show its run tab/plot.

        The predictor Service (not in the user's list) and any name absent from
        the list are skipped — there is nothing to navigate to. If the running
        provider is already selected, keep the user's current edit/run sub-tab.
        """
        self._live_predictor_flux_idx = idx
        self._sync_predictor_dialog_live_value()
        self._active_run_node_name = name
        if not self._ctrl.get_auto_follow_tabs():
            return
        self._follow_run_node(name, force_focus=False)

    def _on_auto_follow_changed(self, enabled: bool) -> None:
        if not enabled or not self._run_active:
            return
        if self._active_run_node_name is None:
            self._auto_follow_navigation = True
            try:
                self._detail.focus_run()
            finally:
                self._auto_follow_navigation = False
            return
        followed = self._follow_run_node(self._active_run_node_name, force_focus=True)
        if not followed:
            self._auto_follow_navigation = True
            try:
                self._detail.focus_run()
            finally:
                self._auto_follow_navigation = False

    def _on_user_detail_tab_changed(self, _index: int) -> None:
        if not self._auto_follow_navigation:
            self._disable_auto_follow_from_user_action()

    def _disable_auto_follow_from_user_action(self) -> None:
        if not self._run_active or not self._ctrl.get_auto_follow_tabs():
            return
        self._ctrl.set_auto_follow_tabs(False)
        self._list.refresh_preferences()

    def _follow_run_node(self, name: str, *, force_focus: bool) -> bool:
        names = self._ctrl.state.node_names()
        if name not in names:
            return False  # a Service (predictor) or unknown name → no navigation
        target = names.index(name)
        already_selected = self._list.selected_index == target
        self._auto_follow_navigation = True
        try:
            if force_focus or not already_selected:
                self._detail.focus_run()
            self._list.select_index(target)  # → _on_select shows its canvas
        finally:
            self._auto_follow_navigation = False
        return True

    def _on_row_updated(self, name: str, idx: int, emitted_at: float) -> None:
        """Main-thread: a Result row was filled → redraw that provider's Plotter."""
        try:
            queue_ms = elapsed_ms(emitted_at)
            entry = self._plots.get(name)
            if entry is None:
                return
            plotter = entry[1]
            result = self._ctrl.state.run_results.get(name)
            if result is not None and plotter is not None:
                profile_start = perf_now()
                try:
                    plotter.update(result, idx)
                except Exception:
                    logger.exception(
                        "autofluxdep plot update failed: node=%s idx=%d", name, idx
                    )
                    canvas, _plotter = entry
                    self._plots[name] = (canvas, None)
                    return
                self._row_update_perf.record(
                    elapsed_ms(profile_start),
                    queue_ms=queue_ms,
                    detail=f"node={name} idx={idx}",
                )
        finally:
            self._bridge.row_rendered(name, idx)

    def _on_point_done(self, idx: int) -> None:
        self._live_predictor_flux_idx = idx
        self._sync_predictor_dialog_live_value()

    def _on_run_progress_changed(self) -> None:
        profile_start = perf_now()
        collect_start = perf_now()
        models = tuple(
            model
            for _handle, model in self._progress_control.progress_bars(
                RUN_PROGRESS_OWNER_ID
            )
        )
        self._progress_collect_perf.record(
            elapsed_ms(collect_start),
            detail=f"bars={len(models)}",
        )
        flux_model = next(
            (model for model in models if model.label == FLUX_PROGRESS_LABEL), None
        )
        if flux_model is not None:
            flux_start = perf_now()
            maximum = flux_model.qt_maximum()
            fmt = flux_model.format()
            value = flux_model.qt_value()
            self._apply_flux_progress_snapshot((maximum, fmt, value))
            self._progress_flux_perf.record(
                elapsed_ms(flux_start),
                detail=f"value={value} maximum={maximum}",
            )

        round_models = tuple(
            model for model in models if model.label != FLUX_PROGRESS_LABEL
        )
        stack_start = perf_now()
        self._round_progress.render_models(round_models)
        self._progress_stack_perf.record(
            elapsed_ms(stack_start),
            detail=f"bars={len(round_models)}",
        )
        self._progress_perf.record(
            elapsed_ms(profile_start),
            detail=f"bars={len(models)}",
        )

    def _record_progress_stack_step(
        self, label: str, duration_ms: float, detail: str
    ) -> None:
        perf = self._progress_stack_step_perf.get(label)
        if perf is None:
            perf = PerfStats(f"main.progress_stack.{label}", logger, slow_ms=20.0)
            self._progress_stack_step_perf[label] = perf
        perf.record(duration_ms, detail=detail)

    def _apply_flux_progress_snapshot(self, snapshot: _ProgressSnapshot) -> None:
        old = self._flux_progress_snapshot
        if old == snapshot:
            return
        maximum, fmt, value = snapshot
        if old is None or old[0] != maximum:
            self._progress.setMaximum(maximum)
        if old is None or old[1] != fmt:
            self._progress.setFormat(fmt)
        if old is None or old[2] != value:
            self._progress.setValue(value)
        self._flux_progress_snapshot = snapshot

    def _on_run_done(self) -> None:
        self._reset_run_ui()
        self._finish_close_after_run_terminal()

    def _reset_run_ui(self, *, switch_tab: bool | None = None) -> None:
        self._run_active = False
        self._run_paused = False
        self._active_run_node_name = None
        self._live_predictor_flux_idx = None
        self._list.set_run_state("idle")
        if switch_tab is None:
            switch_tab = self._ctrl.get_auto_follow_tabs()
        self._detail.set_running(False, switch_tab=switch_tab)
        self._refresh_toolbar_buttons()
        self._sync_predictor_dialog_live_state()
        self._sync_inspect_dialog_read_only()

    def _on_run_failed(self, message: str) -> None:
        """A Node's produce raised mid-sweep → unlock the UI + surface the error.

        Same unlock path as a stop/finish (the run is over), plus a message box so
        the user sees why the sweep aborted (e.g. an unconfigured Node Fast-Failed)
        rather than the run silently ending."""
        closing_after_terminal = self._close_after_run_terminal
        self._on_run_done()
        if closing_after_terminal:
            return
        self._dialog_presenter.warning(self, "Run failed", message)

    def _finish_close_after_run_terminal(self) -> None:
        if not self._close_after_run_terminal or self._closing:
            return
        self._close_after_run_terminal = False
        QTimer.singleShot(0, self._perform_close)

    def take_window_screenshot(self) -> bytes:
        """Grab the main window as PNG bytes for the read-only remote view."""
        from qtpy.QtCore import QBuffer, QIODevice  # type: ignore[attr-defined]

        pixmap = self.grab()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buf, "PNG"):
            raise RuntimeError("Qt failed to encode the autofluxdep window as PNG")
        return bytes(buf.data().data())  # type: ignore[arg-type]

    # --- session status / toolbar state ---

    def _on_soc_changed(self, _payload: SocChangedPayload) -> None:
        self._refresh_session_dependents()

    def _on_device_changed(self, _payload: DeviceChangedPayload) -> None:
        self._refresh_session_dependents()
        self._refresh_cfg_editor_from_session(_payload)

    def _on_predictor_changed(self, _payload: PredictorChangedPayload) -> None:
        self._refresh_session_status()
        self._sync_predictor_dialog_live_value()

    def _current_live_predictor_value(self) -> float | None:
        idx = self._live_predictor_flux_idx
        values = self._ctrl.state.flux_values
        if idx is None or idx < 0 or idx >= len(values):
            return None
        return float(values[idx])

    def _sync_predictor_dialog_live_state(self) -> None:
        dlg = self._predictor_dialog
        if dlg is None:
            return
        if self._run_active:
            dlg.set_live_mode(True)
            self._sync_predictor_dialog_live_value()
        else:
            dlg.set_live_mode(False)

    def _sync_predictor_dialog_live_value(self) -> None:
        dlg = self._predictor_dialog
        if dlg is None or not self._run_active:
            return
        value = self._current_live_predictor_value()
        if value is None:
            return
        dlg.set_live_device_value(value)

    def _sync_inspect_dialog_read_only(self) -> None:
        dlg = self._dialog_refs.get("inspect")
        if dlg is None:
            return
        set_read_only = getattr(dlg, "set_read_only", None)
        if not callable(set_read_only):
            return
        set_read_only(self._run_active or self._ctrl.is_running or self._ctrl.is_paused)

    def _on_context_switched(self, _payload: ContextSwitchedPayload) -> None:
        self._refresh_session_dependents()
        self._refresh_cfg_editor_from_session(_payload)

    def _on_md_changed(self, _payload: MdChangedPayload) -> None:
        self._refresh_cfg_editor_from_session(_payload)

    def _on_ml_changed(self, _payload: MlChangedPayload) -> None:
        self._refresh_cfg_editor_from_session(_payload)

    def _refresh_cfg_editor_from_session(self, payload: SessionPayload) -> None:
        self._detail.refresh_cfg_editor(payload.EVENT)

    def _refresh_toolbar_buttons(self) -> None:
        editing = not (
            self._ctrl.is_running or self._ctrl.is_paused or self._run_paused
        )
        self._setup_btn.setEnabled(editing)
        self._devices_btn.setEnabled(editing)
        self._predictor_btn.setEnabled(editing or self._ctrl.is_running)
        # Inspect stays enabled during a run: the inspector reflects the live
        # context and is non-modal, so it never blocks the event loop.
        self._inspect_btn.setEnabled(True)

    def _refresh_session_status(self) -> None:
        ctx = self._ctrl.state.exp_context
        if ctx.is_active() and ctx.active_label:
            ctx_text = ctx.active_label
        elif ctx.has_context():
            ctx_text = f"{ctx.chip_name}/{ctx.qub_name} (draft)"
        else:
            ctx_text = "(none)"
        self._ctx_label.setText(ctx_text)
        self._setup_label.setText("connected" if ctx.has_soc() else "no SoC")
        self._predictor_label.setText("active" if ctx.predictor is not None else "none")

    def _refresh_session_dependents(self) -> None:
        self._refresh_session_status()
        self._list.refresh_run_availability()
        self._list.refresh_flux_sources()
