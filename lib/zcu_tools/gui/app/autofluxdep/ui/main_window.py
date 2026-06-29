"""MainWindow — the autofluxdep-gui shell.

A QMainWindow holding a measure-gui-style top session row, a left/right split
(``NodeListPane`` + ``NodeDetailPane``), and a global flux progress bar in the
status bar. It wires the edit↔run state switch and owns the liveplot integration:

- At **Run start** it allocates each provider's sweep Result (``Controller
  .prepare_run_results``) and, for every Result, builds a bare matplotlib
  ``Figure`` + the provider's ``Plotter`` + a ``FigureCanvasQTAgg`` — all
  sweep-lived, so auto-follow can show any provider's plot at any time.
- It calls ``Controller.start_run`` as an OperationRunner client, passing a
  ``notify(name, idx)`` callback. The shared BackgroundRunner fills the Result
  rows in place and fires ``notify``; ``notify`` and the EventBus run events are
  marshalled to the Qt main thread by ``_RunBridge``. A main-thread slot then calls
  ``plotter.update(result, idx)`` — all drawing stays on the main thread
  (ADR-0017: the worker never touches matplotlib).

The run acquires against a flux-aware MockSoc (offline) or real hardware; Setup
builds a MockSoc + FakeDevice + a SimplePredictor.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
)

from .node_detail import NodeDetailPane
from .node_list import NodeListPane


class _RunBridge(QObject):
    """Marshals worker-thread run events + row notifications onto the main thread.

    The controller emits EventBus events on the worker thread; this bridge
    subscribes and re-emits as Qt signals (delivered on the main thread because
    they are connected there). ``row_updated`` is the row-updated notification
    the worker's round_hook fires (a provider name + flux index — no figure);
    ``node_entered`` is the auto-follow notification (a provider started running).
    """

    run_started = Signal()
    node_entered = Signal(str, int)
    point_done = Signal(int)
    run_finished = Signal()
    run_stopped = Signal()
    run_failed = Signal(str)
    row_updated = Signal(str, int)

    def __init__(self, ctrl: Controller) -> None:
        super().__init__()
        bus = ctrl.bus
        bus.subscribe(RunStartedPayload, lambda p: self.run_started.emit())
        bus.subscribe(NodeEnteredPayload, self._on_node_entered)
        bus.subscribe(PointDonePayload, lambda p: self.point_done.emit(p.idx))
        bus.subscribe(RunFinishedPayload, lambda p: self.run_finished.emit())
        bus.subscribe(RunStoppedPayload, lambda p: self.run_stopped.emit())
        bus.subscribe(RunFailedPayload, lambda p: self.run_failed.emit(p.message))

    def _on_node_entered(self, p: NodeEnteredPayload) -> None:
        self.node_entered.emit(p.name, p.idx)

    def notify(self, name: str, idx: int) -> None:
        """The worker-thread notify callback — re-emits as a queued Qt signal."""
        self.row_updated.emit(name, idx)


class MainWindow(QMainWindow):
    """autofluxdep-gui main window: node list + node detail + flux progress."""

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        # per-provider sweep-lived liveplot state: name -> (canvas, plotter)
        self._plots: dict[str, tuple[QWidget, Any]] = {}
        # The single live (non-modal) context inspector, or None when closed. The
        # base auto-refreshes off the shared session event bus, so the open dialog
        # tracks md/ml edits live without this window pushing to it.
        self._inspect_dialog: QDialog | None = None
        self._setup_dialog: QDialog | None = None
        self._devices_dialog: QDialog | None = None
        self._predictor_dialog: QDialog | None = None
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
        self._list = NodeListPane(ctrl)
        self._detail = NodeDetailPane()
        self._detail.set_canvas_park(self._canvas_park)
        split.addWidget(self._list)
        split.addWidget(self._detail)
        split.setStretchFactor(1, 1)
        main_layout.addWidget(split, stretch=1)

        # global flux progress in the status bar
        self._progress = QProgressBar()
        self._progress.setFormat("flux %v/%m")
        self.statusBar().addPermanentWidget(self._progress)  # type: ignore[union-attr]

        # selection → right pane follows
        self._list.selection_changed.connect(self._on_select)
        self._list.run_requested.connect(self._start)
        self._list.stop_requested.connect(self._stop)

        # run bridge (worker thread → main thread)
        self._bridge = _RunBridge(ctrl)
        self._bridge.run_started.connect(self._on_run_started)
        self._bridge.node_entered.connect(self._on_node_entered)
        self._bridge.point_done.connect(self._on_point_done)
        self._bridge.row_updated.connect(self._on_row_updated)
        self._bridge.run_finished.connect(self._on_run_done)
        self._bridge.run_stopped.connect(self._on_run_done)
        self._bridge.run_failed.connect(self._on_run_failed)

        # Shared session changes refresh the top status row and the flux source
        # picker (mock setup can auto-provision fake_flux).
        ctrl.bus.subscribe(SocChangedPayload, self._on_soc_changed)
        ctrl.bus.subscribe(DeviceChangedPayload, self._on_device_changed)
        ctrl.bus.subscribe(PredictorChangedPayload, self._on_predictor_changed)
        ctrl.bus.subscribe(ContextSwitchedPayload, self._on_context_switched)

        self._refresh_toolbar_buttons()
        self._refresh_session_status()
        self._on_select(self._list.selected_index)

    def restore_workflow_view(self) -> None:
        """Refresh list/detail/flux widgets after controller-level restore."""
        self._list.refresh_from_state()
        self._on_select(self._list.selected_index)

    def closeEvent(self, event: Any) -> None:
        self._ctrl.persist_all()
        super().closeEvent(event)

    # --- selection ---

    def _on_select(self, row: int) -> None:
        nodes = self._ctrl.state.nodes
        node = nodes[row] if 0 <= row < len(nodes) else None
        self._detail.show_node(self._ctrl, node, row)
        # show this Node's live canvas (if a run built one)
        canvas = None
        if node is not None and node.name in self._plots:
            canvas = self._plots[node.name][0]
        self._detail.show_run_canvas(canvas)

    # --- inspect (non-modal context inspector) ---

    def _on_setup_clicked(self) -> None:
        from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

        existing = self._setup_dialog
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return

        # Non-blocking open() keeps the Qt event loop (and the control socket)
        # alive while the dialog is visible. WA_DeleteOnClose + instance ref
        # prevent premature GC; finished clears the ref and refreshes state.
        dlg = SetupDialog(self._ctrl, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(
            lambda _r: (
                setattr(self, "_setup_dialog", None),
                self._refresh_session_status(),
                self._list._refresh_buttons(),
                self._list.refresh_flux_sources(),
            )
        )
        self._setup_dialog = dlg
        dlg.open()

    def _on_devices_clicked(self) -> None:
        from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

        existing = self._devices_dialog
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return

        # The shared device dialog manages all instruments (a flux source among
        # them) through the same SessionControllerPort the setup dialog uses.
        dlg = DeviceDialog(self._ctrl, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(
            lambda _r: (
                setattr(self, "_devices_dialog", None),
                self._refresh_session_status(),
                self._list._refresh_buttons(),
                self._list.refresh_flux_sources(),
            )
        )
        self._devices_dialog = dlg
        dlg.open()

    def _on_predictor_clicked(self) -> None:
        from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

        existing = self._predictor_dialog
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return

        # The shared predictor dialog loads a FluxoniumPredictor into the active
        # context; the run reads exp_context.predictor.
        dlg = PredictorDialog(self._ctrl, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(
            lambda _r: (
                setattr(self, "_predictor_dialog", None),
                self._refresh_session_status(),
            )
        )
        self._predictor_dialog = dlg
        dlg.open()

    def _on_inspect(self) -> None:
        """Open the shared context inspector, or raise it if already open.

        autofluxdep reuses ``InspectDialogBase`` as-is (the measure-only ml
        create/modify path drags the CfgEditor and is deliberately excluded —
        Phase 160c). The dialog subscribes to the shared session event bus, so it
        auto-refreshes on every md/ml edit without this window pushing to it.
        A single instance is kept alive (non-modal, WA_DeleteOnClose); a second
        request just raises the existing one.
        """
        existing = self._inspect_dialog
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            if not existing.isVisible():
                existing.show()
            return

        from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase

        dlg = InspectDialogBase(self._ctrl, self._ctrl.get_bus(), parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # ``finished`` fires for both accept and reject; drop the reference so the
        # next request rebuilds a fresh dialog (the C++ object is deleted on close).
        dlg.finished.connect(lambda _status: setattr(self, "_inspect_dialog", None))
        self._inspect_dialog = dlg
        dlg.open()

    # --- run lifecycle ---

    def _start(self) -> None:
        self._build_plots()
        self._progress.setMaximum(max(1, len(self._ctrl.state.flux_values)))
        self._progress.setValue(0)
        self._ctrl.start_run(notify=self._bridge.notify)

    def _build_plots(self) -> None:
        """Allocate Results + build each provider's Figure / Plotter / canvas.

        Main-thread, Run start. Mirrors CONTEXT.md's Ownership: the main thread
        builds the empty Result containers (via the controller) and the
        UI-owned Plotters/canvases bound to them; the worker then fills rows.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # tear down any previous run's canvases (detach the shown one first, then
        # destroy every canvas — they are discarded, never drawn again)
        self._detail.show_run_canvas(None)
        for canvas, _ in self._plots.values():
            canvas.setParent(None)
            canvas.deleteLater()
        self._plots = {}

        results = self._ctrl.prepare_run_results()
        for node in self._ctrl.state.nodes:
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

    def _stop(self) -> None:
        self._ctrl.stop_run()

    def _on_run_started(self) -> None:
        self._list.set_running(True)
        self._detail.set_running(True)
        self._refresh_toolbar_buttons()
        self._detail.focus_run()

    def _on_node_entered(self, name: str, idx: int) -> None:
        """Auto-follow: a provider started → select it + show its run tab/plot.

        The predictor Service (not in the user's list) and any name absent from
        the list are skipped — there is nothing to navigate to. If the running
        provider is already selected, keep the user's current edit/run sub-tab.
        """
        del idx
        names = self._ctrl.state.node_names()
        if name not in names:
            return  # a Service (predictor) or unknown name → no navigation
        target = names.index(name)
        already_selected = self._list.selected_index == target
        self._list.select_index(target)  # → _on_select shows its canvas
        if not already_selected:
            self._detail.focus_run()

    def _on_row_updated(self, name: str, idx: int) -> None:
        """Main-thread: a Result row was filled → redraw that provider's Plotter."""
        entry = self._plots.get(name)
        if entry is None:
            return
        plotter = entry[1]
        result = self._ctrl.state.run_results.get(name)
        if result is not None and plotter is not None:
            plotter.update(result, idx)

    def _on_point_done(self, idx: int) -> None:
        self._progress.setValue(idx + 1)

    def _on_run_done(self) -> None:
        self._list.set_running(False)
        self._detail.set_running(False)
        self._refresh_toolbar_buttons()

    def _on_run_failed(self, message: str) -> None:
        """A Node's produce raised mid-sweep → unlock the UI + surface the error.

        Same unlock path as a stop/finish (the run is over), plus a message box so
        the user sees why the sweep aborted (e.g. an unconfigured Node Fast-Failed)
        rather than the run silently ending."""
        from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

        self._on_run_done()
        QMessageBox.warning(self, "Run failed", message)

    # --- session status / toolbar state ---

    def _on_soc_changed(self, _payload: SocChangedPayload) -> None:
        self._refresh_session_status()
        self._list._refresh_buttons()
        self._list.refresh_flux_sources()

    def _on_device_changed(self, _payload: DeviceChangedPayload) -> None:
        self._list.refresh_flux_sources()

    def _on_predictor_changed(self, _payload: PredictorChangedPayload) -> None:
        self._refresh_session_status()

    def _on_context_switched(self, _payload: ContextSwitchedPayload) -> None:
        self._refresh_session_status()

    def _refresh_toolbar_buttons(self) -> None:
        editing = not self._ctrl.is_running
        self._setup_btn.setEnabled(editing)
        self._devices_btn.setEnabled(editing)
        self._predictor_btn.setEnabled(editing)
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
