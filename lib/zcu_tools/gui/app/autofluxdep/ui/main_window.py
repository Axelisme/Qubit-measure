"""MainWindow — the autofluxdep-gui shell.

A QMainWindow holding a left/right split (``NodeListPane`` + ``NodeDetailPane``)
with a global flux progress bar in the status bar. It wires the edit↔run state
switch: pressing Run starts a worker thread that calls ``Controller.start_run``;
run-lifecycle EventBus events (emitted on the worker thread) are marshalled to
the Qt main thread by a ``_RunBridge`` QObject, which drives auto-follow (select
the running Node + show its run tab), the global progress bar, and the
edit↔run lock.

Prototype: the run is fake (dry data, no hardware); Setup builds fake resources.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QMainWindow,
    QProgressBar,
    QSplitter,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import Event, EventType

from .node_detail import NodeDetailPane
from .node_list import NodeListPane


class _RunWorker(QThread):
    """Runs the (fake) sweep off the main thread so Stop stays responsive."""

    def __init__(self, ctrl: Controller) -> None:
        super().__init__()
        self._ctrl = ctrl

    def run(self) -> None:  # noqa: D401 - QThread entry point
        self._ctrl.start_run()


class _RunBridge(QObject):
    """Marshals worker-thread EventBus run events onto the Qt main thread.

    The controller emits events on the worker thread; this bridge subscribes and
    re-emits as Qt signals, which (being connected on the main thread) are
    delivered there. The MainWindow connects its UI slots to these signals.
    """

    run_started = Signal()
    node_started = Signal(str, int)
    point_done = Signal(int)
    run_finished = Signal()
    run_stopped = Signal()

    def __init__(self, ctrl: Controller) -> None:
        super().__init__()
        bus = ctrl.bus
        bus.subscribe(EventType.RUN_STARTED, lambda e: self.run_started.emit())
        bus.subscribe(EventType.NODE_STARTED, self._on_node_started)
        bus.subscribe(
            EventType.POINT_DONE, lambda e: self.point_done.emit(int(e.payload))
        )
        bus.subscribe(EventType.RUN_FINISHED, lambda e: self.run_finished.emit())
        bus.subscribe(EventType.RUN_STOPPED, lambda e: self.run_stopped.emit())

    def _on_node_started(self, e: Event) -> None:
        name, idx = e.payload
        self.node_started.emit(str(name), int(idx))


class MainWindow(QMainWindow):
    """autofluxdep-gui main window: node list + node detail + flux progress."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._worker: Optional[_RunWorker] = None
        self.setWindowTitle("autofluxdep-gui")
        self.resize(1100, 800)

        split = QSplitter()
        self._list = NodeListPane(ctrl)
        self._detail = NodeDetailPane()
        split.addWidget(self._list)
        split.addWidget(self._detail)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

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
        self._bridge.node_started.connect(self._on_node_started)
        self._bridge.point_done.connect(self._on_point_done)
        self._bridge.run_finished.connect(self._on_run_done)
        self._bridge.run_stopped.connect(self._on_run_done)

        # also reflect workflow edits in the setup light / run-enabled
        ctrl.bus.subscribe(
            EventType.SETUP_DONE, lambda e: self._list._refresh_buttons()
        )

        self._on_select(self._list.selected_index)

    # --- selection ---

    def _on_select(self, row: int) -> None:
        nodes = self._ctrl.state.nodes
        node = nodes[row] if 0 <= row < len(nodes) else None
        self._detail.show_node(node)

    # --- run lifecycle ---

    def _start(self) -> None:
        self._progress.setMaximum(max(1, len(self._ctrl.state.flux_values)))
        self._progress.setValue(0)
        self._worker = _RunWorker(self._ctrl)
        self._worker.start()

    def _stop(self) -> None:
        self._ctrl.stop_run()

    def _on_run_started(self) -> None:
        self._list.set_running(True)
        self._detail.set_running(True)

    def _on_node_started(self, name: str, idx: int) -> None:
        # auto-follow: select the running Node + show its run tab
        names = self._ctrl.state.node_names()
        if name in names:
            self._list.select_index(names.index(name))
        self._detail.focus_run(name, idx)

    def _on_point_done(self, idx: int) -> None:
        self._progress.setValue(idx + 1)

    def _on_run_done(self) -> None:
        self._list.set_running(False)
        self._detail.set_running(False)
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
