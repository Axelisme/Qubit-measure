"""NodeListPane — the left side: node list + flux + Run/Stop.

Owns workflow editing (select / add / remove / reorder Nodes), the flux sweep
fields, the flux-source picker, and the single Run/Stop toggle button (▶ Run in
edit state, ■ Stop while running). Global session actions (Setup / Devices /
Predictor / Inspect) live in MainWindow's top toolbar, matching measure-gui's
layout. Drives the Controller; emits a Qt signal when the selection changes so
the right pane follows.
"""

from __future__ import annotations

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.registry import available_node_types


class NodeListPane(QWidget):
    """Left pane: node management + flux + setup + run/stop."""

    selection_changed = Signal(int)  # selected node index, or -1
    run_requested = Signal()
    stop_requested = Signal()

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._running = False

        root = QVBoxLayout(self)

        root.addWidget(QLabel("Nodes"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(lambda _it: self._on_rename())
        root.addWidget(self._list, 1)

        # add / remove / reorder / rename
        btns = QHBoxLayout()
        self._add_btn = _btn("+", self._on_add)
        self._del_btn = _btn("−", self._on_remove)
        self._up_btn = _btn("↑", lambda: self._on_move(-1))
        self._down_btn = _btn("↓", lambda: self._on_move(+1))
        self._rename_btn = _btn("✎", self._on_rename)
        for b in (
            self._add_btn,
            self._del_btn,
            self._up_btn,
            self._down_btn,
            self._rename_btn,
        ):
            btns.addWidget(b)
        root.addLayout(btns)

        # flux sweep
        root.addWidget(QLabel("flux sweep"))
        flux_row = QHBoxLayout()
        self._flux_start = QLineEdit("2e-3")
        self._flux_stop = QLineEdit("-0.2e-3")
        self._flux_npts = QLineEdit("101")
        for w in (self._flux_start, self._flux_stop, self._flux_npts):
            flux_row.addWidget(w)
        root.addLayout(flux_row)

        # flux source: which connected device the sweep is applied through (its
        # unit labels the flux axis). Populated from the connected device list.
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("flux source"))
        self._flux_source = QComboBox()
        self._flux_source.currentIndexChanged.connect(self._on_flux_source_changed)
        src_row.addWidget(self._flux_source, 1)
        self._flux_unit = QLabel("")
        src_row.addWidget(self._flux_unit)
        root.addLayout(src_row)

        # run/stop toggle (single button)
        self._run_btn = _btn("▶ Run", self._on_run_stop)
        root.addWidget(self._run_btn)

        self.refresh_list()
        self._refresh_buttons()
        self.refresh_flux_sources()

    # --- list / selection ---

    def refresh_list(self) -> None:
        prev = self._list.currentRow()
        self._list.blockSignals(True)
        self._list.clear()
        for n in self._ctrl.state.nodes:
            self._list.addItem(n.name)
        if 0 <= prev < self._list.count():
            self._list.setCurrentRow(prev)
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._on_row_changed(self._list.currentRow())

    def select_index(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    @property
    def selected_index(self) -> int:
        return self._list.currentRow()

    def _on_row_changed(self, row: int) -> None:
        self.selection_changed.emit(row)

    # --- workflow editing ---

    def _on_add(self) -> None:
        types = available_node_types()
        name, ok = QInputDialog.getItem(self, "Add Node", "Node type:", types, 0, False)
        if ok and name:
            self._ctrl.add_node_by_type(name)
            self.refresh_list()
            self._list.setCurrentRow(self._list.count() - 1)

    def _on_remove(self) -> None:
        idx = self._list.currentRow()
        if 0 <= idx < len(self._ctrl.state.nodes):
            self._ctrl.remove_node(self._ctrl.state.nodes[idx].name)
            self.refresh_list()

    def _on_move(self, delta: int) -> None:
        idx = self._list.currentRow()
        new_idx = self._ctrl.reorder(idx, delta)
        self.refresh_list()
        self.select_index(new_idx)

    def _on_rename(self) -> None:
        if self._running:
            return  # names are locked while a run is in progress
        idx = self._list.currentRow()
        if not (0 <= idx < len(self._ctrl.state.nodes)):
            return
        current = self._ctrl.state.nodes[idx].name
        new_name, ok = QInputDialog.getText(
            self, "Rename Node", "Instance name:", text=current
        )
        if ok:
            self._ctrl.rename_node(idx, new_name)
            self.refresh_list()
            self.select_index(idx)

    # --- flux source ---

    def refresh_flux_sources(self) -> None:
        """Repopulate the flux-source combo from the connected devices, keeping
        the current selection if its device is still connected."""
        current = self._ctrl.get_flux_device()
        names = [d.name for d in self._ctrl.list_devices()]
        self._flux_source.blockSignals(True)
        self._flux_source.clear()
        self._flux_source.addItem("(none)", None)
        for n in names:
            self._flux_source.addItem(n, n)
        idx = self._flux_source.findData(current) if current in names else 0
        self._flux_source.setCurrentIndex(max(0, idx))
        self._flux_source.blockSignals(False)
        self._sync_flux_device()

    def _on_flux_source_changed(self, _idx: int) -> None:
        self._sync_flux_device()

    def _sync_flux_device(self) -> None:
        """Push the selected flux source into State + label the axis with its unit."""
        name = self._flux_source.currentData()
        self._ctrl.set_flux_device(name)
        unit = self._ctrl.get_device_unit(name) if name else ""
        self._flux_unit.setText(f"[{unit}]" if unit and unit != "none" else "")

    # --- run / stop ---

    def _on_run_stop(self) -> None:
        if self._running:
            self.stop_requested.emit()
        else:
            self._commit_flux()
            self.run_requested.emit()

    def _commit_flux(self) -> None:
        import numpy as np

        start = float(self._flux_start.text())
        stop = float(self._flux_stop.text())
        npts = int(self._flux_npts.text())
        self._ctrl.set_flux_values(np.linspace(start, stop, npts).tolist())

    def set_running(self, running: bool) -> None:
        self._running = running
        self._run_btn.setText("■ Stop" if running else "▶ Run")
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        editing = not self._running
        for b in (
            self._add_btn,
            self._del_btn,
            self._up_btn,
            self._down_btn,
            self._rename_btn,
            self._flux_start,
            self._flux_stop,
            self._flux_npts,
            self._flux_source,
        ):
            b.setEnabled(editing)
        # Run enabled only when set up; Stop always enabled while running
        self._run_btn.setEnabled(self._running or self._ctrl.state.has_setup)


def _btn(text: str, slot) -> QPushButton:
    b = QPushButton(text)
    b.clicked.connect(slot)
    return b
