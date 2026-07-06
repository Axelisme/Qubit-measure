"""NodeListPane — the left side: node list + flux + Run/Stop.

Owns workflow editing (select / add / remove / reorder Nodes), the flux sweep
fields, the flux-source picker, and the single Run/Stop toggle button (▶ Run in
edit state, ■ Stop while running). Global session actions (Setup / Devices /
Predictor / Inspect) live in MainWindow's top toolbar, matching measure-gui's
layout. Drives the Controller; emits a Qt signal when the selection changes so
the right pane follows.
"""

from __future__ import annotations

import math
from typing import Literal

from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.cfg import DirectValue, EvalValue, ScalarSpec
from zcu_tools.gui.app.autofluxdep.cfg.form import (
    LiveModelEnv,
    ScalarLiveField,
    ScalarWidget,
)
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.registry import available_node_types
from zcu_tools.gui.widgets import DialogPresenter, QtDialogPresenter

RunUiState = Literal["idle", "running", "pausing", "paused"]


class _FluxScalarEditor(QWidget):
    """Flux-sweep scalar editor backed by the shared ScalarWidget eval UX."""

    textChanged = Signal(str)

    def __init__(
        self,
        ctrl: Controller,
        *,
        label: str,
        type_: type,
        text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._label = label
        self._type = type_
        self._torn_down = False
        self._field = ScalarLiveField(
            ScalarSpec(label=label, type=type_, decimals=6 if type_ is float else None),
            LiveModelEnv(ctrl=ctrl),
            initial_val=_flux_value_from_text(text, type_),
        )
        self._field.on_change.connect(self._on_field_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._widget = ScalarWidget(self._field, self)
        layout.addWidget(self._widget)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt-style test helper
        self._field.set_value(_flux_value_from_text(text, self._type))

    def expression_text(self) -> str:
        value = self._field.get_value()
        if isinstance(value, EvalValue):
            return value.expr
        if value is None:
            return ""
        if not isinstance(value, DirectValue):
            raise RuntimeError(f"Unexpected flux field value {type(value).__name__}")
        if value.value is None:
            return ""
        return str(value.value)

    def teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        self._field.on_change.disconnect(self._on_field_changed)
        self._widget.teardown()

    def _on_field_changed(self, value: object) -> None:
        if isinstance(value, DirectValue) and value.value is None:
            self._field.set_value(DirectValue(_direct_default_for_type(self._type)))
            return
        if not self.signalsBlocked():
            self.textChanged.emit(self.expression_text())


def _flux_value_from_text(text: str, type_: type) -> DirectValue | EvalValue:
    stripped = text.strip()
    if stripped == "":
        return EvalValue(expr="")
    try:
        if type_ is int:
            return DirectValue(int(stripped))
        if type_ is float:
            value = float(stripped)
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            return DirectValue(value)
    except (TypeError, ValueError):
        return EvalValue(expr=stripped)
    raise RuntimeError(f"Unsupported flux field type {type_!r}")


def _direct_default_for_type(type_: type) -> int | float:
    if type_ is int:
        return 0
    if type_ is float:
        return 0.0
    raise RuntimeError(f"Unsupported flux field type {type_!r}")


class NodeListPane(QWidget):
    """Left pane: node management + flux + setup + run/stop."""

    selection_changed = Signal(int)  # selected node index, or -1
    run_requested = Signal()
    pause_requested = Signal()
    continue_requested = Signal()
    restart_requested = Signal()
    abort_requested = Signal()
    sample_export_requested = Signal()
    auto_follow_changed = Signal(bool)

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget | None = None,
        *,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._dialog_presenter = dialog_presenter or QtDialogPresenter()
        self._run_state: RunUiState = "idle"
        self._torn_down = False

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
        start_expr, stop_expr, npts_expr = self._ctrl.get_flux_sweep_expressions()
        self._flux_start = _FluxScalarEditor(
            self._ctrl, label="start", type_=float, text=start_expr
        )
        self._flux_stop = _FluxScalarEditor(
            self._ctrl, label="stop", type_=float, text=stop_expr
        )
        self._flux_npts = _FluxScalarEditor(
            self._ctrl, label="points", type_=int, text=npts_expr
        )
        for w in (self._flux_start, self._flux_stop, self._flux_npts):
            w.textChanged.connect(lambda _text: self._sync_flux_expressions())
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
        self._auto_follow_tabs = QCheckBox("Auto-follow active tab")
        self._auto_follow_tabs.setChecked(self._ctrl.get_auto_follow_tabs())
        self._auto_follow_tabs.toggled.connect(self._on_auto_follow_toggled)
        root.addWidget(self._auto_follow_tabs)

        self._run_btn = _btn("▶ Run", self._on_run_stop)
        self._restart_btn = _btn("↻ Restart", self._on_restart)
        self._abort_btn = _btn("■ Abort", self._on_abort)
        self._export_sample_btn = _btn("Export sample", self._on_export_sample)
        self._run_action_row = QHBoxLayout()
        self._run_action_row.addWidget(self._run_btn, 1)
        self._run_action_row.addWidget(self._restart_btn, 1)
        root.addLayout(self._run_action_row)
        root.addWidget(self._abort_btn)
        root.addWidget(self._export_sample_btn)

        self.refresh_list()
        self.refresh_run_availability()
        self.refresh_flux_sources()

    # --- list / selection ---

    def refresh_from_state(self) -> None:
        """Refresh all view state after a controller-level workflow restore."""
        self.refresh_list()
        self.refresh_flux_fields()
        self.refresh_flux_sources()
        self.refresh_preferences()

    def teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        self._flux_start.teardown()
        self._flux_stop.teardown()
        self._flux_npts.teardown()

    def refresh_list(self) -> None:
        prev = self._list.currentRow()
        self._list.blockSignals(True)
        self._list.clear()
        for index, node in enumerate(self._ctrl.state.nodes):
            item = QListWidgetItem()
            item.setToolTip(node.name)
            item.setData(Qt.ItemDataRole.UserRole, node.name)  # type: ignore[attr-defined]
            widget = _NodeRowWidget(
                index=index,
                name=node.name,
                enabled=node.enabled,
                on_toggled=self._on_node_enabled_toggled,
            )
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
        if 0 <= prev < self._list.count():
            self._list.setCurrentRow(prev)
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._on_row_changed(self._list.currentRow())
        self.refresh_run_availability()

    def select_index(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    @property
    def selected_index(self) -> int:
        return self._list.currentRow()

    def _on_row_changed(self, row: int) -> None:
        self.selection_changed.emit(row)

    def _on_node_enabled_toggled(self, index: int, enabled: bool) -> None:
        self._ctrl.set_node_enabled(index, enabled)
        self.refresh_run_availability()

    def refresh_flux_fields(self) -> None:
        start_expr, stop_expr, npts_expr = self._ctrl.get_flux_sweep_expressions()
        fields = (
            (self._flux_start, start_expr),
            (self._flux_stop, stop_expr),
            (self._flux_npts, npts_expr),
        )
        for field, text in fields:
            field.blockSignals(True)
            field.setText(text)
            field.blockSignals(False)

    def refresh_preferences(self) -> None:
        self._auto_follow_tabs.blockSignals(True)
        self._auto_follow_tabs.setChecked(self._ctrl.get_auto_follow_tabs())
        self._auto_follow_tabs.blockSignals(False)

    def _on_auto_follow_toggled(self, enabled: bool) -> None:
        self._ctrl.set_auto_follow_tabs(enabled)
        self.auto_follow_changed.emit(enabled)

    def _sync_flux_expressions(self) -> None:
        self._ctrl.set_flux_sweep_expressions(
            self._flux_start.expression_text(),
            self._flux_stop.expression_text(),
            self._flux_npts.expression_text(),
        )

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
        if self._run_state != "idle":
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
        if self._run_state == "idle":
            self._ctrl.set_flux_device(name)
        unit = self._ctrl.get_device_unit(name) if name else ""
        self._flux_unit.setText(f"[{unit}]" if unit and unit != "none" else "")
        self.refresh_run_availability()

    # --- run / stop ---

    def _on_run_stop(self) -> None:
        if self._run_state == "running":
            self.pause_requested.emit()
        elif self._run_state == "paused":
            self.continue_requested.emit()
        elif self._run_state == "idle":
            try:
                self._commit_flux()
            except Exception as exc:
                self._dialog_presenter.warning(self, "Invalid flux sweep", str(exc))
                return
            self.run_requested.emit()

    def _on_abort(self) -> None:
        self.abort_requested.emit()

    def _on_restart(self) -> None:
        self.restart_requested.emit()

    def _on_export_sample(self) -> None:
        self.sample_export_requested.emit()

    def _commit_flux(self) -> None:
        self._ctrl.commit_flux_sweep(
            self._flux_start.expression_text(),
            self._flux_stop.expression_text(),
            self._flux_npts.expression_text(),
        )

    def set_running(self, running: bool) -> None:
        self.set_run_state("running" if running else "idle")

    def set_run_state(self, state: RunUiState) -> None:
        self._run_state = state
        self._run_btn.setText(
            {
                "idle": "▶ Run",
                "running": "⏸ Pause",
                "pausing": "Pausing...",
                "paused": "▶ Continue",
            }[state]
        )
        self.refresh_run_availability()

    def refresh_run_availability(self) -> None:
        editing = self._run_state == "idle"
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
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            widget = self._list.itemWidget(item)
            if isinstance(widget, _NodeRowWidget):
                widget.set_editing(editing)
        reason = self._run_disabled_reason()
        self._run_btn.setEnabled(
            (self._run_state == "idle" and reason is None)
            or self._run_state in {"running", "paused"}
        )
        self._run_btn.setToolTip("" if self._run_btn.isEnabled() else (reason or ""))
        self._abort_btn.setVisible(self._run_state in {"running", "pausing", "paused"})
        self._abort_btn.setEnabled(self._run_state in {"running", "pausing", "paused"})
        self._restart_btn.setVisible(self._run_state == "paused")
        self._restart_btn.setEnabled(self._run_state == "paused")
        can_export = self._run_state == "idle" and self._ctrl.can_export_sample_table()
        self._export_sample_btn.setVisible(can_export)
        self._export_sample_btn.setEnabled(can_export)

    def _run_disabled_reason(self) -> str | None:
        return self._ctrl.run_readiness()


class _NodeRowWidget(QWidget):
    """A compact node-list row with an enable checkbox and stable label."""

    def __init__(
        self,
        *,
        index: int,
        name: str,
        enabled: bool,
        on_toggled,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setToolTip(name)
        self._checkbox = QCheckBox()
        self._checkbox.setToolTip("Include this node in future runs")
        self._label = QLabel(name)
        self._label.setToolTip(name)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(6)
        layout.addWidget(self._checkbox)
        layout.addWidget(self._label, 1)

        self._checkbox.blockSignals(True)
        self._checkbox.setChecked(enabled)
        self._checkbox.blockSignals(False)
        self._label.setEnabled(enabled)
        self._checkbox.toggled.connect(
            lambda checked, row=index: self._handle_toggled(row, checked, on_toggled)
        )

    def set_editing(self, editing: bool) -> None:
        self._checkbox.setEnabled(editing)

    def _handle_toggled(self, index: int, checked: bool, on_toggled) -> None:
        self._label.setEnabled(checked)
        on_toggled(index, checked)


def _btn(text: str, slot) -> QPushButton:
    b = QPushButton(text)
    b.clicked.connect(slot)
    return b
