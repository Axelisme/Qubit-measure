"""SetupDialog — combined project setup and ZCU connection in one resizable dialog."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.services import StartupConnectionRequest, StartupProjectRequest

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


@runtime_checkable
class SocConfigProtocol(Protocol):
    """Protocol for QICK Soc configuration objects."""

    def description(self) -> str: ...


class SetupDialog(QDialog):
    """Resizable dialog combining project setup (left) and ZCU connection (right)."""

    def __init__(
        self,
        controller: "Controller",
        parent: Optional[QWidget] = None,
        startup_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._startup_mode = startup_mode
        self.setWindowTitle("Setup" if startup_mode else "Setup / Context")
        self.resize(900, 600)

        root_layout = QVBoxLayout(self)

        # ── horizontal splitter: left = project, right = connection ──────
        splitter = QSplitter(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, stretch=1)

        # ── Left panel: Project ──────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # chip / qub / res name section
        name_group = QGroupBox("Chip & Qubit & Resonator")
        name_form = QFormLayout(name_group)

        self._chip_edit = QLineEdit("unknown_chip")
        self._chip_edit.setPlaceholderText("e.g. Q5_2D")
        self._chip_edit.textChanged.connect(self._on_names_changed)
        name_form.addRow("Chip name:", self._chip_edit)

        self._qub_edit = QLineEdit("unknown_qubit")
        self._qub_edit.setPlaceholderText("e.g. Q1")
        self._qub_edit.textChanged.connect(self._on_names_changed)
        name_form.addRow("Qubit name:", self._qub_edit)

        self._res_edit = QLineEdit("unknown_resonator")
        self._res_edit.setPlaceholderText("e.g. R1")
        name_form.addRow("Resonator name:", self._res_edit)

        left_layout.addWidget(name_group)

        # derived paths section
        paths_group = QGroupBox("Derived paths (editable)")
        paths_form = QFormLayout(paths_group)

        result_dir_row = QHBoxLayout()
        self._result_dir_edit = QLineEdit()
        self._result_dir_edit.setPlaceholderText("/path/to/result_dir")
        result_dir_row.addWidget(self._result_dir_edit)
        browse_dir_btn = QPushButton("Browse…")
        browse_dir_btn.clicked.connect(self._on_browse_dir)
        result_dir_row.addWidget(browse_dir_btn)
        paths_form.addRow("Result dir:", result_dir_row)

        db_path_row = QHBoxLayout()
        self._db_path_edit = QLineEdit()
        self._db_path_edit.setPlaceholderText("/path/to/database")
        db_path_row.addWidget(self._db_path_edit)
        browse_db_btn = QPushButton("Browse…")
        browse_db_btn.clicked.connect(self._on_browse_db)
        db_path_row.addWidget(browse_db_btn)
        paths_form.addRow("Database path:", db_path_row)

        left_layout.addWidget(paths_group)

        # apply startup context button
        self._apply_btn = QPushButton(
            "Apply & Setup" if startup_mode else "Apply startup context"
        )
        self._apply_btn.clicked.connect(self._on_apply_startup_clicked)
        left_layout.addWidget(self._apply_btn)

        # context list
        ctx_group = QGroupBox("Contexts (requires file-backed project)")
        ctx_layout = QVBoxLayout(ctx_group)
        self._ctx_list = QListWidget()
        self._ctx_list.setMaximumHeight(120)
        ctx_layout.addWidget(self._ctx_list)

        switch_row = QHBoxLayout()
        self._switch_btn = QPushButton("Switch to selected")
        self._switch_btn.clicked.connect(self._on_switch_clicked)
        switch_row.addWidget(self._switch_btn)
        ctx_layout.addLayout(switch_row)
        left_layout.addWidget(ctx_group)

        # new context
        new_group = QGroupBox("New context (requires file-backed project)")
        new_form = QFormLayout(new_group)

        device_row = QHBoxLayout()
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(160)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        device_row.addWidget(self._device_combo)
        refresh_dev_btn = QPushButton("↻")
        refresh_dev_btn.setFixedWidth(28)
        refresh_dev_btn.setToolTip("Refresh device list")
        refresh_dev_btn.clicked.connect(self._refresh_device_list)
        device_row.addWidget(refresh_dev_btn)
        new_form.addRow("Track device:", device_row)

        self._unit_label = QLabel("—")
        new_form.addRow("Unit:", self._unit_label)

        self._clone_check = QCheckBox("Clone from current context")
        new_form.addRow("", self._clone_check)

        self._new_ctx_btn = QPushButton("Create new context")
        self._new_ctx_btn.clicked.connect(self._on_new_ctx_clicked)
        new_form.addRow("", self._new_ctx_btn)
        left_layout.addWidget(new_group)

        # project status label
        self._project_status = QLabel("")
        left_layout.addWidget(self._project_status)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ── Right panel: Connection ──────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        conn_group = QGroupBox("ZCU Connection")
        conn_form = QFormLayout(conn_group)

        self._ip_edit = QLineEdit("192.168.10.1")
        conn_form.addRow("IP address:", self._ip_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(8887)
        conn_form.addRow("Port:", self._port_spin)

        right_layout.addWidget(conn_group)

        self._mock_check = QCheckBox("Use MockSoc (offline, no hardware)")
        self._mock_check.stateChanged.connect(self._on_mock_toggled)
        right_layout.addWidget(self._mock_check)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        right_layout.addWidget(self._connect_btn)

        self._conn_status = QLabel("")
        right_layout.addWidget(self._conn_status)

        # soccfg description — hidden until connection succeeds
        self._cfg_text = QPlainTextEdit()
        self._cfg_text.setReadOnly(True)
        self._cfg_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # type: ignore[attr-defined]
        self._cfg_text.setVisible(False)
        right_layout.addWidget(self._cfg_text, stretch=1)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        splitter.setSizes([450, 450])

        # ── Close button ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        root_layout.addWidget(btn_box)

        # initialise
        self._on_names_changed()
        self._prefill_from_persistence()
        self._refresh_device_list()
        self._refresh_context_list()
        self._maybe_show_current_cfg()

        # EventBus subscriptions for live updates
        from zcu_tools.gui.event_bus import GuiEvent

        bus = controller.get_bus()
        bus.subscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_context_switched)
        bus.subscribe(GuiEvent.SOC_CHANGED, self._on_bus_soc_changed)
        bus.subscribe(GuiEvent.DEVICE_CHANGED, self._on_bus_device_changed)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            logger.debug("_cleanup_bus_subscriptions called but already inactive")
            return
        from zcu_tools.gui.event_bus import GuiEvent

        bus = self._ctrl.get_bus()
        bus.unsubscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_context_switched)
        bus.unsubscribe(GuiEvent.SOC_CHANGED, self._on_bus_soc_changed)
        bus.unsubscribe(GuiEvent.DEVICE_CHANGED, self._on_bus_device_changed)
        self._bus_subs_active = False

    def _on_bus_context_switched(self, payload: object) -> None:
        del payload
        self._refresh_context_list()

    def _on_bus_soc_changed(self, payload: object) -> None:
        from zcu_tools.gui.event_bus import SocChangedPayload

        self._connect_btn.setEnabled(True)
        if isinstance(payload, SocChangedPayload) and payload.soc is not None:
            self._set_conn_status("Connected", error=False)
            self._maybe_show_current_cfg()
            self._mock_check.blockSignals(True)
            self._mock_check.setChecked(payload.is_mock)
            self._mock_check.blockSignals(False)
        else:
            self._set_conn_status("Disconnected", error=False)
            self._cfg_text.setVisible(False)

    def _on_bus_device_changed(self, payload: object) -> None:
        del payload
        self._refresh_device_list()

    # ------------------------------------------------------------------
    # Project panel handlers
    # ------------------------------------------------------------------

    def _prefill_from_persistence(self) -> None:
        from zcu_tools.gui.services.startup import derive_project_paths

        data = self._ctrl.get_persisted_startup()
        if data is None:
            return
        if data.chip_name:
            self._chip_edit.setText(data.chip_name)
        if data.qub_name:
            self._qub_edit.setText(data.qub_name)
        if data.res_name:
            self._res_edit.setText(data.res_name)
        # Re-derive from chip/qub so the dated database_path lands in *today's*
        # folder (the persisted one may carry a stale date from a prior session);
        # fall back to the persisted paths only when chip/qub are absent.
        if data.chip_name and data.qub_name:
            result_dir, database_path = derive_project_paths(
                data.chip_name, data.qub_name, os.getcwd()
            )
            self._result_dir_edit.setText(result_dir)
            self._db_path_edit.setText(database_path)
        else:
            if data.result_dir:
                self._result_dir_edit.setText(data.result_dir)
            if data.database_path:
                self._db_path_edit.setText(data.database_path)
        if data.ip:
            self._ip_edit.setText(data.ip)
        if data.port:
            self._port_spin.setValue(data.port)

    def _on_names_changed(self) -> None:
        from zcu_tools.gui.services.startup import derive_project_paths

        chip = self._chip_edit.text().strip()
        qub = self._qub_edit.text().strip()
        if chip and qub:
            result_dir, database_path = derive_project_paths(chip, qub, os.getcwd())
            self._result_dir_edit.setText(result_dir)
            self._db_path_edit.setText(database_path)

    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select result directory")
        if path:
            self._result_dir_edit.setText(path)

    def _on_browse_db(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select database directory")
        if path:
            self._db_path_edit.setText(path)

    def _refresh_device_list(self) -> None:
        entries = self._ctrl.list_devices()

        current = self._device_combo.currentText()
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("(none)")
        for entry in entries:
            if entry.is_connected:
                self._device_combo.addItem(
                    f"{entry.name}  [{entry.type_name}]", userData=entry.name
                )
        self._device_combo.blockSignals(False)

        idx = self._device_combo.findText(current)
        self._device_combo.setCurrentIndex(max(idx, 0))
        self._on_device_changed(self._device_combo.currentText())

    def _on_device_changed(self, _text: str) -> None:
        device_name = self._device_combo.currentData()
        if device_name:
            self._unit_label.setText(self._ctrl.get_device_unit(device_name))
        else:
            self._unit_label.setText("—")

    def _on_apply_startup_clicked(self) -> None:
        chip = self._chip_edit.text().strip() or "unknown_chip"
        qub = self._qub_edit.text().strip() or "unknown_qubit"
        res = self._res_edit.text().strip() or "unknown_resonator"
        result_dir = self._result_dir_edit.text().strip()
        db_path = self._db_path_edit.text().strip()
        if not self._ctrl.apply_startup_project(
            StartupProjectRequest(
                chip_name=chip,
                qub_name=qub,
                res_name=res,
                result_dir=result_dir,
                database_path=db_path,
            )
        ):
            return
        self._set_project_status(f"Startup context applied: {chip}/{qub} (res={res})")
        logger.info(
            "SetupDialog: startup context applied chip=%r qub=%r res=%r",
            chip,
            qub,
            res,
        )

        if result_dir:
            self._refresh_context_list()
            logger.info("SetupDialog: project available result_dir=%r", result_dir)

    def _on_switch_clicked(self) -> None:
        item = self._ctx_list.currentItem()
        if item is None:
            self._set_project_status("Select a context first", error=True)
            return
        label = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        self._ctrl.use_context(label)
        self._refresh_context_list()
        self._set_project_status(f"Switched to context: {label}")
        logger.info("SetupDialog: switched to context=%r", label)

    def _on_new_ctx_clicked(self) -> None:
        clone = self._clone_check.isChecked()
        device_name: Optional[str] = self._device_combo.currentData()
        unit = self._unit_label.text() if device_name else "none"
        value: Optional[float] = None
        if device_name:
            value = self._ctrl.get_device_value_for_new_context(device_name)
        self._ctrl.new_context(value=value, unit=unit, clone_from_current=clone)
        self._refresh_context_list()
        val_str = f"{value} {unit}" if value is not None else "NoValue"
        self._set_project_status(f"Created new context ({val_str})")
        logger.info(
            "SetupDialog: new_context value=%r unit=%r clone=%r device=%r",
            value,
            unit,
            clone,
            device_name,
        )

    def _refresh_context_list(self) -> None:
        self._ctx_list.clear()
        labels = self._ctrl.get_context_labels()
        active = self._ctrl.get_active_context_label()
        bold_font = QFont()
        bold_font.setBold(True)
        active_idx = -1
        for i, label in enumerate(labels):
            is_active = label == active
            display = f"▶ {label}" if is_active else f"   {label}"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, label)  # type: ignore[attr-defined]
            if is_active:
                item.setFont(bold_font)
                active_idx = i
            self._ctx_list.addItem(item)
        if active_idx >= 0:
            self._ctx_list.setCurrentRow(active_idx)
        elif self._ctx_list.count() > 0:
            self._ctx_list.setCurrentRow(0)

    def _set_project_status(self, msg: str, error: bool = False) -> None:
        self._project_status.setText(msg)
        color = "red" if error else "green"
        self._project_status.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------------
    # Connection panel handlers
    # ------------------------------------------------------------------

    def _maybe_show_current_cfg(self) -> None:
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(soccfg.description())
            self._set_conn_status("Currently connected", error=False)

    def _on_mock_toggled(self, state: int) -> None:
        use_mock = bool(state)
        self._ip_edit.setEnabled(not use_mock)
        self._port_spin.setEnabled(not use_mock)

    def _on_connect_clicked(self) -> None:
        from zcu_tools.gui.services.connection import (
            ConnectMockRequest,
            ConnectRemoteRequest,
        )

        use_mock = self._mock_check.isChecked()
        req = (
            ConnectMockRequest()
            if use_mock
            else ConnectRemoteRequest(
                ip=self._ip_edit.text().strip(), port=self._port_spin.value()
            )
        )

        self._ctrl.bind_connection_outcome(
            self._on_connect_finished,
            self._on_connect_failed,
        )

        if not use_mock:
            self._ctrl.remember_startup_connection(
                StartupConnectionRequest(
                    ip=self._ip_edit.text().strip(),
                    port=self._port_spin.value(),
                )
            )
        self._connect_btn.setEnabled(False)
        self._set_conn_status("Connecting…", error=False)
        self._ctrl.start_connect(req)

    def _on_connect_finished(self) -> None:
        self._connect_btn.setEnabled(True)
        self._set_conn_status("Connected", error=False)
        soccfg = self._ctrl.get_soccfg()
        if isinstance(soccfg, SocConfigProtocol):
            self._show_cfg(soccfg.description())

    def _on_connect_failed(self, message: str) -> None:
        self._connect_btn.setEnabled(True)
        self._set_conn_status(message, error=True)

    def _show_cfg(self, text: str) -> None:
        self._cfg_text.setPlainText(text)
        self._cfg_text.setVisible(True)

    def _set_conn_status(self, msg: str, error: bool = False) -> None:
        self._conn_status.setText(msg)
        color = "red" if error else "green"
        self._conn_status.setStyleSheet(f"color: {color};")
