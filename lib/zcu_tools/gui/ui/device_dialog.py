"""DeviceDialog — register devices and inspect/control selected device."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QColor  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.event_bus import (
    DeviceChangedPayload,
    DeviceSetupChangedPayload,
    GuiEvent,
)
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSetupSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
    list_supported_device_types,
)

from .progress_stack import ProgressStack
from .widgets import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


@runtime_checkable
class DevicePanelProtocol(Protocol):
    """Protocol for device detail panels."""

    def load(self, info: Any) -> None: ...
    def read(self) -> Any: ...


# ---------------------------------------------------------------------------
# Per-device detail panels
# ---------------------------------------------------------------------------


class _FakeDevicePanel(QWidget):
    """Info + control panel for FakeDevice."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        form.addRow("Type:", self._type_label)

        self._address_label = QLabel()
        form.addRow("Address:", self._address_label)

        self._output_combo = QComboBox()
        self._output_combo.addItems(["on", "off"])
        form.addRow("Output:", self._output_combo)

        self._value_spin = TrimDoubleSpinBox()
        self._value_spin.setRange(-1e9, 1e9)
        self._value_spin.setDecimals(6)
        self._value_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Value:", self._value_spin)

        self._rampstep_spin = TrimDoubleSpinBox()
        self._rampstep_spin.setRange(1e-9, 1e9)
        self._rampstep_spin.setDecimals(9)
        self._rampstep_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Ramp step:", self._rampstep_spin)

    def load(self, info: Any) -> None:
        from zcu_tools.device.fake import FakeDeviceInfo

        assert isinstance(info, FakeDeviceInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._value_spin.setValue(info.value)
        self._rampstep_spin.setValue(info.rampstep)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_spin.value(),
            "rampstep": self._rampstep_spin.value(),
        }


class _YOKOGS200Panel(QWidget):
    """Info + control panel for YOKOGS200."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        form.addRow("Type:", self._type_label)

        self._address_label = QLabel()
        form.addRow("Address:", self._address_label)

        self._mode_label = QLabel()
        form.addRow("Mode:", self._mode_label)

        self._output_combo = QComboBox()
        self._output_combo.addItems(["on", "off"])
        form.addRow("Output:", self._output_combo)

        self._value_spin = TrimDoubleSpinBox()
        self._value_spin.setRange(-1e9, 1e9)
        self._value_spin.setDecimals(6)
        self._value_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Value:", self._value_spin)

    def load(self, info: Any) -> None:
        from zcu_tools.device.yoko import YOKOGS200Info

        assert isinstance(info, YOKOGS200Info)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._mode_label.setText(info.mode)
        self._output_combo.setCurrentText(info.output)
        self._value_spin.setValue(info.value)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_spin.value(),
        }


class _SGS100APanel(QWidget):
    """Info + control panel for SGS100A."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        form.addRow("Type:", self._type_label)

        self._address_label = QLabel()
        form.addRow("Address:", self._address_label)

        self._output_combo = QComboBox()
        self._output_combo.addItems(["on", "off"])
        form.addRow("Output:", self._output_combo)

        self._freq_spin = TrimDoubleSpinBox()
        self._freq_spin.setRange(1e6, 20e9)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Freq (Hz):", self._freq_spin)

        self._pow_spin = TrimDoubleSpinBox()
        self._pow_spin.setRange(-120, 30)
        self._pow_spin.setDecimals(2)
        self._pow_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Power (dBm):", self._pow_spin)

    def load(self, info: Any) -> None:
        from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo

        assert isinstance(info, RohdeSchwarzSGS100AInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._freq_spin.setValue(info.freq_Hz)
        self._pow_spin.setValue(info.power_dBm)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "freq_Hz": self._freq_spin.value(),
            "power_dBm": self._pow_spin.value(),
        }


class _MemoryDevicePanel(QWidget):
    """Read-only info panel for a remembered-but-not-connected device."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        form.addRow("Type:", self._type_label)

        self._name_label = QLabel()
        form.addRow("Name:", self._name_label)

        self._addr_label = QLabel()
        form.addRow("Address:", self._addr_label)

        note = QLabel("Not connected. Press Reconnect to connect.")
        note.setStyleSheet("color: gray;")
        form.addRow(note)

    def load_memory(self, type_name: str, name: str, address: str) -> None:
        self._type_label.setText(type_name)
        self._name_label.setText(name)
        self._addr_label.setText(address or "(none)")


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class DeviceDialog(QDialog):
    """Resizable dialog combining device listing (left) and detail control (right)."""

    def __init__(
        self, controller: Controller, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Manage Hardware Devices")
        self.resize(800, 500)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        layout.addWidget(splitter, stretch=1)

        # --- Left side: List + Management ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(QLabel("Registered Devices:"))
        left_layout.addWidget(self._list, stretch=1)

        # Add device form
        add_box = QGroupBox("Register New Device")
        add_form = QFormLayout(add_box)

        self._type_combo = QComboBox()
        self._type_combo.addItems(list_supported_device_types())
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        add_form.addRow("Type:", self._type_combo)

        self._name_edit = QLineEdit()
        add_form.addRow("Name:", self._name_edit)

        self._addr_edit = QLineEdit()
        self._addr_edit.setPlaceholderText("TCPIP::192.168.1.1::INSTR")
        add_form.addRow("Address:", self._addr_edit)

        self._add_btn = QPushButton("Add Device")
        self._add_btn.clicked.connect(self._on_add_clicked)
        add_form.addRow(self._add_btn)

        self._add_status = QLabel("")
        self._add_status.setWordWrap(True)
        add_form.addRow(self._add_status)

        left_layout.addWidget(add_box)
        splitter.addWidget(left_widget)

        # --- Right side: Detail Panel Stack ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._stack.addWidget(QLabel("Select a device to configure."))  # Page 0: Idle
        self._stack.addWidget(_FakeDevicePanel())  # Page 1
        self._stack.addWidget(_YOKOGS200Panel())  # Page 2
        self._stack.addWidget(_SGS100APanel())  # Page 3
        self._memory_panel = _MemoryDevicePanel()
        self._stack.addWidget(self._memory_panel)  # Page 4: memory-only
        right_layout.addWidget(self._stack, stretch=1)

        # Bottom buttons for right side
        btn_row = QHBoxLayout()
        self._drop_btn = QPushButton("Drop")
        self._drop_btn.setStyleSheet("color: red;")
        self._drop_btn.clicked.connect(self._on_forget_clicked)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.clicked.connect(self._on_apply_or_stop_clicked)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._drop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._refresh_btn)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(close_btn)
        right_layout.addLayout(btn_row)

        self._active_setup: Optional[DeviceSetupSnapshot] = None

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])

        # Status & Progress
        self._progress = ProgressStack()
        layout.addWidget(self._progress)

        bus = self._ctrl.get_bus()
        bus.subscribe(GuiEvent.DEVICE_CHANGED, self._on_device_changed)
        bus.subscribe(GuiEvent.DEVICE_SETUP_CHANGED, self._on_setup_changed)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

        self._ctrl.get_device_progress_model().attach_stack(self._progress)

        self._refresh_list()
        self._render_setup(self._ctrl.get_active_device_setup())

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            return
        self._ctrl.get_bus().unsubscribe(
            GuiEvent.DEVICE_SETUP_CHANGED, self._on_setup_changed
        )
        self._ctrl.get_bus().unsubscribe(
            GuiEvent.DEVICE_CHANGED, self._on_device_changed
        )
        self._ctrl.get_device_progress_model().detach_stack()
        self._bus_subs_active = False

    def _refresh_list(self, select_name: Optional[str] = None) -> None:
        self._list.clear()
        entries = self._ctrl.list_devices()
        for entry in entries:
            if entry.is_connected:
                item = QListWidgetItem(f"{entry.name} ({entry.type_name})")
            else:
                item = QListWidgetItem(
                    f"{entry.name} ({entry.type_name}) [not connected]"
                )
                item.setForeground(QColor("gray"))
            item.setData(Qt.ItemDataRole.UserRole, entry.name)  # type: ignore[attr-defined]
            self._list.addItem(item)

        if select_name is not None:
            for row in range(self._list.count()):
                it = self._list.item(row)
                if it is not None and it.data(Qt.ItemDataRole.UserRole) == select_name:  # type: ignore[attr-defined]
                    self._list.setCurrentRow(row)
                    break

        self._on_selection_changed(self._list.currentRow())
        # refresh default name so it stays unique after any list change
        dtype = self._type_combo.currentText()
        existing = {e.name for e in entries}
        self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _on_selection_changed(self, _row: int) -> None:
        item = self._list.currentItem()
        if item is None:
            self._stack.setCurrentIndex(0)
            if self._active_setup is None:
                self._drop_btn.setEnabled(False)
                self._refresh_btn.setEnabled(False)
                self._apply_btn.setEnabled(False)
            return

        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        snapshot = self._ctrl.get_device_snapshot(name)
        if snapshot is None:
            return
        is_memory = snapshot.status is DeviceStatus.MEMORY_ONLY
        is_busy = snapshot.status not in {
            DeviceStatus.MEMORY_ONLY,
            DeviceStatus.CONNECTED,
        }

        if self._active_setup is None:
            self._drop_btn.setEnabled(not is_busy)
            self._drop_btn.setText("Forget" if is_memory else "Drop")
            self._refresh_btn.setEnabled(not is_memory and not is_busy)
            self._apply_btn.setEnabled(not is_busy)
            self._apply_btn.setText("Reconnect" if is_memory else "Apply Changes")
            self._apply_btn.setStyleSheet("")

        if is_memory:
            self._memory_panel.load_memory(
                snapshot.type_name, snapshot.name, snapshot.address
            )
            self._stack.setCurrentIndex(4)
            return

        info = snapshot.info
        if info is None:
            return

        # Map info.type to stack page
        page_map = {"FakeDevice": 1, "YOKOGS200": 2, "RohdeSchwarzSGS100A": 3}
        page = page_map.get(getattr(info, "type", ""), 0)
        self._stack.setCurrentIndex(page)

        panel = self._stack.currentWidget()
        if page > 0 and isinstance(panel, DevicePanelProtocol):
            panel.load(info)

    @staticmethod
    def _unique_name(base: str, existing: set[str]) -> str:
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _on_type_changed(self, dtype: str) -> None:
        existing = {e.name for e in self._ctrl.list_devices()}
        self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _on_add_clicked(self) -> None:
        dtype = self._type_combo.currentText()
        name = self._name_edit.text().strip() or dtype.lower()
        addr = self._addr_edit.text().strip()
        self._add_status.setText("")

        self._ctrl.start_connect_device(
            ConnectDeviceRequest(type_name=dtype, name=name, address=addr)
        )
        self._add_status.setStyleSheet("color: gray;")
        self._add_status.setText(f"Connecting {name}...")

    def _on_forget_clicked(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if self._ctrl.is_memory_device(name):
            # Remove from memory entirely — won't appear after restart
            self._ctrl.forget_device(name)
        else:
            # Disconnect only — keep in startup memory so it reappears as gray on next launch
            self._ctrl.start_disconnect_device(DisconnectDeviceRequest(name=name))

    def _on_refresh_clicked(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
            self._ctrl.get_device_info(name)
        self._on_selection_changed(self._list.currentRow())

    def _on_apply_or_stop_clicked(self) -> None:
        if self._active_setup is not None:
            self._ctrl.cancel_device_operation(self._active_setup.device_name)
            return

        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]

        if self._ctrl.is_memory_device(name):
            self._add_status.setText("")
            self._ctrl.start_reconnect_device(name)
            self._add_status.setStyleSheet("color: gray;")
            self._add_status.setText(f"Reconnecting {name}...")
            return

        panel = self._stack.currentWidget()
        if not isinstance(panel, DevicePanelProtocol):
            return

        updates = panel.read()
        info = self._ctrl.get_device_info(name)
        if info is None:
            return
        from zcu_tools.device.base import BaseDeviceInfo

        if not isinstance(info, BaseDeviceInfo):
            return
        new_info = info.with_updates(**updates)
        self._ctrl.start_setup_device(SetupDeviceRequest(name=name, info=new_info))

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        name = payload.name
        self._refresh_list(select_name=name)
        if name is None:
            return
        snapshot = self._ctrl.get_device_snapshot(name)
        if snapshot is None:
            return
        if snapshot.error is not None:
            self._add_status.setStyleSheet("color: red;")
            self._add_status.setText(snapshot.error)
        elif snapshot.status is DeviceStatus.CONNECTED:
            self._add_status.setStyleSheet("color: green;")
            self._add_status.setText(f"Connected {name}.")
        elif snapshot.status in {
            DeviceStatus.CONNECTING,
            DeviceStatus.DISCONNECTING,
            DeviceStatus.SETTING_VALUE,
        }:
            self._add_status.setStyleSheet("color: gray;")
            self._add_status.setText(
                f"{snapshot.status.value.replace('_', ' ').title()}: {name}..."
            )

    def _on_setup_changed(self, payload: DeviceSetupChangedPayload) -> None:
        self._render_setup(payload.active_setup)

    def _render_setup(self, setup: Optional[DeviceSetupSnapshot]) -> None:
        self._active_setup = setup
        if setup is not None:
            for row in range(self._list.count()):
                item = self._list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == setup.device_name:  # type: ignore[attr-defined]
                    self._list.setCurrentRow(row)
                    break
            self._progress.render_snapshot(setup.progress)
            self._progress.show()
            self._set_setup_running(True)
            return
        self._progress.hide()
        self._set_setup_running(False)
        self._on_selection_changed(self._list.currentRow())

    def _set_setup_running(self, running: bool) -> None:
        has_selection = self._list.currentItem() is not None
        self._drop_btn.setEnabled(has_selection and not running)
        self._refresh_btn.setEnabled(has_selection and not running)
        self._list.setEnabled(not running)
        self._type_combo.setEnabled(not running)
        self._name_edit.setEnabled(not running)
        self._addr_edit.setEnabled(not running)
        self._add_btn.setEnabled(not running)
        self._apply_btn.setEnabled(has_selection)
        if running:
            self._apply_btn.setText("Stop")
            self._apply_btn.setStyleSheet("color: red;")
        else:
            self._apply_btn.setText("Apply Changes")
            self._apply_btn.setStyleSheet("")
