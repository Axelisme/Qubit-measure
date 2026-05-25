"""DeviceDialog — register devices and inspect/control selected device."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
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

from zcu_tools.gui.services.device import DeviceProtocol, _DeviceSetupWorker

from .progress_stack import ProgressStack
from .widgets import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

# Known device types: display name → (class_path, requires_address)
_DEVICE_TYPES: dict[str, tuple[str, bool]] = {
    "FakeDevice": ("zcu_tools.device.fake.FakeDevice", False),
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}


@runtime_checkable
class DevicePanelProtocol(Protocol):
    """Protocol for device detail panels."""

    def load(self, info: Any) -> None: ...
    def read(self) -> Any: ...


def _instantiate_device(type_name: str, address: str) -> object:
    class_path, requires_address = _DEVICE_TYPES[type_name]
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    if requires_address:
        import pyvisa  # type: ignore[import-untyped]

        rm = pyvisa.ResourceManager()
        return cls(address, rm)
    return cls()


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
        self._type_combo.addItems(list(_DEVICE_TYPES.keys()))
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
        right_layout.addWidget(self._stack, stretch=1)

        # Bottom buttons for right side
        btn_row = QHBoxLayout()
        self._drop_btn = QPushButton("Drop Selected")
        self._drop_btn.setStyleSheet("color: red;")
        self._drop_btn.clicked.connect(self._on_drop_clicked)
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.clicked.connect(self._on_apply_or_stop_clicked)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._drop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(close_btn)
        right_layout.addLayout(btn_row)

        self._active_worker: Optional[_DeviceSetupWorker] = None

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])

        # Status & Progress
        self._progress = ProgressStack()
        layout.addWidget(self._progress)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self._list.clear()
        devices = self._ctrl.list_devices()
        for name, type_ in devices.items():
            item = QListWidgetItem(f"{name} ({type_})")
            item.setData(Qt.ItemDataRole.UserRole, name)  # type: ignore[attr-defined]
            self._list.addItem(item)

        self._on_selection_changed(self._list.currentRow())
        # refresh default name so it stays unique after any list change
        dtype = self._type_combo.currentText()
        existing = set(devices.keys())
        self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _on_selection_changed(self, row: int) -> None:
        item = self._list.currentItem()
        if item is None:
            self._stack.setCurrentIndex(0)
            self._drop_btn.setEnabled(False)
            self._apply_btn.setEnabled(False)
            return

        self._drop_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)

        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        info = self._ctrl.get_device_info(name)
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
        existing = set(self._ctrl.list_devices().keys())
        self._name_edit.setText(self._unique_name(dtype.lower(), existing))

    def _on_add_clicked(self) -> None:
        dtype = self._type_combo.currentText()
        name = self._name_edit.text().strip() or dtype.lower()
        addr = self._addr_edit.text().strip()
        self._add_status.setText("")

        try:
            dev = _instantiate_device(dtype, addr)
            self._ctrl.register_device(name, cast(DeviceProtocol, dev))
            self._ctrl.setup_device(name, {"address": addr})
            self._add_status.setStyleSheet("color: green;")
            self._add_status.setText(f"Added {name}")
            self._refresh_list()
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            self._add_status.setStyleSheet("color: red;")
            self._add_status.setText(f"Connection failed: {e}")
        except Exception as e:
            self._add_status.setStyleSheet("color: red;")
            self._add_status.setText(f"Error: {e}")

    def _on_drop_clicked(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        self._ctrl.drop_device(name)
        self._refresh_list()

    def _on_apply_or_stop_clicked(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
            return

        item = self._list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
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
        from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

        pbar_factory = QtProgressBarFactory(self._progress)
        worker = self._ctrl.setup_device(name, new_info, pbar_factory)
        self._active_worker = worker
        worker.finished.connect(lambda _: self._on_setup_done())
        worker.failed.connect(lambda _name, _msg: self._on_setup_done())
        worker.cancelled.connect(lambda _: self._on_setup_done())
        self._set_setup_running(True)

    def _on_setup_done(self) -> None:
        self._active_worker = None
        self._set_setup_running(False)
        self._on_selection_changed(self._list.currentRow())

    def _set_setup_running(self, running: bool) -> None:
        self._drop_btn.setEnabled(not running)
        self._list.setEnabled(not running)
        if running:
            self._apply_btn.setText("Stop")
            self._apply_btn.setStyleSheet("color: red;")
            self._progress.show()
        else:
            self._apply_btn.setText("Apply Changes")
            self._apply_btn.setStyleSheet("")
            self._progress.hide()
