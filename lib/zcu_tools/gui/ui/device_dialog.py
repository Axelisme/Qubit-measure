"""DeviceDialog — register devices and inspect/control selected device."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QDialogButtonBox,
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

from .progress_stack import ProgressStack

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

# Known device types: display name → (class_path, requires_address)
_DEVICE_TYPES: dict[str, tuple[str, bool]] = {
    "FakeDevice": ("zcu_tools.device.fake.FakeDevice", False),
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}


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

        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(-1e9, 1e9)
        self._value_spin.setDecimals(6)
        self._value_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Value:", self._value_spin)

        self._rampstep_spin = QDoubleSpinBox()
        self._rampstep_spin.setRange(1e-9, 1e9)
        self._rampstep_spin.setDecimals(9)
        self._rampstep_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Ramp step:", self._rampstep_spin)

    def load(self, info: object) -> None:
        from zcu_tools.device.fake import FakeDeviceInfo

        assert isinstance(info, FakeDeviceInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._value_spin.setValue(info.value)
        self._rampstep_spin.setValue(info.rampstep)

    def read(self) -> dict:
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_spin.value(),
            "rampstep": self._rampstep_spin.value(),
        }


class _YOKOPanel(QWidget):
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

        self._value_spin = QDoubleSpinBox()
        self._value_spin.setDecimals(6)
        self._value_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Value:", self._value_spin)

        self._rampstep_spin = QDoubleSpinBox()
        self._rampstep_spin.setRange(1e-9, 1.0)
        self._rampstep_spin.setDecimals(9)
        self._rampstep_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Ramp step:", self._rampstep_spin)

        self._mode: str = "voltage"

    def load(self, info: object) -> None:
        from zcu_tools.device.yoko import YOKOGS200Info

        assert isinstance(info, YOKOGS200Info)
        self._mode = info.mode
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._mode_label.setText(info.mode)
        self._output_combo.setCurrentText(info.output)

        # range depends on mode
        if info.mode == "voltage":
            self._value_spin.setRange(-7.0, 7.0)
            self._value_spin.setDecimals(6)
            self._value_spin.setSuffix(" V")
        else:
            self._value_spin.setRange(-7e-3, 7e-3)
            self._value_spin.setDecimals(8)
            self._value_spin.setSuffix(" A")

        self._value_spin.setValue(info.value)
        self._rampstep_spin.setValue(info.rampstep)

    def read(self) -> dict:
        # mode intentionally excluded — changing mode requires device reconnect
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_spin.value(),
            "rampstep": self._rampstep_spin.value(),
        }


class _SGSPanel(QWidget):
    """Info + control panel for RohdeSchwarzSGS100A."""

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

        self._iq_combo = QComboBox()
        self._iq_combo.addItems(["on", "off"])
        form.addRow("IQ:", self._iq_combo)

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1e6, 20e9)
        self._freq_spin.setDecimals(0)
        self._freq_spin.setSuffix(" Hz")
        self._freq_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Frequency:", self._freq_spin)

        self._power_spin = QDoubleSpinBox()
        self._power_spin.setRange(-120.0, 25.0)
        self._power_spin.setDecimals(2)
        self._power_spin.setSuffix(" dBm")
        self._power_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        form.addRow("Power:", self._power_spin)

    def load(self, info: object) -> None:
        from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo

        assert isinstance(info, RohdeSchwarzSGS100AInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._iq_combo.setCurrentText(info.IQ)
        self._freq_spin.setValue(info.freq_Hz)
        self._power_spin.setValue(info.power_dBm)

    def read(self) -> dict:
        return {
            "output": self._output_combo.currentText(),
            "IQ": self._iq_combo.currentText(),
            "freq_Hz": self._freq_spin.value(),
            "power_dBm": self._power_spin.value(),
        }


# ---------------------------------------------------------------------------
# DeviceDialog
# ---------------------------------------------------------------------------

# DeviceInfo.type discriminator value → stack page index
_TYPE_TO_PAGE: dict[str, int] = {
    "FakeDevice": 1,
    "YOKOGS200": 2,
    "RohdeSchwarzSGS100A": 3,
}


class DeviceDialog(QDialog):
    """Modal dialog for device registration and per-device control."""

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Devices")
        self.resize(720, 480)

        root_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, stretch=1)

        # ── Left panel ───────────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        list_group = QGroupBox("Registered devices")
        list_layout = QVBoxLayout(list_group)
        self._device_list = QListWidget()
        self._device_list.currentItemChanged.connect(self._on_device_selected)
        list_layout.addWidget(self._device_list)

        drop_row = QHBoxLayout()
        self._drop_btn = QPushButton("Remove selected")
        self._drop_btn.clicked.connect(self._on_drop_clicked)
        drop_row.addWidget(self._drop_btn)
        drop_row.addStretch()
        list_layout.addLayout(drop_row)
        left_layout.addWidget(list_group)

        add_group = QGroupBox("Add device")
        add_form = QFormLayout(add_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. flux_coil")
        add_form.addRow("Name:", self._name_edit)

        self._type_combo = QComboBox()
        self._type_combo.addItems(list(_DEVICE_TYPES.keys()))
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        add_form.addRow("Type:", self._type_combo)

        self._address_edit = QLineEdit()
        self._address_edit.setPlaceholderText("GPIB0::1::INSTR")
        add_form.addRow("Address:", self._address_edit)

        self._add_btn = QPushButton("Add device")
        self._add_btn.clicked.connect(self._on_add_clicked)
        add_form.addRow("", self._add_btn)
        left_layout.addWidget(add_group)

        self._left_status = QLabel("")
        left_layout.addWidget(self._left_status)
        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ── Right panel ──────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._stack = QStackedWidget()

        # page 0: placeholder
        placeholder = QLabel("Select a device to view details")
        placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._stack.addWidget(placeholder)  # index 0

        # page 1: FakeDevice
        self._fake_panel = _FakeDevicePanel()
        self._stack.addWidget(self._fake_panel)  # index 1

        # page 2: YOKOGS200
        self._yoko_panel = _YOKOPanel()
        self._stack.addWidget(self._yoko_panel)  # index 2

        # page 3: RohdeSchwarzSGS100A
        self._sgs_panel = _SGSPanel()
        self._stack.addWidget(self._sgs_panel)  # index 3

        right_layout.addWidget(self._stack, stretch=1)

        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        apply_row.addStretch()
        apply_row.addWidget(self._apply_btn)
        right_layout.addLayout(apply_row)

        self._right_status = QLabel("")
        right_layout.addWidget(self._right_status)
        splitter.addWidget(right_widget)

        splitter.setSizes([240, 480])

        # ── Progress bar (idle height = 0) ────────────────────────────────
        self._pbar_stack = ProgressStack()
        root_layout.addWidget(self._pbar_stack)

        # ── Close button ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        self._close_btn = btn_box.button(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        root_layout.addWidget(btn_box)

        self._worker = None  # holds active _DeviceSetupWorker to prevent GC

        self._on_type_changed(self._type_combo.currentText())
        self._refresh_device_list()

    # ------------------------------------------------------------------
    # Left panel handlers
    # ------------------------------------------------------------------

    def _on_type_changed(self, type_name: str) -> None:
        _, requires_address = _DEVICE_TYPES.get(type_name, ("", True))
        self._address_edit.setEnabled(requires_address)

    def _on_add_clicked(self) -> None:
        name = self._name_edit.text().strip()
        type_name = self._type_combo.currentText()
        address = self._address_edit.text().strip()
        if not name:
            self._set_left_status("Device name cannot be empty", error=True)
            return
        try:
            device = _instantiate_device(type_name, address)
            self._ctrl.register_device(name, device)
            self._refresh_device_list(select_name=name)
            self._name_edit.clear()
            self._set_left_status(f"Added: {name} ({type_name})")
            logger.info(
                "DeviceDialog: register_device name=%r type=%r", name, type_name
            )
        except Exception as exc:
            self._set_left_status(str(exc), error=True)
            logger.warning("DeviceDialog: add failed: %r", exc)

    def _on_drop_clicked(self) -> None:
        name = self._current_device_name()
        if name is None:
            return
        try:
            self._ctrl.drop_device(name)
            self._refresh_device_list()
            self._stack.setCurrentIndex(0)
            self._apply_btn.setEnabled(False)
            self._set_left_status(f"Removed: {name}")
        except Exception as exc:
            self._set_left_status(str(exc), error=True)

    def _refresh_device_list(self, select_name: Optional[str] = None) -> None:
        """Rebuild the list; select_name overrides the previous selection."""
        current_name = (
            select_name if select_name is not None else self._current_device_name()
        )
        self._device_list.blockSignals(True)
        self._device_list.clear()
        devices = self._ctrl.list_devices()
        restore_idx = -1
        for i, (name, type_str) in enumerate(devices.items()):
            item = QListWidgetItem(f"{name}  [{type_str}]")
            item.setData(Qt.UserRole, name)  # type: ignore[attr-defined]
            self._device_list.addItem(item)
            if name == current_name:
                restore_idx = i
        self._device_list.blockSignals(False)
        if restore_idx >= 0:
            self._device_list.setCurrentRow(restore_idx)
        elif self._device_list.count() > 0:
            self._device_list.setCurrentRow(0)
        else:
            self._stack.setCurrentIndex(0)
            self._apply_btn.setEnabled(False)
            return
        # manually trigger right-panel refresh since blockSignals suppressed the signal
        selected = self._current_device_name()
        if selected is not None:
            self._load_device_info(selected)

    def _current_device_name(self) -> Optional[str]:
        item = self._device_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Right panel handlers
    # ------------------------------------------------------------------

    def _on_device_selected(
        self, current: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            self._stack.setCurrentIndex(0)
            self._apply_btn.setEnabled(False)
            return
        name: str = current.data(Qt.UserRole)  # type: ignore[attr-defined]
        self._load_device_info(name)

    def _load_device_info(self, name: str) -> None:
        try:
            info = self._ctrl.get_device_info(name)
        except Exception as exc:
            self._set_right_status(str(exc), error=True)
            self._stack.setCurrentIndex(0)
            self._apply_btn.setEnabled(False)
            return

        type_name: str = info.type
        page = _TYPE_TO_PAGE.get(type_name, 0)
        self._stack.setCurrentIndex(page)

        panel = self._stack.currentWidget()
        if page > 0 and hasattr(panel, "load"):
            try:
                panel.load(info)  # type: ignore[union-attr]
                self._apply_btn.setEnabled(True)
                self._set_right_status("")
            except Exception as exc:
                self._set_right_status(str(exc), error=True)
                self._apply_btn.setEnabled(False)
        else:
            # unknown device type — show placeholder
            self._stack.setCurrentIndex(0)
            self._apply_btn.setEnabled(False)
            self._set_right_status(f"Unknown device type: {type_name}", error=True)

    def _on_apply_clicked(self) -> None:
        name = self._current_device_name()
        if name is None:
            return

        page = self._stack.currentIndex()
        if page == 0:
            return
        panel = self._stack.currentWidget()
        if not hasattr(panel, "read"):
            return

        try:
            info = self._ctrl.get_device_info(name)
            updates = panel.read()  # type: ignore[union-attr]
            new_info = info.with_updates(**updates)
        except Exception as exc:
            self._set_right_status(str(exc), error=True)
            logger.warning("DeviceDialog: build info failed name=%r: %r", name, exc)
            return

        from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

        factory = QtProgressBarFactory(self._pbar_stack)
        self._set_apply_busy(True)
        self._set_right_status(f"Applying to {name}…")
        try:
            worker = self._ctrl.setup_device(name, new_info, pbar_factory=factory)
        except Exception as exc:
            self._set_apply_busy(False)
            self._set_right_status(str(exc), error=True)
            logger.warning("DeviceDialog: setup_device rejected name=%r: %r", name, exc)
            return

        self._worker = worker
        worker.finished.connect(self._on_apply_finished)
        worker.failed.connect(self._on_apply_failed)
        logger.info(
            "DeviceDialog: setup_device started name=%r updates=%r", name, updates
        )

    def _on_apply_finished(self, name: str) -> None:
        self._worker = None
        self._set_apply_busy(False)
        self._load_device_info(name)
        self._set_right_status(f"Applied to {name}")
        logger.info("DeviceDialog: setup_device finished name=%r", name)

    def _on_apply_failed(self, name: str, msg: str) -> None:
        self._worker = None
        self._set_apply_busy(False)
        self._set_right_status(msg, error=True)
        logger.warning("DeviceDialog: setup_device failed name=%r: %r", name, msg)

    def _set_apply_busy(self, busy: bool) -> None:
        enabled = not busy
        self._apply_btn.setEnabled(enabled)
        self._drop_btn.setEnabled(enabled)
        self._add_btn.setEnabled(enabled)
        if self._close_btn is not None:
            self._close_btn.setEnabled(enabled)
        # disable the current panel so user cannot modify fields during apply
        panel = self._stack.currentWidget()
        if panel is not None:
            panel.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_left_status(self, msg: str, error: bool = False) -> None:
        self._left_status.setText(msg)
        self._left_status.setStyleSheet(f"color: {'red' if error else 'green'};")

    def _set_right_status(self, msg: str, error: bool = False) -> None:
        self._right_status.setText(msg)
        color = "red" if error else ("green" if msg else "")
        self._right_status.setStyleSheet(f"color: {color};" if color else "")
