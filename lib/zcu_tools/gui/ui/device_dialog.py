"""DeviceDialog — register devices and get/set device values."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

# Known device types: display name → (class_path, requires_address)
_DEVICE_TYPES: dict[str, tuple[str, bool]] = {
    "FakeDevice": ("zcu_tools.device.fake.FakeDevice", False),
    "YOKOGS200": ("zcu_tools.device.yoko.YOKOGS200", True),
    "RohdeSchwarzSGS100A": ("zcu_tools.device.sgs100a.RohdeSchwarzSGS100A", True),
}


def _instantiate_device(type_name: str, address: str) -> object:
    """Import and construct a device by type name."""
    class_path, requires_address = _DEVICE_TYPES[type_name]
    module_path, cls_name = class_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    if requires_address:
        import pyvisa  # type: ignore[import-untyped]

        rm = pyvisa.ResourceManager()
        return cls(address, rm)
    return cls()


class DeviceDialog(QDialog):
    """Modal dialog for device registration and control."""

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Devices")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        # ── registered devices ───────────────────────────────────────────
        list_group = QGroupBox("Registered devices")
        list_layout = QVBoxLayout(list_group)
        self._device_list = QListWidget()
        self._device_list.setMaximumHeight(100)
        self._device_list.currentRowChanged.connect(self._on_device_selected)
        list_layout.addWidget(self._device_list)

        drop_row = QHBoxLayout()
        self._drop_btn = QPushButton("Remove selected")
        self._drop_btn.clicked.connect(self._on_drop_clicked)
        drop_row.addWidget(self._drop_btn)
        drop_row.addStretch()
        list_layout.addLayout(drop_row)
        layout.addWidget(list_group)

        # ── get / set value ──────────────────────────────────────────────
        ctrl_group = QGroupBox("Control selected device")
        ctrl_form = QFormLayout(ctrl_group)

        self._get_value_label = QLabel("—")
        get_row = QHBoxLayout()
        get_row.addWidget(self._get_value_label)
        get_btn = QPushButton("Get")
        get_btn.clicked.connect(self._on_get_value)
        get_row.addWidget(get_btn)
        ctrl_form.addRow("Current value:", get_row)

        set_row = QHBoxLayout()
        self._set_value_spin = QDoubleSpinBox()
        self._set_value_spin.setRange(-1e9, 1e9)
        self._set_value_spin.setDecimals(6)
        set_row.addWidget(self._set_value_spin)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._on_set_value)
        set_row.addWidget(set_btn)
        ctrl_form.addRow("Set value:", set_row)
        layout.addWidget(ctrl_group)

        # ── add new device ───────────────────────────────────────────────
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
        layout.addWidget(add_group)

        # ── status ───────────────────────────────────────────────────────
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._on_type_changed(self._type_combo.currentText())
        self._refresh_device_list()

    # ------------------------------------------------------------------

    def _on_type_changed(self, type_name: str) -> None:
        _, requires_address = _DEVICE_TYPES.get(type_name, ("", True))
        self._address_edit.setEnabled(requires_address)

    def _on_device_selected(self, row: int) -> None:
        _ = row  # selection tracked via currentItem()

    def _current_device_name(self) -> Optional[str]:
        item = self._device_list.currentItem()
        if item is None:
            return None
        return item.text().split(" [")[0]

    def _on_get_value(self) -> None:
        name = self._current_device_name()
        if name is None:
            self._set_status("Select a device first", error=True)
            return
        try:
            val = self._ctrl.get_device_value(name)
            self._get_value_label.setText(str(val))
            self._set_status(f"{name} = {val}")
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _on_set_value(self) -> None:
        name = self._current_device_name()
        if name is None:
            self._set_status("Select a device first", error=True)
            return
        value = self._set_value_spin.value()
        try:
            actual = self._ctrl.set_device_value(name, value)
            self._get_value_label.setText(str(actual))
            self._set_status(f"{name} set to {actual}")
            logger.info(
                "DeviceDialog: set_device_value name=%r value=%r → %r",
                name,
                value,
                actual,
            )
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _on_drop_clicked(self) -> None:
        name = self._current_device_name()
        if name is None:
            return
        try:
            self._ctrl.drop_device(name)
            self._refresh_device_list()
            self._set_status(f"Removed device: {name}")
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def _on_add_clicked(self) -> None:
        name = self._name_edit.text().strip()
        type_name = self._type_combo.currentText()
        address = self._address_edit.text().strip()
        if not name:
            self._set_status("Device name cannot be empty", error=True)
            return
        try:
            device = _instantiate_device(type_name, address)
            self._ctrl.register_device(name, device)
            self._refresh_device_list()
            self._name_edit.clear()
            self._set_status(f"Added device: {name} ({type_name})")
            logger.info(
                "DeviceDialog: register_device name=%r type=%r", name, type_name
            )
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("DeviceDialog: add failed: %r", exc)

    def _refresh_device_list(self) -> None:
        self._device_list.clear()
        devices = self._ctrl.list_devices()
        for name, type_str in devices.items():
            self._device_list.addItem(f"{name} [{type_str}]")

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
