"""ProjectDialog — project setup: chip/qub name derivation + result_dir + context switch."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

# Unit reported for each device type when no live query is possible
_DEVICE_DEFAULT_UNITS: dict[str, str] = {
    "FakeDevice": "A",
    "YOKOGS200": "A",  # refined at runtime by get_mode()
}


def _detect_unit(device_name: str) -> str:
    """Try to read unit from the live device; fall back to defaults."""
    try:
        from zcu_tools.device import GlobalDeviceManager

        dev = GlobalDeviceManager.get_device(device_name)
        dev_type = type(dev).__name__
        if dev_type == "YOKOGS200":
            mode = dev.get_mode()  # type: ignore[attr-defined]
            return "V" if mode == "voltage" else "A"
        return _DEVICE_DEFAULT_UNITS.get(dev_type, "A")
    except Exception:
        return "A"


class ProjectDialog(QDialog):
    """Modal dialog for project setup: chip/qub name → result_dir derivation, context switch/new."""

    def __init__(
        self,
        controller: "Controller",
        parent: Optional[QWidget] = None,
        startup_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._startup_mode = startup_mode
        self.setWindowTitle("Project Setup" if startup_mode else "Project / Context")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        # ── chip / qub / res name section ────────────────────────────────
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

        layout.addWidget(name_group)

        # ── derived paths section ─────────────────────────────────────────
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

        layout.addWidget(paths_group)

        # ── apply startup context button ──────────────────────────────────
        self._apply_btn = QPushButton(
            "Apply & Setup" if startup_mode else "Apply startup context"
        )
        self._apply_btn.clicked.connect(self._on_apply_startup_clicked)
        layout.addWidget(self._apply_btn)

        # ── context list ─────────────────────────────────────────────────
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
        layout.addWidget(ctx_group)

        # ── new context ───────────────────────────────────────────────────
        new_group = QGroupBox("New context (requires file-backed project)")
        new_form = QFormLayout(new_group)

        # device selector — populated from GlobalDeviceManager on open
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
        layout.addWidget(new_group)

        # ── status label ─────────────────────────────────────────────────
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # ── close button ─────────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # trigger path derivation and populate lists
        self._on_names_changed()
        self._refresh_device_list()
        self._refresh_context_list()

    # ------------------------------------------------------------------

    def _on_names_changed(self) -> None:
        chip = self._chip_edit.text().strip()
        qub = self._qub_edit.text().strip()
        if chip and qub:
            cwd = os.getcwd()
            self._result_dir_edit.setText(os.path.join(cwd, "result", chip, qub))
            self._db_path_edit.setText(os.path.join(cwd, "Database", chip, qub))

    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select result directory")
        if path:
            self._result_dir_edit.setText(path)

    def _on_browse_db(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select database directory")
        if path:
            self._db_path_edit.setText(path)

    def _refresh_device_list(self) -> None:
        """Repopulate the device combo from GlobalDeviceManager."""
        try:
            from zcu_tools.device import GlobalDeviceManager

            devices = GlobalDeviceManager.get_all_devices()
        except Exception:
            devices = {}

        current = self._device_combo.currentText()
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("(none)")
        for name in sorted(devices):
            dev_type = type(devices[name]).__name__
            self._device_combo.addItem(f"{name}  [{dev_type}]", userData=name)
        self._device_combo.blockSignals(False)

        # restore previous selection if still present
        idx = self._device_combo.findText(current)
        self._device_combo.setCurrentIndex(max(idx, 0))
        self._on_device_changed(self._device_combo.currentText())

    def _on_device_changed(self, _text: str) -> None:
        device_name = self._device_combo.currentData()
        if device_name:
            unit = _detect_unit(device_name)
            self._unit_label.setText(unit)
        else:
            self._unit_label.setText("—")

    def _on_apply_startup_clicked(self) -> None:
        from zcu_tools.meta_tool import MetaDict, ModuleLibrary

        chip = self._chip_edit.text().strip() or "unknown_chip"
        qub = self._qub_edit.text().strip() or "unknown_qubit"
        res = self._res_edit.text().strip() or "unknown_resonator"
        result_dir = self._result_dir_edit.text().strip()
        db_path = self._db_path_edit.text().strip()
        md = MetaDict()
        ml = ModuleLibrary()
        try:
            self._ctrl.set_startup_context(
                md,
                ml,
                chip_name=chip,
                qub_name=qub,
                res_name=res,
                result_dir=result_dir,
                database_path=db_path,
            )
            self._set_status(f"Startup context applied: {chip}/{qub} (res={res})")
            logger.info(
                "ProjectDialog: startup context applied chip=%r qub=%r res=%r",
                chip,
                qub,
                res,
            )
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ProjectDialog: apply startup failed: %r", exc)
            return

        # auto-setup file-backed project from result_dir (silently skip if dir missing)
        if result_dir:
            try:
                self._ctrl.setup_project(result_dir)
                self._refresh_context_list()
                logger.info("ProjectDialog: auto-setup result_dir=%r", result_dir)
            except Exception as exc:
                logger.info("ProjectDialog: auto-setup skipped (%r)", exc)

    def _on_switch_clicked(self) -> None:
        item = self._ctx_list.currentItem()
        if item is None:
            self._set_status("Select a context first", error=True)
            return
        label = item.text()
        try:
            self._ctrl.use_context(label)
            self._set_status(f"Switched to context: {label}")
            logger.info("ProjectDialog: switched to context=%r", label)
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ProjectDialog: switch failed: %r", exc)

    def _on_new_ctx_clicked(self) -> None:
        clone = self._clone_check.isChecked()
        device_name: Optional[str] = self._device_combo.currentData()
        unit = self._unit_label.text() if device_name else "A"
        # read current value from device if available
        value: Optional[float] = None
        if device_name:
            try:
                from zcu_tools.device import GlobalDeviceManager

                dev = GlobalDeviceManager.get_device(device_name)
                value = float(dev.get_value())  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            self._ctrl.new_context(value=value, unit=unit, clone_from_current=clone)
            self._refresh_context_list()
            val_str = f"{value} {unit}" if value is not None else "NoValue"
            self._set_status(f"Created new context ({val_str})")
            logger.info(
                "ProjectDialog: new_context value=%r unit=%r clone=%r device=%r",
                value,
                unit,
                clone,
                device_name,
            )
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ProjectDialog: new_context failed: %r", exc)

    def _refresh_context_list(self) -> None:
        self._ctx_list.clear()
        labels = self._ctrl.get_context_labels()
        for label in labels:
            self._ctx_list.addItem(label)
        active = self._ctrl.get_active_context_label()
        if active:
            items = self._ctx_list.findItems(active, Qt.MatchExactly)  # type: ignore[attr-defined]
            if items:
                self._ctx_list.setCurrentItem(items[0])

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
