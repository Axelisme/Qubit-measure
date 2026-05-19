"""ProjectDialog — set result_dir, switch or create context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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


class ProjectDialog(QDialog):
    """Modal dialog for project setup: result_dir, context switch/new."""

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Project / Context")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        # ── result_dir ───────────────────────────────────────────────────
        dir_group = QGroupBox("Project directory")
        dir_form = QFormLayout(dir_group)
        dir_row = QHBoxLayout()
        self._result_dir_edit = QLineEdit()
        self._result_dir_edit.setPlaceholderText("/path/to/result_dir")
        dir_row.addWidget(self._result_dir_edit)
        browse_dir_btn = QPushButton("Browse…")
        browse_dir_btn.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(browse_dir_btn)
        dir_form.addRow("Result dir:", dir_row)
        self._setup_btn = QPushButton("Setup")
        self._setup_btn.clicked.connect(self._on_setup_clicked)
        dir_form.addRow("", self._setup_btn)
        layout.addWidget(dir_group)

        # ── context list ─────────────────────────────────────────────────
        ctx_group = QGroupBox("Contexts")
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
        new_group = QGroupBox("New context")
        new_form = QFormLayout(new_group)

        self._new_value_spin = QDoubleSpinBox()
        self._new_value_spin.setRange(-1e6, 1e6)
        self._new_value_spin.setDecimals(6)
        self._new_value_spin.setValue(0.0)
        new_form.addRow("Flux value:", self._new_value_spin)

        self._new_unit_edit = QLineEdit("A")
        self._new_unit_edit.setMaximumWidth(60)
        new_form.addRow("Unit:", self._new_unit_edit)

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

        # populate if already set up
        self._refresh_context_list()

    # ------------------------------------------------------------------

    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select result directory")
        if path:
            self._result_dir_edit.setText(path)

    def _on_setup_clicked(self) -> None:
        result_dir = self._result_dir_edit.text().strip()
        if not result_dir:
            self._set_status("Result dir cannot be empty", error=True)
            return
        try:
            self._ctrl.setup_project(result_dir)
            self._refresh_context_list()
            self._set_status(f"Project set up: {result_dir}")
            logger.info("ProjectDialog: setup result_dir=%r", result_dir)
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ProjectDialog: setup failed: %r", exc)

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
        value = self._new_value_spin.value()
        unit = self._new_unit_edit.text().strip() or "A"
        clone = self._clone_check.isChecked()
        try:
            self._ctrl.new_context(value=value, unit=unit, clone_from_current=clone)
            self._refresh_context_list()
            self._set_status(f"Created new context (value={value} {unit})")
            logger.info(
                "ProjectDialog: new_context value=%r unit=%r clone=%r",
                value,
                unit,
                clone,
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
