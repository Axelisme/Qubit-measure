"""InspectDialog — overview of MetaDict parameters (editable) and ModuleLibrary (read-only)."""

from __future__ import annotations

import ast
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import yaml
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.adapter import CfgSchema, make_default_value, schema_to_dict
from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
from zcu_tools.gui.event_bus import GuiEvent, Payload
from zcu_tools.gui.specs.waveform import make_waveform_spec_by_style
from zcu_tools.gui.ui.cfg_form import CfgFormWidget
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.event_bus import EventBus

logger = logging.getLogger(__name__)

_MONO_FONT = QFont("Monospace")
_MONO_FONT.setStyleHint(QFont.StyleHint.Monospace)

_MAX_VALUE_LEN = 80


class _MlAddDialog(QDialog):
    """Helper dialog for adding a new Module or Waveform to the ModuleLibrary."""

    def __init__(
        self,
        ctrl: "Controller",
        item_kind: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._item_kind = item_kind  # "module" or "waveform"
        self.setWindowTitle(f"Add {item_kind.capitalize()}")
        self.setMinimumSize(560, 500)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._type_combo = QComboBox()

        form.addRow("Name:", self._name_edit)
        form.addRow("Type:" if item_kind == "module" else "Style:", self._type_combo)
        layout.addLayout(form)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._form_widget = CfgFormWidget()
        self._scroll.setWidget(self._form_widget)
        layout.addWidget(self._scroll, stretch=1)

        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: red;")
        layout.addWidget(self._warning_label)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        if item_kind == "module":
            self._type_combo.addItems(list(_MODULE_SPEC_FACTORIES.keys()))
        else:
            self._type_combo.addItems(
                ["const", "cosine", "gauss", "drag", "flat_top", "arb"]
            )

        self._name_edit.textChanged.connect(self._validate)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        self._form_widget.validity_changed.connect(self._validate)
        cancel_btn.clicked.connect(self.reject)
        self._save_btn.clicked.connect(self._on_save)

        self._on_type_changed(self._type_combo.currentText())
        self._validate()

    def _on_type_changed(self, type_str: str) -> None:
        if self._item_kind == "module":
            spec = _MODULE_SPEC_FACTORIES[type_str]()
        else:
            spec = make_waveform_spec_by_style(type_str)
        value = make_default_value(spec)
        schema = CfgSchema(spec=spec, value=value)
        self._form_widget.populate(schema, self._ctrl)
        self._validate()

    def _validate(self, *_: Any) -> None:
        name = self._name_edit.text().strip()
        ml = self._ctrl.get_current_ml()
        valid = True
        warn_text = ""

        if not name:
            valid = False
        else:
            if self._item_kind == "module" and ml is not None and name in ml.modules:
                warn_text = "Name already exists in modules!"
                valid = False
            elif (
                self._item_kind == "waveform"
                and ml is not None
                and name in ml.waveforms
            ):
                warn_text = "Name already exists in waveforms!"
                valid = False

        if valid and not self._form_widget.is_valid():
            valid = False
            warn_text = "Configuration is invalid."

        self._warning_label.setText(warn_text)
        self._save_btn.setEnabled(valid)

    def _on_save(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return

        try:
            schema = self._form_widget.read_schema()
            ml = self._ctrl.get_current_ml()
            raw_dict = schema_to_dict(schema, ml)

            if self._item_kind == "module":
                obj = ModuleCfgFactory.from_raw(raw_dict, ml=ml)
                self._ctrl.set_ml_module(name, obj)
            else:
                obj = WaveformCfgFactory.from_raw(raw_dict, ml=ml)
                self._ctrl.set_ml_waveform(name, obj)

            self.accept()
        except Exception as e:
            logger.exception("Error saving %s '%s'", self._item_kind, name)
            QMessageBox.critical(self, "Validation Error", str(e))


class InspectDialog(QDialog):
    """Non-modal dialog showing MetaDict and ModuleLibrary contents."""

    def __init__(
        self,
        ctrl: "Controller",
        bus: "EventBus",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._bus = bus
        self.setWindowTitle("Inspect Context")
        self.resize(700, 500)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.Window  # type: ignore[attr-defined]
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- tabs ---
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_md_tab(), "Parameters")
        self._tabs.addTab(self._build_ml_tab(), "Modules")
        layout.addWidget(self._tabs)

        # --- bottom bar ---
        bottom = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        bottom.addWidget(refresh_btn)
        bottom.addStretch()
        self._status_label = QLabel("No context")
        bottom.addWidget(self._status_label)
        layout.addLayout(bottom)

        # Subscribe to EventBus for auto-refresh
        bus.subscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_refresh)
        bus.subscribe(GuiEvent.MD_CHANGED, self._on_bus_refresh)
        bus.subscribe(GuiEvent.ML_CHANGED, self._on_bus_refresh)
        # Unsubscribe when dialog is destroyed
        self.destroyed.connect(
            lambda: bus.unsubscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_refresh)
        )
        self.destroyed.connect(
            lambda: bus.unsubscribe(GuiEvent.MD_CHANGED, self._on_bus_refresh)
        )
        self.destroyed.connect(
            lambda: bus.unsubscribe(GuiEvent.ML_CHANGED, self._on_bus_refresh)
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_md_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)

        self._md_table = QTableWidget(0, 2)
        self._md_table.setHorizontalHeaderLabels(["Key", "Value"])
        header = self._md_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._md_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._md_table.setAlternatingRowColors(True)
        self._md_table.setSortingEnabled(True)
        self._md_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._md_table.cellClicked.connect(self._on_md_row_clicked)
        layout.addWidget(self._md_table)

        # Edit bar
        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Key:"))
        self._edit_key = QLineEdit()
        self._edit_key.setPlaceholderText("key")
        self._edit_key.setFixedWidth(120)
        self._edit_key.textChanged.connect(self._on_edit_key_changed)
        edit_row.addWidget(self._edit_key)
        edit_row.addWidget(QLabel("Value:"))
        self._edit_value = QLineEdit()
        self._edit_value.setPlaceholderText("value (Python literal or plain string)")
        edit_row.addWidget(self._edit_value)
        self._set_btn = QPushButton("Set")
        self._set_btn.setEnabled(False)
        self._set_btn.clicked.connect(self._on_set_clicked)
        edit_row.addWidget(self._set_btn)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        edit_row.addWidget(self._delete_btn)
        layout.addLayout(edit_row)

        return w

    def _build_ml_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)

        # left: tree (modules / waveforms groups)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._ml_tree = QTreeWidget()
        self._ml_tree.setHeaderHidden(True)
        self._ml_tree.setRootIsDecorated(True)
        self._ml_tree.currentItemChanged.connect(self._on_ml_item_changed)
        left_layout.addWidget(self._ml_tree)

        btn_layout = QHBoxLayout()
        self._add_mod_btn = QPushButton("Add Module...")
        self._add_wav_btn = QPushButton("Add Waveform...")
        self._del_ml_btn = QPushButton("Delete")
        self._del_ml_btn.setEnabled(False)

        btn_layout.addWidget(self._add_mod_btn)
        btn_layout.addWidget(self._add_wav_btn)
        btn_layout.addWidget(self._del_ml_btn)
        left_layout.addLayout(btn_layout)

        self._add_mod_btn.clicked.connect(self._on_add_module_clicked)
        self._add_wav_btn.clicked.connect(self._on_add_waveform_clicked)
        self._del_ml_btn.clicked.connect(self._on_delete_ml_clicked)

        splitter.addWidget(left_panel)

        # right: YAML text
        self._ml_text = QPlainTextEdit()
        self._ml_text.setReadOnly(True)
        self._ml_text.setFont(_MONO_FONT)
        splitter.addWidget(self._ml_text)

        splitter.setSizes([200, 450])
        layout.addWidget(splitter)
        return w

    # ------------------------------------------------------------------
    # Populate helpers
    # ------------------------------------------------------------------

    def _populate_md(self, md: Any) -> None:
        self._md_table.setSortingEnabled(False)
        self._md_table.setRowCount(0)

        if md is None:
            self._md_table.setSortingEnabled(True)
            return

        for key, value in md.items():
            row = self._md_table.rowCount()
            self._md_table.insertRow(row)

            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)  # type: ignore[attr-defined]

            value_str = str(value)
            display_str = (
                value_str
                if len(value_str) <= _MAX_VALUE_LEN
                else value_str[:_MAX_VALUE_LEN] + "…"
            )
            value_item = QTableWidgetItem(display_str)
            value_item.setFlags(  # type: ignore[arg-type]
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable  # type: ignore[attr-defined]
            )
            if len(value_str) > _MAX_VALUE_LEN:
                value_item.setToolTip(value_str)

            self._md_table.setItem(row, 0, key_item)
            self._md_table.setItem(row, 1, value_item)

        self._md_table.resizeColumnToContents(0)
        self._md_table.setSortingEnabled(True)

    def _populate_ml(self, ml: Any) -> None:
        # Remember current selection to restore after rebuild
        prev_item = self._ml_tree.currentItem()
        prev_data = (
            prev_item.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
            if prev_item is not None
            else None
        )

        self._ml_tree.blockSignals(True)
        self._ml_tree.clear()

        if ml is None:
            self._ml_tree.blockSignals(False)
            self._ml_text.setPlainText("")
            return

        bold = QFont()
        bold.setBold(True)

        restore_item: Optional[QTreeWidgetItem] = None
        for group_label, store in [
            ("modules", ml.modules),
            ("waveforms", ml.waveforms),
        ]:
            if not store:
                continue
            group_item = QTreeWidgetItem(self._ml_tree, [group_label])
            group_item.setFont(0, bold)
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # type: ignore[attr-defined]
            group_item.setExpanded(True)
            for name in sorted(store.keys()):
                child = QTreeWidgetItem(group_item, [name])
                child.setData(0, Qt.ItemDataRole.UserRole, (group_label, name))  # type: ignore[attr-defined]
                if prev_data == (group_label, name):
                    restore_item = child

        self._ml_tree.blockSignals(False)

        if restore_item is not None:
            self._ml_tree.setCurrentItem(restore_item)
        else:
            self._ml_text.setPlainText("")

    def _on_md_row_clicked(self, row: int, _col: int) -> None:
        key_item = self._md_table.item(row, 0)
        val_item = self._md_table.item(row, 1)
        if key_item is None:
            return
        self._edit_key.setText(key_item.text())
        # Use full value from tooltip if truncated, else display text
        full_val = (
            val_item.toolTip()
            if (val_item and val_item.toolTip())
            else (val_item.text() if val_item else "")
        )
        self._edit_value.setText(full_val)

    def _on_edit_key_changed(self, text: str) -> None:
        has_key = bool(text.strip())
        self._set_btn.setEnabled(has_key)
        self._delete_btn.setEnabled(has_key)

    @staticmethod
    def _parse_value(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    def _on_set_clicked(self) -> None:
        key = self._edit_key.text().strip()
        if not key:
            return
        value = self._parse_value(self._edit_value.text())
        self._ctrl.set_md_attr(key, value)
        self._populate_md(self._ctrl.get_current_md())
        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.setText(f"Last updated: {now}")

    def _on_delete_clicked(self) -> None:
        key = self._edit_key.text().strip()
        if not key:
            return
        self._ctrl.del_md_attr(key)
        self._edit_key.clear()
        self._edit_value.clear()
        self._populate_md(self._ctrl.get_current_md())
        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.setText(f"Last updated: {now}")

    def _on_ml_item_changed(
        self, current: Optional[QTreeWidgetItem], _previous: Any
    ) -> None:
        self._del_ml_btn.setEnabled(False)
        if current is None:
            self._ml_text.setPlainText("")
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            self._ml_text.setPlainText("")
            return

        self._del_ml_btn.setEnabled(True)

        group, name = data
        ml = self._ctrl.get_current_ml()
        if ml is None:
            self._ml_text.setPlainText("")
            return

        store = ml.modules if group == "modules" else ml.waveforms
        cfg = store[name]
        text = yaml.dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
        self._ml_text.setPlainText(text)

    def _on_add_module_clicked(self) -> None:
        dlg = _MlAddDialog(self._ctrl, "module", self)
        dlg.exec()

    def _on_add_waveform_clicked(self) -> None:
        dlg = _MlAddDialog(self._ctrl, "waveform", self)
        dlg.exec()

    def _on_delete_ml_clicked(self) -> None:
        current = self._ml_tree.currentItem()
        if current is None:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            return

        group, name = data
        ans = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {group[:-1]} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        if group == "modules":
            self._ctrl.del_ml_module(name)
        else:
            self._ctrl.del_ml_waveform(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _on_bus_refresh(self, payload: Payload) -> None:
        """EventBus subscriber wrapper; payload is ignored, delegates to refresh()."""
        del payload
        self.refresh()

    def refresh(self) -> None:
        md = self._ctrl.get_current_md()
        ml = self._ctrl.get_current_ml()

        self._populate_md(md)
        self._populate_ml(ml)

        if md is None and ml is None:
            self._status_label.setText("No context")
        else:
            now = datetime.now().strftime("%H:%M:%S")
            self._status_label.setText(f"Last updated: {now}")

        logger.debug("InspectDialog: refreshed")
