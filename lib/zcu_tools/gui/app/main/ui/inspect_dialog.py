"""InspectDialog — overview of MetaDict parameters (editable) and ModuleLibrary (read-only)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

import yaml
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
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

from zcu_tools.gui.app.main.adapter import CfgSchema
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
    SessionPayload,
)
from zcu_tools.gui.session.services.context import MdValueError, MlEntryValidationError

from .cfg_form import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.event_bus import EventBus

logger = logging.getLogger(__name__)

_MONO_FONT = QFont("Monospace")
_MONO_FONT.setStyleHint(QFont.StyleHint.Monospace)

_MAX_VALUE_LEN = 80


_MlItemKind = Literal["module", "waveform"]


class _MlModifyDialog(QDialog):
    """Edit an EXISTING ModuleLibrary entry (fixed shape).

    Name and type/style are read-only — modify never changes shape (to change
    shape, delete the entry and create a new one from a role). Creating new
    entries goes through ``_MlCreateDialog`` / ``create_from_role``.
    """

    def __init__(
        self,
        ctrl: "Controller",
        item_kind: _MlItemKind,
        name: str,
        cfg: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not name or cfg is None:
            raise ValueError("Modify requires both name and cfg.")

        self._ctrl = ctrl
        self._item_kind = item_kind
        self._name = name
        self.setWindowTitle(f"Modify {item_kind.capitalize()}")
        self.setMinimumSize(560, 500)

        layout = QVBoxLayout(self)

        # ADR-0006: modify an existing ml entry is the UI twin of the agent's
        # open(from_name) → edit → commit flow. Open a committable session loaded
        # from the live ml; Save commits via the single write authority. No
        # UI-side schema build / lowering / raw write.
        self._cfg_editor_owner = f"inspect-{uuid.uuid4().hex[:8]}"
        editor_id, _ = self._ctrl.open_cfg_editor(
            item_kind, from_name=name, gc=False, owner_key=self._cfg_editor_owner
        )
        root = self._ctrl.get_cfg_editor_root(editor_id)
        discriminator = self._read_discriminator(
            CfgSchema(spec=root.spec, value=root.get_value())
        )

        form = QFormLayout()
        form.addRow("Name:", QLabel(name))
        form.addRow(
            "Type:" if item_kind == "module" else "Style:", QLabel(discriminator)
        )
        layout.addLayout(form)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # CfgFormWidget attaches to the service-owned LiveModel (ADR-0008); edits
        # land in that draft and enter the live ModuleLibrary only on commit.
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

        self._form_widget.validity_changed.connect(self._validate)
        cancel_btn.clicked.connect(self.reject)
        self._save_btn.clicked.connect(self._on_save)
        # Detach + tear down the service-owned model when the dialog closes.
        self.finished.connect(self._close_cfg_editor)

        self._form_widget.attach(root)
        self._validate()

    def _close_cfg_editor(self, *_: Any) -> None:
        # Detach the widget, then tear down the service-owned model (ADR-0008).
        self._form_widget.detach()
        editor_id = self._ctrl.editor_id_for_owner(self._cfg_editor_owner)
        if editor_id is not None:
            self._ctrl.teardown_cfg_editor(editor_id)

    @property
    def _discriminator_label(self) -> str:
        return "type" if self._item_kind == "module" else "style"

    def _read_discriminator(self, schema: CfgSchema) -> str:
        value = schema.value.fields[self._discriminator_label]
        raw_value = getattr(value, "value", None)
        if not isinstance(raw_value, str):
            raise RuntimeError(
                f"Invalid {self._discriminator_label} value {raw_value!r}"
            )
        return raw_value

    def _validate(self, *_: Any) -> None:
        if self._form_widget.is_valid():
            self._warning_label.setText("")
            self._save_btn.setEnabled(True)
        else:
            self._warning_label.setText("Configuration is invalid.")
            self._save_btn.setEnabled(False)

    def _on_save(self) -> None:
        # ADR-0006: commit the service-owned session through the single write
        # authority (lowering + register happen there). No UI-side lowering.
        editor_id = self._ctrl.editor_id_for_owner(self._cfg_editor_owner)
        if editor_id is None:
            return
        try:
            self._ctrl.commit_cfg_editor(editor_id, self._name)
        except MlEntryValidationError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return

        self.accept()

    def clear(self) -> None:
        # Teardown of the service-owned model happens in _close_cfg_editor (also
        # wired to `finished`); detach is idempotent, so this just ensures the
        # widget is unbound.
        self._form_widget.detach()


class _MlCreateDialog(QDialog):
    """Create a new ml module/waveform from a role (the single create path).

    One-shot: pick a role + a name → the role's factory seeds the value
    (md-linked defaults for named roles, structural zeros for ``:blank`` roles)
    and registers it directly into ml (no editable form here). To change the
    entry afterwards, use Modify.
    """

    def __init__(self, ctrl: "Controller", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Create ModuleLibrary entry")

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Pick a role and a name. Named roles seed md-linked defaults; "
            "'Blank: …' roles seed an empty shape. Edit afterwards via Modify."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._role_combo = QComboBox()
        catalog = self._ctrl.get_role_catalog()
        # Modules then waveforms, each labelled; role entry stashed on the item.
        for kind in ("module", "waveform"):
            for entry in catalog.entries_for(kind):  # type: ignore[arg-type]
                self._role_combo.addItem(f"{entry.label}  ({kind})", userData=entry)
        form.addRow("Role:", self._role_combo)
        self._name_edit = QLineEdit()
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(create_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        create_btn.clicked.connect(self._on_create)
        cancel_btn.clicked.connect(self.reject)

    def _on_create(self) -> None:
        entry = self._role_combo.currentData()
        if entry is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Entry name must not be empty.")
            return
        try:
            self._ctrl.create_from_role(entry.item_kind, entry.role_id, name)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            QMessageBox.critical(self, "Create failed", str(exc))
            return
        self.accept()


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

        # --- top toolbar: status on the left, Refresh + Close on the right ---
        toolbar = QHBoxLayout()
        self._status_label = QLabel("No context")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        # --- tabs ---
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_md_tab(), "Parameters")
        self._tabs.addTab(self._build_ml_tab(), "Modules")
        layout.addWidget(self._tabs)

        # Subscribe to EventBus for auto-refresh
        bus.subscribe(ContextSwitchedPayload, self._on_bus_refresh)
        bus.subscribe(MdChangedPayload, self._on_bus_refresh)
        bus.subscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = True
        self.finished.connect(self._cleanup_bus_subscriptions)
        self.destroyed.connect(self._cleanup_bus_subscriptions)

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
        self._create_btn = QPushButton("Create...")
        self._modify_ml_btn = QPushButton("Modify...")
        self._modify_ml_btn.setEnabled(False)
        self._rename_ml_btn = QPushButton("Rename...")
        self._rename_ml_btn.setEnabled(False)
        self._del_ml_btn = QPushButton("Delete")
        self._del_ml_btn.setEnabled(False)

        btn_layout.addWidget(self._create_btn)
        btn_layout.addWidget(self._modify_ml_btn)
        btn_layout.addWidget(self._rename_ml_btn)
        btn_layout.addWidget(self._del_ml_btn)
        left_layout.addLayout(btn_layout)

        self._create_btn.clicked.connect(self._on_create_clicked)
        self._modify_ml_btn.clicked.connect(self._on_modify_ml_clicked)
        self._rename_ml_btn.clicked.connect(self._on_rename_ml_clicked)
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

    def _on_set_clicked(self) -> None:
        key = self._edit_key.text().strip()
        if not key:
            return
        try:
            value = self._ctrl.coerce_md_value(key, self._edit_value.text())
        except MdValueError as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
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
        self._modify_ml_btn.setEnabled(False)
        self._rename_ml_btn.setEnabled(False)
        self._del_ml_btn.setEnabled(False)
        if current is None:
            self._ml_text.setPlainText("")
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            self._ml_text.setPlainText("")
            return

        self._modify_ml_btn.setEnabled(True)
        self._rename_ml_btn.setEnabled(True)
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

    def _on_create_clicked(self) -> None:
        dlg = _MlCreateDialog(self._ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()

    def _current_ml_item_data(self) -> Optional[tuple[str, str]]:
        current = self._ml_tree.currentItem()
        if current is None:
            return None
        data = current.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if data is None:
            return None
        group, name = data
        if group not in {"modules", "waveforms"}:
            raise RuntimeError(f"Unsupported ModuleLibrary group {group!r}")
        if not isinstance(name, str):
            raise RuntimeError(f"Invalid ModuleLibrary item name {name!r}")
        return group, name

    def _on_modify_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        group, name = data
        ml = self._ctrl.get_current_ml()
        if ml is None:
            return

        if group == "modules":
            dlg = _MlModifyDialog(
                self._ctrl, "module", name=name, cfg=ml.modules[name], parent=self
            )
        else:
            dlg = _MlModifyDialog(
                self._ctrl, "waveform", name=name, cfg=ml.waveforms[name], parent=self
            )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(lambda _: dlg.clear())
        dlg.open()

    def _on_delete_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
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

    def _on_rename_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        group, name = data
        new, ok = QInputDialog.getText(
            self, "Rename", f"New name for {group[:-1]} '{name}':", text=name
        )
        new = new.strip()
        if not ok or not new or new == name:
            return
        try:
            if group == "modules":
                self._ctrl.rename_ml_module(name, new)
            else:
                self._ctrl.rename_ml_waveform(name, new)
        except Exception as exc:  # noqa: BLE001 — surface failure to the user
            QMessageBox.critical(self, "Rename failed", str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            logger.debug("_cleanup_bus_subscriptions called but already inactive")
            return
        self._bus.unsubscribe(ContextSwitchedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MdChangedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = False

    def _on_bus_refresh(self, payload: SessionPayload) -> None:
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
