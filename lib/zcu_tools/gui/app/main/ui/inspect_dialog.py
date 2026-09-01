"""InspectDialog — measure's context inspector.

Subclasses the app-agnostic ``InspectDialogBase`` (md tab + ml view/rename/delete)
and adds the dense measure-only md property grid plus the ml *create / modify*
path, which drags the CfgEditor (a measure concern) and so cannot live in the
session core. The md create dialog and the two ml edit dialogs stay here with it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QKeyEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.session.services.context import MdValueError, MlEntryValidationError
from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase
from zcu_tools.gui.session.ui.value_source_input import (
    SessionValueSourceInputHost,
    ValueSourceInputController,
)
from zcu_tools.gui.widgets.cfg import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_control import ContextControlPort

logger = logging.getLogger(__name__)


_MlItemKind = Literal["module", "waveform"]


class _MdTable(QTableWidget):
    """Measure Parameters table with a direct, table-only Delete shortcut."""

    def __init__(
        self, on_delete: Callable[[], None], parent: QWidget | None = None
    ) -> None:
        super().__init__(0, 2, parent)
        self._on_delete = on_delete

    def keyPressEvent(self, e: QKeyEvent | None) -> None:
        if e is None:
            super().keyPressEvent(e)
            return
        # A delegate editor receives its own key events, so Delete remains normal
        # text deletion while a cell QLineEdit is active. Restrict the shortcut to
        # the view or viewport itself rather than any other dialog child.
        focus = QApplication.focusWidget()
        table_has_focus = focus is self or focus is self.viewport()
        if (
            table_has_focus and e.key() == Qt.Key.Key_Delete  # type: ignore[attr-defined]
        ):
            self._on_delete()
            e.accept()
            return
        super().keyPressEvent(e)


class _MdCreateDialog(QDialog):
    """Retained non-modal dialog for one validated scalar MetaDict entry."""

    def __init__(
        self,
        ctrl: ContextControlPort,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("New Parameter")
        self.setModal(False)

        layout = QVBoxLayout(self)
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Key:"))
        self._key_edit = QLineEdit()
        self._key_edit.setPlaceholderText("key")
        input_row.addWidget(self._key_edit, stretch=1)
        input_row.addWidget(QLabel("Value:"))
        self._value_edit = QLineEdit()
        self._value_edit.setPlaceholderText("scalar value")
        self._value_source_input = ValueSourceInputController(
            self._value_edit,
            SessionValueSourceInputHost(self._ctrl),
            parent=self._value_edit,
        )
        self._value_source_input.resolve_failed.connect(  # type: ignore[attr-defined]
            self._value_edit.setToolTip
        )
        input_row.addWidget(self._value_edit, stretch=2)
        layout.addLayout(input_row)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        layout.addWidget(self._error_label)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self._create_btn = QPushButton("Create")
        self._cancel_btn = QPushButton("Cancel")
        action_row.addWidget(self._create_btn)
        action_row.addWidget(self._cancel_btn)
        layout.addLayout(action_row)

        self._create_btn.clicked.connect(self._on_create)
        self._cancel_btn.clicked.connect(self.reject)
        self.finished.connect(self._cleanup)
        self._key_edit.setFocus()

    def _cleanup(self, *_args: object) -> None:
        self._value_source_input.detach()

    def _on_create(self) -> None:
        key = self._key_edit.text().strip()
        if not key:
            self._show_error("Key must not be empty.")
            return

        try:
            value = self._ctrl.coerce_md_value(key, self._value_edit.text())
        except MdValueError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surface validation failures
            self._show_error(str(exc))
            return

        try:
            self._ctrl.create_md_attr(key, value)
        except Exception as exc:  # noqa: BLE001 — surface domain validation failures
            self._show_error(str(exc))
            return
        self.accept()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        QMessageBox.warning(self, "Invalid parameter", message)


class _MlModifyDialog(QDialog):
    """Edit an EXISTING ModuleLibrary entry (fixed shape).

    Name and type/style are read-only — modify never changes shape (to change
    shape, delete the entry and create a new one from a role). Creating new
    entries goes through ``_MlCreateDialog`` / ``create_from_role``.
    """

    def __init__(
        self,
        ctrl: Controller,
        item_kind: _MlItemKind,
        name: str,
        cfg: Any,
        parent: QWidget | None = None,
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
        draft = self._ctrl.get_cfg_editor_draft(editor_id)
        discriminator = self._read_discriminator(draft.snapshot())

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
        self._form_widget = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(ctrl)
        )
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

        self._form_widget.attach(draft)
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

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Create ModuleLibrary entry")

        # The (item_kind, name) of a successful create, read by the parent after
        # the dialog closes to chain straight into Modify. None until create wins.
        self.created: tuple[_MlItemKind, str] | None = None
        # True once the user has typed into the name field by hand: from then on
        # switching role must not clobber their name (least surprise). Qt only
        # fires textEdited on user keystrokes, never on programmatic setText, so
        # seeding the suggestion below does not set this flag.
        self._name_edited = False

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

        # Seed the name with the initial role's convention-based suggestion.
        initial = self._role_combo.currentData()
        if initial is not None:
            self._name_edit.setText(self._suggest_name(initial))

        self._name_edit.textEdited.connect(self._on_name_edited)
        self._role_combo.currentIndexChanged.connect(self._on_role_changed)

        btn_row = QHBoxLayout()
        create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(create_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        create_btn.clicked.connect(self._on_create)
        cancel_btn.clicked.connect(self.reject)

    def _suggest_name(self, entry: Any) -> str:
        """Convention-based name suggestion for ``entry``, de-duplicated.

        Blank roles carry no ``default_name`` -> empty (the user must name it).
        Otherwise append ``_2``/``_3``/… until the name is free in the live ml.
        """
        base = entry.default_name
        if not base:
            return ""
        name = base
        suffix = 2
        while self._ctrl.has_ml_entry(entry.item_kind, name):
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def _on_role_changed(self, _index: int) -> None:
        # Only re-suggest while the name is still the auto-filled one; once the
        # user has typed their own, leave it alone (least surprise).
        if self._name_edited:
            return
        entry = self._role_combo.currentData()
        if entry is not None:
            self._name_edit.setText(self._suggest_name(entry))

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
        self.created = (entry.item_kind, name)
        self.accept()


class InspectDialog(InspectDialogBase):
    """Measure inspector with dense md editing and measure-only ml actions.

    The base owns the autofluxdep-compatible md/ml surface; this subclass
    replaces only the measure md tab and injects CfgEditor-backed ml actions
    through the base's two template-method hooks.
    """

    def __init__(
        self,
        ctrl: Controller,
        bus: BaseEventBus,
        parent: QWidget | None = None,
    ) -> None:
        # The base only needs the shared context facet. This subclass keeps the
        # concrete app controller for measure-only CfgEditor and role-catalog
        # commands that deliberately stay outside session core.
        self._app_ctrl = ctrl
        self._new_md_dialog: _MdCreateDialog | None = None
        self._md_syncing = False
        super().__init__(ctrl.context_control, bus, parent=parent)

    def _build_md_tab(self) -> QWidget:
        """Build measure's dense inline-editable Parameters property grid.

        The shared base intentionally keeps the autofluxdep presentation and
        read-only hooks unchanged; only measure opts into this composition.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        self._md_table = _MdTable(self._on_md_delete_requested)
        self._md_table.setHorizontalHeaderLabels(["Key", "Value"])
        header = self._md_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._md_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self._md_table.setAlternatingRowColors(True)
        self._md_table.setSortingEnabled(True)
        self._md_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._md_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._md_table.setWordWrap(False)
        self._md_table.cellClicked.connect(self._on_md_row_clicked)
        self._md_table.itemChanged.connect(self._on_md_item_changed)
        self._md_table.itemSelectionChanged.connect(self._on_md_selection_changed)

        delegate = self._md_table.itemDelegate()
        if delegate is not None:
            delegate.closeEditor.connect(self._on_md_editor_closed)  # type: ignore[attr-defined]
        layout.addWidget(self._md_table)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self._new_btn = QPushButton("New")
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        action_row.addWidget(self._new_btn)
        action_row.addWidget(self._delete_btn)
        layout.addLayout(action_row)

        self._new_btn.clicked.connect(self._on_new_md_clicked)
        self._delete_btn.clicked.connect(self._on_md_delete_clicked)
        return widget

    def _populate_md(self, md: Any) -> None:
        """Populate the measure grid while retaining the selected key."""
        selected_key = self._selected_md_key()
        self._md_syncing = True
        self._md_table.blockSignals(True)
        sorting_enabled = self._md_table.isSortingEnabled()
        self._md_table.setSortingEnabled(False)
        try:
            self._md_table.setRowCount(0)
            if md is not None:
                for key, value in md.items():
                    row = self._md_table.rowCount()
                    self._md_table.insertRow(row)

                    key_text = str(key)
                    key_item = QTableWidgetItem(key_text)
                    key_item.setData(
                        Qt.ItemDataRole.UserRole,
                        key_text,  # type: ignore[attr-defined]
                    )
                    key_item.setFlags(  # type: ignore[arg-type]
                        Qt.ItemFlag.ItemIsEnabled  # type: ignore[attr-defined]
                        | Qt.ItemFlag.ItemIsSelectable  # type: ignore[attr-defined]
                        | Qt.ItemFlag.ItemIsEditable  # type: ignore[attr-defined]
                    )

                    value_text = str(value)
                    value_item = QTableWidgetItem(value_text)
                    value_item.setData(
                        Qt.ItemDataRole.UserRole,
                        value_text,  # type: ignore[attr-defined]
                    )
                    value_item.setFlags(  # type: ignore[arg-type]
                        Qt.ItemFlag.ItemIsEnabled  # type: ignore[attr-defined]
                        | Qt.ItemFlag.ItemIsSelectable  # type: ignore[attr-defined]
                        | Qt.ItemFlag.ItemIsEditable  # type: ignore[attr-defined]
                    )
                    if len(value_text) > 80:
                        value_item.setToolTip(value_text)

                    self._md_table.setItem(row, 0, key_item)
                    self._md_table.setItem(row, 1, value_item)

            self._md_table.resizeColumnToContents(0)
        finally:
            self._md_table.setSortingEnabled(sorting_enabled)
            self._md_table.blockSignals(False)
            self._md_syncing = False

        if selected_key is not None:
            for row in range(self._md_table.rowCount()):
                item = self._md_table.item(row, 0)
                if item is not None and item.text() == selected_key:
                    self._md_table.selectRow(row)
                    break
        self._on_md_selection_changed()

    def _selected_md_key(self) -> str | None:
        row = self._md_table.currentRow()
        if row < 0:
            return None
        selection = self._md_table.selectionModel()
        if selection is None or not selection.hasSelection():
            return None
        item = self._md_table.item(row, 0)
        if item is None:
            return None
        key = item.text().strip()
        return key or None

    def _on_md_selection_changed(self) -> None:
        self._delete_btn.setEnabled(self._selected_md_key() is not None)

    def _on_md_row_clicked(self, row: int, _column: int) -> None:
        # Keep selection row-oriented; editing starts through the table's normal
        # delegate triggers, not through a second edit bar.
        if 0 <= row < self._md_table.rowCount():
            self._md_table.selectRow(row)

    def _on_md_item_changed(self, item: QTableWidgetItem) -> None:
        if self._md_syncing:
            return
        row = item.row()
        if row < 0:
            return
        if item.column() == 0:
            self._commit_md_key(item, row)
        elif item.column() == 1:
            self._commit_md_value(item, row)

    def _commit_md_key(self, item: QTableWidgetItem, row: int) -> None:
        old_key = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if not isinstance(old_key, str):
            return
        new_key = item.text().strip()
        if not new_key:
            self._reject_md_edit(item, old_key, "Key must not be empty.")
            return
        if new_key == old_key:
            self._finish_md_edit(old_key)
            return

        try:
            self._ctrl.rename_md_attr(old_key, new_key)
        except Exception as exc:  # noqa: BLE001 — surface domain validation failures
            self._reject_md_edit(item, old_key, str(exc))
            return
        self._refresh_md_after_edit()
        self._finish_md_edit(new_key)

    def _commit_md_value(self, item: QTableWidgetItem, row: int) -> None:
        key_item = self._md_table.item(row, 0)
        if key_item is None:
            return
        key = key_item.text().strip()
        old_value = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if not isinstance(old_value, str):
            old_value = item.text()
        try:
            value = self._ctrl.coerce_md_value(key, item.text())
            self._ctrl.set_md_attr(key, value)
        except MdValueError as exc:
            self._reject_md_edit(item, old_value, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surface domain failures
            self._reject_md_edit(item, old_value, str(exc))
            return
        self._refresh_md_after_edit()
        self._finish_md_edit(key)

    def _reject_md_edit(
        self, item: QTableWidgetItem, original: str, message: str
    ) -> None:
        self._md_syncing = True
        try:
            item.setText(original)
        finally:
            self._md_syncing = False
        QMessageBox.warning(self, "Invalid parameter", message)
        self._md_table.setFocus(Qt.FocusReason.OtherFocusReason)  # type: ignore[attr-defined]

    def _refresh_md_after_edit(self) -> None:
        # The real bus refreshes synchronously; the explicit refresh also keeps
        # composition tests with a passive bus faithful to the live ContextService.
        self._md_syncing = True
        try:
            self._populate_md(self._ctrl.get_current_md())
        finally:
            self._md_syncing = False

    def _finish_md_edit(self, key: str) -> None:
        for row in range(self._md_table.rowCount()):
            item = self._md_table.item(row, 0)
            if item is not None and item.text() == key:
                self._md_table.selectRow(row)
                break
        self._md_table.setFocus(Qt.FocusReason.OtherFocusReason)  # type: ignore[attr-defined]

    def _on_md_editor_closed(self, *_args: Any) -> None:
        # Both Submit (Enter) and Cancel (Escape) return keyboard focus to the
        # table. Escape never reaches itemChanged, so it remains a true cancel.
        self._md_table.setFocus(Qt.FocusReason.OtherFocusReason)  # type: ignore[attr-defined]

    def _on_new_md_clicked(self) -> None:
        existing = self._new_md_dialog
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                if not existing.isVisible():
                    existing.show()
                return
            except RuntimeError:
                self._new_md_dialog = None

        dialog = _MdCreateDialog(self._ctrl, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        self._new_md_dialog = dialog
        dialog.finished.connect(
            lambda _result, created=dialog: self._on_new_md_finished(created)
        )
        dialog.open()

    def _on_new_md_finished(self, dialog: _MdCreateDialog) -> None:
        if self._new_md_dialog is dialog:
            self._new_md_dialog = None

    def _on_md_delete_clicked(self) -> None:
        key = self._selected_md_key()
        if key is not None:
            self._delete_md_key(key, confirm=True)

    def _on_md_delete_requested(self) -> None:
        # The table-only Delete shortcut is intentionally confirmation-free.
        key = self._selected_md_key()
        if key is not None:
            self._delete_md_key(key, confirm=False)

    def _delete_md_key(self, key: str, *, confirm: bool) -> None:
        if confirm:
            answer = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete parameter '{key}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._ctrl.del_md_attr(key)
        except Exception as exc:  # noqa: BLE001 — surface domain failures
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self._refresh_md_after_edit()
        self._md_table.setFocus(Qt.FocusReason.OtherFocusReason)  # type: ignore[attr-defined]

    def _build_extra_toolbar_buttons(self, toolbar: QHBoxLayout) -> None:
        self._arb_waveform_btn = QPushButton("Arb Waveforms…")
        toolbar.addWidget(self._arb_waveform_btn)
        self._arb_waveform_btn.clicked.connect(self._on_arb_waveform_clicked)

    def _build_extra_ml_buttons(self, btn_layout: QHBoxLayout) -> None:
        self._create_btn = QPushButton("Create...")
        self._modify_ml_btn = QPushButton("Modify...")
        self._modify_ml_btn.setEnabled(False)
        btn_layout.addWidget(self._create_btn)
        btn_layout.addWidget(self._modify_ml_btn)
        self._create_btn.clicked.connect(self._on_create_clicked)
        self._modify_ml_btn.clicked.connect(self._on_modify_ml_clicked)

    def _on_arb_waveform_clicked(self) -> None:
        # InspectDialog is always parented to MainWindow, which provides open_dialog.
        # The local fallback was dead code that could create a registry-invisible
        # dialog; removed for Fast-Fail clarity (ADR-0002).
        opener = getattr(self.parent(), "open_dialog", None)
        if not callable(opener):
            raise RuntimeError(
                "InspectDialog must be parented to a window providing open_dialog()"
            )
        opener(DialogName.ARB_WAVEFORM)

    def _on_ml_selection_changed(self, enabled: bool) -> None:
        self._modify_ml_btn.setEnabled(enabled)

    def _on_create_clicked(self) -> None:
        dlg = _MlCreateDialog(self._app_ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # On a successful create, chain straight into Modify so the user can
        # immediately tweak the freshly-seeded entry. Open Modify only after the
        # create dialog has closed (non-modal, no stacked modals).
        dlg.finished.connect(lambda _: self._after_create(dlg))
        dlg.open()

    def _after_create(self, dlg: _MlCreateDialog) -> None:
        created = dlg.created
        if created is None:
            return
        item_kind, name = created
        group = "modules" if item_kind == "module" else "waveforms"
        self._open_ml_modify(group, name)

    def _on_modify_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        group, name = data
        self._open_ml_modify(group, name)

    def _open_ml_modify(self, group: str, name: str) -> None:
        # Shared by selection-driven Modify and the auto-open after Create. Re-read
        # the live ml so a just-created entry's cfg is present (create -> ML_CHANGED
        # already refreshed the store).
        ml = self._app_ctrl.get_current_ml()
        if ml is None:
            return

        if group == "modules":
            dlg = _MlModifyDialog(
                self._app_ctrl, "module", name=name, cfg=ml.modules[name], parent=self
            )
        else:
            dlg = _MlModifyDialog(
                self._app_ctrl,
                "waveform",
                name=name,
                cfg=ml.waveforms[name],
                parent=self,
            )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(lambda _: dlg.clear())
        dlg.open()
