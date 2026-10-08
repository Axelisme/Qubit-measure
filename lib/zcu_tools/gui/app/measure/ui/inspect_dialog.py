"""InspectDialog — measure's context inspector.

Subclasses the app-agnostic ``InspectDialogBase`` with a component tree.
The measure Modules tab owns the selected service-backed
CfgEditor draft.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont, QKeyEvent  # type: ignore[attr-defined]
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
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase
from zcu_tools.gui.widgets.cfg import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.resources.context.library import LibrarySnapshot

logger = logging.getLogger(__name__)


_MlItemKind = Literal["module", "waveform"]


class _MlTree(QTreeWidget):
    """ModuleLibrary tree with a direct Delete shortcut at the tree boundary."""

    def __init__(
        self, on_delete: Callable[[], None], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._on_delete = on_delete

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            super().keyPressEvent(event)
            return
        focus = QApplication.focusWidget()
        tree_has_focus = focus is self or focus is self.viewport()
        if (
            tree_has_focus and event.key() == Qt.Key.Key_Delete  # type: ignore[attr-defined]
        ):
            self._on_delete()
            event.accept()
            return
        super().keyPressEvent(event)


class _MlCreateDialog(QDialog):
    """Create a new ml module/waveform from a template (the single create path).

    One-shot: pick a template + a name → the template's factory seeds the value
    (md-linked defaults for named templates, structural zeros for ``:blank`` templates)
    and registers it directly into ml. The embedded editor handles later edits.
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Create ModuleLibrary entry")
        self.setModal(False)

        # The (item_kind, name) of a successful create, read by the parent after
        # the dialog closes so it can select the new embedded editor. None until
        # create wins.
        self.created: tuple[_MlItemKind, str] | None = None
        # True once the user has typed into the name field by hand: from then on
        # switching template must not clobber their name (least surprise). Qt only
        # fires textEdited on user keystrokes, never on programmatic setText, so
        # seeding the suggestion below does not set this flag.
        self._name_edited = False

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Pick a template and a name. Named templates seed md-linked defaults; "
            "'Blank: …' templates seed an empty shape. Edit afterwards in Inspect."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._template_combo = QComboBox()
        catalog = self._ctrl.get_template_catalog()
        # Modules then waveforms, each labelled; template entry stashed on the item.
        for kind in ("module", "waveform"):
            for entry in catalog.entries_for(kind):  # type: ignore[arg-type]
                self._template_combo.addItem(f"{entry.label}  ({kind})", userData=entry)
        form.addRow("Template:", self._template_combo)
        self._name_edit = QLineEdit()
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        # Seed the name with the initial template's convention-based suggestion.
        initial = self._template_combo.currentData()
        if initial is not None:
            self._name_edit.setText(self._suggest_name(initial))

        self._name_edit.textEdited.connect(self._on_name_edited)
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)

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

        Blank templates carry no ``default_name`` -> empty (the user must name it).
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

    def _on_template_changed(self, _index: int) -> None:
        # Only re-suggest while the name is still the auto-filled one; once the
        # user has typed their own, leave it alone (least surprise).
        if self._name_edited:
            return
        entry = self._template_combo.currentData()
        if entry is not None:
            self._name_edit.setText(self._suggest_name(entry))

    def _on_create(self) -> None:
        entry = self._template_combo.currentData()
        if entry is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Entry name must not be empty.")
            return
        try:
            self._ctrl.create_from_template(entry.item_kind, entry.template_id, name)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            QMessageBox.critical(self, "Create failed", str(exc))
            return
        self.created = (entry.item_kind, name)
        self.accept()


class InspectDialog(InspectDialogBase):
    """Measure inspector with component Parameters and embedded ml editing.

    The base owns the generic presentation; this subclass
    replaces Parameters and Modules presentation only for measure.
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
        self._new_ml_dialog: _MlCreateDialog | None = None
        self._ml_editor_owner = f"inspect-ml-{uuid.uuid4().hex[:8]}"
        self._ml_editor_id: str | None = None
        self._ml_selected_data: tuple[str, str] | None = None
        self._ml_baseline_schema: Any | None = None
        self._ml_baseline_name: str | None = None
        self._ml_live_ml: LibrarySnapshot | None = None
        self._ml_dirty_hint = False
        self._ml_tree_syncing = False
        self._ml_applying = False
        self._ml_refresh_pending = False
        super().__init__(ctrl.context_control, bus, parent=parent)
        self.finished.connect(self._on_finished)

    def _build_md_tab(self) -> QWidget:
        """Use the session component-tree editor for measure parameters."""
        self._component_tree = ComponentTreeWidget(self._ctrl)
        return self._component_tree

    def _build_extra_toolbar_buttons(self, toolbar: QHBoxLayout) -> None:
        self._arb_waveform_btn = QPushButton("Arb Waveforms…")
        toolbar.addWidget(self._arb_waveform_btn)
        self._arb_waveform_btn.clicked.connect(self._on_arb_waveform_clicked)

    def _build_ml_tab(self) -> QWidget:
        """Build measure's tree plus one embedded service-owned cfg editor."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._ml_tree = _MlTree(self._on_ml_delete_requested)
        self._ml_tree.setHeaderHidden(True)
        self._ml_tree.setRootIsDecorated(True)
        self._ml_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._ml_tree.currentItemChanged.connect(self._on_ml_item_changed)
        left_layout.addWidget(self._ml_tree)

        collection_row = QHBoxLayout()
        self._create_btn = QPushButton("New")
        self._del_ml_btn = QPushButton("Delete")
        self._del_ml_btn.setEnabled(False)
        collection_row.addWidget(self._create_btn)
        collection_row.addWidget(self._del_ml_btn)
        collection_row.addStretch()
        left_layout.addLayout(collection_row)
        self._create_btn.clicked.connect(self._on_create_clicked)
        self._del_ml_btn.clicked.connect(self._on_delete_ml_clicked)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        editor_row = QHBoxLayout()
        editor_row.addWidget(QLabel("Name:"))
        self._ml_name_edit = QLineEdit()
        self._ml_name_edit.setPlaceholderText("ModuleLibrary name")
        self._ml_name_edit.setEnabled(False)
        editor_row.addWidget(self._ml_name_edit, stretch=1)
        self._ml_status_label = QLabel("Saved")
        editor_row.addWidget(self._ml_status_label)
        self._ml_revert_btn = QPushButton("Revert")
        self._ml_revert_btn.setEnabled(False)
        editor_row.addWidget(self._ml_revert_btn)
        self._ml_apply_btn = QPushButton("Apply")
        self._ml_apply_btn.setEnabled(False)
        editor_row.addWidget(self._ml_apply_btn)
        right_layout.addLayout(editor_row)

        self._ml_form_widget = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(self._app_ctrl)
        )
        right_layout.addWidget(self._ml_form_widget, stretch=1)
        self._ml_form_widget.set_editing_enabled(False)
        self._ml_name_edit.textEdited.connect(self._on_ml_name_edited)
        self._ml_form_widget.schema_changed.connect(self._on_ml_schema_changed)
        self._ml_form_widget.validity_changed.connect(self._on_ml_validity_changed)
        self._ml_revert_btn.clicked.connect(self._on_ml_revert_clicked)
        self._ml_apply_btn.clicked.connect(self._on_ml_apply_clicked)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([240, 460])
        layout.addWidget(splitter)
        return widget

    def _populate_ml(self, ml: LibrarySnapshot | None) -> None:
        """Populate the measure tree without dropping an unresolved draft."""
        if self._ml_applying:
            self._ml_refresh_pending = True
            return
        if self._ml_has_changes():
            # An external ML_CHANGED must not silently replace a user draft. The
            # next explicit Apply/Revert/Discard action will refresh the tree.
            self._ml_refresh_pending = True
            return

        desired = self._ml_selected_data
        if ml is None:
            desired = None
        live_changed = (
            self._ml_live_ml is not None
            and ml is not None
            and self._ml_live_ml is not ml
        )
        if self._ml_editor_id is not None and (
            desired is None
            or ml is None
            or not self._ml_entry_exists(ml, desired)
            or live_changed
        ):
            self._close_ml_editor()
            if desired is not None and (
                ml is None or not self._ml_entry_exists(ml, desired)
            ):
                desired = None
                self._ml_selected_data = None

        self._ml_tree_syncing = True
        self._ml_tree.blockSignals(True)
        self._ml_tree.clear()
        restore_item: QTreeWidgetItem | None = None
        if ml is not None:
            bold = QFont()
            bold.setBold(True)
            for group, store in (
                ("modules", ml.module_keys()),
                ("waveforms", ml.waveform_keys()),
            ):
                if not store:
                    continue
                group_item = QTreeWidgetItem(self._ml_tree, [group])
                group_item.setFont(0, bold)
                group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # type: ignore[attr-defined]
                group_item.setExpanded(True)
                for name in sorted(store):
                    child = QTreeWidgetItem(group_item, [name])
                    child.setData(
                        0,
                        Qt.ItemDataRole.UserRole,
                        (group, name),  # type: ignore[attr-defined]
                    )
                    if desired == (group, name):
                        restore_item = child
        self._ml_tree.blockSignals(False)
        self._ml_tree_syncing = False
        self._ml_live_ml = ml
        self._ml_refresh_pending = False

        if restore_item is None:
            if desired is not None:
                self._ml_selected_data = None
                self._clear_ml_editor()
            else:
                self._update_ml_editor_controls()
            return

        # Keep the restored selection from triggering a prompt while the tree is
        # rebuilt. Re-run the normal selection path once it is coherent; it either
        # keeps the current session or opens a fresh one.
        self._ml_tree_syncing = True
        try:
            self._ml_tree.setCurrentItem(restore_item)
        finally:
            self._ml_tree_syncing = False
        self._on_ml_item_changed(restore_item, None)

    @staticmethod
    def _ml_item_data(item: QTreeWidgetItem | None) -> tuple[str, str] | None:
        if item is None:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        group, name = data
        if group not in {"modules", "waveforms"} or not isinstance(name, str):
            return None
        return group, name

    def _current_ml_item_data(self) -> tuple[str, str] | None:
        return self._ml_item_data(self._ml_tree.currentItem())

    @staticmethod
    def _ml_entry_exists(ml: LibrarySnapshot, data: tuple[str, str]) -> bool:
        group, name = data
        store = ml.module_keys() if group == "modules" else ml.waveform_keys()
        return name in store

    def _find_ml_item(self, data: tuple[str, str]) -> QTreeWidgetItem | None:
        for index in range(self._ml_tree.topLevelItemCount()):
            group_item = self._ml_tree.topLevelItem(index)
            if group_item is None:
                continue
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if self._ml_item_data(child) == data:
                    return child
        return None

    def _on_ml_item_changed(
        self, current: QTreeWidgetItem | None, previous: Any
    ) -> None:
        del previous
        if self._ml_tree_syncing:
            return
        next_data = self._ml_item_data(current)
        if next_data == self._ml_selected_data and self._ml_editor_id is not None:
            self._update_ml_editor_controls()
            return

        if self._ml_has_changes():
            if not self._resolve_dirty_ml(select_result=False):
                self._restore_ml_selection(self._ml_selected_data)
                return

        self._close_ml_editor()
        self._ml_selected_data = next_data
        if next_data is None:
            self._clear_ml_editor()
        else:
            self._open_ml_editor(next_data)
        self._flush_pending_ml_refresh()

    def _restore_ml_selection(self, data: tuple[str, str] | None) -> None:
        item = None if data is None else self._find_ml_item(data)
        self._ml_tree_syncing = True
        try:
            self._ml_tree.setCurrentItem(item)
        finally:
            self._ml_tree_syncing = False

    def _flush_pending_ml_refresh(self) -> None:
        if not self._ml_refresh_pending or self._ml_has_changes():
            return
        self._ml_refresh_pending = False
        self._populate_ml(self._ctrl.read_context().library)

    def _open_ml_editor(self, data: tuple[str, str]) -> None:
        group, name = data
        item_kind = "module" if group == "modules" else "waveform"
        editor_id: str | None = None
        try:
            editor_id, _ = self._app_ctrl.open_cfg_editor(
                item_kind,
                from_name=name,
                gc=False,
                owner_key=self._ml_editor_owner,
            )
            draft = self._app_ctrl.get_cfg_editor_draft(editor_id)
            baseline = draft.snapshot()
            self._ml_form_widget.attach(draft)
        except Exception as exc:
            logger.exception("Unable to open ModuleLibrary editor for %s", data)
            if editor_id is not None:
                self._app_ctrl.teardown_cfg_editor(editor_id)
            QMessageBox.warning(self, "Editor unavailable", str(exc))
            self._ml_selected_data = None
            self._clear_ml_editor()
            return

        self._ml_editor_id = editor_id
        self._ml_baseline_schema = baseline
        self._ml_baseline_name = name
        self._ml_dirty_hint = False
        self._ml_name_edit.blockSignals(True)
        self._ml_name_edit.setText(name)
        self._ml_name_edit.blockSignals(False)
        self._ml_form_widget.set_editing_enabled(True)
        self._update_ml_editor_controls()

    def _close_ml_editor(self) -> None:
        editor_id = self._ml_editor_id
        self._ml_editor_id = None
        self._ml_baseline_schema = None
        self._ml_baseline_name = None
        self._ml_dirty_hint = False
        self._ml_form_widget.detach()
        self._ml_form_widget.set_editing_enabled(False)
        self._ml_name_edit.blockSignals(True)
        self._ml_name_edit.clear()
        self._ml_name_edit.blockSignals(False)
        if editor_id is not None:
            self._app_ctrl.teardown_cfg_editor(editor_id)
        self._update_ml_editor_controls()

    def _clear_ml_editor(self) -> None:
        self._close_ml_editor()
        self._ml_status_label.setText("Saved")
        self._del_ml_btn.setEnabled(False)

    def _ml_has_changes(self) -> bool:
        if self._ml_editor_id is None:
            return False
        if self._ml_dirty_hint:
            return True
        if self._ml_baseline_name != self._ml_name_edit.text():
            return True
        baseline = self._ml_baseline_schema
        if baseline is None:
            return True
        try:
            return self._ml_form_widget.read_schema().value != baseline.value
        except RuntimeError:
            return True

    def _update_ml_editor_controls(self) -> None:
        has_editor = self._ml_editor_id is not None
        dirty = self._ml_has_changes()
        valid = has_editor and self._ml_form_widget.is_valid()
        has_name = bool(self._ml_name_edit.text().strip())
        self._ml_name_edit.setEnabled(has_editor)
        self._ml_revert_btn.setEnabled(has_editor and dirty)
        self._ml_apply_btn.setEnabled(has_editor and dirty and valid and has_name)
        self._ml_status_label.setText("Unsaved" if dirty else "Saved")
        self._del_ml_btn.setEnabled(self._current_ml_item_data() is not None)

    def _on_ml_name_edited(self, _text: str) -> None:
        self._ml_dirty_hint = True
        self._update_ml_editor_controls()

    def _on_ml_schema_changed(self, _schema: Any) -> None:
        self._ml_dirty_hint = True
        self._update_ml_editor_controls()

    def _on_ml_validity_changed(self, _valid: bool) -> None:
        self._update_ml_editor_controls()

    def _on_ml_revert_clicked(self) -> None:
        data = self._ml_selected_data
        if data is None:
            return
        self._close_ml_editor()
        self._ml_selected_data = data
        self._open_ml_editor(data)
        self._flush_pending_ml_refresh()

    def _on_ml_apply_clicked(self) -> None:
        self._apply_ml_draft(select_result=True)

    def _apply_ml_draft(self, *, select_result: bool) -> bool:
        data = self._ml_selected_data
        editor_id = self._ml_editor_id
        if data is None or editor_id is None:
            return False
        if not self._ml_form_widget.is_valid():
            self._update_ml_editor_controls()
            return False
        new_name = self._ml_name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid name", "Entry name must not be empty.")
            return False

        group, old_name = data
        self._ml_applying = True
        try:
            self._app_ctrl.replace_cfg_editor(editor_id, old_name, new_name)
        except Exception as exc:  # noqa: BLE001 — retain draft on every failure
            QMessageBox.warning(self, "Apply failed", str(exc))
            return False
        finally:
            self._ml_applying = False

        # The service removes the old session only after the ContextService write
        # succeeds. Reopen from live ml so a successful Apply always leaves a
        # fresh clean draft selected rather than reusing the committed model.
        self._close_ml_editor()
        self._ml_selected_data = (group, new_name)
        self._ml_dirty_hint = False
        # Keep a refresh requested by the synchronous ML_CHANGED fact. A
        # selection transaction (or the next explicit refresh) consumes it after
        # the resulting target is selected, so the tree cannot retain old names.
        if select_result:
            self._populate_ml(self._ctrl.read_context().library)
        return True

    def _resolve_dirty_ml(self, *, select_result: bool) -> bool:
        if not self._ml_has_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved ModuleLibrary changes",
            "Apply the current draft before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._apply_ml_draft(select_result=select_result)
        if answer == QMessageBox.StandardButton.Discard:
            self._close_ml_editor()
            return True
        return False

    def _on_delete_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        if self._ml_has_changes() and not self._resolve_dirty_ml(select_result=False):
            return
        data = self._ml_selected_data or data
        self._delete_ml_entry(data, confirm=True)

    def _on_ml_delete_requested(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        if self._ml_has_changes() and not self._resolve_dirty_ml(select_result=False):
            return
        data = self._ml_selected_data or data
        self._delete_ml_entry(data, confirm=False)

    def _delete_ml_entry(self, data: tuple[str, str], *, confirm: bool) -> None:
        group, name = data
        if confirm:
            answer = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete {group[:-1]} '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            if group == "modules":
                self._app_ctrl.del_ml_module(name)
            else:
                self._app_ctrl.del_ml_waveform(name)
        except Exception as exc:  # noqa: BLE001 — surface domain failures
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        if self._ml_selected_data == data:
            self._ml_selected_data = None
        self._populate_ml(self._ctrl.read_context().library)

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

    def _on_create_clicked(self) -> None:
        existing = self._new_ml_dialog
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                if not existing.isVisible():
                    existing.show()
                return
            except RuntimeError:
                self._new_ml_dialog = None

        dialog = _MlCreateDialog(self._app_ctrl, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        self._new_ml_dialog = dialog
        dialog.finished.connect(
            lambda _result, created=dialog: self._on_ml_create_finished(created)
        )
        dialog.open()

    def _on_ml_create_finished(self, dialog: _MlCreateDialog) -> None:
        if self._new_ml_dialog is dialog:
            self._new_ml_dialog = None
        created = dialog.created
        if created is None or self._ml_has_changes():
            if created is not None:
                self._ml_refresh_pending = True
            return
        item_kind, name = created
        selected_data = (
            "modules" if item_kind == "module" else "waveforms",
            name,
        )
        if self._ml_selected_data != selected_data:
            self._close_ml_editor()
        self._ml_selected_data = selected_data
        self._populate_ml(self._ctrl.read_context().library)

    def _on_finished(self, *_args: Any) -> None:
        self._close_ml_editor()

    def _prepare_close(self) -> bool:
        if not self._ml_has_changes():
            self._close_ml_editor()
            return True
        return self._resolve_dirty_ml(select_result=False)

    def closeEvent(self, a0: Any) -> None:
        if not self._prepare_close():
            a0.ignore()
            return
        a0.accept()
        super().closeEvent(a0)

    def reject(self) -> None:
        if not self._prepare_close():
            return
        super().reject()
