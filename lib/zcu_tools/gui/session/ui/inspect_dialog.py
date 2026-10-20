"""InspectDialog — the context inspector shared by every session host.

Parameters edits components through the component tree. Modules lists the
library and edits one selected entry with the shared FormEditor; it also
creates entries from templates and opens the host's ARB waveform dialog.
Aliases edits the ordered global alias table.
Every collaborator is a required constructor argument, so measure and
workflows open the same dialog with the same buttons.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent, QFont, QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError, FailedPreconditionError
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
    SessionPayload,
)
from zcu_tools.gui.session.library_entry import (
    LIBRARY_WRITE_ERRORS,
    LibraryEntryInvalid,
    LibraryKind,
    ModuleEntryForm,
    WaveformEntryForm,
    open_library_entry,
)
from zcu_tools.gui.session.ui.alias_panel import AliasPanel
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.session.ui.new_entry_dialog import NewEntryDialog
from zcu_tools.gui.widgets.form import FormEditor

if TYPE_CHECKING:
    from zcu_tools.cfg_model import TemplateDef
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.form import FormModel
    from zcu_tools.gui.session.alias_presets import AliasPresets
    from zcu_tools.gui.session.arb_waveform import ArbWaveformKeys
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.widgets.dialog_presenter import DialogPresenter
    from zcu_tools.resources.context.library import LibrarySnapshot

logger = logging.getLogger(__name__)

type _EntryKey = tuple[LibraryKind, str]

_GROUPS: tuple[tuple[LibraryKind, str], ...] = (
    ("module", "modules"),
    ("waveform", "waveforms"),
)


class _LibraryTree(QTreeWidget):
    """Library tree whose Delete key deletes directly while the tree has focus."""

    def __init__(
        self, on_delete: Callable[[], None], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._on_delete = on_delete

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        focus = QApplication.focusWidget()
        if (
            event is not None
            and event.key() == Qt.Key.Key_Delete
            and (focus is self or focus is self.viewport())
        ):
            self._on_delete()
            event.accept()
            return
        super().keyPressEvent(event)


class InspectDialog(QDialog):
    """Non-modal inspector for components and library entries of one session.

    Public child object names (buttons are found by their text):
      - Toolbar: "Arb Waveforms…", "Refresh", "Close".
      - Tabs "Parameters" (a ComponentTreeWidget), "Modules" and "Aliases"
        (an AliasPanel; inspect.aliases.* names as AliasPanel documents).
      - inspect.library: the QTreeWidget of library roots, grouped under
        "modules" and "waveforms"; the "New" and "Delete" buttons below it.
      - inspect.entry.name: the Name QLineEdit of the selected entry;
        inspect.entry.status: a QLabel "Saved" or "Unsaved";
        inspect.entry.revert and inspect.entry.apply: the Revert and Apply
        QPushButtons; inspect.entry.scroll: the QScrollArea holding the
        selected entry's FormEditor (form.* names as FormEditor documents).

    Selecting an entry opens it as a library entry form at one fixed capture
    (see open_library_entry). The entry is Unsaved while its Name differs
    from the stored name, its FormState input differs from the opened base,
    or the editor holds unparsed text. Revert resets the state to the opened
    base and the Name. Apply writes the Spec under Name in one replace and
    reopens the result clean; a rejected write is shown through
    dialog_presenter and keeps the draft. Switching entries, deleting, or
    closing while Unsaved asks Save, Discard or Cancel through
    dialog_presenter. Library and parameter publications rebuild the tree
    and reopen the selected entry when the library changed, except while
    Unsaved; the refresh then waits for the next explicit decision.

    The Delete button asks for confirmation; the Delete key on the focused
    tree deletes without asking. Both pin the capture before acting. New
    opens a NewEntryDialog over templates and selects the created entry.
    "Arb Waveforms…" calls the host's open_arb_waveforms.

    Aliases fixes its draft to the loaded entry/point. Clean drafts follow
    session switches; dirty drafts stay shown and Apply rejects a different
    current entry/point. Revert loads the current point's table. Apply saves
    only that selected point. Its Load preset menu replaces the whole draft
    with one of alias_presets without committing. Closing with an Unsaved module entry
    or alias draft asks Save, Discard or Cancel for each in turn; a rejected
    alias Save keeps the dialog open on the Aliases page with its error.
    """

    def __init__(
        self,
        ctrl: ContextControlPort,
        bus: BaseEventBus,
        *,
        templates: Mapping[str, TemplateDef],
        arb_waveforms: ArbWaveformKeys,
        open_arb_waveforms: Callable[[], None],
        dialog_presenter: DialogPresenter,
        alias_presets: AliasPresets | None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog, subscribe to session facts and show the current context.

        ctrl reads captures and commits component and library changes. bus
        delivers context, parameter and library facts; the subscriptions end
        when the dialog finishes or is destroyed. templates is the persistent
        template catalog New offers. arb_waveforms lists the arb asset names
        offered to arb data fields when an entry opens. open_arb_waveforms
        opens the host's ARB waveform dialog. dialog_presenter shows every
        question and failure of this dialog. alias_presets is the composition's
        named alias catalog offered by the Aliases page's Load preset; None
        means no catalog and leaves Load preset disabled.
        """
        super().__init__(parent)
        self._ctrl = ctrl
        self._bus = bus
        self._templates = templates
        self._arb_waveforms = arb_waveforms
        self._open_arb_waveforms = open_arb_waveforms
        self._presenter = dialog_presenter
        self._read_only = False
        self._selected: _EntryKey | None = None
        self._entry: ModuleEntryForm | WaveformEntryForm | None = None
        self._editor: FormEditor[FormModel] | None = None
        self._unsubscribe_state: Callable[[], None] | None = None
        # The library the tree was last built from; a different one reopens.
        self._library: LibrarySnapshot | None = None
        # A publication arrived while Unsaved or during Apply; rebuild later.
        self._refresh_pending = False
        self._applying = False
        self._new_dialog: NewEntryDialog | None = None
        self.setWindowTitle("Inspect Context")
        self.resize(760, 520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(self._build_toolbar())

        self._tabs = QTabWidget()
        self._component_tree = ComponentTreeWidget(ctrl)
        self._aliases = AliasPanel(ctrl, presets=alias_presets)
        self._tabs.addTab(self._component_tree, "Parameters")
        self._tabs.addTab(self._build_modules_tab(), "Modules")
        self._tabs.addTab(self._aliases, "Aliases")
        layout.addWidget(self._tabs)

        bus.subscribe(ContextSwitchedPayload, self._on_bus_refresh)
        bus.subscribe(MdChangedPayload, self._on_bus_refresh)
        bus.subscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = True
        self.finished.connect(self._on_finished)
        self.destroyed.connect(self._cleanup_bus_subscriptions)
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        self._status_label = QLabel("No context")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()
        self._arb_waveform_btn = QPushButton("Arb Waveforms…")
        self._arb_waveform_btn.clicked.connect(self._on_arb_waveforms_clicked)
        toolbar.addWidget(self._arb_waveform_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        return toolbar

    def _build_modules_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_library_panel())
        splitter.addWidget(self._build_entry_panel())
        splitter.setSizes([240, 500])
        layout.addWidget(splitter)
        self._update_controls()
        return widget

    def _build_library_panel(self) -> QWidget:
        """Build the library tree with its New and Delete buttons."""
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._tree = _LibraryTree(self._on_delete_key)
        self._tree.setObjectName("inspect.library")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._tree.currentItemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self._tree)
        row = QHBoxLayout()
        self._new_btn = QPushButton("New")
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        row.addWidget(self._new_btn)
        row.addWidget(self._delete_btn)
        row.addStretch()
        left_layout.addLayout(row)
        self._new_btn.clicked.connect(self._on_new_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        return left

    def _build_entry_panel(self) -> QWidget:
        """Build the Name row and the scroll area of the selected entry's form."""
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        entry_row = QHBoxLayout()
        entry_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setObjectName("inspect.entry.name")
        self._name_edit.setPlaceholderText("ModuleLibrary name")
        self._name_edit.setEnabled(False)
        entry_row.addWidget(self._name_edit, stretch=1)
        self._entry_status = QLabel("Saved")
        self._entry_status.setObjectName("inspect.entry.status")
        entry_row.addWidget(self._entry_status)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("inspect.entry.revert")
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("inspect.entry.apply")
        entry_row.addWidget(self._revert_btn)
        entry_row.addWidget(self._apply_btn)
        right_layout.addLayout(entry_row)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("inspect.entry.scroll")
        self._scroll.setWidgetResizable(True)
        right_layout.addWidget(self._scroll, stretch=1)
        self._name_edit.textChanged.connect(self._update_controls)
        self._revert_btn.clicked.connect(self._on_revert_clicked)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        return right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_read_only(self, *, read_only: bool) -> None:
        """Lock or unlock every input while keeping the dialog readable.

        Read-only disables New, Delete, Revert, Apply, the component tree's
        edits and the Aliases page's writes and Load preset; the Name, the
        entry form and the alias table stay readable, scrollable and foldable
        but ignore input. Arb Waveforms stays clickable: the host opens the
        ARB dialog read-only under its own rule. A draft made before locking stays shown and closing
        no longer asks about it. Call again with False to unlock.
        """
        self._read_only = read_only
        self._component_tree.set_read_only(read_only=read_only)
        self._aliases.set_read_only(read_only=read_only)
        self._new_btn.setEnabled(not read_only)
        if self._editor is not None:
            self._editor.set_editing_enabled(enabled=not read_only)
        self._update_controls()

    def refresh(self) -> None:
        """Refresh components/library and follow alias selection only while clean.

        Unsaved alias drafts keep their loaded entry/point and reject Apply
        after selection changes. Revert explicitly loads the current point.
        """
        self._aliases.refresh()
        self._component_tree.refresh()
        try:
            parameters = self._ctrl.read_context()
        except FailedPreconditionError:
            self._populate(None)
            self._status_label.setText("No context")
            return
        self._populate(parameters.library)
        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.setText(f"Last updated: {now}")

    # ------------------------------------------------------------------
    # Library tree
    # ------------------------------------------------------------------

    def _populate(self, library: LibrarySnapshot | None) -> None:
        """Rebuild the tree; reopen the selection when the library changed."""
        if self._applying or self._is_unsaved():
            self._refresh_pending = True
            return
        self._refresh_pending = False
        changed = library is not self._library
        self._library = library
        desired = self._selected
        if desired is not None and (
            library is None or not _has_entry(library, desired)
        ):
            desired = None
        self._rebuild_tree(library, desired)
        if desired is None:
            self._close_entry()
            self._selected = None
        elif self._entry is None or changed:
            self._close_entry()
            self._open_entry(desired)
        self._update_controls()

    def _rebuild_tree(
        self, library: LibrarySnapshot | None, selected: _EntryKey | None
    ) -> None:
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            if library is None:
                return
            bold = QFont()
            bold.setBold(True)
            for kind, label in _GROUPS:
                keys = (
                    library.module_keys()
                    if kind == "module"
                    else library.waveform_keys()
                )
                if not keys:
                    continue
                group = QTreeWidgetItem(self._tree, [label])
                group.setFont(0, bold)
                group.setFlags(Qt.ItemFlag.ItemIsEnabled)
                group.setExpanded(True)
                for name in sorted(keys):
                    item = QTreeWidgetItem(group, [name])
                    item.setData(0, Qt.ItemDataRole.UserRole, (kind, name))
                    if selected == (kind, name):
                        self._tree.setCurrentItem(item)
        finally:
            self._tree.blockSignals(False)

    def _current_key(self) -> _EntryKey | None:
        item = self._tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return None
        kind, name = data
        return kind, name

    def _select_in_tree(self, key: _EntryKey | None) -> None:
        self._tree.blockSignals(True)
        try:
            self._tree.setCurrentItem(_find_item(self._tree, key))
        finally:
            self._tree.blockSignals(False)

    def _on_item_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        del current
        key = self._current_key()
        if key == self._selected:
            self._update_controls()
            return
        if not self._resolve_unsaved():
            self._select_in_tree(self._selected)
            return
        self._close_entry()
        self._selected = key
        if key is not None:
            self._open_entry(key)
        if self._refresh_pending:
            self._populate_from_context()
        self._update_controls()

    def _populate_from_context(self) -> None:
        try:
            library = self._ctrl.read_context().library
        except FailedPreconditionError:
            library = None
        self._populate(library)

    # ------------------------------------------------------------------
    # Entry editing
    # ------------------------------------------------------------------

    def _open_entry(self, key: _EntryKey) -> None:
        kind, name = key
        try:
            entry = open_library_entry(
                self._ctrl.read_context(),
                kind,
                name,
                arb_names=self._arb_waveforms.list_data_keys(),
            )
        except ExpectedError as exc:
            logger.info("Library entry %s unavailable: %s", key, exc)
            self._presenter.warning(self, "Editor unavailable", str(exc))
            self._selected = None
            self._select_in_tree(None)
            return
        editor = FormEditor(entry.state, entry.env)
        editor.set_editing_enabled(enabled=not self._read_only)
        editor.validity_changed.connect(self._update_controls)
        # setWidget deletes the previous form, so folds and scroll start fresh.
        self._scroll.setWidget(editor)
        self._entry = entry
        self._editor = editor
        self._unsubscribe_state = entry.state.subscribe(self._update_controls)
        self._name_edit.blockSignals(True)
        self._name_edit.setText(name)
        self._name_edit.blockSignals(False)

    def _close_entry(self) -> None:
        if self._unsubscribe_state is not None:
            self._unsubscribe_state()
            self._unsubscribe_state = None
        if self._editor is not None:
            self._editor.detach()
            self._editor = None
            self._scroll.setWidget(QWidget())
        self._entry = None
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)

    def _is_unsaved(self) -> bool:
        entry, editor, key = self._entry, self._editor, self._selected
        if entry is None or editor is None or key is None:
            return False
        return (
            self._name_edit.text() != key[1]
            or entry.state.is_dirty
            or editor.local_parse_errors() is not None
        )

    def _update_controls(self, *_args: object) -> None:
        editable = self._entry is not None and not self._read_only
        unsaved = self._is_unsaved()
        valid = self._editor is not None and self._editor.errors() is None
        self._name_edit.setEnabled(self._entry is not None)
        self._name_edit.setReadOnly(not editable)
        self._revert_btn.setEnabled(editable and unsaved)
        self._apply_btn.setEnabled(
            editable and unsaved and valid and bool(self._name_edit.text().strip())
        )
        self._entry_status.setText("Unsaved" if unsaved else "Saved")
        self._delete_btn.setEnabled(
            not self._read_only and self._current_key() is not None
        )

    def _on_revert_clicked(self) -> None:
        entry, editor, key = self._entry, self._editor, self._selected
        if self._read_only or entry is None or editor is None or key is None:
            return
        entry.state.reset(entry.state.base)
        editor.discard_local_text()
        self._name_edit.setText(key[1])
        self._update_controls()

    def _on_apply_clicked(self) -> None:
        if self._read_only:
            return
        if self._apply():
            self._populate_from_context()

    def _apply(self) -> bool:
        """Write the draft under Name; True after the write, False to keep it."""
        entry, editor, key = self._entry, self._editor, self._selected
        if entry is None or editor is None or key is None:
            return False
        new_name = self._name_edit.text().strip()
        if not new_name:
            self._presenter.warning(
                self, "Invalid name", "Entry name must not be empty."
            )
            return False
        self._applying = True
        try:
            local_errors = editor.local_parse_errors()
            if local_errors is not None:
                raise LibraryEntryInvalid(entry.kind, local_errors)
            self._ctrl.apply_library_change(
                entry.parameters, entry.replace(key[1], new_name)
            )
        except LIBRARY_WRITE_ERRORS as exc:
            logger.info("Library entry apply failed: %s", exc)
            self._presenter.warning(self, "Apply failed", str(exc))
            return False
        finally:
            self._applying = False
        self._close_entry()
        self._selected = (key[0], new_name)
        # The write published a refresh while applying; the caller rebuilds.
        self._refresh_pending = True
        return True

    def _resolve_unsaved(self) -> bool:
        """Settle an Unsaved draft; True when the caller may continue."""
        if self._read_only or not self._is_unsaved():
            return True
        answer = self._presenter.save_discard_cancel(
            self,
            "Unsaved ModuleLibrary changes",
            "Apply the current draft before continuing?",
        )
        if answer == "save":
            return self._apply()
        if answer == "discard":
            self._close_entry()
            return True
        return False

    # ------------------------------------------------------------------
    # Delete, New and ARB
    # ------------------------------------------------------------------

    def _on_delete_clicked(self) -> None:
        self._delete(confirm=True)

    def _on_delete_key(self) -> None:
        self._delete(confirm=False)

    def _delete(self, *, confirm: bool) -> None:
        key = self._current_key()
        if self._read_only or key is None:
            return
        if not self._resolve_unsaved():
            return
        kind, name = key
        # Pin the target before any question: a context switch while asking
        # must not redirect the deletion into another point or entry.
        capture = self._ctrl.read_context()
        if confirm and not self._presenter.confirm(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {kind} '{name}'?",
        ):
            return
        try:
            self._ctrl.apply_library_change(
                capture, LibraryChange(kind, "delete", name)
            )
        except LIBRARY_WRITE_ERRORS as exc:
            logger.info("Library entry delete failed: %s", exc)
            self._presenter.warning(self, "Delete failed", str(exc))
            return
        if self._selected == key:
            self._close_entry()
            self._selected = None
        self._populate_from_context()

    def _on_arb_waveforms_clicked(self) -> None:
        # Also while read-only: the host opens ARB under its own read-only rule.
        self._open_arb_waveforms()

    def _on_new_clicked(self) -> None:
        if self._read_only:
            return
        if self._new_dialog is not None:
            self._new_dialog.raise_()
            self._new_dialog.activateWindow()
            return
        dialog = NewEntryDialog(
            self._ctrl,
            self._templates,
            arb_waveforms=self._arb_waveforms,
            dialog_presenter=self._presenter,
            parent=self,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda _result: self._on_new_finished(dialog))
        self._new_dialog = dialog
        dialog.open()

    def _on_new_finished(self, dialog: NewEntryDialog) -> None:
        self._new_dialog = None
        created = dialog.created
        if created is None:
            return
        if self._is_unsaved():
            self._refresh_pending = True
            return
        self._close_entry()
        self._selected = created
        self._populate_from_context()

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------

    def _on_bus_refresh(self, payload: SessionPayload) -> None:
        del payload
        self.refresh()

    def _cleanup_bus_subscriptions(self, *_args: object) -> None:
        if not self._bus_subs_active:
            return
        self._bus.unsubscribe(ContextSwitchedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MdChangedPayload, self._on_bus_refresh)
        self._bus.unsubscribe(MlChangedPayload, self._on_bus_refresh)
        self._bus_subs_active = False

    def _on_finished(self, *_args: object) -> None:
        self._close_entry()
        self._cleanup_bus_subscriptions()

    def _resolve_unsaved_aliases(self) -> bool:
        """Settle an Unsaved alias draft; True when closing may continue."""
        if self._read_only or not self._aliases.is_unsaved():
            return True
        answer = self._presenter.save_discard_cancel(
            self,
            "Unsaved alias changes",
            "Apply the alias table before closing?",
        )
        if answer == "save":
            if self._aliases.apply():
                return True
            # The rejection is shown inline on the page; bring it into view.
            self._tabs.setCurrentWidget(self._aliases)
            return False
        if answer == "discard":
            self._aliases.revert()
            return True
        return False

    def _resolve_before_close(self) -> bool:
        return self._resolve_unsaved() and self._resolve_unsaved_aliases()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if not self._resolve_before_close():
            if a0 is not None:
                a0.ignore()
            return
        self._close_entry()
        super().closeEvent(a0)

    def reject(self) -> None:
        if not self._resolve_before_close():
            return
        super().reject()


def _has_entry(library: LibrarySnapshot, key: _EntryKey) -> bool:
    kind, name = key
    keys = library.module_keys() if kind == "module" else library.waveform_keys()
    return name in keys


def _find_item(tree: QTreeWidget, key: _EntryKey | None) -> QTreeWidgetItem | None:
    if key is None:
        return None
    for index in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(index)
        if group is None:
            continue
        for child_index in range(group.childCount()):
            child = group.child(child_index)
            if child is not None and child.data(0, Qt.ItemDataRole.UserRole) == key:
                return child
    return None


__all__ = ["InspectDialog"]
