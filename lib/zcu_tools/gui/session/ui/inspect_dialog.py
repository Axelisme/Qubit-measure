"""InspectDialog — the context inspector shared by every session host.

Parameters edits components through a filterable, initially collapsed tree.
Modules browses dotted library keys as a filterable hierarchy and edits one
selected entry with the shared FormEditor; it also
creates entries from templates and opens the host's ARB waveform dialog.
Aliases edits the ordered global alias table.
Every collaborator is a required constructor argument, so measure and
workflows open the same dialog with the same buttons.
Driver filter targets are inspect.parameters.filter and inspect.library.filter;
the hierarchical library tree remains inspect.library.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import JsonValue
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
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
from zcu_tools.gui.form import FormWritePath
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
    PointFileChangedPayload,
    SessionPayload,
)
from zcu_tools.gui.session.library_entry import (
    LIBRARY_WRITE_ERRORS,
    LibraryKind,
    ModuleEntryForm,
    WaveformEntryForm,
    open_library_entry,
)
from zcu_tools.gui.session.ui.alias_panel import AliasPanel
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.session.ui.new_entry_dialog import NewEntryDialog
from zcu_tools.gui.session.ui.point_file_banner import PointFileBanner
from zcu_tools.gui.session.ui.tree_filter import TreeFilter
from zcu_tools.gui.widgets import set_foreground, success_colour
from zcu_tools.gui.widgets.form import FormEditor

if TYPE_CHECKING:
    from zcu_tools.cfg_model import TemplateDef
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.form import FormModel
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
      - inspect.parameters.filter and inspect.library.filter: live Filter
        QLineEdits matching names and dotted paths, ignoring case. Matches and
        ancestors stay shown and expand; clearing restores browsing expansion.
      - inspect.library: the QTreeWidget of dotted-key segments, grouped under
        "modules" and "waveforms"; the "New" and "Delete" buttons below it.
      - inspect.entry.name: the selected entry's Name QLineEdit.
        inspect.entry.error: the QLabel for rejected writes.
        inspect.entry.scroll: the QScrollArea holding its FormEditor.

    Selecting an entry captures its expression environment. Enter, ordinary
    focus loss, dropdown changes and checkbox toggles replace Name and the valid
    Spec together through apply_library_change. Typing alone never writes.
    Invalid text stays marked in the form. Owner/revision failures appear in
    inspect.entry.error and keep the input editable without changing the file.
    PointFileBanner offers Keep/Reload for stale revisions. Switching entries or
    points, deleting, or closing discards uncommitted input without questions.
    Successful writes keep the mounted form and focus; no undo is retained.

    The Delete button asks for confirmation; the Delete key on the focused
    tree deletes without asking. Both pin the capture before acting. New
    opens a NewEntryDialog over templates and selects the created entry.
    "Arb Waveforms…" calls the host's open_arb_waveforms.

    Aliases commits the complete valid table after a cell edit or Remove.
    New incomplete rows stay local. Point switches discard invalid rows and
    load the selected point. Closing never asks to save or discard.
    """

    # tolerated: Hosts inject distinct context/event, template, ARB, presenter and alias capabilities; no shared options record.
    def __init__(  # noqa: PLR0913
        self,
        ctrl: ContextControlPort,
        bus: BaseEventBus,
        *,
        templates: Mapping[str, TemplateDef],
        arb_waveforms: ArbWaveformKeys,
        open_arb_waveforms: Callable[[], None],
        dialog_presenter: DialogPresenter,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog, subscribe to session facts and show the current context.

        ctrl reads captures and commits component and library changes. bus
        delivers context, parameter and library facts; the subscriptions end
        when the dialog finishes or is destroyed. templates is the persistent
        template catalog New offers. arb_waveforms lists the arb asset names
        offered to arb data fields when an entry opens. open_arb_waveforms
        opens the host's ARB waveform dialog. dialog_presenter shows every
        deletion question and New/Delete failures. Entry write failures stay inline.
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
        self._scope = LifetimeScope(self)
        self._entry_scope = LifetimeScope(self)
        self._scope.add_cleanup(self._close_entry)
        # The library the tree was last built from; a different one reopens.
        self._library: LibrarySnapshot | None = None
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
        self._aliases = AliasPanel(ctrl)
        self._scope.add_cleanup(self._aliases.detach)
        self._tabs.addTab(self._component_tree, "Parameters")
        self._tabs.addTab(self._build_modules_tab(), "Modules")
        self._tabs.addTab(self._aliases, "Aliases")
        layout.addWidget(self._tabs)
        footer = QHBoxLayout()
        self._saved_hint = QLabel()
        self._saved_hint.setObjectName("inspect.saved")
        set_foreground(self._saved_hint, success_colour(self))
        footer.addWidget(self._saved_hint, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._scope.connect(buttons.rejected, lambda: self.close())
        footer.addWidget(buttons)
        layout.addLayout(footer)

        self._scope.add_cleanup(
            bus.subscribe(
                ContextSwitchedPayload, self._scope.callback(self._on_bus_refresh)
            ).unsubscribe
        )
        self._scope.add_cleanup(
            bus.subscribe(
                MdChangedPayload, self._scope.callback(self._on_bus_refresh)
            ).unsubscribe
        )
        self._scope.add_cleanup(
            bus.subscribe(
                MlChangedPayload, self._scope.callback(self._on_bus_refresh)
            ).unsubscribe
        )
        self._scope.add_cleanup(
            bus.subscribe(
                PointFileChangedPayload, self._scope.callback(self._on_file_changed)
            ).unsubscribe
        )
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
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
        self._scope.connect(
            self._arb_waveform_btn.clicked, self._on_arb_waveforms_clicked
        )
        toolbar.addWidget(self._arb_waveform_btn)
        refresh_btn = QPushButton("Refresh")
        self._scope.connect(refresh_btn.clicked, self.refresh)
        toolbar.addWidget(refresh_btn)
        return toolbar

    def _build_modules_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        self._library_sync = PointFileBanner(
            self._ctrl,
            "library",
            self._scope.callback(self._reload_library),
            parent=widget,
        )
        self._library_revision = self._library_sync.capture()
        layout.addWidget(self._library_sync)
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
        self._scope.connect(self._tree.currentItemChanged, self._on_item_changed)
        self._library_filter = TreeFilter(
            self._tree, object_name="inspect.library.filter", default_expanded=True
        )
        left_layout.addWidget(self._library_filter)
        left_layout.addWidget(self._tree)
        row = QHBoxLayout()
        self._new_btn = QPushButton("New")
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setEnabled(False)
        row.addWidget(self._new_btn)
        row.addWidget(self._delete_btn)
        row.addStretch()
        left_layout.addLayout(row)
        self._scope.connect(self._new_btn.clicked, self._on_new_clicked)
        self._scope.connect(self._delete_btn.clicked, self._on_delete_clicked)
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
        right_layout.addLayout(entry_row)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("inspect.entry.scroll")
        self._scroll.setWidgetResizable(True)
        right_layout.addWidget(self._scroll, stretch=1)
        self._scope.connect(self._name_edit.editingFinished, self._commit_entry)
        self._entry_error = QLabel()
        self._entry_error.setObjectName("inspect.entry.error")
        self._entry_error.setWordWrap(True)
        right_layout.addWidget(self._entry_error)
        return right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_read_only(self, *, read_only: bool) -> None:
        """Lock or unlock every input while keeping the dialog readable.

        Read-only disables New, Delete, the component tree's
        edits and the Aliases page's writes; the Name, the
        entry form and the alias table stay readable, scrollable and foldable
        but ignore input. Arb Waveforms stays clickable: the host opens the
        ARB dialog read-only under its own rule. Input made before locking stays
        shown. Call again with False to unlock.
        """
        self._read_only = read_only
        self._library_sync.set_read_only(read_only=read_only)
        self._component_tree.set_read_only(read_only=read_only)
        self._aliases.set_read_only(read_only=read_only)
        self._new_btn.setEnabled(not read_only)
        if self._editor is not None:
            self._editor.set_editing_enabled(enabled=not read_only)
        self._update_controls()

    def refresh(self) -> None:
        """Refresh installed values, preserving local edits only on the same point.

        Context-switch publications discard local input before refreshing.
        Source changes retain input for the file banner's Keep/Reload decision.
        """
        self._aliases.refresh()
        self._component_tree.refresh()
        self._library_sync.refresh(has_draft=self._has_library_draft())
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
        if self._applying or self._has_library_draft():
            return
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
                nodes: dict[tuple[str, ...], QTreeWidgetItem] = {(): group}
                for name in sorted(keys):
                    path = tuple(name.split("."))
                    for depth in range(1, len(path) + 1):
                        prefix = path[:depth]
                        if prefix not in nodes:
                            node = QTreeWidgetItem(nodes[prefix[:-1]], [prefix[-1]])
                            node.setFlags(Qt.ItemFlag.ItemIsEnabled)
                            nodes[prefix] = node
                    item = nodes[path]
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    item.setData(0, Qt.ItemDataRole.UserRole, (kind, name))
                    item.setToolTip(0, name)
                    if selected == (kind, name):
                        self._tree.setCurrentItem(item)
        finally:
            self._library_filter.refresh()
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
        self._close_entry()
        self._selected = key
        if key is not None:
            self._open_entry(key)
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

        def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
            entry.state.write(batch, entry.env)
            self._commit_entry()

        def select(path: FormWritePath, value: JsonValue) -> None:
            entry.state.select(path, value, entry.env)
            self._commit_entry()

        def run_command(path: FormWritePath, name: str) -> None:
            entry.state.run_command(path, name, entry.env)
            self._commit_entry()

        editor = FormEditor(
            entry.state,
            entry.env,
            write=self._entry_scope.callback(write),
            select=self._entry_scope.callback(select),
            run_command=self._entry_scope.callback(run_command),
        )
        editor.set_idle_commit(enabled=False)
        editor.set_editing_enabled(enabled=not self._read_only)
        self._entry_scope.add_cleanup(editor.detach)
        # setWidget deletes the previous form, so folds and scroll start fresh.
        self._scroll.setWidget(editor)
        self._entry = entry
        self._library_revision = self._library_sync.capture()
        self._editor = editor
        self._name_edit.blockSignals(True)
        self._name_edit.setText(name)
        self._name_edit.blockSignals(False)

    def _close_entry(self) -> None:
        self._entry_scope.close()
        self._entry_scope = LifetimeScope(self)
        if self._editor is not None:
            self._editor = None
            self._scroll.setWidget(QWidget())
        self._entry = None
        self._entry_error.clear()
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)

    def _has_library_draft(self) -> bool:
        entry, editor, key = self._entry, self._editor, self._selected
        if entry is None or editor is None or key is None:
            return False
        return (
            self._name_edit.text() != key[1]
            or entry.state.is_dirty
            or editor.has_pending_text()
        )

    def _update_controls(self, *_args: object) -> None:
        editable = self._entry is not None and not self._read_only
        self._name_edit.setEnabled(self._entry is not None)
        self._name_edit.setReadOnly(not editable)
        self._delete_btn.setEnabled(
            not self._read_only and self._current_key() is not None
        )

    def _commit_entry(self) -> None:
        """Replace a valid edited entry once; retain rejected input inline."""
        entry, editor, key = self._entry, self._editor, self._selected
        if (
            self._read_only
            or self._applying
            or entry is None
            or editor is None
            or key is None
            or not self._has_library_draft()
        ):
            return
        if editor.errors() is not None:
            return
        new_name = self._name_edit.text().strip()
        if not new_name:
            self._entry_error.setText("Entry name must not be empty.")
            return
        scope = self._entry_scope
        self._applying = True
        try:
            self._library_sync.check(self._library_revision)
            self._ctrl.apply_library_change(
                entry.parameters, entry.replace(key[1], new_name)
            )
        except LIBRARY_WRITE_ERRORS as exc:
            logger.info("Library entry write failed: %s", exc)
            if not scope.closed:
                self._entry_error.setText(str(exc))
            return
        finally:
            self._applying = False
        if scope.closed:
            return
        # Keep the mounted editor, focus and captured expression environment.
        # Reset only the accepted base; no undo history is retained.
        entry.state.reset(entry.state.input)
        self._selected = (key[0], new_name)
        self._name_edit.setText(new_name)
        self._library_revision = self._library_sync.capture()
        self._library = self._ctrl.read_context().library
        self._rebuild_tree(self._library, self._selected)
        self._entry_error.clear()

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
        self._scope.connect(
            dialog.finished, lambda _result: self._on_new_finished(dialog)
        )
        self._new_dialog = dialog
        dialog.open()

    def _on_new_finished(self, dialog: NewEntryDialog) -> None:
        self._new_dialog = None
        created = dialog.created
        if created is None:
            return
        self._close_entry()
        self._selected = created
        self._populate_from_context()

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------

    def _reload_library(self) -> None:
        self._close_entry()
        self._populate_from_context()

    def _on_file_changed(self, payload: PointFileChangedPayload) -> None:
        del payload
        # Source observations alone must not rebuild an active parameter editor.
        self._library_sync.refresh(has_draft=self._has_library_draft())
        self._aliases.refresh()
        self._component_tree.refresh_file_status()

    def _on_bus_refresh(self, payload: SessionPayload) -> None:
        if isinstance(payload, ContextSwitchedPayload):
            self._close_entry()
            self._saved_hint.clear()
        self.refresh()
        if isinstance(payload, (MdChangedPayload, MlChangedPayload)):
            set_foreground(self._saved_hint, success_colour(self))
            self._saved_hint.setText(f"Saved at {datetime.now():%H:%M:%S}")

    def done(self, a0: int) -> None:
        """Retire editor and bus callbacks before any accepted/rejected finish."""
        self._scope.close()
        super().done(a0)


def _has_entry(library: LibrarySnapshot, key: _EntryKey) -> bool:
    kind, name = key
    keys = library.module_keys() if kind == "module" else library.waveform_keys()
    return name in keys


def _find_item(tree: QTreeWidget, key: _EntryKey | None) -> QTreeWidgetItem | None:
    if key is None:
        return None
    for item in tree.findItems(
        "*", Qt.MatchFlag.MatchWildcard | Qt.MatchFlag.MatchRecursive
    ):
        if item.data(0, Qt.ItemDataRole.UserRole) == key:
            return item
    return None


__all__ = ["InspectDialog"]
