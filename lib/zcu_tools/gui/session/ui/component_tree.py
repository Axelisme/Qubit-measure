"""Component tree frontend for the session's parameter owner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import yaml
from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from qtpy.QtGui import QBrush, QKeySequence, QPalette, QShortcut
from qtpy.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import PointFileRevision, SavedPoint
from zcu_tools.gui.session.ui.add_component_dialog import AddComponentDialog
from zcu_tools.gui.session.ui.component_editing import (
    edit_choices,
    edit_completions,
    field_tooltip,
    parse_field_input,
    set_error_foreground,
    show_error,
)
from zcu_tools.gui.session.ui.point_file_banner import PointFileBanner
from zcu_tools.gui.session.ui.tree_filter import TreeFilter
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.resources.entry import (
    FieldValueKind,
    component_field_specs,
    document_fields,
    document_meta,
    document_value,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument


logger = logging.getLogger(__name__)

_FIELD, _VALUE, _UNIT, _METADATA = range(4)
_UNSET_TEXT = "(unset)"

# A row key is the component/general root followed by its Field-column texts.
type _RowKey = tuple[str, ...]


@dataclass(frozen=True)
class _EditTarget:
    """What an in-place edit of one Value cell submits.

    component names the owning component or "general" root; segments is the
    field path inside it. value_kind selects the parsing rule for the submitted text. choices
    are drop-down options; empty means a free-text editor. completions are
    component names a free-text editor offers while typing (empty for none).
    initial is the editor's start text, "" for an unset field. write_depth is
    how many leading segments the owner path addresses; the remaining
    segments nest inside the written value because their ancestor is unset.
    """

    component: str
    segments: tuple[str, ...]
    value_kind: FieldValueKind
    choices: tuple[str, ...]
    completions: tuple[str, ...]
    initial: str
    write_depth: int
    revision: PointFileRevision | None = None
    setup: bool = False
    entry_id: str | None = None
    point: SavedPoint | None = None
    draft: str | None = None


class ComponentTreeWidget(QWidget):
    """Inspect and edit components and general extensions through ContextControlPort.

    The widget owns presentation only. The tree has Field, Value, Unit and
    Metadata columns. Columns can be resized by dragging their headers; Metadata
    fills spare width, and refresh preserves column widths. Metadata shows a
    row's opaque caller mapping as one-line flow YAML with a block YAML tooltip
    on each field row, including group
    and unset rows, blank when absent or empty; no key is interpreted. Each component row shows its kind; its children follow
    component_field_specs(kind), so unset fields stay visible as an italic
    gray "(unset)" and nested models expand into child rows. Document data
    the schema does not declare follows the declared rows. A separate general/ext
    branch shows the selected document's extension fields, including nested
    mappings; it is not a registered component. Roots start collapsed. The Filter
    box matches names and dotted paths without case sensitivity and reveals matches
    with their ancestors. Clearing it restores browsing expansion, including across
    refreshes for this widget's lifetime. Double-click or F2
    opens an in-place editor on an editable Value cell: Enter or focus loss
    submits, Esc cancels. Choice, boolean and role reference fields edit
    through a drop-down (references list the document's components matching
    the role); other text fields complete component names. All writes use the owner; errors stay visible in
    red below the tree as "<field>: <message>", with the full owner text as
    the tooltip, without changing the rejected document. A point edit captures
    its file revision when the editor opens. A rejected value remains visible
    and can be edited again; Keep never advances its baseline. Reload discards
    retained text only after the owner validates and installs the source.
    A setup/current
    selector controls the editing destination; "Add component…" opens an
    AddComponentDialog for that destination.
    """

    def __init__(
        self,
        control: ContextControlPort,
        parent: QWidget | None = None,
    ) -> None:
        """Bind the owner-thread control and render its current document.

        Without an entry or parameter context the tree is empty, the owner's
        message shows as a hint and "Add component…" is disabled. Field edits
        go through the source owner; setup mode edits only the entry template.
        set_read_only disables writes while retaining tree browsing.
        """
        super().__init__(parent)
        self._control = control
        self._read_only = False
        self._edit_targets: dict[_RowKey, _EditTarget] = {}
        self._drafts: dict[_RowKey, str] = {}
        layout = QVBoxLayout(self)
        self._sync = PointFileBanner(
            control, "parameters", self._reload_file, parent=self
        )
        layout.addWidget(self._sync)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Destination:"))
        self._destination = QComboBox()
        self._destination.addItems(["Current point", "Setup template"])
        self._destination.currentIndexChanged.connect(self.refresh)
        toolbar.addWidget(self._destination)
        toolbar.addStretch(1)
        self._add_btn = QPushButton("Add component…")
        self._add_btn.setObjectName("component_add")
        self._add_btn.clicked.connect(self.open_add_dialog)
        toolbar.addWidget(self._add_btn)
        layout.addLayout(toolbar)
        self._tree = self._build_tree()
        self._filter = TreeFilter(
            self._tree,
            object_name="inspect.parameters.filter",
            default_expanded=False,
        )
        layout.addWidget(self._filter)
        layout.addWidget(self._tree, stretch=1)
        self._error = QLabel()
        make_label_selectable(self._error)
        self._error.setObjectName("component_error")
        self._error.setWordWrap(True)
        set_error_foreground(self._error, error=True)
        layout.addWidget(self._error)
        self._hint = QLabel()
        self._hint.setObjectName("component_hint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self.refresh()

    def set_read_only(self, *, read_only: bool) -> None:
        """Block Value editing and Add while preserving tree/destination browsing.

        Lock any active Value editor without discarding its text. Late
        delegate submissions cannot write while read_only is True.
        """
        self._read_only = read_only
        self._sync.set_read_only(read_only=read_only)
        for editor in self._tree.findChildren(QLineEdit):
            editor.setReadOnly(read_only)
        for combo in self._tree.findChildren(QComboBox):
            combo.setEnabled(not read_only)
        self._add_btn.setEnabled(not read_only)
        self.refresh()

    def refresh(self) -> None:
        """Render the cached selected document; never implicitly refresh files.

        An active or rejected draft keeps its tree and original baseline.
        Otherwise expanded rows and the current row survive a refresh and the
        error text is cleared. FailedPreconditionError (no entry or no parameter context)
        empties the tree, shows the owner's message as a hint and disables Add.
        """
        self.refresh_file_status()
        if self._drafts:
            return
        current = self._tree.currentItem()
        selected = _row_key(current) if current is not None else None
        self._tree.clear()
        self._edit_targets.clear()
        self._clear_error()
        self._add_btn.setEnabled(False)
        try:
            document = self._control.inspect_document(setup=self._is_setup())
        except FailedPreconditionError as exc:
            self._show_hint(str(exc))
            return
        self._show_hint("")
        self._add_btn.setEnabled(not self._read_only)
        items: dict[_RowKey, QTreeWidgetItem] = {}
        self._render_general_ext(document, items)
        for name, component in document.components.items():
            root = QTreeWidgetItem(self._tree, [name, component.kind, "", ""])
            items[(name,)] = root
            self._render_declared(document, name, component.kind, items)
            self._render_undeclared(document, name, component.kind, items)
        self._filter.refresh()
        if selected in items:
            self._tree.setCurrentItem(items[selected])

    def refresh_file_status(self) -> None:
        """Update source notices without closing an active or rejected draft."""
        self._sync.refresh()

    def _reload_file(self) -> None:
        self._drafts.clear()
        self.refresh()

    def open_add_dialog(self) -> AddComponentDialog:
        """Open the Add component dialog for the current destination and return it.

        The dialog is window-modal and opens without blocking (QDialog.open).
        When it adds a component, the tree refreshes and selects the new
        component's row. The dialog deletes itself after it closes.
        """
        dialog = AddComponentDialog(
            self._control,
            setup=self._is_setup(),
            on_added=self._show_added,
            parent=self,
        )
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()
        return dialog

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> bool:
        """Add a component through the owner without the dialog; True on success.

        Success refreshes the tree and selects the new component. Recoverable
        validation/resource failures display red text in the component_error
        QLabel and return False; no default or empty value hides the failure.
        """
        try:
            self._control.add_component(name, kind, fields, setup=setup)
        except Exception as exc:
            logger.exception("Component addition failed")
            show_error(self._error, exc)
            return False
        self._show_added(name)
        return True

    def edit_parameter(
        self,
        path: str,
        value: YamlValue,
        *,
        setup: bool = False,
        target: SavedPoint | None = None,
    ) -> bool:
        """Edit a concrete field through its owner; report errors as for Add.

        target pins a point edit to its original destination. When omitted,
        capture the currently selected point. setup edits the entry template.
        """
        try:
            if setup:
                self._control.set_setup_parameter(path, value)
            else:
                point = target or self._control.read_context().target
                self._control.set_parameter(point, path, value)
        except Exception as exc:
            logger.exception("Parameter edit failed")
            show_error(self._error, exc)
            return False
        self.refresh()
        return True

    def _is_setup(self) -> bool:
        return self._destination.currentIndex() == 1

    def _show_added(self, name: str) -> None:
        self.refresh()
        for index in range(self._tree.topLevelItemCount()):
            root = self._tree.topLevelItem(index)
            if root is not None and root.text(_FIELD) == name:
                self._tree.setCurrentItem(root)
                return
        raise LookupError(f"Added component {name!r} is missing from the tree")

    def _clear_error(self) -> None:
        self._error.clear()
        self._error.setToolTip("")

    def _show_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))

    # ------------------------------------------------------------------
    # Tree rendering
    # ------------------------------------------------------------------

    def _build_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setObjectName("component_tree")
        tree.setHeaderLabels(["Field", "Value", "Unit", "Metadata"])
        # Editing opens only on Value cells of editable rows, never on names.
        tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        delegate = _ValueDelegate(self._edit_target, self._submit_edit, tree)
        delegate.closeEditor.connect(self._editor_closed)
        tree.setItemDelegateForColumn(_VALUE, delegate)
        tree.itemDoubleClicked.connect(self._on_double_click)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_F2), tree)
        shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        shortcut.activated.connect(self._edit_current)
        header = tree.header()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        # Seed widths only once so document rebuilds retain the user's sizing.
        tree.setColumnWidth(_FIELD, 220)
        tree.setColumnWidth(_VALUE, 240)
        tree.setColumnWidth(_UNIT, 80)
        return tree

    def _render_general_ext(
        self,
        document: SetupDocument | PointDocument,
        items: dict[_RowKey, QTreeWidgetItem],
    ) -> None:
        """Render general extensions without treating general as a component kind."""
        general = QTreeWidgetItem(self._tree, ["general", "", "", ""])
        items[("general",)] = general
        items[("general", "ext")] = QTreeWidgetItem(general, ["ext", "", "", ""])
        for field in document_fields(document):
            key = field.segments
            if key[:2] != ("general", "ext"):
                continue
            for depth in range(2, len(key) + 1):
                prefix = key[:depth]
                if prefix not in items:
                    items[prefix] = QTreeWidgetItem(
                        items[prefix[:-1]], [prefix[-1], "", "", ""]
                    )
                item = items[prefix]
                if _addressable(prefix):
                    path = ".".join(prefix)
                    item.setToolTip(_FIELD, path)
                    self._show_metadata(item, document_meta(document, path))
            item = items[key]
            item.setText(_VALUE, _display(field.value))
            self._show_metadata(item, field.metadata)
            if field.editable_path is not None:
                self._add_target(
                    item,
                    _EditTarget(
                        component="general",
                        segments=key[1:],
                        value_kind="text" if isinstance(field.value, str) else "yaml",
                        choices=(),
                        completions=(),
                        initial=_display(field.value),
                        write_depth=len(key) - 1,
                    ),
                )

    def _render_declared(
        self,
        document: SetupDocument | PointDocument,
        name: str,
        kind: str,
        items: dict[_RowKey, QTreeWidgetItem],
    ) -> None:
        """Add one row per declared field below the component's root row."""
        component_value = document_value(document, name)
        for spec in component_field_specs(kind):
            key = (name, *spec.segments)
            present, value = _value_at(component_value, spec.segments)
            item = QTreeWidgetItem(
                items[key[:-1]], [spec.segments[-1], "", spec.unit or "", ""]
            )
            items[key] = item
            path = ".".join(key)
            item.setToolTip(_FIELD, field_tooltip(spec, path))
            self._show_metadata(item, document_meta(document, path))
            if spec.value_kind == "group":
                if not present:
                    self._mark_unset(item)
                continue
            if present:
                item.setText(_VALUE, _display(value))
            else:
                self._mark_unset(item)
            if _addressable(key):
                self._add_target(
                    item,
                    _EditTarget(
                        component=name,
                        segments=spec.segments,
                        value_kind=spec.value_kind,
                        choices=edit_choices(document, spec),
                        completions=edit_completions(document, spec),
                        initial=_display(value) if present else "",
                        write_depth=_write_depth(component_value, spec.segments),
                    ),
                )

    def _render_undeclared(
        self,
        document: SetupDocument | PointDocument,
        name: str,
        kind: str,
        items: dict[_RowKey, QTreeWidgetItem],
    ) -> None:
        """Keep document data that the kind's schema does not declare visible."""
        specs = component_field_specs(kind)
        groups = {spec.segments for spec in specs if spec.value_kind == "group"}
        leaves = {spec.segments for spec in specs if spec.value_kind != "group"}
        for field in document_fields(document):
            if field.segments[0] != name:
                continue
            relative = field.segments[1:]
            prefixes = [relative[:count] for count in range(1, len(relative) + 1)]
            if (
                relative == ("kind",)
                or relative in groups
                or any(prefix in leaves for prefix in prefixes)
            ):
                continue
            depth = max(
                count
                for count in range(len(relative))
                if (name, *relative[:count]) in items
            )
            label = ".".join(relative[depth:])
            item = QTreeWidgetItem(
                items[(name, *relative[:depth])], [label, _display(field.value), "", ""]
            )
            items[(name, *relative[:depth], label)] = item
            item.setToolTip(_FIELD, field.editable_path or ".".join(field.segments))
            self._show_metadata(item, field.metadata)
            if field.editable_path is not None:
                self._add_target(
                    item,
                    _EditTarget(
                        component=name,
                        segments=relative,
                        value_kind="text" if isinstance(field.value, str) else "yaml",
                        choices=(),
                        completions=(),
                        initial=_display(field.value),
                        write_depth=len(relative),
                    ),
                )

    def _add_target(self, item: QTreeWidgetItem, target: _EditTarget) -> None:
        if not self._read_only:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._edit_targets[_row_key(item)] = target

    def _mark_unset(self, item: QTreeWidgetItem) -> None:
        item.setText(_VALUE, _UNSET_TEXT)
        font = item.font(_VALUE)
        font.setItalic(True)
        item.setFont(_VALUE, font)
        gray = self._tree.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text
        )
        item.setForeground(_VALUE, QBrush(gray))

    @staticmethod
    def _show_metadata(item: QTreeWidgetItem, metadata: YamlMap | None) -> None:
        if not metadata:
            return
        # Caller key order is kept; one line avoids PyYAML's default wrapping.
        summary = yaml.dump(
            metadata,
            Dumper=_OneLineDumper,
            default_flow_style=True,
            sort_keys=False,
            width=sys.maxsize,
        )
        item.setText(_METADATA, summary.strip())
        details = yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False)
        item.setToolTip(_METADATA, details.strip())

    # ------------------------------------------------------------------
    # In-place editing
    # ------------------------------------------------------------------

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if not self._read_only and _row_key(item) in self._edit_targets:
            self._tree.editItem(item, _VALUE)

    def _edit_current(self) -> None:
        item = self._tree.currentItem()
        if item is not None:
            self._on_double_click(item, _VALUE)

    def _edit_target(self, index: QModelIndex) -> _EditTarget:
        item = self._tree.itemFromIndex(index)
        if item is None:
            raise LookupError("Edited index has no tree row")
        key = _row_key(item)
        target = self._edit_targets[key]
        if key not in self._drafts:
            setup = self._is_setup()
            entry = self._control.get_entry_info()
            target = replace(
                target,
                setup=setup,
                entry_id=None if entry is None else entry.entry_id,
                point=None if setup else self._control.read_context().target,
                revision=None if setup else self._sync.capture(),
            )
            self._edit_targets[key] = target
            self._drafts[key] = target.initial
        return replace(target, draft=self._drafts[key])

    def _editor_closed(self, *_args: object) -> None:
        # Esc or an unchanged commit leaves no unsaved value behind.
        for key in tuple(self._drafts):
            if self._drafts[key] == self._edit_targets[key].initial:
                self._drafts.pop(key)

    def _submit_edit(self, target: _EditTarget, raw: str) -> None:
        key = (target.component, *target.segments)
        self._drafts[key] = raw
        item = self._tree.currentItem()
        if item is not None and _row_key(item) == key:
            item.setText(_VALUE, raw)
        if raw == target.initial:
            self._drafts.pop(key, None)
            return  # Unchanged text writes nothing.
        if self._read_only:
            return  # Closing a locked editor retains text, but cannot submit.
        try:
            entry = self._control.get_entry_info()
            current_revision = self._sync.capture()
            if (
                target.setup != self._is_setup()
                or target.entry_id != (None if entry is None else entry.entry_id)
                or (
                    not target.setup
                    and (
                        current_revision is None
                        or target.revision is None
                        or current_revision.path != target.revision.path
                    )
                )
            ):
                raise FailedPreconditionError(
                    "Parameter draft belongs to another destination. "
                    "Return to its source or Reload before editing this destination."
                )
            self._sync.check(target.revision)
        except FailedPreconditionError as exc:
            self._sync.show_conflict(str(exc))
            show_error(self._error, exc)
            return
        try:
            value = parse_field_input(target.value_kind, raw)
        except ValueError as exc:
            show_error(self._error, exc, field=".".join(target.segments))
            return
        for segment in reversed(target.segments[target.write_depth :]):
            nested: YamlMap = {segment: value}
            value = nested
        path = ".".join((target.component, *target.segments[: target.write_depth]))
        if self.edit_parameter(path, value, setup=target.setup, target=target.point):
            self._drafts.pop(key, None)
            self.refresh()


class _ValueDelegate(QStyledItemDelegate):
    """Value-column editor that submits through the widget, never the item model.

    Qt's delegate commits on Enter or focus loss and closes without a commit
    on Esc. target(index) returns the edited row's _EditTarget; submit(target,
    raw) receives the editor text. The host retains rejected text in the tree
    and supplies it again when the user reopens that draft.
    """

    def __init__(
        self,
        target: Callable[[QModelIndex], _EditTarget],
        submit: Callable[[_EditTarget, str], None],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._target = target
        self._submit = submit

    def createEditor(
        self, parent: QWidget | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget | None:
        del option
        target = self._target(index)
        if not target.choices:
            editor = QLineEdit(parent)
            editor.setText(target.initial if target.draft is None else target.draft)
            if target.completions:
                editor.setCompleter(QCompleter(list(target.completions), editor))
            return editor
        combo = QComboBox(parent)
        if not target.initial:
            combo.addItem("")  # Blank start option: leaving it selected writes nothing.
        elif target.initial not in target.choices:
            combo.addItem(target.initial)  # Keep a dangling reference selectable.
        combo.addItems(list(target.choices))
        combo.setCurrentText(target.initial if target.draft is None else target.draft)
        combo.activated.connect(lambda _row: self._finish(combo))
        return combo

    def setEditorData(self, editor: QWidget | None, index: QModelIndex) -> None:
        # createEditor already chose the start text; the cell shows display text.
        del editor, index

    def setModelData(
        self,
        editor: QWidget | None,
        model: QAbstractItemModel | None,
        index: QModelIndex,
    ) -> None:
        del model
        if isinstance(editor, QComboBox):
            raw = editor.currentText()
        elif isinstance(editor, QLineEdit):
            raw = editor.text()
        else:
            raise TypeError(f"Unexpected Value editor {editor!r}")
        self._submit(self._target(index), raw)

    def _finish(self, combo: QComboBox) -> None:
        self.commitData.emit(combo)
        self.closeEditor.emit(combo, QAbstractItemDelegate.EndEditHint.NoHint)


class _OneLineDumper(yaml.SafeDumper):
    """SafeDumper whose strings with line breaks use escaped double quotes."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # Plain and single-quoted scalars write a newline as a real line break.
    style = '"' if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_OneLineDumper.add_representer(str, _represent_str)


def _row_key(item: QTreeWidgetItem) -> _RowKey:
    texts: list[str] = []
    node: QTreeWidgetItem | None = item
    while node is not None:
        texts.append(node.text(_FIELD))
        node = node.parent()
    return tuple(reversed(texts))


def _value_at(
    component: YamlValue, segments: tuple[str, ...]
) -> tuple[bool, YamlValue]:
    """Return (present, value) for segments inside a logical component mapping.

    Explicit null counts as present; a missing key or non-mapping ancestor
    returns (False, None).
    """
    node = component
    for segment in segments:
        if not isinstance(node, dict) or segment not in node:
            return False, None
        node = node[segment]
    return True, node


def _write_depth(component: YamlValue, segments: tuple[str, ...]) -> int:
    """Return how many leading segments an owner set path may address.

    Owner paths need every ancestor to hold a mapping. The first missing or
    null ancestor ends the path; the remaining segments nest inside the value.
    """
    node = component
    for index, segment in enumerate(segments[:-1]):
        node = node.get(segment) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return index + 1
    return len(segments)


def _addressable(key: _RowKey) -> bool:
    # Dotted owner paths cannot name empty keys or keys containing a dot.
    return all(part and "." not in part for part in key)


def _display(value: YamlValue) -> str:
    if isinstance(value, str):
        return value
    return yaml.safe_dump(value, default_flow_style=True).removesuffix("...\n").strip()
