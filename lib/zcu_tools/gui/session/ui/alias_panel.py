"""Editor panel for the ordered global alias table, shown as Inspect's Aliases page."""

from __future__ import annotations

from io import StringIO

from pydantic import TypeAdapter, ValidationError
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import (
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.ui.point_file_banner import PointFileBanner
from zcu_tools.gui.widgets import error_colour, secondary_colour
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.resources.entry.aliases import AliasValue

_EDIT_TRIGGERS = (
    QAbstractItemView.EditTrigger.DoubleClicked
    | QAbstractItemView.EditTrigger.EditKeyPressed
    | QAbstractItemView.EditTrigger.AnyKeyPressed
)


class AliasPanel(QWidget):
    """Write completed alias edits through the session owner.

    Public child object names: inspect.aliases.table (QTableWidget, columns
    Alias, Value, Resolves to), inspect.aliases.add and inspect.aliases.remove
    (QPushButtons), and inspect.aliases.error (QLabel for rejected writes).

    Construct on the control's owner thread. Initial read failures propagate.
    Completing a cell edit or removing rows sends the entire valid table to
    apply_alias_rows. Add stays local until the new row has a valid alias;
    an empty Value is a legal unbound alias. Invalid rows are red and never
    write. Owner/storage errors keep the input and installed table unchanged.
    The read-only resolution column projects installed rows, not invalid input.
    Without a point it stays blank; source faults appear in that column.
    refresh discards local rows on point/entry switches and otherwise keeps
    uncommitted input for PointFileBanner's Keep/Reload revision flow.
    The panel has no Apply, Revert or close-time questions.
    """

    def __init__(
        self,
        control: ContextControlPort,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Build the panel and load the installed rows from control."""
        super().__init__(parent)
        self._control = control
        self._base: tuple[AliasRow, ...] = ()
        self._target: SavedPoint | None = None
        self._applying = False
        self._read_only = False
        layout = QVBoxLayout(self)
        self._sync = PointFileBanner(control, "aliases", self._load, parent=self)
        layout.addWidget(self._sync)
        hint = QLabel(
            "Enter a component name, leave a value empty, or use a $ link "
            "to an earlier row (for example $qubit.resonator), or an ordered "
            "list such as [Q3, Q1]. An empty list is unbound."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(0, 3, self)
        self._table.setObjectName("inspect.aliases.table")
        self._table.setHorizontalHeaderLabels(["Alias", "Value", "Resolves to"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(_EDIT_TRIGGERS)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setStretchLastSection(True)
        layout.addWidget(self._table)

        row_actions = QHBoxLayout()
        self._add_btn = self._action_button("Add", "inspect.aliases.add")
        self._add_btn.clicked.connect(self._add_row)
        self._remove_btn = self._action_button("Remove", "inspect.aliases.remove")
        self._remove_btn.clicked.connect(self._remove_rows)
        row_actions.addWidget(self._add_btn)
        row_actions.addWidget(self._remove_btn)
        row_actions.addStretch()
        layout.addLayout(row_actions)

        self._error = QLabel(self)
        self._error.setObjectName("inspect.aliases.error")
        self._error.setWordWrap(True)
        make_label_selectable(self._error)
        layout.addWidget(self._error)
        self._load()
        self._table.cellChanged.connect(self._cell_finished)

    def _action_button(self, text: str, name: str) -> QPushButton:
        button = QPushButton(text, self)
        button.setObjectName(name)
        return button

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _has_draft(self) -> bool:
        """Read pending rows and live text without committing the active cell."""
        try:
            return self._draft_rows() != self._base
        except InvalidInputError:
            return True

    def _commit(self) -> None:
        """Write a valid complete table; keep failures and input on screen."""
        if self._read_only or self._applying:
            return
        self._applying = True
        try:
            rows = self._validated_rows()
            self._sync.check(self._revision)
            self._control.apply_alias_rows(rows)
        except (FailedPreconditionError, InvalidInputError, OSError) as error:
            self._error.setText(str(error))
            return
        finally:
            self._applying = False
        self._base = rows
        self._revision = self._sync.capture()
        self._error.clear()
        self._refresh_resolutions()

    def _load(self) -> None:
        """Discard local input and load the currently selected point."""
        self._target = self._current_target()
        self._base = self._control.get_alias_rows()
        self._revision = self._sync.capture()
        self._show_rows(self._base)
        self._refresh_resolutions()

    def refresh(self) -> None:
        """Follow point/entry switches and preserve same-point local input.

        Called after session facts on the owner thread. Read failures propagate.
        """
        if self._applying:
            return
        if self._current_target() != self._target or not self._has_draft():
            self._load()
        self._sync.refresh(has_draft=self._has_draft())

    def _current_target(self) -> SavedPoint | None:
        info = self._control.get_entry_info()
        label = self._control.current_point_label()
        if info is None or label is None:
            return None
        return SavedPoint(entry_id=info.entry_id, point=label)

    def set_read_only(self, *, read_only: bool) -> None:
        """Lock or unlock edits while keeping the rows selectable and scrollable.

        Read-only disables Add and Remove and stops cell editing.
        A draft made before locking stays shown.
        """
        self._read_only = read_only
        self._sync.set_read_only(read_only=read_only)
        for button in (self._add_btn, self._remove_btn):
            button.setEnabled(not read_only)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
            if read_only
            else _EDIT_TRIGGERS
        )

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _cell_finished(self, _row: int, column: int) -> None:
        if column < 2:
            self._commit()

    def _validated_rows(self) -> tuple[AliasRow, ...]:
        """Use the alias owner to validate and mark each invalid row."""
        with QSignalBlocker(self._table):
            for index in range(self._table.rowCount()):
                for column in (0, 1):
                    item = self._table.item(index, column)
                    if item is not None:
                        item.setForeground(self._table.palette().text())
                        item.setToolTip("")
            rows: list[AliasRow] = []
            first_error: InvalidInputError | None = None
            for index in range(self._table.rowCount()):
                alias = self._table.item(index, 0)
                value = self._table.item(index, 1)
                if alias is None or value is None:
                    raise RuntimeError("Alias editor row is missing a cell")
                try:
                    row = AliasRow(alias.text(), _parse_value(value.text()))
                    AliasTable([*rows, row])
                except InvalidInputError as error:
                    if first_error is None:
                        first_error = error
                    for item in (alias, value):
                        item.setForeground(error_colour(self._table))
                        item.setToolTip(str(error))
                else:
                    rows.append(row)
            if first_error is not None:
                raise first_error
            return tuple(rows)

    def _draft_rows(self) -> tuple[AliasRow, ...]:
        # Read live input without committing it or triggering cellChanged.
        editor = self._table.focusWidget()
        rows: list[AliasRow] = []
        for index in range(self._table.rowCount()):
            alias = self._table.item(index, 0)
            value = self._table.item(index, 1)
            if alias is None or value is None:
                raise RuntimeError("Alias editor row is missing a cell")
            alias_text, value_text = alias.text(), value.text()
            if (
                isinstance(editor, QLineEdit)
                and editor.isVisible()
                and self._table.currentRow() == index
            ):
                if self._table.currentColumn() == 0:
                    alias_text = editor.text()
                elif self._table.currentColumn() == 1:
                    value_text = editor.text()
            rows.append(AliasRow(alias=alias_text, value=_parse_value(value_text)))
        return tuple(rows)

    def _show_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Replace every draft row with rows and clear the error."""
        with QSignalBlocker(self._table):
            self._table.setRowCount(0)
            for row in rows:
                self._append_row(row)
        self._error.clear()

    def _append_row(self, row: AliasRow) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._table.setItem(index, 0, QTableWidgetItem(row.alias))
        self._table.setItem(index, 1, QTableWidgetItem(_display_value(row.value)))
        result = QTableWidgetItem()
        result.setFlags(result.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(index, 2, result)

    def _result_item(self, index: int) -> QTableWidgetItem:
        """Return row index's resolution cell, which _append_row always creates."""
        item = self._table.item(index, 2)
        if item is None:
            raise RuntimeError(f"Alias row {index} has no resolution cell")
        return item

    def _refresh_resolutions(self) -> None:
        with QSignalBlocker(self._table):
            self._show_resolutions()

    def _show_resolutions(self) -> None:
        for index in range(self._table.rowCount()):
            item = self._result_item(index)
            item.setText("")
            item.setToolTip("")
            item.setForeground(self._table.palette().text())
        try:
            rows = self._control.read_context().resolved_aliases()
        except ContextSourceUnavailableError as error:
            # A source fault prevents the whole projection; keep it visible.
            for index in range(self._table.rowCount()):
                item = self._result_item(index)
                item.setText(str(error))
                item.setToolTip(str(error))
                item.setForeground(error_colour(self._table))
            return
        except FailedPreconditionError:
            # No entry or selected point: there is no applied projection to show.
            return
        for index, row in enumerate(rows[: self._table.rowCount()]):
            item = self._result_item(index)
            if not row.value:
                item.setText("(unbound)")
                item.setForeground(secondary_colour(self._table))
            elif row.error is not None:
                item.setText(row.error)
                item.setForeground(error_colour(self._table))
            else:
                item.setText(
                    _display_value(row.resolved) if row.resolved is not None else ""
                )
                item.setForeground(self._table.palette().text())
            item.setToolTip(
                row.error
                or (_display_value(row.resolved) if row.resolved is not None else "")
            )

    def _add_row(self) -> None:
        with QSignalBlocker(self._table):
            self._append_row(AliasRow(alias="", value=""))
        index = self._table.rowCount() - 1
        self._table.setCurrentCell(index, 0)
        item = self._table.item(index, 0)
        assert item is not None
        self._table.editItem(item)

    def _remove_rows(self) -> None:
        rows = {index.row() for index in self._table.selectedIndexes()}
        for index in sorted(rows, reverse=True):
            self._table.removeRow(index)
        self._commit()


def _display_value(value: AliasValue) -> str:
    if isinstance(value, str):
        return value
    yaml = YAML(typ="safe")
    yaml.default_flow_style = True
    stream = StringIO()
    yaml.dump(value, stream)
    return stream.getvalue().strip()


def _parse_value(text: str) -> AliasValue:
    if not text.lstrip().startswith("["):
        return text
    try:
        raw = YAML(typ="safe").load(text)
        return TypeAdapter(list[str]).validate_python(raw, strict=True)
    except (YAMLError, ValidationError) as error:
        raise InvalidInputError(
            f"Invalid alias list: {error}",
            reason_code="alias_invalid",
        ) from error


__all__ = ["AliasPanel"]
