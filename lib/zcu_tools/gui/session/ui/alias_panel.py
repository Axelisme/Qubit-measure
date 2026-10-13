"""Editor panel for the ordered global alias table, shown as Inspect's Aliases page."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError

_EDIT_TRIGGERS = (
    QAbstractItemView.EditTrigger.DoubleClicked
    | QAbstractItemView.EditTrigger.EditKeyPressed
    | QAbstractItemView.EditTrigger.AnyKeyPressed
)


class AliasPanel(QWidget):
    """Edit authored alias rows and explicitly confirm through the session owner.

    Public child object names: inspect.aliases.table (QTableWidget with the
    columns Alias, Value and Resolves to), the QPushButtons
    inspect.aliases.add, inspect.aliases.remove and inspect.aliases.apply,
    and inspect.aliases.error (the QLabel showing the last rejected Apply).

    Construct on the control's owner thread. The table starts with
    control.get_alias_rows(); initial read failures propagate. Add, Remove and
    cell edits change only this panel's draft. The read-only resolution column
    projects the installed table when rows load and after a successful Apply,
    never the draft. Without a selected point it stays blank; source faults
    appear as diagnostics in the column. The draft is unsaved while its rows
    differ from the rows last loaded or applied. The panel never asks questions;
    its host decides what to do with an unsaved draft.
    """

    def __init__(
        self,
        control: ContextControlPort,
        parent: QWidget | None = None,
    ) -> None:
        """Build the panel and load the installed rows from control."""
        super().__init__(parent)
        self._control = control
        self._base: tuple[AliasRow, ...] = ()
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Enter a component name, leave a value empty, or use a $ link "
            "to an earlier row (for example $qubit.resonator)."
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
        self._apply_btn = self._action_button("Apply", "inspect.aliases.apply")
        self._apply_btn.clicked.connect(self.apply)
        row_actions.addWidget(self._apply_btn)
        layout.addLayout(row_actions)

        self._error = QLabel(self)
        self._error.setObjectName("inspect.aliases.error")
        self._error.setWordWrap(True)
        layout.addWidget(self._error)
        self.revert()

    def _action_button(self, text: str, name: str) -> QPushButton:
        button = QPushButton(text, self)
        button.setObjectName(name)
        return button

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_unsaved(self) -> bool:
        """Return True while the shown rows differ from those last loaded or applied."""
        return self._draft_rows() != self._base

    def apply(self) -> bool:
        """Submit every shown row through control.apply_alias_rows().

        The owner saves to the opened entry or stays memory-only in EMPTY.
        Return True after the table is installed; the resolution column then
        reprojects it. Input and storage errors return False, appear in
        inspect.aliases.error and keep both the installed table and the draft.
        """
        rows = self._draft_rows()
        try:
            self._control.apply_alias_rows(rows)
        except (InvalidInputError, OSError) as error:
            self._error.setText(str(error))
            return False
        self._base = rows
        self._error.clear()
        self._refresh_resolutions()
        return True

    def revert(self) -> None:
        """Replace the draft with the installed rows and clear the error.

        Used to load the rows and to discard a draft. An earlier successful
        Apply is not undone; read failures propagate.
        """
        self._base = self._control.get_alias_rows()
        self._table.setRowCount(0)
        for row in self._base:
            self._append_row(row)
        self._error.clear()
        self._refresh_resolutions()

    def set_read_only(self, *, read_only: bool) -> None:
        """Lock or unlock edits while keeping the rows selectable and scrollable.

        Read-only disables Add, Remove and Apply and stops cell editing. A draft made before locking stays shown.
        """
        for button in (self._add_btn, self._remove_btn, self._apply_btn):
            button.setEnabled(not read_only)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
            if read_only
            else _EDIT_TRIGGERS
        )

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _draft_rows(self) -> tuple[AliasRow, ...]:
        rows: list[AliasRow] = []
        for index in range(self._table.rowCount()):
            alias = self._table.item(index, 0)
            value = self._table.item(index, 1)
            if alias is None or value is None:
                raise RuntimeError("Alias editor row is missing a cell")
            rows.append(AliasRow(alias=alias.text(), value=value.text()))
        return tuple(rows)

    def _append_row(self, row: AliasRow) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._table.setItem(index, 0, QTableWidgetItem(row.alias))
        self._table.setItem(index, 1, QTableWidgetItem(row.value))
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
                item.setForeground(QColor("red"))
            return
        except FailedPreconditionError:
            # No entry or selected point: there is no applied projection to show.
            return
        for index, row in enumerate(rows):
            item = self._result_item(index)
            if not row.value:
                item.setText("(unbound)")
                item.setForeground(QColor("gray"))
            elif row.error is not None:
                item.setText(row.error)
                item.setForeground(QColor("red"))
            else:
                item.setText(row.resolved or "")
                item.setForeground(self._table.palette().text())
            item.setToolTip(row.error or row.resolved or "")

    def _add_row(self) -> None:
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


__all__ = ["AliasPanel"]
