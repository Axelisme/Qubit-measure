"""Session-only editor for the ordered global alias table."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.gui.widgets.remote_edit_gate import (
    bind_control_enabled,
    dispose_control_watches,
)


class AliasDialog(QDialog):
    """Edit authored alias rows and explicitly confirm through the session owner.

    Construct on the control's owner thread. The table starts with
    control.get_alias_rows(); initial read failures propagate. Add, Remove and
    cell edits change only this dialog's draft. The read-only resolution column
    projects the installed table on open and after successful Apply, never the
    draft. Without a selected point it stays blank. Source faults appear as
    diagnostics in the column. Apply submits all rows through
    control.apply_alias_rows(), which saves to the opened entry or remains
    memory-only in EMPTY. Apply stays open and shows input/storage errors inline
    without changing the installed table or discarding edits. Cancel or closing
    discards unapplied edits; an earlier successful Apply is not undone.
    Optional control_identity gates content and Apply, never Cancel or closing.
    Omitted authority preserves the behavior of apps without control identity.
    """

    def __init__(
        self,
        control: ContextControlPort,
        parent: QWidget | None = None,
        *,
        control_identity: ControlIdentity | None = None,
    ) -> None:
        super().__init__(parent)
        self._control = control
        self.setWindowTitle("Aliases")
        self.resize(700, 360)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Enter a component name, leave a value empty, or use a $ link "
            "to an earlier row (for example $qubit.resonator)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Alias", "Value", "Resolves to"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self._table.horizontalHeader()
        assert header is not None
        header.setStretchLastSection(True)
        layout.addWidget(self._table)
        for row in control.get_alias_rows():
            self._append_row(row)
        self._refresh_resolutions()

        row_actions = QHBoxLayout()
        add = QPushButton("Add", self)
        add.clicked.connect(self._add_row)
        row_actions.addWidget(add)
        remove = QPushButton("Remove", self)
        remove.clicked.connect(self._remove_rows)
        row_actions.addWidget(remove)
        row_actions.addStretch()
        layout.addLayout(row_actions)

        self._error = QLabel(self)
        self._error.setWordWrap(True)
        layout.addWidget(self._error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        assert apply_button is not None
        apply_button.clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for content in (self._table, add, remove, apply_button):
            bind_control_enabled(content, control_identity, retirement_dialog=self)

    def dispose_control_watches(self) -> None:
        """Retire owned authority projections before final view destruction.

        Hiding a persistent dialog is not retirement. Call on the owner thread
        before its parent shuts down or schedules the view's Qt deletion.
        """
        dispose_control_watches(self)

    def deleteLater(self) -> None:
        """Release authority watches before direct owner-thread Qt deletion."""
        self.dispose_control_watches()
        super().deleteLater()

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

    def _apply(self) -> None:
        rows: list[AliasRow] = []
        for index in range(self._table.rowCount()):
            alias = self._table.item(index, 0)
            value = self._table.item(index, 1)
            if alias is None or value is None:
                raise RuntimeError("Alias editor row is missing a cell")
            rows.append(AliasRow(alias=alias.text(), value=value.text()))
        try:
            self._control.apply_alias_rows(tuple(rows))
        except (InvalidInputError, OSError) as error:
            self._error.setText(str(error))
            return
        self._error.clear()
        self._refresh_resolutions()
