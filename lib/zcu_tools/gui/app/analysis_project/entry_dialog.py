"""Entry and fluxonium-component selection shared by standalone analysis apps."""

import logging
from pathlib import Path

from pydantic import ValidationError
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.analysis_project.entry import (
    AnalysisEntry,
    AnalysisTarget,
)
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.resources.entry import RoleResolutionError

logger = logging.getLogger(__name__)


class EntryDialog(QDialog):
    """Open existing entries under project_root/result and choose their target.

    Open reads the entry and its qubit alias. Component choices are fluxonium
    qubits only. Read/validation failures remain visible in the dialog. OK returns
    the selected AnalysisTarget; Cancel returns no target. Never creates entries
    or saves to Database. set_writable gates every write-confirming control.
    """

    def __init__(
        self,
        target: AnalysisTarget | None,
        parent: QWidget | None = None,
        *,
        project_root: str | Path,
    ) -> None:
        super().__init__(parent)
        self._scope = LifetimeScope(self)
        self.setWindowTitle("Analysis entry")
        self.resize(620, 250)
        self._opened: AnalysisEntry | None = None
        self._writable = True
        self._content = QWidget()
        form = QFormLayout(self._content)
        root = QVBoxLayout(self)
        root.addWidget(self._content)
        current_root = (
            target.result_root.parent if target is not None else Path(project_root)
        )
        self._root = QLineEdit(str(current_root))
        self._root.setObjectName("analysisProjectRoot")
        browse = QPushButton("Browse…")
        self._scope.connect(browse.clicked, self._browse)
        row = QHBoxLayout()
        row.addWidget(self._root)
        row.addWidget(browse)
        form.addRow("Project root", row)
        self._entry = QComboBox()
        self._entry.setObjectName("analysisEntry")
        self._entry.setEditable(True)
        form.addRow("Entry", self._entry)
        refresh = QPushButton("Refresh entries")
        self._scope.connect(refresh.clicked, self._refresh_entries)
        form.addRow(refresh)
        self._open_button = QPushButton("Open entry")
        self._scope.connect(self._open_button.clicked, self._open_entry)
        form.addRow(self._open_button)
        self._component = QComboBox()
        self._component.setObjectName("analysisComponent")
        form.addRow("Component", self._component)
        self._error = QLabel()
        self._error.setWordWrap(True)
        root.addWidget(self._error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None
        self._ok = ok
        self._scope.connect(buttons.accepted, lambda: self.accept())
        self._scope.connect(buttons.rejected, lambda: self.reject())
        root.addWidget(buttons)
        self._connect_inputs()
        self._refresh_entries()
        if target is not None:
            self._entry.setCurrentText(target.entry)
            self._open_entry()
            self._component.setCurrentText(target.component)
        self._update_ok()

    def _connect_inputs(self) -> None:
        self._scope.connect(self._root.textChanged, self._invalidate)
        self._scope.connect(self._entry.currentTextChanged, self._invalidate)
        self._scope.connect(self._component.currentTextChanged, self._update_ok)

    def done(self, a0: int) -> None:
        """Retire input callbacks before accepting or cancelling the selection."""
        self._scope.close()
        super().done(a0)

    def set_writable(self, *, writable: bool) -> None:
        """Gate entry editing and OK while leaving Cancel available."""
        self._writable = writable
        self._content.setEnabled(writable)
        self._update_ok()

    def selected_target(self) -> AnalysisTarget | None:
        """Return the opened target with the current component, or None if unopened."""
        component = self._component.currentText()
        if self._opened is None or not component:
            return None
        return self._opened.select(component)

    def _invalidate(self, _text: str = "") -> None:
        self._opened = None
        self._component.clear()
        self._update_ok()

    def _update_ok(self) -> None:
        selected = self._opened is not None and bool(self._component.currentText())
        self._ok.setEnabled(self._writable and selected)
        if selected:
            self._error.clear()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Project root",
            self._root.text(),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self._root.setText(folder)
            self._refresh_entries()

    def _refresh_entries(self) -> None:
        current = self._entry.currentText()
        self._entry.clear()
        try:
            self._entry.addItems(AnalysisEntry.list_names(self._root.text()))
        except Exception as error:
            logger.exception("Unable to list analysis entries")
            self._error.setText(str(error))
        self._entry.setCurrentText(current)

    def _open_entry(self) -> None:
        self._invalidate()
        try:
            opened = AnalysisEntry.open(
                self._root.text().strip(), self._entry.currentText().strip()
            )
        except Exception as error:
            logger.exception("Unable to open analysis target")
            self._error.setText(str(error))
            return
        self._opened = opened
        self._component.addItems(opened.components)
        self._component.setCurrentIndex(-1)
        try:
            target = opened.select()
        except (InvalidInputError, RoleResolutionError, ValidationError) as error:
            self._error.setText(str(error))
        else:
            self._component.setCurrentText(target.component)
            self._error.clear()
        self._update_ok()
