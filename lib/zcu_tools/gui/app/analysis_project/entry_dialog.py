"""Entry and fluxonium-component selection shared by standalone analysis apps."""

import logging
from pathlib import Path

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
    AnalysisTarget,
    fluxonium_components,
)
from zcu_tools.resources.entry import ResultEntry

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
        self.setWindowTitle("Analysis entry")
        self.resize(620, 250)
        self._target: AnalysisTarget | None = None
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
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self._root)
        row.addWidget(browse)
        form.addRow("Project root", row)
        self._entry = QComboBox()
        self._entry.setObjectName("analysisEntry")
        self._entry.setEditable(True)
        form.addRow("Entry", self._entry)
        refresh = QPushButton("Refresh entries")
        refresh.clicked.connect(self._refresh_entries)
        form.addRow(refresh)
        self._open_button = QPushButton("Open entry")
        self._open_button.clicked.connect(self._open_entry)
        form.addRow(self._open_button)
        self._component = QComboBox()
        self._component.setObjectName("analysisComponent")
        form.addRow("Component", self._component)
        self._error = QLabel()
        self._error.setWordWrap(True)
        root.addWidget(self._error)
        self._write_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self._write_buttons.accepted.connect(self.accept)
        cancel = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        cancel.rejected.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self._write_buttons)
        buttons.addWidget(cancel)
        root.addLayout(buttons)
        self._root.textChanged.connect(self._invalidate)
        self._entry.currentTextChanged.connect(self._invalidate)
        self._refresh_entries()
        if target is not None:
            self._entry.setCurrentText(target.entry)
            self._open_entry()
            self._component.setCurrentText(target.component)
        self._update_ok()

    def set_writable(self, *, writable: bool) -> None:
        """Gate entry editing and OK while leaving Cancel available."""
        self._writable = writable
        self._content.setEnabled(writable)
        self._update_ok()

    def selected_target(self) -> AnalysisTarget | None:
        """Return the opened target with the current component, or None if unopened."""
        if self._target is None:
            return None
        return AnalysisTarget.open(
            self._root.text().strip(), self._target.entry, self._component.currentText()
        )

    def _invalidate(self, _text: str = "") -> None:
        self._target = None
        self._component.clear()
        self._update_ok()

    def _update_ok(self) -> None:
        self._write_buttons.setEnabled(self._writable and self._target is not None)

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
            self._entry.addItems(
                ResultEntry.list_names(result_root=Path(self._root.text()) / "result")
            )
        except Exception as error:
            logger.exception("Unable to list analysis entries")
            self._error.setText(str(error))
        self._entry.setCurrentText(current)

    def _open_entry(self) -> None:
        self._invalidate()
        try:
            target = AnalysisTarget.open(
                self._root.text().strip(), self._entry.currentText().strip()
            )
            names = fluxonium_components(target.resource)
        except Exception as error:
            logger.exception("Unable to open analysis target")
            self._error.setText(str(error))
            return
        self._target = target
        self._component.addItems(names)
        self._component.setCurrentText(target.component)
        self._error.clear()
        self._update_ok()
