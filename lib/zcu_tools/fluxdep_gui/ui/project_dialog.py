"""ProjectDialog — set the project's chip / qubit names (and optional paths).

The chip / qubit names locate the default output layout
(``result/<chip>/<qubit>/...``). The ``result_dir`` auto-derives from the names
(``result/<chip>/<qubit>``) unless the user edits / browses it; ``database_path``
is a free root for raw spectra. Both have a Browse button (choose a folder).
Returns a ``ProjectInfo`` (or None on cancel).
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from zcu_tools.fluxdep_gui.services.export import default_result_dir
from zcu_tools.fluxdep_gui.state import ProjectInfo


class ProjectDialog(QDialog):
    """Edit the project chip/qubit names (+ result/database roots, with Browse)."""

    def __init__(self, project: ProjectInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project")
        self.resize(560, 220)

        # The result dir tracks chip/qubit until the user edits / browses it.
        derived = default_result_dir(project.chip_name, project.qub_name)
        self._result_overridden = bool(
            project.result_dir and project.result_dir != derived
        )

        form = QFormLayout(self)
        self._chip_edit = QLineEdit(project.chip_name)
        self._qub_edit = QLineEdit(project.qub_name)
        self._chip_edit.textChanged.connect(self._on_names_changed)
        self._qub_edit.textChanged.connect(self._on_names_changed)
        form.addRow("Chip name", self._chip_edit)
        form.addRow("Qubit name", self._qub_edit)

        self._result_edit = QLineEdit(project.result_dir or derived)
        self._result_edit.textEdited.connect(self._on_result_edited)
        form.addRow("Result dir", self._dir_row(self._result_edit, self._browse_result))

        self._database_edit = QLineEdit(project.database_path)
        form.addRow(
            "Database path", self._dir_row(self._database_edit, self._browse_database)
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _dir_row(self, edit: QLineEdit, on_browse) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(on_browse)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # --- auto-derivation -------------------------------------------------

    def _on_names_changed(self, _text: str) -> None:
        if self._result_overridden:
            return
        self._result_edit.setText(
            default_result_dir(
                self._chip_edit.text().strip(), self._qub_edit.text().strip()
            )
        )

    def _on_result_edited(self, _text: str) -> None:
        # a manual edit detaches the result dir from the chip/qubit derivation
        self._result_overridden = True

    # --- browse ----------------------------------------------------------

    def _browse_result(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select result dir", self._result_edit.text()
        )
        if path:
            self._result_overridden = True
            self._result_edit.setText(path)

    def _browse_database(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select database path", self._database_edit.text()
        )
        if path:
            self._database_edit.setText(path)

    # --- result ----------------------------------------------------------

    def result_project(self) -> ProjectInfo:
        """The edited project info (names trimmed)."""
        return ProjectInfo(
            chip_name=self._chip_edit.text().strip(),
            qub_name=self._qub_edit.text().strip(),
            result_dir=self._result_edit.text().strip(),
            database_path=self._database_edit.text().strip(),
        )
