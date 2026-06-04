"""ProjectDialog — set the project's chip / qubit names (and optional paths).

The chip / qubit names locate the default output layout
(``result/<chip>/<qubit>/data/fluxdep/...``) used by Export. Setting them here
(once) keeps the Export dialog to just a path field. ``result_dir`` /
``database_path`` are optional roots (left blank, the notebook-layout default
applies). Returns a ``ProjectInfo`` (or None on cancel).
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)

from zcu_tools.fluxdep_gui.state import ProjectInfo


class ProjectDialog(QDialog):
    """Edit the project chip/qubit names (+ optional result/database roots)."""

    def __init__(self, project: ProjectInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project")
        self.resize(480, 200)

        form = QFormLayout(self)
        self._chip_edit = QLineEdit(project.chip_name)
        self._qub_edit = QLineEdit(project.qub_name)
        self._result_edit = QLineEdit(project.result_dir)
        self._database_edit = QLineEdit(project.database_path)
        form.addRow("Chip name", self._chip_edit)
        form.addRow("Qubit name", self._qub_edit)
        form.addRow("Result dir (optional)", self._result_edit)
        form.addRow("Database path (optional)", self._database_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_project(self) -> ProjectInfo:
        """The edited project info (names trimmed)."""
        return ProjectInfo(
            chip_name=self._chip_edit.text().strip(),
            qub_name=self._qub_edit.text().strip(),
            result_dir=self._result_edit.text().strip(),
            database_path=self._database_edit.text().strip(),
        )
