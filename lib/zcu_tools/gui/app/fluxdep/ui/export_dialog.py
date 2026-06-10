"""ExportSpectrumsDialog — confirm the export path for spectrums.hdf5.

The default path comes from the project's result dir (set/derived via the
Project… dialog) as ``<result_dir>/data/fluxdep/spectrums.hdf5``. This dialog only
shows that path and lets the user override it via Browse. Returns the resolved
filepath (or None on cancel).
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.fluxdep.services.export import default_export_path


class ExportSpectrumsDialog(QDialog):
    """Show the default export path (from the project) and confirm / override it."""

    def __init__(
        self,
        result_dir: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export spectrums.hdf5")
        self.resize(560, 130)
        self._build_ui(default_export_path(result_dir))

    def _build_ui(self, default_path: str) -> None:
        root = QVBoxLayout(self)

        root.addWidget(
            QLabel(
                "Set chip / qubit (or result dir) via Project… to change the default."
            )
        )

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self._path_edit = QLineEdit(default_path)
        path_row.addWidget(self._path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        path_row.addWidget(browse)
        root.addLayout(path_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_browse(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export spectrums.hdf5",
            self._path_edit.text(),
            filter="HDF5 (*.hdf5 *.h5);;All files (*)",
        )
        if filepath:
            self._path_edit.setText(filepath)

    def export_path(self) -> str | None:
        """The chosen export path, or None if empty."""
        path = self._path_edit.text().strip()
        return path or None
