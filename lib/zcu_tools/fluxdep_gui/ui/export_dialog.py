"""ExportSpectrumsDialog — choose chip/qubit + export path for spectrums.hdf5.

chip_name / qub_name drive the default path
``result/<chip>/<qubit>/data/fluxdep/spectrums.hdf5`` (defaulting to
``unknown_chip`` / ``unknown_qubit``); typing in either field updates the shown
path live. The user can override the path entirely via Browse. Returns the
resolved filepath (or None on cancel).
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

from zcu_tools.fluxdep_gui.services.export import (
    DEFAULT_CHIP,
    DEFAULT_QUBIT,
    default_export_path,
)


class ExportSpectrumsDialog(QDialog):
    """Pick chip/qubit (→ default path) and confirm the export filepath."""

    def __init__(
        self,
        chip_name: str = "",
        qub_name: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export spectrums.hdf5")
        self.resize(560, 160)
        # whether the path was manually overridden (stop auto-updating it then)
        self._path_overridden = False
        self._build_ui(chip_name or DEFAULT_CHIP, qub_name or DEFAULT_QUBIT)

    def _build_ui(self, chip: str, qub: str) -> None:
        root = QVBoxLayout(self)

        names_row = QHBoxLayout()
        names_row.addWidget(QLabel("Chip:"))
        self._chip_edit = QLineEdit(chip)
        self._chip_edit.textChanged.connect(self._on_names_changed)
        names_row.addWidget(self._chip_edit)
        names_row.addWidget(QLabel("Qubit:"))
        self._qub_edit = QLineEdit(qub)
        self._qub_edit.textChanged.connect(self._on_names_changed)
        names_row.addWidget(self._qub_edit)
        root.addLayout(names_row)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self._path_edit = QLineEdit(default_export_path(chip, qub))
        self._path_edit.textEdited.connect(self._on_path_edited)
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

    def _on_names_changed(self, _text: str) -> None:
        if self._path_overridden:
            return
        self._path_edit.setText(
            default_export_path(self._chip_edit.text(), self._qub_edit.text())
        )

    def _on_path_edited(self, _text: str) -> None:
        # a manual edit detaches the path from the chip/qubit auto-derivation
        self._path_overridden = True

    def _on_browse(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export spectrums.hdf5",
            self._path_edit.text(),
            filter="HDF5 (*.hdf5 *.h5);;All files (*)",
        )
        if filepath:
            self._path_overridden = True
            self._path_edit.setText(filepath)

    def export_path(self) -> Optional[str]:
        """The chosen export path, or None if empty."""
        path = self._path_edit.text().strip()
        return path or None
