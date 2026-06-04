"""LoadOnetoneDialog — pick a one-tone hdf5 + axis orientation with a live preview.

dispersive loads a single one-tone spectrum, so this is fluxdep's LoadSpectrumDialog
trimmed to just file + transpose + preview (no type / inherit). The dialog reads the
chosen hdf5 best-effort and shows a signal-amplitude preview so the user can judge
the axis orientation; a "Transpose axes" toggle flips the preview's x/y. The chosen
``transpose_axes`` flag is passed to the Controller only on Load, so State always
stores the canonical (x=flux, y=freq) layout.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
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

from zcu_tools.utils.datasaver import load_data

logger = logging.getLogger(__name__)


class LoadOnetoneRequest(NamedTuple):
    filepath: str
    transpose_axes: bool


class LoadOnetoneDialog(QDialog):
    """Modal dialog returning a LoadOnetoneRequest (or None on cancel)."""

    def __init__(self, parent: Optional[QWidget] = None, start_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Load one-tone spectrum")
        self.resize(640, 480)
        self._filepath = ""
        self._transpose = False
        self._start_dir = start_dir
        # best-effort cache of the raw (signals, x, y) for preview
        self._raw: Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None

        self._build_ui()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # File row.
        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("Choose a one-tone hdf5…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        file_row.addWidget(QLabel("File:"))
        file_row.addWidget(self._file_edit, stretch=1)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # Transpose toggle.
        opt_row = QHBoxLayout()
        self._transpose_btn = QPushButton("Transpose axes")
        self._transpose_btn.setCheckable(True)
        self._transpose_btn.toggled.connect(self._on_transpose_toggled)
        opt_row.addWidget(self._transpose_btn)
        opt_row.addWidget(
            QLabel("(toggle if the preview shows frequency on x instead of flux)")
        )
        opt_row.addStretch(1)
        root.addLayout(opt_row)

        # Preview canvas.
        self._figure = Figure(figsize=(5, 3))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(1, 1, 1)
        root.addWidget(self._canvas, stretch=1)
        self._render_preview()

        # OK / Cancel.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if self._ok_button is not None:
            self._ok_button.setEnabled(False)  # no file chosen yet
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # --- actions ---------------------------------------------------------

    def _on_browse(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load one-tone",
            self._start_dir,
            filter="HDF5 (*.hdf5 *.h5);;All files (*)",
        )
        if not filepath:
            return
        self._filepath = filepath
        self._file_edit.setText(filepath)
        if self._ok_button is not None:
            self._ok_button.setEnabled(True)
        self._raw = self._read_best_effort(filepath)
        self._render_preview()

    def _on_transpose_toggled(self, checked: bool) -> None:
        self._transpose = checked
        self._render_preview()

    @staticmethod
    def _read_best_effort(
        filepath: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        try:
            signals, x, y = load_data(filepath, return_comment=False)
            return signals, x, y
        except Exception:  # noqa: BLE001 — preview is best-effort; never crash the dialog
            logger.exception("preview read failed for %r", filepath)
            return None

    def _render_preview(self) -> None:
        self._ax.clear()
        if self._raw is None:
            self._ax.text(
                0.5,
                0.5,
                "No preview\n(choose a 2D one-tone file)",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
            )
            self._ax.set_axis_off()
            self._canvas.draw_idle()
            return

        signals, x, y = self._raw
        self._ax.set_axis_on()
        if y is None:
            self._ax.text(
                0.5,
                0.5,
                "Not a 2D spectrum",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
            )
            self._ax.set_axis_off()
            self._canvas.draw_idle()
            return

        # Apply the preview-only transpose so the user sees the orientation the
        # chosen flag will produce (x=flux, y=freq after the swap).
        amp = np.abs(signals)
        if self._transpose:
            amp, x_axis, y_axis = amp.T, y, x
        else:
            x_axis, y_axis = x, y
        self._ax.imshow(
            amp.T,
            aspect="auto",
            origin="lower",
            extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
        )
        self._ax.set_xlabel("Device value (flux)")
        self._ax.set_ylabel("Frequency")
        self._canvas.draw_idle()

    # --- result ----------------------------------------------------------

    def result_request(self) -> Optional[LoadOnetoneRequest]:
        """The chosen LoadOnetoneRequest, or None if no file was selected."""
        if not self._filepath:
            return None
        return LoadOnetoneRequest(
            filepath=self._filepath, transpose_axes=self._transpose
        )
