"""LoadSpectrumDialog — pick a file + type + axis orientation with a live preview.

The dialog reads the chosen hdf5 best-effort and shows a signal-amplitude preview
(blank fallback if it cannot be read / is not 2D). A "Transpose axes" toggle flips
the preview's x/y — this is a pure preview-layer UI state; only on Load is the
chosen ``transpose_axes`` flag passed to the Controller, so State always stores
the spectrum in the canonical (x=flux, y=freq) layout. ``inherit_from`` optionally
seeds the new spectrum's flux alignment from an already-loaded one.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
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

from zcu_tools.fluxdep_gui.state import SpecType
from zcu_tools.utils.datasaver import load_data

logger = logging.getLogger(__name__)


class LoadRequest(NamedTuple):
    filepath: str
    spec_type: SpecType
    inherit_from: Optional[str]
    transpose_axes: bool


class LoadSpectrumDialog(QDialog):
    """Modal dialog returning a LoadRequest (or None on cancel)."""

    def __init__(
        self,
        loaded_names: list[str],
        parent: Optional[QWidget] = None,
        start_dir: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load spectrum")
        self.resize(640, 480)
        self._filepath = ""
        self._transpose = False
        self._start_dir = start_dir
        # best-effort cache of the raw (signals, x, y) for preview
        self._raw: Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None

        self._build_ui(loaded_names)

    # --- construction ----------------------------------------------------

    def _build_ui(self, loaded_names: list[str]) -> None:
        root = QVBoxLayout(self)

        # File row.
        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("Choose a spectrum hdf5…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        file_row.addWidget(QLabel("File:"))
        file_row.addWidget(self._file_edit, stretch=1)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # Type + transpose + inherit row.
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Type:"))
        self._type = QComboBox()
        self._type.addItems(["OneTone", "TwoTone"])
        opt_row.addWidget(self._type)

        self._transpose_btn = QPushButton("Transpose axes")
        self._transpose_btn.setCheckable(True)
        self._transpose_btn.toggled.connect(self._on_transpose_toggled)
        opt_row.addWidget(self._transpose_btn)

        opt_row.addWidget(QLabel("Inherit from:"))
        self._inherit = QComboBox()
        self._inherit.addItem("(none)", userData=None)
        for n in loaded_names:
            self._inherit.addItem(n, userData=n)
        opt_row.addWidget(self._inherit, stretch=1)
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
            "Load spectrum",
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
                "No preview\n(choose a 2D spectrum file)",
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

        # Apply the preview-only transpose so the user sees the orientation that
        # the chosen flag will produce (x=flux, y=freq after the swap).
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

    def result_request(self) -> Optional[LoadRequest]:
        """The chosen LoadRequest, or None if no file was selected."""
        if not self._filepath:
            return None
        spec_type: SpecType = (
            "OneTone" if self._type.currentText() == "OneTone" else "TwoTone"
        )
        inherit = self._inherit.currentData()
        return LoadRequest(
            filepath=self._filepath,
            spec_type=spec_type,
            inherit_from=inherit,
            transpose_axes=self._transpose,
        )
