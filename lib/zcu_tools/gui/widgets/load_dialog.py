"""Shared base for the spectrum-load dialogs — file + transpose + live preview.

Both analysis GUIs load an hdf5 spectrum behind a modal dialog that reads the file
best-effort and shows a signal-amplitude preview so the user can judge the axis
orientation (a "Transpose axes" toggle flips the preview's x/y). Only the *load*
applies the chosen ``transpose_axes`` flag, so State always stores the canonical
(x=flux, y=freq) layout.

This base owns the shared mechanics — the file row + Browse, the transpose toggle,
the best-effort read, the preview canvas + render, and OK-gated-until-a-file. App
extras are added via two hooks:

* ``_build_options(opt_row)`` — populate the option row (the subclass decides where
  the shared ``self._transpose_btn`` sits and adds any extra widgets, e.g. a Type
  combo or an "Inherit from" combo).
* ``result_request()`` (abstract) — build the app's own request object from the
  shared fields (``self._filepath`` / ``self._transpose``) plus the extra widgets.

This module pulls in Qt + matplotlib; import it only after the matplotlib backend
is selected (i.e. when building the UI).
"""

from __future__ import annotations

import logging

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


class LoadDataDialog(QDialog):
    """Base modal dialog: pick an hdf5 + transpose orientation with a live preview.

    Subclasses parameterise the wording via the constructor (``window_title`` /
    ``file_placeholder`` / ``browse_caption`` / ``no_preview_text``), populate the
    option row in ``_build_options`` and return their request from
    ``result_request``.
    """

    def __init__(
        self,
        window_title: str,
        file_placeholder: str,
        browse_caption: str,
        no_preview_text: str,
        parent: QWidget | None = None,
        start_dir: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(640, 480)
        self._file_placeholder = file_placeholder
        self._browse_caption = browse_caption
        self._no_preview_text = no_preview_text
        self._filepath = ""
        self._transpose = False
        self._start_dir = start_dir
        # best-effort cache of the raw (signals, x, y) for preview
        self._raw: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None

        self._build_ui()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # File row.
        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText(self._file_placeholder)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        file_row.addWidget(QLabel("File:"))
        file_row.addWidget(self._file_edit, stretch=1)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # Option row — the shared transpose toggle plus the subclass's extras. The
        # button is created here (wired to the shared state) but placed by the
        # subclass, which owns the row's layout.
        self._transpose_btn = QPushButton("Transpose axes")
        self._transpose_btn.setCheckable(True)
        self._transpose_btn.toggled.connect(self._on_transpose_toggled)
        opt_row = QHBoxLayout()
        self._build_options(opt_row)
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

    def _build_options(self, opt_row: QHBoxLayout) -> None:
        """Populate the option row. Subclasses add ``self._transpose_btn`` (and any
        extra option widgets) in whatever order/layout the app wants."""
        raise NotImplementedError

    # --- actions ---------------------------------------------------------

    def _on_browse(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            self._browse_caption,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
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
                self._no_preview_text,
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
