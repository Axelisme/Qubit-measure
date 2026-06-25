"""LinePickerWidget — pick the half-flux and integer-flux lines on a 2D spectrum.

Thin Qt chrome around the toolkit-agnostic ``TwoLinePicker`` kernel (in
``zcu_tools.analysis.fluxdep``): the canvas is a Qt-embedded FigureCanvasQTAgg,
the conjugate toggle, swap / auto-align buttons and the info label are Qt
widgets wired to the core, and
``get_result`` returns the core's picked positions. The core is passive (it only
mutates state), so this widget repaints (``_repaint``) after each interaction it
drives. measure-gui drives the same core through its own interactive analysis
session, which repaints via the host port instead.

``fold_initial_lines`` / ``find_best_mirror_position`` are re-exported from the
core for backwards compatibility (callers used to import them from here).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QLabel,
    QPushButton,
    QWidget,
)

from zcu_tools.analysis.fluxdep import (
    TwoLinePicker,
    find_best_mirror_position,
    fold_initial_lines,
)

from .base import InteractiveMplWidget

__all__ = [
    "LinePickerWidget",
    "find_best_mirror_position",
    "fold_initial_lines",
]


class LinePickerWidget(InteractiveMplWidget):
    """Drag/align the half-flux (red) and integer-flux (blue) vertical lines."""

    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        flux_half: float | None = None,
        flux_int: float | None = None,
        force_magnitude: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # The magnitude-only projection is fixed by the spectrum type (OneTone
        # True — phase uninformative; TwoTone False) via force_magnitude, applied
        # to the core at construction — there is no runtime toggle.
        self._info: QLabel | None = None

        # The core owns the data, plots, line state and interaction; this widget
        # is only the Qt shell. The core is passive — it never repaints — so the
        # widget calls _repaint (info label + canvas) after each drive.
        self._picker = TwoLinePicker(
            self.figure,
            signals,
            dev_values,
            freqs,
            flux_half=flux_half,
            flux_int=flux_int,
            force_magnitude=force_magnitude,
        )
        self._build_controls()
        self._repaint()  # initial paint

    # --- controls --------------------------------------------------------

    def _build_controls(self) -> None:
        self._conjugate_checkbox = QCheckBox("Conjugate Line")
        # Conjugate is a flag with no visual change until the next drag — no repaint.
        self._conjugate_checkbox.toggled.connect(self._picker.set_conjugate)
        self.controls_layout.addWidget(self._conjugate_checkbox)

        swap_button = QPushButton("Swap Lines")
        swap_button.clicked.connect(self._on_swap)
        self.controls_layout.addWidget(swap_button)

        align_button = QPushButton("Auto Align")
        align_button.clicked.connect(self._on_auto_align)
        self.controls_layout.addWidget(align_button)

        info = QLabel(self._picker.info_text())
        info.setWordWrap(True)
        self._info = info
        self.controls_layout.addWidget(info)

        self.add_finish_button()

    def _repaint(self) -> None:
        # Refresh the info label (reads live positions) and paint the Qt canvas.
        if self._info is not None:
            self._info.setText(self._picker.info_text())
        self.canvas.draw_idle()

    # --- mouse interaction (base dispatches in-axes events here) ----------

    def on_press(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_press(event.xdata)  # selection only — no visual change

    def on_move(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_move(event.xdata)
        self._repaint()

    def on_release(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_release(event.xdata, event.ydata)
        self._repaint()

    # --- button / toggle actions -----------------------------------------

    def _on_swap(self) -> None:
        self._picker.swap()
        self._repaint()

    def _on_auto_align(self) -> None:
        self._picker.auto_align()  # synchronous in fluxdep (small spectra)
        self._repaint()

    # --- result ----------------------------------------------------------

    def get_result(self) -> tuple[float, float]:
        """Selected (flux_half, flux_int)."""
        return self._picker.positions()
