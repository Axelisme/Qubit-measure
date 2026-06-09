"""LinePickerWidget — pick the half-flux and integer-flux lines on a 2D spectrum.

Thin Qt chrome around the toolkit-agnostic ``TwoLinePicker`` core (in
``zcu_tools.notebook.analysis.fluxdep.interactive.two_line_picker``): the canvas
is a Qt-embedded FigureCanvasQTAgg, the conjugate / magnitude toggles, swap /
auto-align buttons and the info label are Qt widgets wired to the core, and
``get_result`` returns the core's picked positions. measure-gui drives the same
core through its own interactive analysis host.

``fold_initial_lines`` / ``find_best_mirror_position`` are re-exported from the
core for backwards compatibility (callers used to import them from here).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QLabel,
    QPushButton,
    QWidget,
)

from zcu_tools.notebook.analysis.fluxdep.interactive.two_line_picker import (
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
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
        force_magnitude: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # OneTone spectra are locked to magnitude-only (phase is uninformative);
        # the magnitude checkbox is hidden in that case.
        self._force_magnitude = force_magnitude
        self._info: Optional[QLabel] = None

        # The core owns the data, plots, line state and interaction; this widget
        # is only the Qt shell. It repaints via _on_redraw (info label + canvas).
        self._picker = TwoLinePicker(
            self.figure,
            signals,
            dev_values,
            freqs,
            flux_half=flux_half,
            flux_int=flux_int,
            force_magnitude=force_magnitude,
            redraw=self._on_redraw,
        )
        self._build_controls()

    # --- controls --------------------------------------------------------

    def _build_controls(self) -> None:
        self._conjugate_checkbox = QCheckBox("Conjugate Line")
        self._conjugate_checkbox.toggled.connect(self._picker.set_conjugate)
        self.controls_layout.addWidget(self._conjugate_checkbox)

        # Magnitude toggle only when not forced (OneTone locks it on).
        if not self._force_magnitude:
            self._magnitude_checkbox = QCheckBox("Magnitude Only")
            self._magnitude_checkbox.toggled.connect(self._picker.set_magnitude_only)
            self.controls_layout.addWidget(self._magnitude_checkbox)

        swap_button = QPushButton("Swap Lines")
        swap_button.clicked.connect(self._on_swap)
        self.controls_layout.addWidget(swap_button)

        align_button = QPushButton("Auto Align")
        align_button.clicked.connect(self._on_auto_align)
        self.controls_layout.addWidget(align_button)

        self._info = QLabel(self._picker.info_text())
        self._info.setWordWrap(True)
        self.controls_layout.addWidget(self._info)

        self.add_finish_button()

    def _on_redraw(self) -> None:
        # The core asks for a repaint after a state change; refresh the info
        # label (it reads live positions) and paint the Qt canvas. The very first
        # redraw fires during core construction, before _info exists — guard it.
        if self._info is not None:
            self._info.setText(self._picker.info_text())
        self.canvas.draw_idle()

    # --- mouse interaction (base dispatches in-axes events here) ----------

    def on_press(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_press(event.xdata)

    def on_move(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_move(event.xdata)

    def on_release(self, event) -> None:  # noqa: ANN001 - MouseEvent via base
        self._picker.on_release(event.xdata, event.ydata)

    # --- button actions --------------------------------------------------

    def _on_swap(self) -> None:
        self._picker.swap()

    def _on_auto_align(self) -> None:
        self._picker.auto_align()

    # --- result ----------------------------------------------------------

    def get_result(self) -> tuple[float, float]:
        """Selected (flux_half, flux_int)."""
        return self._picker.positions()
