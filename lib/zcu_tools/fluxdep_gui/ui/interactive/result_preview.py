"""ResultPreviewWidget — a read-only view of a finished spectrum.

Shown in the editing area once a spectrum's points are selected: the spectrum
(no mask) with the chosen points and the half/integer flux markers. Purely a
display — no controls, no interaction. The user returns to editing via the
"Re-pick lines" / "Re-select points" buttons.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]

from zcu_tools.fluxdep_gui.state import SpectrumEntry
from zcu_tools.notebook.analysis.fluxdep.processing import cast2real_and_norm


def _contrast_limits(amp: np.ndarray) -> tuple[float, float]:
    finite = amp[np.isfinite(amp)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        return float(finite.min()), float(finite.max()) or 1.0
    return float(lo), float(hi)


class ResultPreviewWidget(QWidget):
    """Read-only spectrum + selected points + flux markers."""

    def __init__(self, entry: SpectrumEntry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)
        self._render(entry)

    def _render(self, entry: SpectrumEntry) -> None:
        ax = self._figure.add_subplot(1, 1, 1)
        dev = entry.raw["dev_values"]
        freqs = entry.raw["freqs"]
        amp = cast2real_and_norm(entry.raw["signals"])  # no mask
        vmin, vmax = _contrast_limits(amp)
        ax.imshow(
            amp.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(dev[0], dev[-1], freqs[0], freqs[-1]),
            vmin=vmin,
            vmax=vmax,
        )
        pts = entry.points
        if pts["dev_values"].size:
            ax.scatter(
                pts["dev_values"],
                pts["freqs"],
                color="red",
                s=18,
                edgecolors="white",
                linewidths=0.3,
            )
        if entry.aligned:
            ax.axvline(entry.flux_half, color="red", linestyle="--", linewidth=1)
            ax.axvline(entry.flux_int, color="blue", linestyle="--", linewidth=1)
        ax.set_xlabel("Device value")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"{entry.name}  ({pts['dev_values'].size} points)")
        self._figure.tight_layout()
        self._canvas.draw_idle()
