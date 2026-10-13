"""ResultPreviewWidget — a read-only view of a spectrum's published state.

Shown in the editing area when no picker is open: the spectrum (no mask) with
its published points and, once aligned, the half/integer flux markers. The
figure itself has no interaction and the widget has no controls; the window's
stage bar opens the stages. It keeps the pickers' layout (figure left, control
column right) so the figure does not jump when a picker replaces it.
"""

from __future__ import annotations

from qtpy.QtWidgets import QWidget

from zcu_tools.analysis.fluxdep import (
    cast2real_and_norm,
    contrast_limits,
    onetone_background,
)
from zcu_tools.analysis.fluxdep.line_picker import (
    HALF_FLUX_COLOR,
    INTEGER_FLUX_COLOR,
)
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.plotting.fluxdep.palette import KEPT_COLOR

from .base import InteractiveMplWidget


class ResultPreviewWidget(InteractiveMplWidget):
    """Read-only spectrum + published points + flux markers."""

    def __init__(self, entry: SpectrumEntry, *, parent: QWidget | None = None) -> None:
        """Render entry's raw spectrum, published points and alignment once."""
        super().__init__(parent)
        self._render(entry)

    def _render(self, entry: SpectrumEntry) -> None:
        ax = self.figure.add_subplot(1, 1, 1)
        dev = entry.raw["dev_values"]
        freqs = entry.raw["freqs"]
        signals = entry.raw["signals"]  # no mask
        # OneTone is locked to magnitude (phase uninformative) and drawn like
        # its sample picker; TwoTone uses phase.
        if entry.spec_type == "OneTone":
            shown = onetone_background(signals)
            amp, vmin, vmax = shown.values, shown.vmin, shown.vmax
        else:
            amp = cast2real_and_norm(signals, use_phase=True)
            vmin, vmax = contrast_limits(amp)
        ax.imshow(
            amp.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(dev[0], dev[-1], freqs[0], freqs[-1]),
            cmap="gray_r",  # white→black; orange points stand out on the dark feature
            vmin=vmin,
            vmax=vmax,
        )
        pts = entry.points
        if pts["dev_values"].size:
            ax.scatter(
                pts["dev_values"],
                pts["freqs"],
                color=KEPT_COLOR,
                s=18,
                edgecolors="white",
                linewidths=0.3,
            )
        if entry.aligned:
            ax.axvline(
                entry.flux_half, color=HALF_FLUX_COLOR, linestyle="--", linewidth=1
            )
            ax.axvline(
                entry.flux_int, color=INTEGER_FLUX_COLOR, linestyle="--", linewidth=1
            )
        ax.set_xlabel("Device value")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"{entry.name}  ({pts['dev_values'].size} points)")
        self.figure.tight_layout()
        self.canvas.draw_idle()
