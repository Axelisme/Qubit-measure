"""ResultPreviewWidget — a read-only view of a spectrum's published state.

Shown in the editing area when no picker is open: the spectrum (no mask) with
its published points and, once aligned, the half/integer flux markers. The
figure itself has no interaction. It uses the pickers' layout: the figure on
the left and a control column on the right whose bottom offers only the next
steps the spectrum allows: Align when unaligned; Re-align and Sample when
aligned.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import QPushButton, QWidget

from zcu_tools.analysis.fluxdep import cast2real_and_norm
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.plotting.fluxdep.image import contrast_limits, onetone_dip_depth

from .base import InteractiveMplWidget


class ResultPreviewWidget(InteractiveMplWidget):
    """Read-only spectrum + published points + flux markers, with next steps."""

    def __init__(
        self,
        entry: SpectrumEntry,
        *,
        on_align: Callable[[], None] | None = None,
        on_sample: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Render entry and offer the next steps its alignment allows.

        Unaligned: an Align button (``fluxdep.preview.align``) calls on_align.
        Aligned: Re-align (``fluxdep.preview.realign``) calls on_align and
        Sample (``fluxdep.preview.sample``) calls on_sample. An omitted
        callback omits its button. The buttons sit at the bottom of the right
        control column, top to bottom in that order. set_editable gates the
        buttons only; the figure stays viewable.
        """
        super().__init__(parent)
        steps = (
            (("Re-align", "realign", on_align), ("Sample", "sample", on_sample))
            if entry.aligned
            else (("Align", "align", on_align),)
        )
        for text, action, on_click in steps:
            if on_click is not None:
                self.bottom_layout.addWidget(_step_button(text, action, on_click))
        self._render(entry)

    def _render(self, entry: SpectrumEntry) -> None:
        ax = self.figure.add_subplot(1, 1, 1)
        dev = entry.raw["dev_values"]
        freqs = entry.raw["freqs"]
        signals = entry.raw["signals"]  # no mask
        # OneTone is locked to magnitude (phase uninformative) and drawn like
        # its sample picker; TwoTone uses phase.
        if entry.spec_type == "OneTone":
            amp = onetone_dip_depth(signals)
        else:
            amp = cast2real_and_norm(signals, use_phase=True)
        vmin, vmax = contrast_limits(amp)
        ax.imshow(
            amp.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(dev[0], dev[-1], freqs[0], freqs[-1]),
            cmap="gray_r",  # white→black; red points stand out on the dark feature
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
        self.figure.tight_layout()
        self.canvas.draw_idle()


def _step_button(text: str, action: str, on_click: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(f"fluxdep.preview.{action}")
    button.clicked.connect(on_click)
    return button
