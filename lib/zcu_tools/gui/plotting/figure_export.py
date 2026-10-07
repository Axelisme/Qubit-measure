"""Fixed-geometry PNG rendering for caller-owned native Figures."""

from __future__ import annotations

import io
import math

from matplotlib.figure import Figure


def render_figure_png(
    figure: Figure,
    *,
    figsize: tuple[float, float] = (6.4, 4.8),
    dpi: float = 100.0,
) -> bytes:
    """Render PNG bytes at fixed geometry without changing Figure ownership.

    figsize is a pair of positive finite sizes in inches; dpi is positive finite
    dots per inch. Defaults produce 640 x 480 pixels with Matplotlib's default
    savefig settings, independent of window size. Invalid geometry raises ValueError
    before changing the Figure. Native rendering failures propagate unchanged.

    The caller must exclusively own rendering; use the owner thread for a live
    GUI Figure. Original size, dpi and canvas identity survive success or failure.
    No backend selection, pyplot registration or GUI presentation occurs.
    """
    if len(figsize) != 2 or any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in figsize
    ):
        raise ValueError("figsize must contain two positive finite inch values")
    if isinstance(dpi, bool) or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive and finite")
    original_size = tuple(float(value) for value in figure.get_size_inches())
    output = io.BytesIO()
    try:
        figure.set_size_inches(*figsize, forward=False)
        figure.savefig(output, format="png", dpi=dpi)
    finally:
        figure.set_size_inches(*original_size, forward=False)
    return output.getvalue()
