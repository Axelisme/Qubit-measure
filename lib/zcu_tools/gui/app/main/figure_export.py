"""Fixed-size figure export for save-image and figure-screenshot.

Both paths render the *same* live matplotlib Figure that is embedded in a tab's
Qt canvas. That canvas resizes with the GUI window, which mutates the figure's
``size_inches`` — so a naive ``fig.savefig(path)`` (save) or ``canvas.grab()``
(screenshot) produces an image whose dimensions track the current window shape.

To make exports window-independent, both paths funnel through here: the live
figure's size is temporarily pinned to a fixed value, rendered, then restored
(try/finally). All calls must run on the Qt main thread (the live figure is
GUI-owned); the existing save/screenshot entry points already do.
"""

from __future__ import annotations

import io

from matplotlib.figure import Figure

# Fixed export geometry — independent of the GUI window size.
SAVE_FIGSIZE: tuple[float, float] = (8.0, 5.0)  # inches
SAVE_DPI: int = 150


def _render_with_fixed_size(
    fig: Figure, sink: object, **savefig_kwargs: object
) -> None:
    """Pin fig to the fixed export size, savefig to ``sink``, then restore.

    ``sink`` is anything ``Figure.savefig`` accepts (a path str or a binary
    file-like). Restore is in a finally so the GUI-displayed figure keeps its
    on-screen size even if savefig raises.
    """
    orig_w, orig_h = (float(v) for v in fig.get_size_inches())
    try:
        fig.set_size_inches(*SAVE_FIGSIZE)
        fig.savefig(sink, dpi=SAVE_DPI, **savefig_kwargs)  # type: ignore[arg-type]
    finally:
        fig.set_size_inches(orig_w, orig_h)


def save_figure_to_path(fig: Figure, path: str) -> None:
    """Save ``fig`` to ``path`` at the fixed export size/dpi (window-independent)."""
    _render_with_fixed_size(fig, path)


def render_figure_png(fig: Figure) -> bytes:
    """Render ``fig`` to PNG bytes at the fixed export size/dpi.

    Replaces ``canvas.grab()`` for figure screenshots so the result is the same
    fixed geometry as a saved image, not the current widget pixel size.
    """
    buf = io.BytesIO()
    _render_with_fixed_size(fig, buf, format="png")
    return buf.getvalue()
