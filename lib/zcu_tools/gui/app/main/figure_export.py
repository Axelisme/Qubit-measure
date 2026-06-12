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

# Fixed export geometry for SAVED images — full quality, independent of the GUI
# window size. This is what the user gets on disk, so it stays large.
SAVE_FIGSIZE: tuple[float, float] = (8.0, 5.0)  # inches
SAVE_DPI: int = 150

# Fixed export geometry for AGENT SCREENSHOTS (tab.get_current_figure). The agent
# only needs to eyeball the plot, so this is deliberately small to keep the
# base64 PNG token-light (~640x480). Distinct from the save path, which must keep
# full quality. Tune here if the screenshot is too small/large to read.
SCREENSHOT_FIGSIZE: tuple[float, float] = (6.4, 4.8)  # inches -> 640x480 at dpi=100
SCREENSHOT_DPI: int = 100


def _render_with_fixed_size(
    fig: Figure,
    sink: object,
    figsize: tuple[float, float],
    dpi: int,
    **savefig_kwargs: object,
) -> None:
    """Pin fig to ``figsize``, savefig to ``sink`` at ``dpi``, then restore.

    ``sink`` is anything ``Figure.savefig`` accepts (a path str or a binary
    file-like). The original on-screen size is restored in a finally so the
    GUI-displayed figure is never permanently resized, even if savefig raises.
    """
    orig_w, orig_h = (float(v) for v in fig.get_size_inches())
    try:
        fig.set_size_inches(*figsize)
        fig.savefig(sink, dpi=dpi, **savefig_kwargs)  # type: ignore[arg-type]
    finally:
        fig.set_size_inches(orig_w, orig_h)


def save_figure_to_path(fig: Figure, path: str) -> None:
    """Save ``fig`` to ``path`` at the full-quality save size/dpi (window-independent)."""
    _render_with_fixed_size(fig, path, SAVE_FIGSIZE, SAVE_DPI)


def render_figure_png(fig: Figure) -> bytes:
    """Render ``fig`` to PNG bytes at the small screenshot size/dpi.

    Used for agent figure screenshots (tab.get_current_figure). Replaces
    ``canvas.grab()`` so the result has a fixed, window-independent geometry; it
    is intentionally smaller/lower-dpi than a saved image to stay token-light.
    """
    buf = io.BytesIO()
    _render_with_fixed_size(fig, buf, SCREENSHOT_FIGSIZE, SCREENSHOT_DPI, format="png")
    return buf.getvalue()
