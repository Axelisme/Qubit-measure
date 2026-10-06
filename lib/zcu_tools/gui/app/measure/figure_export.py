"""Measure save and Data Preview geometry for live GUI-owned Figures.

Calls run on the Figure's owner thread. Size is temporarily pinned and restored
after rendering; agent screenshots use the shared gui.plotting renderer.
"""

from __future__ import annotations

import io
import os.path

from matplotlib import rcParams
from matplotlib.figure import Figure

# Fixed export geometry for SAVED images — full quality, independent of the GUI
# window size. The 4:3 canvas matches the preview geometry while providing enough
# physical area that labels do not dominate multi-panel figures.
SAVE_FIGSIZE: tuple[float, float] = (12.0, 9.0)  # 1800x1350 at dpi=150
SAVE_DPI: int = 150

# Data Preview uses the same logical canvas as the saved image so typography and
# layout are WYSIWYG, but rasterizes at 640x480 to keep gallery refresh lightweight.
DATA_PREVIEW_FIGSIZE: tuple[float, float] = SAVE_FIGSIZE
DATA_PREVIEW_DPI: float = 160.0 / 3.0


def _render_with_fixed_size(
    fig: Figure,
    sink: object,
    figsize: tuple[float, float],
    dpi: float,
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


def resolve_figure_path(path: str) -> str:
    """Make Matplotlib's implicit filename extension explicit before saving."""
    if os.path.splitext(path)[1].lstrip("."):
        return path
    return f"{path.rstrip('.')}.{rcParams['savefig.format']}"


def save_figure_to_path(fig: Figure, path: str) -> None:
    """Save ``fig`` to ``path`` at the full-quality save size/dpi (window-independent)."""
    _render_with_fixed_size(fig, path, SAVE_FIGSIZE, SAVE_DPI)


def render_figure_preview_png(fig: Figure) -> bytes:
    """Render a 640x480 Data Preview with the saved image's logical geometry."""
    buf = io.BytesIO()
    _render_with_fixed_size(
        fig,
        buf,
        DATA_PREVIEW_FIGSIZE,
        DATA_PREVIEW_DPI,
        format="png",
    )
    return buf.getvalue()
