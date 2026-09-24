"""Fixed-size figure export: output geometry is window-independent, and the
live figure's on-screen size is restored after export. The agent screenshot path
(render_figure_png) uses a SMALL fixed geometry to stay token-light; the save
path (save_figure_to_path) keeps the full-quality save geometry."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from zcu_tools.gui.app.main.figure_export import (  # noqa: E402
    DATA_PREVIEW_DPI,
    DATA_PREVIEW_FIGSIZE,
    SAVE_DPI,
    SAVE_FIGSIZE,
    SCREENSHOT_DPI,
    SCREENSHOT_FIGSIZE,
    render_figure_png,
    render_figure_preview_png,
    save_figure_to_path,
)

_SAVE_PX = (int(SAVE_FIGSIZE[0] * SAVE_DPI), int(SAVE_FIGSIZE[1] * SAVE_DPI))
_SHOT_PX = (
    int(SCREENSHOT_FIGSIZE[0] * SCREENSHOT_DPI),
    int(SCREENSHOT_FIGSIZE[1] * SCREENSHOT_DPI),
)
_PREVIEW_PX = (
    round(DATA_PREVIEW_FIGSIZE[0] * DATA_PREVIEW_DPI),
    round(DATA_PREVIEW_FIGSIZE[1] * DATA_PREVIEW_DPI),
)


def test_screenshot_is_smaller_than_save():
    """The agent screenshot must be strictly smaller than a saved image so it
    stays token-light; only the screenshot path was shrunk (consolidation)."""
    assert _SHOT_PX[0] < _SAVE_PX[0]
    assert _SHOT_PX[1] < _SAVE_PX[1]


def test_render_png_is_fixed_small_size_regardless_of_figure_size():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig.set_size_inches(20, 12)  # simulate a figure stretched by a big window
    try:
        png = render_figure_png(fig)
        img = Image.open(io.BytesIO(png))
        assert img.size == _SHOT_PX
        # on-screen size restored, not left at the export size
        assert tuple(fig.get_size_inches()) == (20.0, 12.0)
    finally:
        plt.close(fig)


def test_render_png_independent_of_window_two_sizes():
    sizes = []
    for w, h in [(6, 4), (18, 11)]:
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        fig.set_size_inches(w, h)
        try:
            img = Image.open(io.BytesIO(render_figure_png(fig)))
            sizes.append(img.size)
        finally:
            plt.close(fig)
    assert sizes[0] == sizes[1] == _SHOT_PX


def test_data_preview_uses_save_logical_geometry_at_small_raster_size():
    fig, ax = plt.subplots()
    ax.set_title("Preview geometry")
    fig.set_size_inches(5, 4)
    drawn_sizes: list[tuple[float, float]] = []
    fig.canvas.mpl_connect(
        "draw_event",
        lambda _event: drawn_sizes.append(
            (float(fig.get_size_inches()[0]), float(fig.get_size_inches()[1]))
        ),
    )
    try:
        png = render_figure_preview_png(fig)
        img = Image.open(io.BytesIO(png))
        assert img.size == _PREVIEW_PX == (640, 480)
        assert drawn_sizes[-1] == SAVE_FIGSIZE
        assert tuple(fig.get_size_inches()) == (5.0, 4.0)
    finally:
        plt.close(fig)


def test_save_to_path_keeps_full_save_size(tmp_path):
    assert SAVE_FIGSIZE == (12.0, 9.0)
    assert SAVE_DPI == 150
    assert _SAVE_PX == (1800, 1350)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig.set_size_inches(15, 9)
    out = tmp_path / "plot.png"
    try:
        save_figure_to_path(fig, str(out))
        img = Image.open(out)
        assert img.size == _SAVE_PX
        assert tuple(fig.get_size_inches()) == (15.0, 9.0)
    finally:
        plt.close(fig)
