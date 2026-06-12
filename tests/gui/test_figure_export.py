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
    SAVE_DPI,
    SAVE_FIGSIZE,
    SCREENSHOT_DPI,
    SCREENSHOT_FIGSIZE,
    render_figure_png,
    save_figure_to_path,
)

_SAVE_PX = (int(SAVE_FIGSIZE[0] * SAVE_DPI), int(SAVE_FIGSIZE[1] * SAVE_DPI))
_SHOT_PX = (
    int(SCREENSHOT_FIGSIZE[0] * SCREENSHOT_DPI),
    int(SCREENSHOT_FIGSIZE[1] * SCREENSHOT_DPI),
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


def test_save_to_path_keeps_full_save_size(tmp_path):
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
