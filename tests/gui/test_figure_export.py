"""Measure save and Data Preview preserve their app-owned geometry."""

from __future__ import annotations

import io

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from zcu_tools.gui.app.measure.figure_export import (
    DATA_PREVIEW_DPI,
    DATA_PREVIEW_FIGSIZE,
    SAVE_DPI,
    SAVE_FIGSIZE,
    render_figure_preview_png,
    render_save_figure_png,
    save_figure_to_path,
)

_SAVE_PX = (int(SAVE_FIGSIZE[0] * SAVE_DPI), int(SAVE_FIGSIZE[1] * SAVE_DPI))
_PREVIEW_PX = (
    round(DATA_PREVIEW_FIGSIZE[0] * DATA_PREVIEW_DPI),
    round(DATA_PREVIEW_FIGSIZE[1] * DATA_PREVIEW_DPI),
)


def test_data_preview_uses_save_logical_geometry_at_small_raster_size():
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.subplots()
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
        fig.clear()


def test_native_png_encoder_preserves_owner_geometry_and_canvas():
    figure = Figure(figsize=(5, 4), dpi=80)
    canvas = FigureCanvasAgg(figure)
    try:
        png = render_save_figure_png(figure)
        assert Image.open(io.BytesIO(png)).size == (1800, 1350)
        assert tuple(figure.get_size_inches()) == (5, 4)
        assert figure.dpi == 80
        assert figure.canvas is canvas
    finally:
        figure.clear()


def test_native_png_render_failure_restores_display(monkeypatch: pytest.MonkeyPatch):
    figure = Figure(figsize=(5, 4), dpi=80)
    canvas = FigureCanvasAgg(figure)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr(figure, "savefig", fail)
    try:
        with pytest.raises(RuntimeError, match="render failed"):
            render_save_figure_png(figure)
        assert tuple(figure.get_size_inches()) == (5, 4)
        assert figure.dpi == 80
        assert figure.canvas is canvas
    finally:
        figure.clear()


def test_save_to_path_keeps_full_save_size(tmp_path):
    assert SAVE_FIGSIZE == (12.0, 9.0)
    assert SAVE_DPI == 150
    assert _SAVE_PX == (1800, 1350)
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([1, 2, 3])
    fig.set_size_inches(15, 9)
    out = tmp_path / "plot.png"
    try:
        save_figure_to_path(fig, str(out))
        img = Image.open(out)
        assert img.size == _SAVE_PX
        assert tuple(fig.get_size_inches()) == (15.0, 9.0)
    finally:
        fig.clear()
