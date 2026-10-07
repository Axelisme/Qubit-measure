"""Shared PNG geometry and caller-owned Figure lifecycle contracts."""

from __future__ import annotations

import io

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from zcu_tools.gui.plotting.figure_export import render_figure_png


def test_render_png_is_fixed_small_size_regardless_of_figure_size():
    figure = Figure(figsize=(20, 12), dpi=137)
    canvas = FigureCanvasAgg(figure)
    figure.subplots().plot([1, 2, 3])

    png = render_figure_png(figure)

    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (640, 480)
        assert image.format == "PNG"
    assert tuple(figure.get_size_inches()) == (20.0, 12.0)
    assert figure.dpi == 137
    assert figure.canvas is canvas


def test_render_png_independent_of_window_two_sizes():
    sizes = []
    for width, height in [(6, 4), (18, 11)]:
        figure = Figure(figsize=(width, height))
        FigureCanvasAgg(figure)
        figure.subplots().plot([1, 2])
        with Image.open(io.BytesIO(render_figure_png(figure))) as image:
            sizes.append(image.size)
        assert tuple(figure.get_size_inches()) == (width, height)
    assert sizes == [(640, 480), (640, 480)]


def test_custom_geometry_preserves_figure_size_dpi_and_canvas():
    figure = Figure(figsize=(11, 7), dpi=121)
    canvas = FigureCanvasAgg(figure)
    figure.subplots().set_title("Custom geometry")

    png = render_figure_png(figure, figsize=(4, 3), dpi=80)

    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (320, 240)
    assert tuple(figure.get_size_inches()) == (11.0, 7.0)
    assert figure.dpi == 121
    assert figure.canvas is canvas


def test_native_render_failure_restores_figure(monkeypatch):
    figure = Figure(figsize=(9, 5), dpi=143)
    canvas = FigureCanvasAgg(figure)
    failure = OSError("PNG sink failed")

    def fail_save(*args, **kwargs):
        raise failure

    monkeypatch.setattr(figure, "savefig", fail_save)
    with pytest.raises(OSError, match="PNG sink failed") as caught:
        render_figure_png(figure, figsize=(3, 2), dpi=90)
    assert caught.value is failure
    assert tuple(figure.get_size_inches()) == (9.0, 5.0)
    assert figure.dpi == 143
    assert figure.canvas is canvas


@pytest.mark.parametrize(
    "figsize,dpi",
    [
        ((0, 4), 100),
        ((-1, 4), 100),
        ((6, float("inf")), 100),
        ((float("nan"), 4), 100),
        ((6, 4), 0),
        ((6, 4), -1),
        ((6, 4), float("inf")),
        ((6, 4), float("nan")),
        ((True, 4), 100),
        ((6, 4), True),
    ],
)
def test_invalid_geometry_rejects_without_changing_figure(figsize, dpi):
    figure = Figure(figsize=(8, 6), dpi=123)
    canvas = FigureCanvasAgg(figure)
    with pytest.raises(ValueError, match="positive.*finite"):
        render_figure_png(figure, figsize=figsize, dpi=dpi)
    assert tuple(figure.get_size_inches()) == (8.0, 6.0)
    assert figure.dpi == 123
    assert figure.canvas is canvas
