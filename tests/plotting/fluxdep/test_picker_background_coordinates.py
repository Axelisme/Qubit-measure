"""Picker backgrounds and masks keep samples at their native coordinates."""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
from zcu_tools.analysis.fluxdep.onetone import OneToneInputs, OneTonePickState
from zcu_tools.analysis.fluxdep.processing import onetone_background
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickState,
    project_twotone_pick,
)
from zcu_tools.plotting.fluxdep.onetone import make_onetone_pick_figure
from zcu_tools.plotting.fluxdep.twotone import TwoTonePickPlot


@pytest.fixture(params=["forward", "reverse", "nonuniform", "nonuniform-reverse"])
def spectrum(request: pytest.FixtureRequest) -> FluxPickInputs:
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([5.0, 6.0, 7.0])
    if "nonuniform" in request.param:
        xs = np.array([0.0, 0.2, 0.9, 2.0, 4.0])
        ys = np.array([5.0, 5.2, 7.0])
    signals = np.array(
        [[1, 4, 2], [7, 3, 9], [2, 8, 5], [8, 1, 6], [3, 9, 4]],
        dtype=np.complex128,
    )
    if "reverse" in request.param:
        xs, ys, signals = xs[::-1], ys[::-1], signals[::-1, ::-1]
    return FluxPickInputs(signals, xs, ys)


def _assert_sample_pixels(
    canvas: FigureCanvasAgg,
    axes: Axes,
    image: AxesImage,
    spectrum: FluxPickInputs,
    expected: NDArray[np.float64],
    opacity: float = 1.0,
) -> None:
    """Observe nearest-cell colours at samples and inside their neighbours' midpoints.

    Sampling an artist's public Agg raster avoids grid/marker overlays. Probes
    either side of a sample distinguish centered cells from endpoint extents,
    even when both render the same colour at the sample itself.
    """
    xs, ys = spectrum.dev_values, spectrum.freqs
    dx = np.min(np.abs(np.diff(xs)))
    dy = np.min(np.abs(np.diff(ys)))
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    try:
        axes.set_xlim(float(np.min(xs) - dx), float(np.max(xs) + dx))
        axes.set_ylim(float(np.min(ys) - dy), float(np.max(ys) + dy))
        canvas.draw()
        rgba, left, bottom, _ = image.make_image(canvas.get_renderer())
        assert rgba is not None
        for row, x in enumerate(xs):
            for column, y in enumerate(ys):
                colour = image.to_rgba(expected[row, column], alpha=opacity, bytes=True)
                for offset_x, offset_y in (
                    (0.0, 0.0),
                    (-0.4 * dx, -0.4 * dy),
                    (0.4 * dx, 0.4 * dy),
                ):
                    px, py = axes.transData.transform((x + offset_x, y + offset_y))
                    ix, iy = int(px - left), int(py - bottom)
                    assert 0 <= ix < rgba.shape[1] and 0 <= iy < rgba.shape[0], (
                        f"sample ({x}, {y}) neighbourhood lies outside the image"
                    )
                    actual = rgba[iy, ix]
                    assert actual[3] == colour[3]
                    if colour[3]:
                        np.testing.assert_allclose(actual[:3], colour[:3], atol=1)
    finally:
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)


def test_onetone_background_cells_are_centered_on_native_samples(
    spectrum: FluxPickInputs,
) -> None:
    figure = make_onetone_pick_figure(
        OneToneInputs(spectrum), OneTonePickState(threshold=0.1, peak_indices=(1, 3))
    )
    axes = figure.axes[0]
    assert axes.xaxis_inverted() == (spectrum.dev_values[-1] < spectrum.dev_values[0])
    assert axes.yaxis_inverted() == (spectrum.freqs[-1] < spectrum.freqs[0])
    assert axes.get_ylim() == pytest.approx(
        (7.5, 4.5) if axes.yaxis_inverted() else (4.5, 7.5)
    )
    image = axes.get_images()[0]
    assert image.get_cmap().name == "gray_r"
    expected = onetone_background(spectrum.signals)
    assert image.get_clim() == pytest.approx((expected.vmin, expected.vmax))
    _assert_sample_pixels(
        FigureCanvasAgg(figure), axes, image, spectrum, expected.values
    )


@pytest.mark.parametrize("show_origin", [True, False])
def test_twotone_background_and_mask_follow_native_samples_on_updates(
    spectrum: FluxPickInputs, show_origin: bool
) -> None:
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    inputs = TwoToneInputs(spectrum)
    renderer = TwoTonePickPlot(figure, inputs)
    mask = np.indices(spectrum.signals.shape).sum(axis=0) % 2 == 0
    state = TwoTonePickState(mask, sigma=0.1, smooth_method="gaussian")
    for selected in (mask, ~mask):
        state = replace(state, mask=selected)
        view = project_twotone_pick(inputs, state)
        renderer.show_state(view, show_mask=True, show_origin=show_origin)
        assert figure.canvas is canvas
        axes = figure.axes[0]
        assert axes.xaxis_inverted() == (
            spectrum.dev_values[-1] < spectrum.dev_values[0]
        )
        assert axes.yaxis_inverted() == (spectrum.freqs[-1] < spectrum.freqs[0])
        assert axes.get_xlim() == pytest.approx(
            (4.5, -0.5) if axes.xaxis_inverted() else (-0.5, 4.5)
        )
        assert axes.get_ylim() == pytest.approx(
            (7.5, 4.5) if axes.yaxis_inverted() else (4.5, 7.5)
        )
        background, overlay = axes.get_images()
        assert background.get_cmap().name == "gray_r"
        assert overlay.get_cmap().name == "Blues"
        assert overlay.get_visible()
        expected = view.result.real_signals.copy()
        if not show_origin:
            expected[~selected] = np.nan
        _assert_sample_pixels(canvas, axes, background, spectrum, expected)
        _assert_sample_pixels(
            canvas,
            axes,
            overlay,
            spectrum,
            np.where(selected, 1.0, np.nan),
            opacity=0.15,
        )
        np.testing.assert_array_equal(state.mask, selected)
