"""Two-line backgrounds render native samples across projections and loss refreshes."""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    mirror_loss_at,
)
from zcu_tools.analysis.fluxdep.processing import (
    cast2real_and_norm,
    onetone_background,
)


@pytest.fixture(params=["uniform", "nonuniform"])
def spectrum(request: pytest.FixtureRequest) -> FluxPickInputs:
    devs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    freqs = np.array([5.0, 6.0, 7.0])
    if request.param == "nonuniform":
        devs = np.array([0.0, 0.2, 0.9, 2.0, 4.0])
        freqs = np.array([5.0, 5.2, 7.0])
    magnitude = np.array([[1, 4, 2], [7, 3, 9], [2, 8, 5], [8, 1, 6], [3, 9, 4]])
    phase = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0]])
    return FluxPickInputs(magnitude * np.exp(1j * phase), devs, freqs)


@pytest.fixture(
    params=["forward", "reverse-device", "reverse-frequency", "reverse-both"]
)
def scan(spectrum: FluxPickInputs, request: pytest.FixtureRequest) -> FluxPickInputs:
    xs, ys, signals = spectrum.dev_values, spectrum.freqs, spectrum.signals
    if request.param in ("reverse-device", "reverse-both"):
        xs, signals = xs[::-1], signals[::-1, :]
    if request.param in ("reverse-frequency", "reverse-both"):
        ys, signals = ys[::-1], signals[:, ::-1]
    return FluxPickInputs(signals, xs, ys)


def _assert_sample_colours(
    canvas: FigureCanvasAgg,
    axes: Axes,
    scan: FluxPickInputs,
    expected: NDArray[np.float64],
) -> None:
    """Observe each background raster, without lines/dots hiding sample colours."""
    image = axes.get_images()[0]
    assert image.get_cmap().name == "gray_r"
    xs, ys = scan.dev_values, scan.freqs
    dx = float(np.min(np.abs(np.diff(xs))))
    dy = float(np.min(np.abs(np.diff(ys))))
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    try:
        axes.set_xlim(float(np.min(xs) - dx), float(np.max(xs) + dx))
        axes.set_ylim(float(np.min(ys) - dy), float(np.max(ys) + dy))
        canvas.draw()
        rgba, left, bottom, _ = image.make_image(canvas.get_renderer())
        assert rgba is not None
        for row, x in enumerate(xs):
            for column, y in enumerate(ys):
                colour = image.to_rgba(expected[row, column], bytes=True)
                for ox, oy in (
                    (0.0, 0.0),
                    (-0.4 * dx, -0.4 * dy),
                    (0.4 * dx, 0.4 * dy),
                ):
                    px, py = axes.transData.transform((x + ox, y + oy))
                    ix, iy = int(px - left), int(py - bottom)
                    assert 0 <= ix < rgba.shape[1] and 0 <= iy < rgba.shape[0]
                    np.testing.assert_allclose(
                        rgba[iy, ix],
                        colour,
                        atol=1,
                        err_msg=f"sample ({x}, {y}), offset ({ox}, {oy})",
                    )
    finally:
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)


@pytest.mark.parametrize("onetone", [False, True], ids=["projection", "onetone"])
@pytest.mark.parametrize("panel", [0, 1], ids=["main", "loss"])
def test_initial_backgrounds_follow_native_sample_centres(
    scan: FluxPickInputs, onetone: bool, panel: int
) -> None:
    figure = Figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(figure)
    TwoLinePicker(figure, scan.signals, scan.dev_values, scan.freqs, onetone=onetone)
    axes = figure.axes[panel]
    assert axes.xaxis_inverted() == (scan.dev_values[-1] < scan.dev_values[0])
    assert axes.yaxis_inverted() == (scan.freqs[-1] < scan.freqs[0])
    expected = cast2real_and_norm(scan.signals, use_phase=True)
    if onetone and panel == 0:
        shown = onetone_background(scan.signals)
        expected = shown.values
        assert axes.get_images()[0].get_clim() == pytest.approx(
            (shown.vmin, shown.vmax)
        )
    _assert_sample_colours(canvas, axes, scan, expected)


@pytest.mark.parametrize("onetone", [False, True], ids=["projection", "onetone"])
def test_projection_and_state_updates_keep_sample_coordinates(
    scan: FluxPickInputs, onetone: bool
) -> None:
    figure = Figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(figure)
    picker = TwoLinePicker(
        figure, scan.signals, scan.dev_values, scan.freqs, onetone=onetone
    )
    picker.set_magnitude_only(on=True)
    for magnitude_only in (True, False):
        state = FluxPickState(
            flux_half=1.0, flux_int=3.0, magnitude_only=magnitude_only
        )
        if not magnitude_only:
            picker.show_state(replace(state, conjugate=True))
        expected = (
            onetone_background(scan.signals).values
            if onetone
            else cast2real_and_norm(scan.signals, use_phase=not magnitude_only)
        )
        _assert_sample_colours(canvas, figure.axes[0], scan, expected)
        assert picker.magnitude_only == magnitude_only


def test_loss_refresh_keeps_native_sample_coordinates(scan: FluxPickInputs) -> None:
    figure = Figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(figure)
    picker = TwoLinePicker(figure, scan.signals, scan.dev_values, scan.freqs)
    for magnitude_only, center in ((False, 2.0), (True, 1.0)):
        picker.set_magnitude_only(on=magnitude_only)
        picker.show_loss("half", center, 6.0)
        projected = cast2real_and_norm(scan.signals, use_phase=not magnitude_only)
        expected, mean = mirror_loss_at(scan.dev_values, projected, center)
        assert figure.axes[1].get_title() == f"mirror loss: {mean:.4f}"
        _assert_sample_colours(canvas, figure.axes[1], scan, expected)
