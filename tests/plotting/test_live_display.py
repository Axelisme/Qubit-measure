"""Live templates show only measured cells and readable scan coordinates."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools import plotting
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@pytest.mark.parametrize("uniform", [False, True])
def test_heatmap_is_blank_until_first_update(uniform: bool) -> None:
    plots = Plots(NonPresentingHost())
    viewer = plotting.live_heatmap_with_line(
        plots, "scan", "Flux (A)", "Frequency (MHz)", uniform=uniform
    )
    image = plots["scan"].axes[0].images[0]
    initial = image.get_array()
    assert (
        not image.get_visible()
        or initial is None
        or np.ma.masked_invalid(initial).count() == 0
    )

    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    viewer.update(np.array([0.0, 0.1]), np.array([6999.0, 7001.0]), values)
    assert image.get_visible()
    np.testing.assert_array_equal(image.get_array(), values.T)
    assert image.get_clim() == (1.0, 4.0)
    plots.finish()
    plots.release()


def test_narrow_mhz_scan_line_ticks_are_readable_without_changing_data() -> None:
    plots = Plots(NonPresentingHost())
    figure, _ = plots.subplots("scan", ncols=2, figsize=(11.0, 4.8), dpi=100)
    viewer = plotting.live_heatmap_with_line(
        plots,
        "scan",
        "Flux (A)",
        "Frequency (MHz)",
        axes=(figure.axes[0], figure.axes[1]),
    )
    frequencies = np.linspace(6999.0, 7001.0, 11)
    signals = np.arange(22, dtype=float).reshape(2, 11)
    viewer.update(np.array([0.0, 0.1]), frequencies, signals)
    figure.canvas.draw()
    line_axes = figure.axes[1]
    boxes = [
        label.get_window_extent()
        for tick, label in zip(
            line_axes.get_xticks(), line_axes.get_xticklabels(), strict=True
        )
        if line_axes.get_xlim()[0] <= tick <= line_axes.get_xlim()[1]
        and label.get_visible()
    ]
    assert len(boxes) >= 2
    assert all(
        left.x1 < right.x0 for left, right in zip(boxes, boxes[1:], strict=False)
    )
    assert line_axes.get_xlabel() == "Frequency (MHz)"
    np.testing.assert_array_equal(line_axes.lines[-1].get_xdata(), frequencies)
    np.testing.assert_array_equal(line_axes.lines[-1].get_ydata(), signals[-1])
    plots.finish()
    plots.release()
