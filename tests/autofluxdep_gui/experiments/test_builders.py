"""Per-experiment Builder structural tests — catalog + Plotter layout.

The Nodes' real produce path (set flux device -> acquire -> fit) is covered
end-to-end against the flux-aware MockSoc by the ``test_*_acquire.py`` integration
tests. These tests stay at the structural level the integration tests do not
touch: the catalog exposes every experiment type, and each Builder's Plotter
embeds the runner module's subplot layout (the matching number of matplotlib
axes).
"""

from __future__ import annotations

import numpy as np
import pytest

# --- catalog exposes all measurement types ---


def test_catalog_exposes_all_experiments():
    from zcu_tools.gui.app.autofluxdep.experiments.catalog import names

    types = set(names())
    assert types == {
        "qubit_freq",
        "lenrabi",
        "ro_optimize",
        "t1",
        "t2ramsey",
        "t2echo",
        "mist",
    }


# --- liveplot alignment: each Builder builds the runner module's subplot layout ---


@pytest.mark.parametrize(
    ("type_name", "n_axes"),
    [
        ("qubit_freq", 3),  # fit_freq (1) + detune 2DwithLine (2)
        ("lenrabi", 2),  # rabi_curve 2DwithLine (2D + line)
        ("ro_optimize", 1),  # snr 2D landscape
        ("t1", 2),  # scalar scatter + current-point curve
        ("t2ramsey", 2),
        ("t2echo", 2),
        ("mist", 2),  # flux×gain 2DwithLine
    ],
)
def test_make_plotter_builds_aligned_subplots(type_name, n_axes):
    # each experiment's Plotter embeds the same LivePlot panels the runner module
    # draws, so the figure has the matching number of axes.
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.experiments.catalog import create_placement

    builder = create_placement(type_name).builder
    figure = Figure()
    plotter = builder.make_plotter(figure)
    assert plotter is not None
    assert len(figure.axes) == n_axes


def test_ro_optimize_plotter_marks_latest_best_point():
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.experiments._support.plotters import (
        Landscape2DPlotter,
    )
    from zcu_tools.gui.app.autofluxdep.experiments._support.result import (
        Sweep2DResult,
    )

    result = Sweep2DResult.allocate(
        np.array([0.0, 0.1, 0.2]),
        np.array([5995.0, 6000.0, 6005.0]),
        np.array([0.2, 0.5, 0.8]),
    )
    result.signal[:] = 1.0
    result.best_freq[0] = 5995.0
    result.best_gain[0] = 0.2
    result.best_freq[1] = 6000.0
    result.best_gain[1] = 0.5

    figure = Figure()
    plotter = Landscape2DPlotter(figure)
    plotter.update(result, 2)

    marker = figure.axes[0].collections[-1]
    offsets = np.asarray(marker.get_offsets(), dtype=np.float64)
    np.testing.assert_allclose(offsets, [[6000.0, 0.5]])

    result.best_freq[2] = 6005.0
    result.best_gain[2] = 0.8
    plotter.update(result, 2)

    offsets = np.asarray(marker.get_offsets(), dtype=np.float64)
    np.testing.assert_allclose(offsets, [[6005.0, 0.8]])


def test_qubit_freq_plotter_title_shows_current_snr():
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.experiments._support.result import (
        QubitFreqResult,
    )
    from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqPlotter

    result = QubitFreqResult.allocate(
        np.array([0.0, 0.1]),
        np.array([-1.0, 0.0, 1.0]),
    )
    result.signal[:] = 1.0
    result.snr[1] = 12.34

    figure = Figure()
    plotter = QubitFreqPlotter(figure)
    plotter.update(result, 1)

    assert "snr = 12.3" in figure.axes[1].get_title()


def test_sweep1d_plotter_title_shows_current_snr():
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.experiments._support.plotters import (
        ColormapLinePlotter,
    )
    from zcu_tools.gui.app.autofluxdep.experiments._support.result import (
        Sweep1DResult,
    )

    result = Sweep1DResult.allocate(
        np.array([0.0, 0.1]),
        np.array([0.0, 0.5, 1.0]),
        x_label="pulse length (us)",
    )
    result.signal[:] = 1.0
    result.snr[1] = 23.45

    figure = Figure()
    plotter = ColormapLinePlotter(
        figure,
        title="lenrabi",
        y_label="Pulse length (us)",
    )
    plotter.update(result, 1)

    assert "snr = 23.4" in figure.axes[0].get_title()
