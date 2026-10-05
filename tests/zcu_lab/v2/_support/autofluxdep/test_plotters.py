"""Shared Autofluxdep plotter behavior, independent of experiment Builders."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.plotting.plots import NonPresentingHost, Plots

pytestmark = pytest.mark.usefixtures("qapp", "drain_qt_events")


def test_decay_plotter_updates_named_scalar_and_current_curve():
    from io import BytesIO

    from matplotlib.figure import Figure

    from zcu_lab.v2._support.autofluxdep.plotters import Decay1DPlotter
    from zcu_lab.v2._support.autofluxdep.result import Sweep1DResult

    plots = Plots(NonPresentingHost())
    figure = Figure()
    plots.adopt("decay", figure)
    plotter = Decay1DPlotter(plots, "decay", "decay", "Lifetime", "Delay")
    result = Sweep1DResult.allocate(
        np.array([0.0, 0.1]), np.array([0.0, 0.5, 1.0]), x_label="Delay"
    )
    result.fit_value[:] = [4.0, 5.0]
    result.signal[:] = [[1.0, 0.5, 0.2], [2.0, 1.0, 0.4]]
    result.fit_curve[:] = [[1.1, 0.6, 0.3], [2.1, 1.1, 0.5]]
    plotter.update(result, 1)

    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[0].get_ydata()), [4.0, 5.0]
    )
    np.testing.assert_allclose(
        np.asarray(figure.axes[1].lines[0].get_ydata()), result.signal[1]
    )
    np.testing.assert_allclose(
        np.asarray(figure.axes[1].lines[1].get_ydata()), result.fit_curve[1]
    )
    retained = plots.finish()
    plots.release()
    saved = BytesIO()
    retained["decay"].savefig(saved, format="png")
    assert saved.getvalue().startswith(b"\x89PNG")


def test_sweep1d_plotter_title_shows_current_snr():
    from matplotlib.figure import Figure

    from zcu_lab.v2._support.autofluxdep.plotters import ColormapLinePlotter
    from zcu_lab.v2._support.autofluxdep.result import Sweep1DResult

    result = Sweep1DResult.allocate(
        np.array([0.0, 0.1]),
        np.array([0.0, 0.5, 1.0]),
        x_label="pulse length (us)",
    )
    result.signal[:] = 1.0
    result.snr[1] = 23.45

    figure = Figure()
    plots = Plots(NonPresentingHost())
    plots.adopt("scan", figure)
    plotter = ColormapLinePlotter(
        plots,
        "scan",
        title="lenrabi",
        y_label="Pulse length (us)",
        marker_of=lambda current: float(current.fit_value[1]),
    )
    result.fit_value[1] = 0.5
    plotter.update(result, 1)

    assert "snr = 23.4" in figure.axes[0].get_title()
    marker = figure.axes[1].lines[-1]
    np.testing.assert_allclose(np.asarray(marker.get_xdata()), [0.5])
    result.fit_value[1] = np.nan
    plotter.update(result, 1)
    assert np.isnan(marker.get_xdata()).all()
