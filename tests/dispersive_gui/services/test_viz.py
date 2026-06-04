"""Tests for dispersive VizService — the tune figure renders without raising."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from zcu_tools.gui.app.dispersive.services.viz import (  # noqa: E402
    render_tune_figure,
    set_dispersion_lines,
    update_bare_line,
)


def _axes():
    fluxs = np.linspace(0.0, 1.0, 30).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, 20).astype(np.float64)
    norm = np.random.RandomState(0).rand(30, 20)
    return fluxs, freqs, norm


def test_render_tune_figure_draws_background_and_rf_only():
    fluxs, freqs, norm = _axes()
    fig = Figure()
    artists = render_tune_figure(fig, fluxs, freqs, norm, 5.3)
    assert artists.figure is fig
    assert artists.image is not None  # norm-phase background
    assert artists.line_bare is not None  # r_f line
    assert artists.line_ground is None  # no dispersion lines yet
    assert artists.g is None


def test_update_bare_line_moves_rf_in_place():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    update_bare_line(artists, 5.45)
    ydata = np.asarray(artists.line_bare.get_ydata())
    assert float(ydata[0]) == 5450.0  # MHz
    assert "5450.0" in artists.ax.get_title()


def test_set_dispersion_lines_adds_then_updates():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    t = fluxs[::2]
    rf0 = np.full(len(t), 5.4)
    rf1 = np.full(len(t), 5.7)

    set_dispersion_lines(artists, t, rf0, rf1, 0.06, 5.3)
    assert artists.line_ground is not None and artists.line_excited is not None
    assert artists.g == 0.06
    assert "g = 60.0" in artists.ax.get_title()

    # a second call updates in place (no new lines)
    g_line = artists.line_ground
    set_dispersion_lines(artists, t, rf0 + 0.1, rf1 + 0.1, 0.07, 5.35)
    assert artists.line_ground is g_line  # same artist, updated
    assert artists.g == 0.07

    # after a prediction, the r_f slider's title keeps g
    update_bare_line(artists, 5.40)
    assert "g = 70.0" in artists.ax.get_title()
