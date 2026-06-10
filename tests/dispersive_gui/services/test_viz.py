"""Tests for dispersive VizService — the tune figure renders without raising."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from zcu_tools.gui.app.dispersive.services.viz import (  # noqa: E402
    add_sample_line,
    move_sample_line,
    remove_sample_line,
    render_tune_figure,
    set_dispersion_lines,
    update_bare_line,
    update_sample_dots,
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


# --- sample-flux lines -----------------------------------------------------


def test_add_sample_line_appends_vertical_line():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    s = add_sample_line(artists, 0.3)
    assert artists.samples == [s]
    assert s.flux == 0.3
    assert s.dot_ground is None and s.dot_excited is None
    # the line is vertical at x = flux
    xdata = np.asarray(s.line.get_xdata())
    assert float(xdata[0]) == 0.3 and float(xdata[1]) == 0.3


def test_move_sample_line_updates_flux_and_x():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    s = add_sample_line(artists, 0.3)
    move_sample_line(s, 0.45)
    assert s.flux == 0.45
    xdata = np.asarray(s.line.get_xdata())
    assert float(xdata[0]) == 0.45


def test_update_sample_dots_creates_then_moves_with_matching_colours():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    s = add_sample_line(artists, 0.3)

    update_sample_dots(artists, s, 5.41, 5.62)  # GHz
    assert s.dot_ground is not None and s.dot_excited is not None
    gy = np.asarray(s.dot_ground.get_ydata())
    ey = np.asarray(s.dot_excited.get_ydata())
    assert float(gy[0]) == 5410.0 and float(ey[0]) == 5620.0  # MHz
    # colours match the dispersion lines: ground = blue, excited = red
    assert s.dot_ground.get_color() == "blue"
    assert s.dot_excited.get_color() == "red"

    # second call moves them in place (same artists)
    g_dot = s.dot_ground
    update_sample_dots(artists, s, 5.40, 5.60)
    assert s.dot_ground is g_dot
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == 5400.0


def test_remove_sample_line_drops_line_and_dots():
    fluxs, freqs, norm = _axes()
    artists = render_tune_figure(Figure(), fluxs, freqs, norm, 5.3)
    s = add_sample_line(artists, 0.3)
    update_sample_dots(artists, s, 5.4, 5.6)
    remove_sample_line(artists, s)
    assert artists.samples == []
