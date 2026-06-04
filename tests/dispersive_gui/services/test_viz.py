"""Tests for dispersive VizService — the tune figure renders without raising."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from zcu_tools.gui.app.dispersive.services.viz import (  # noqa: E402
    render_tune_figure,
    update_tune_lines,
)


def _axes():
    fluxs = np.linspace(0.0, 1.0, 30).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, 20).astype(np.float64)
    norm = np.random.RandomState(0).rand(30, 20)
    return fluxs, freqs, norm


def test_render_tune_figure_and_update_in_place():
    fluxs, freqs, norm = _axes()
    t = fluxs[::2]
    rf0 = np.full(len(t), 5.4)
    rf1 = np.full(len(t), 5.7)
    fig = Figure()

    artists = render_tune_figure(fig, fluxs, freqs, norm, t, rf0, rf1, 0.06, 5.3)
    assert artists.figure is fig
    # update in place with new lines (no raise, title updates)
    update_tune_lines(artists, t, rf0 + 0.1, rf1 + 0.1, 0.07, 5.35)
    assert "0.07" in artists.ax.get_title() or "70.0" in artists.ax.get_title()
