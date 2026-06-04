"""Tests for dispersive VizService — matplotlib renders draw without raising.

Uses Agg (headless) figures. ``calculate_chi_vs_flux`` (scqubits) is monkeypatched
for the dispersive-shift figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import zcu_tools.gui.app.dispersive.services.viz as viz_mod  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from zcu_tools.gui.app.dispersive.services.viz import (  # noqa: E402
    render_dispersive_shift,
    render_dispersive_with_onetone,
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


def test_render_dispersive_with_onetone_dejumps():
    fluxs, freqs, norm = _axes()
    t = np.linspace(0.0, 1.0, 50).astype(np.float64)
    # a line with a spurious single-point jump
    rf = np.linspace(5.2, 5.4, 50)
    rf[25] = 5.9  # spike → should be NaN'd by the de-jump logic
    fig = Figure()
    render_dispersive_with_onetone(fig, 5.3, 0.06, t, (rf,), fluxs, freqs, norm)
    assert len(fig.axes) == 1


def test_compute_chi_vs_flux_passthrough(monkeypatch):
    from zcu_tools.gui.app.dispersive.services.viz import compute_chi_vs_flux

    def fake_chi(params, fluxs, bare_rf, g, *, progress=False, res_dim=7):
        return np.outer(np.ones(len(fluxs)), np.arange(res_dim)) * 0.01

    monkeypatch.setattr(viz_mod, "calculate_chi_vs_flux", fake_chi)
    t = np.linspace(0.0, 1.0, 40).astype(np.float64)
    chi = compute_chi_vs_flux((4.0, 1.0, 0.5), t, 5.3, 0.06, upto=5)
    assert chi.shape == (40, 7)  # res_dim = upto + 2


def test_render_dispersive_shift_draws_precomputed_chi():
    # render takes a precomputed chi (the heavy compute is split out); no scqubits.
    t = np.linspace(0.0, 1.0, 40).astype(np.float64)
    chi = np.outer(np.ones(40), np.arange(7)) * 0.01
    fig = Figure()
    render_dispersive_shift(fig, (4.0, 1.0, 0.5), t, chi, upto=5)
    assert len(fig.axes) == 1
